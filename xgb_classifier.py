import os
import json
import optuna
import xgboost as xgb
from metrics import calculate_recall_at_top_k


class OptunaXGBClassifier:
    """
    Wrapper class that combines Optuna hyperparameter optimization
    and XGBoost classification targeting top-k recall.

    Early stopping is driven by recall@top-k (the metric we optimize),
    NOT logloss. Supports saving/loading the trained model + config.
    """

    def __init__(
        self,
        n_trials=100,
        capacity_pct=0.25,
        random_state=42,
        n_estimators=1000,
        early_stopping_rounds=30,
        n_jobs=-1,
    ):
        self.n_trials = n_trials
        self.capacity_pct = capacity_pct
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.n_jobs = n_jobs

        self.fixed_params = {
            "scale_pos_weight": 1.0,
            "random_state": self.random_state,
            "tree_method": "hist",
            "enable_categorical": True,
            "n_jobs": self.n_jobs,
        }

        self.study = None
        self.best_params = None
        self.model = None

    def _recall_at_k_metric(self, y_true, y_pred):
        """
        Custom eval metric. Returns -recall (XGBoost minimizes custom metrics).
        """
        recall = calculate_recall_at_top_k(
            y_true=y_true, y_probs=y_pred, capacity_ratio=self.capacity_pct
        )
        return -recall

    def _make_estimator(self, params, n_estimators):
        """
        Build an XGBClassifier wired with the custom recall metric + early stopping.
        """
        return xgb.XGBClassifier(
            **params,
            n_estimators=n_estimators,
            eval_metric=self._recall_at_k_metric,
            early_stopping_rounds=self.early_stopping_rounds,
        )

    def _objective(self, trial, X_train, y_train, X_val, y_val):
        params = {
            **self.fixed_params,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 40),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
        }

        model = self._make_estimator(params, n_estimators=self.n_estimators)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        val_probs = model.predict_proba(X_val)[:, 1]
        return calculate_recall_at_top_k(
            y_true=y_val, y_probs=val_probs, capacity_ratio=self.capacity_pct
        )

    def fit_and_tune(self, X_train, y_train, X_val, y_val, verbose=50):
        """
        Runs hyperparameter optimization with Optuna and trains the final model.
        """
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # FIX: seed Optuna's sampler so trial suggestions are reproducible.
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        self.study = optuna.create_study(direction="maximize", sampler=sampler)

        self.study.optimize(
            lambda trial: self._objective(trial, X_train, y_train, X_val, y_val),
            n_trials=self.n_trials,
            show_progress_bar=True,
        )

        print("\n" + "=" * 50)
        print(
            f"Best Validation Recall @ Top {int(self.capacity_pct*100)}%: {self.study.best_value * 100:.2f}%"
        )
        print("=" * 50)

        # Ensure random_state is always present in the final params.
        self.best_params = {
            **self.fixed_params,
            **self.study.best_params,
            "random_state": self.random_state,
        }

        print("Training final model with best hyperparameters...")
        self.model = self._make_estimator(
            self.best_params, n_estimators=self.n_estimators
        )
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=verbose,
        )
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    # ---------- Save / Load ----------

    def save(self, dir_path):
        """
        Persist the trained model + config to `dir_path`.
        Writes:
            - model.json    (XGBoost booster, portable & version-stable)
            - config.json   (wrapper config + best_params)
        """
        if self.model is None:
            raise RuntimeError("No trained model to save. Call fit_and_tune first.")

        os.makedirs(dir_path, exist_ok=True)

        # Use XGBoost's native serialization (recommended over pickle).
        self.model.save_model(os.path.join(dir_path, "model.json"))

        # best_params may contain the callable eval_metric — strip it before JSON.
        serializable_params = {
            k: v for k, v in (self.best_params or {}).items() if not callable(v)
        }

        config = {
            "n_trials": self.n_trials,
            "capacity_pct": self.capacity_pct,
            "random_state": self.random_state,
            "n_estimators": self.n_estimators,
            "early_stopping_rounds": self.early_stopping_rounds,
            "n_jobs": self.n_jobs,
            "best_params": serializable_params,
            "best_value": self.study.best_value if self.study is not None else None,
        }
        with open(os.path.join(dir_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        print(f"Saved model and config to: {dir_path}")

    @classmethod
    def load(cls, dir_path):
        """
        Reconstruct an OptunaXGBClassifier from a saved directory.
        """
        with open(os.path.join(dir_path, "config.json"), "r") as f:
            config = json.load(f)

        obj = cls(
            n_trials=config["n_trials"],
            capacity_pct=config["capacity_pct"],
            random_state=config["random_state"],
            n_estimators=config["n_estimators"],
            early_stopping_rounds=config["early_stopping_rounds"],
            n_jobs=config["n_jobs"],
        )
        obj.best_params = config.get("best_params")

        # Rebuild the estimator shell and load the booster weights.
        model = xgb.XGBClassifier()
        model.load_model(os.path.join(dir_path, "model.json"))
        obj.model = model

        print(f"Loaded model and config from: {dir_path}")
        return obj
