import shap
import preprocess
import pandas as pd
import matplotlib.pyplot as plt
from train import _prepare_data
from xgb_classifier import OptunaXGBClassifier

CAT_COLS = [
    "payer_id",
    "days_g30",
    "unauthorized_proc_flag",
    "missing_documentation_flag",
    "referral_required",
    "referral_required_but_not_present",
    "eligibility_not_verified_flag",
    "out_of_network_flag",
    "prior_auth_required",
    "has_prior_auth",
]
NUM_COLS = ["payment_gap"]


def plot_denial_rate_table(
    df: pd.DataFrame,
    target_col: str = "is_denied",
    cat_cols=CAT_COLS,
    num_cols=NUM_COLS,
    num_bins=5,
):
    cat_cols = cat_cols or []
    num_cols = num_cols or []
    rows = []

    def add(feature, key):
        s = (
            df.groupby(key, observed=False)[target_col]
            .agg(N="count", Denials="sum", Rate="mean")
            .reset_index()
        )
        for _, r in s.iterrows():
            rows.append(
                {
                    "Feature": feature,
                    "Bucket": str(r.iloc[0]),
                    "N (Claims)": int(r["N"]),
                    "Denials": int(r["Denials"]),
                    "Denial Rate": r["Rate"],
                }
            )

    for col in cat_cols:
        add(col, col)

    for col in num_cols:
        tmp = df[[target_col]].copy()
        tmp["__bin__"] = pd.qcut(df[col], q=num_bins, duplicates="drop").astype(str)
        add(col, tmp["__bin__"])

    out = pd.DataFrame(rows).sort_values(
        ["Feature", "Denial Rate"], ascending=[True, False]
    )
    return out.reset_index(drop=True)


def generate_shap_beeswarm_plot(xgb_model, x_val_data, output_path="data/shap_beeswarm.png"):
    print("Generating SHAP Beeswarm Summary Plot...")
    plt.figure()
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(x_val_data)
    shap.plots.beeswarm(shap_values, max_display=12, show=False)

    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Plot saved to {output_path}")


def plot():
    df = preprocess.load_train_data()
    df = df[df["split"] == "train"].copy()
    df = preprocess.add_engineered_features(df=df)

    feature_denial_rate_table = plot_denial_rate_table(df=df)
    feature_denial_rate_table.to_csv("data/feature_denial_rate_table.csv", index=False)

    data = preprocess.load_train_data()
    x_val, _ = _prepare_data(data[data["split"] == "validation"].copy())
    xgb_model = OptunaXGBClassifier.load("xgb_classifier_chkpt")
    generate_shap_beeswarm_plot(xgb_model=xgb_model.model, x_val_data=x_val)
