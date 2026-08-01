import preprocess
import pandas as pd
from xgb_classifier import OptunaXGBClassifier


def _prepare_data(
    df: pd.DataFrame, additional_cols: list = None, keep_target: bool = True
):
    df = df.copy()
    df = preprocess.add_engineered_features(df=df)
    df = preprocess.preprocess_model_features(
        df=df, keep_target=keep_target, additional_cols=additional_cols
    ).reset_index(drop=True)
    if keep_target:
        return df.drop(columns=["is_denied"]), df["is_denied"]
    else:
        return df


def train_model():

    data = preprocess.load_train_data()

    x_train, y_train = _prepare_data(data[data["split"] == "train"].copy())
    x_val, y_val = _prepare_data(data[data["split"] == "validation"].copy())

    op_xgb_classifier = OptunaXGBClassifier()
    op_xgb_classifier.fit_and_tune(
        X_train=x_train, y_train=y_train, X_val=x_val, y_val=y_val, verbose=False
    )
    op_xgb_classifier.save("xgb_classifier_chkpt")
