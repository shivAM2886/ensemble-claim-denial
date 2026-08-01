import pandas as pd
from train import _prepare_data
from xgb_classifier import OptunaXGBClassifier


def predict_claims():
    curr_claims = pd.read_csv("data/current_claims.csv")
    xgb_model = OptunaXGBClassifier.load("xgb_classifier_chkpt")
    x = _prepare_data(curr_claims, keep_target=False)
    y_probs = xgb_model.predict_proba(x)[:, 1]

    curr_claims["denial_probability"] = y_probs
    curr_claims = curr_claims.sort_values(
        by="denial_probability", ascending=False
    ).reset_index(drop=True)
    curr_claims.to_csv("data/current_claims_with_denial_probs.csv", index=False)
