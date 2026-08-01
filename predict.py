import pandas as pd
from train import _prepare_data
from xgb_classifier import OptunaXGBClassifier


def predict_claims():
    curr_claims = pd.read_csv("data/current_claims.csv")
    xgb_model = OptunaXGBClassifier.load("xgb_classifier_chkpt")
    x = _prepare_data(curr_claims, keep_target=False, additional_cols=["claim_id"])
    y_probs = xgb_model.predict_proba(x.drop(columns=["claim_id"]))[:, 1]
    x["predicted_denial_probability"] = y_probs
    
    col_mapping = {}
    for col in x.columns:
        if col not in ["claim_id", "predicted_denial_probability"]:
            col_mapping[col] = f"features_{col}"
    x = x.rename(columns=col_mapping)
    
    curr_claims = pd.merge(curr_claims, x, on="claim_id", how="left")
    curr_claims = curr_claims.sort_values(
        by="predicted_denial_probability", ascending=False
    ).reset_index(drop=True)
    curr_claims.to_csv("data/current_claims_with_denial_probs.csv", index=False)
