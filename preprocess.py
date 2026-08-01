import pandas as pd
from collections import Counter
from typing import List, Optional
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

FEATURES_DICT = {
    "class_features": ["payer_id"],
    "binary_features": [
        "days_g30",
        "referral_required",
        "out_of_network_flag",
        "unauthorized_proc_flag",
        "missing_documentation_flag",
        "eligibility_not_verified_flag",
        "referral_required_but_not_present",
    ],
    "numeric_features": ["payment_gap"],
}


def load_train_data() -> pd.DataFrame:
    data = pd.read_csv("data/claims_history.csv")
    data["service_month"] = pd.to_datetime(data["service_month"], format="%Y-%m")

    print(f"Data shape: {data.shape}")
    print(f"Data Splits: {Counter(data['split'])}\n")
    return data


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates interaction feature variables for claim processing models.
    """
    df = df.copy()

    df["unauthorized_proc_flag"] = (
        (df["prior_auth_required"] == 1) & (df["has_prior_auth"] == 0)
    ).astype(int)

    df["referral_required_but_not_present"] = (
        (df["referral_required"] == 1) & (df["referral_present"] == 0)
    ).astype(int)

    df["days_g30"] = (df["days_to_submit"] >= 30).astype(int)
    df["payment_gap"] = df["total_billed"] - df["expected_payment"]
    df["out_of_network_flag"] = (df["is_in_network"] == 0).astype(int)
    df["eligibility_not_verified_flag"] = (df["eligibility_verified"] == 0).astype(int)

    return df


def preprocess_model_features(
    df: pd.DataFrame,
    additional_cols: Optional[List[str]] = None,
    keep_target: bool = True,
):
    all_cols = (
        FEATURES_DICT["class_features"]
        + FEATURES_DICT["binary_features"]
        + FEATURES_DICT["numeric_features"]
    )

    if keep_target:
        all_cols.append("is_denied")

    if additional_cols:
        all_cols += additional_cols

    df = df[all_cols].copy()

    # cast categorical columns safely
    for col in FEATURES_DICT["class_features"]:
        df[col] = df[col].astype("category")

    # scale numeric columns
    ct = ColumnTransformer(
        transformers=[("num", StandardScaler(), FEATURES_DICT["numeric_features"])],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    ct.set_output(transform="pandas")
    df = ct.fit_transform(df)
    return df
