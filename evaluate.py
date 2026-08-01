import preprocess
import pandas as pd
from train import _prepare_data
from metrics import calculate_recall_at_top_k
from xgb_classifier import OptunaXGBClassifier


def analyze_false_negatives(x, y_probs, y_true):
    x["probs"] = y_probs
    x["is_denied"] = y_true.values
    x = x.sort_values(by="probs", ascending=False)
    x["rank"] = [i + 1 for i in range(len(x))]
    x = x[int(len(x) * 0.25) :]
    x = x[x["is_denied"] == 1]
    denial_reason_counts = x["denial_reason"].value_counts()
    denial_reason_perc = round(
        (denial_reason_counts * 100) / denial_reason_counts.sum(), 2
    )
    denial_reason_df = pd.DataFrame(
        {
            "denial_reason": denial_reason_counts.index,
            "count": denial_reason_counts.values,
            "count_perc": denial_reason_perc.values,
        }
    )
    return denial_reason_df


def evaluate_model():
    data = preprocess.load_train_data()

    x_train, y_train = _prepare_data(
        data[data["split"] == "train"].copy(), additional_cols=["denial_reason"]
    )
    x_val, y_val = _prepare_data(
        data[data["split"] == "validation"].copy(), additional_cols=["denial_reason"]
    )
    x_test, y_test = _prepare_data(
        data[data["split"] == "test"].copy(), additional_cols=["denial_reason"]
    )

    model = OptunaXGBClassifier.load("xgb_classifier_chkpt")

    y_train_probs = model.predict_proba(x_train.drop(columns=["denial_reason"]))[:, 1]
    y_val_probs = model.predict_proba(x_val.drop(columns=["denial_reason"]))[:, 1]
    y_test_probs = model.predict_proba(x_test.drop(columns=["denial_reason"]))[:, 1]

    train_recall = calculate_recall_at_top_k(y_true=y_train, y_probs=y_train_probs)
    val_recall = calculate_recall_at_top_k(y_true=y_val, y_probs=y_val_probs)
    test_recall = calculate_recall_at_top_k(y_true=y_test, y_probs=y_test_probs)

    print(f"Training recall@25%: {train_recall:.4f}")
    print(f"Validation recall@25%: {val_recall:.4f}")
    print(f"Test recall@25%: {test_recall:.4f}")

    drf = []
    for split, x, y_probs, y_true in [
        ("train", x_train, y_train_probs, y_train),
        ("validation", x_val, y_val_probs, y_val),
        ("test", x_test, y_test_probs, y_test),
    ]:
        denial_reason_df = analyze_false_negatives(x, y_probs, y_true)
        denial_reason_df.insert(0, "split", split)
        drf.append(denial_reason_df)
    drf = pd.concat(drf, ignore_index=True)
    drf.to_csv("data/false_negatives_analysis.csv", index=False)
