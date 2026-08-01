import numpy as np


def calculate_recall_at_top_k(y_true, y_probs, capacity_ratio=0.25):
    """
    Calculate recall among the top K highest predicted probability instances.

    Parameters
    ----------
    y_true : array-like
        Binary ground truth labels (1 for positive class, 0 otherwise).
    y_probs : array-like
        Predicted probabilities or confidence scores.
    capacity_ratio : float, default=0.25
        Fraction of total instances to include in top K (e.g., 0.25 for top 25%).

    Returns
    -------
    float
        Recall score within the top K predicted instances.
    """

    y_true = np.array(y_true)
    top_k_count = int(np.ceil(len(y_true) * capacity_ratio))
    sorted_indices = np.argsort(y_probs)[::-1][:top_k_count]

    total_denials = np.sum(y_true == 1)
    tp_top_k = np.sum(y_true[sorted_indices] == 1)
    recall_at_top_k = tp_top_k / total_denials if total_denials > 0 else 0.0
    return recall_at_top_k
