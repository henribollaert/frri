import numpy as np


def balanced_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64 | np.ndarray:
    """
    Calculate the balanced accuracy score given the true labels and predicted labels.
    :param y_true: The true labels.
    :param y_pred: The predicted labels.

    :return: The balanced accuracy score.
    """
    classes = list(np.unique(np.append(y_true, y_pred)))

    y_true_prime = np.array([classes.index(label) for label in y_true])
    y_pred_prime = np.array([classes.index(label) for label in y_pred])

    # Count the number of occurrences of each label in y_true
    label_counts = np.bincount(y_true_prime)

    # Calculate the number of occurrences of each label in y_pred for each true label
    tp_counts = np.bincount(y_true_prime[y_pred_prime == y_true_prime], minlength=len(label_counts))

    # Calculate the balanced accuracy
    balanced_acc = np.mean(
        np.divide(tp_counts, label_counts, out=np.zeros_like(tp_counts, dtype='float64'), where=label_counts != 0),
        where=label_counts != 0
    )

    return balanced_acc
