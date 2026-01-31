from sklearn.metrics import (
    average_precision_score,
    f1_score,
    recall_score,
    precision_score,
    accuracy_score,
)


def evaluate(y_true, y_pred, y_proba=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "pr_auc": average_precision_score(y_true, y_pred),
    }

    if y_proba is not None:
        metrics["pr_auc"] = average_precision_score(y_true, y_proba)

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


def evaluate_on_test(model, data, name):
    X_test = data["X_test"]
    y_test = data["y_test"]

    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]

    print(f"----------- {name} [Test Set] -----------")
    evaluate(y_test, y_pred, y_proba)
