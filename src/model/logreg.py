from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, StratifiedKFold
from data import kfold_data
from evaluate import evaluate


def build_pipeline():
    p = Pipeline(steps=[("scaler", StandardScaler()), ("model", LogisticRegression())])
    return p


def run_baseline(data):
    """
    Runs baseline LR on both the train/val/test set
    as well as a K-fold baseline LR
    """

    p = build_pipeline()
    p.fit(data["X_train"], data["y_train"])

    yval_pred = p.predict(data["X_val"])
    yval_proba = p.predict_proba(data["X_val"])[:, 1]

    print("-----------LogReg Baseline [Normal]--------------")
    evaluate(data["y_val"], yval_pred, yval_proba)
    return p


def run_baseline_cv(data, cv_splits=5):
    p = build_pipeline()
    cv_data = kfold_data(data)

    kf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "pr_auc": "average_precision",
    }

    cv_results = cross_validate(
        p,
        cv_data["X_train"],
        cv_data["y_train"],
        cv=kf,
        scoring=scoring,
        return_train_score=False,
    )

    print(f"--------LogReg Baseline [K-fold (k={cv_splits})]---------")

    for metric in scoring.keys():
        scores = cv_results[f"test_{metric}"]
        print(f"{metric}: {scores.mean():.4f} ± {scores.std():.4f}")

    return p


if __name__ == "__main__":
    pass
