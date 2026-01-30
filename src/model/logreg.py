from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from data import kfold_data

from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
)


def build_pipeline():
    p = Pipeline(steps=[("scaler", StandardScaler()), ("model", LogisticRegression())])
    return p


def evaluate(y, yhat, y_proba):
    f1 = f1_score(y, yhat)
    recall = recall_score(y, yhat)
    precision = precision_score(y, yhat)
    roc = roc_auc_score(y, y_proba)
    precision_curve, recall_curve, _ = precision_recall_curve(y, y_proba)
    pr_auc = auc(recall_curve, precision_curve)

    print(f"""
            F1 Score: {f1:.4f}
            Precision: {precision:.4f}
            Recall: {recall:.4f}
            ROC-AUC: {roc:.4f}
            """)


def run_baseline(data, cv_splits=5):
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

    cv_data = kfold_data(data)

    kf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    y_cv_pred = cross_val_predict(p, cv_data["X_train"], cv_data["y_train"], cv=kf)
    y_cv_proba = cross_val_predict(
        p, cv_data["X_train"], cv_data["y_train"], cv=kf, method="predict_proba"
    )[:, 1]

    print(f"--------LogReg Baseline [K-fold (k={cv_splits})]---------")
    evaluate(cv_data["y_train"], y_cv_pred, y_cv_proba)


if __name__ == "__main__":
    pass
