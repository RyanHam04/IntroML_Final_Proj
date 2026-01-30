from xgboost import XGBClassifier
from model import xgb
from sklearn.model_selection import GridSearchCV
from pprint import pprint
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import StratifiedKFold


param_grid = {
    "model__max_depth": [i * 2 for i in range(1, 6)],
    "model__min_child_weight": [i * 2 for i in range(1, 6)],
    "model__subsample": [0.2 * i for i in range(1, 6)],
    "model__colsample_bytree": [0.2 * i for i in range(1, 6)],
    "model__learning_rate": [0.01, 0.03, 0.1, 0.3],
}


def get_metrics(results, best_idx):
    return {
        "accuracy_mean": results["mean_test_accuracy"][best_idx],
        "accuracy_std": results["std_test_accuracy"][best_idx],
        "precision_mean": results["mean_test_precision"][best_idx],
        "precision_std": results["std_test_precision"][best_idx],
        "recall_mean": results["mean_test_recall"][best_idx],
        "recall_std": results["std_test_recall"][best_idx],
        "roc_auc_mean": results["mean_test_roc_auc"][best_idx],
        "roc_auc_std": results["std_test_roc_auc"][best_idx],
        "f1_mean": results["mean_test_f1"][best_idx],
        "f1_std": results["std_test_f1"][best_idx],
    }


def finetune(data) -> XGBClassifier:
    p = xgb.build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=p,
        param_grid=param_grid,
        scoring={
            "accuracy": "accuracy",
            # This adjustment is needed to prevent div_by_0 errors
            "precision": make_scorer(precision_score, zero_division=0),
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
        },
        refit="recall",  # As we optimize for recall, might change to roc_auc though
        cv=cv,
        n_jobs=-1,  # Uses all cpu cores
        verbose=0,
    )

    grid.fit(data["X_train"], data["y_train"])
    best_idx = grid.best_index_
    metrics = get_metrics(grid.cv_results_, best_idx)

    print("-----------------Best parms-------------")
    pprint(grid.best_params_)
    print("------------------Metrics---------------")
    pprint(metrics)

    return grid.best_estimator_
