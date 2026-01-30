from xgboost import XGBClassifier
from model import xgb
from sklearn.model_selection import GridSearchCV
from pprint import pprint


param_grid = {
    "model__max_depth": [],
    "model__min_child_weight": [],
    "model__subsample": [],
    "model__colsample_bytree": [],
    "model__learning_rate": [],
}


def get_metrics(results, best_idx):
    print(results)
    return {
        "recall_mean": results["mean_test_recall"][best_idx],
        "recall_std": results["std_test_recall"][best_idx],
        "roc_auc_mean": results["mean_test_roc_auc"][best_idx],
        "roc_auc_std": results["std_test_roc_auc"][best_idx],
        "f1_mean": results["mean_test_f1"][best_idx],
        "f1_std": results["std_test_f1"][best_idx],
    }


def finetune(data) -> XGBClassifier:
    # finetunes the model using the parameters, prioritizes recall
    p = xgb.build_pipeline()
    grid = GridSearchCV(
        estimator=p,
        param_grid=param_grid,
        scoring={"f1": "f1", "recall": "recall", "roc_auc": "roc_auc"},
        refit="recall",
        cv=5,
        n_jobs=-1,  # Uses all cpu cores
    )

    grid.fit(data["X_train"], data["y_train"])
    best_idx = grid.best_index_
    metrics = get_metrics(grid.cv_results_, best_idx)

    print("-----------------Best parms-------------")
    pprint(grid.best_params_)
    print("------------------Metrics---------------")
    pprint(metrics)

    return grid.best_estimator_
