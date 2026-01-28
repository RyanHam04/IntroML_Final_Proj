from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score
from typing import List
from data import load_dataset, split_data, kfold_data, validate_dataset
from experiment import Experiment, EXPERIMENTS
from model import xgb, logreg
from pprint import pprint
import numpy as np


def load_data():
    "Loads data"
    df = load_dataset()
    validate_dataset(df)
    return split_data(df, 0.7, 0.2, 0.1)


def k_fold_cv(p: Pipeline, k: int, data):
    data = kfold_data(data)
    cv = StratifiedKFold(n_splits=k)  # Ensuring each fold has proper proportions
    return cross_validate(
        p,
        data["X_train"],
        data["y_train"],
        cv=cv,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
    )

def summarize_cv_scores(scores: dict)-> dict:
    """
    Computes the mean and std for each metric 
    """
    summary = {}

    for key, value in scores.items():
        if key.startswith("test_"):
            metric = key.replace("test_","")
            summary[metric] = {
                "mean": np.mean(value),
                "std": np.std(value),
            }
    return summary

def run_experiments(experiments: List[Experiment], data):
    scores = None
    for exp in experiments:
        if exp.model == "logreg":
            model = logreg.build_pipeline(exp.id, exp.config)
            scores = k_fold_cv(model, exp.k, data)
        elif exp.model == "xgb":
            model = xgb.build_pipeline(exp.id, exp.config)
            scores = k_fold_cv(model, exp.k, data)

        summary = summarize_cv_scores(scores)

        pprint(f"------------{exp.id}----------------- \n {scores}")
        for metric, stats in summary.items():
            print(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")


def main():
    data = load_data()
    run_experiments(EXPERIMENTS, data)


if __name__ == "__main__":
    main()
