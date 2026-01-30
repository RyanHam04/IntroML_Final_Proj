from math import log
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score
from typing import List
from data import load_dataset, split_data, kfold_data, validate_dataset
from tuning import finetune
from model import xgb, logreg
import numpy as np


param_grid_1 = {
    "model__max_depth": [2, 3, 4, 5, 6, 7, 8],
    "model__min_child_weight": [1, 3, 5, 7],
    "model__subsample": [0.1, 0.3, 0.5, 0.7],
    "model__colsample_bytree": [0.1, 0.3, 0.5, 0.7],
    "model__learning_rate": [0.01, 0.03, 0.1, 0.3],
}


# New experiment surrouning the previous optimal hyperparms, more refined
param_grid_2 = {
    "model__max_depth": [2, 3, 4],
    "model__min_child_weight": [4, 5, 6],
    "model__subsample": [0.6, 0.65, 0.7, 0.75, 0.8],
    "model__colsample_bytree": [0.6, 0.65, 0.7, 0.75, 0.8],
    "model__learning_rate": [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
}


def load_data(kfold: bool):
    "Loads data"
    df = load_dataset()
    validate_dataset(df)
    data = split_data(df, 0.7, 0.2, 0.1)
    return kfold_data(data) if kfold else data


def main():
    data = load_data(kfold=False)
    cv_data = load_data(kfold=True)

    logreg.run_baseline(data, cv_splits=5)
    finetune(cv_data, param_grid_1)
    finetune(cv_data, param_grid_2)


if __name__ == "__main__":
    main()
