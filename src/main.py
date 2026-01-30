from math import log
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score
from typing import List
from data import load_dataset, split_data, kfold_data, validate_dataset
from tuning import finetune
from model import xgb, logreg
import numpy as np


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
    finetune(cv_data)


if __name__ == "__main__":
    main()
