from data import load_dataset, split_data, kfold_data, validate_dataset
from evaluate import evaluate_on_test
from tuning import finetune
from model import logreg


def load_data(kfold: bool):
    "Loads data"
    df = load_dataset()
    validate_dataset(df)
    data = split_data(df, 0.7, 0.2, 0.1)
    return kfold_data(data) if kfold else data


def main():
    data = load_data(kfold=False)
    cv_data = load_data(kfold=True)

    lr_base = logreg.run_baseline(data)
    logreg.run_baseline_cv(data, cv_splits=5)  # To determine overfitting

    finetune(cv_data, idx="1")  # Finetunes on initial/broad parameter set
    xgb_model = finetune(cv_data, idx="2")
    evaluate_on_test(lr_base, data, "Logistic Regression")
    evaluate_on_test(xgb_model, data, "XGB")


if __name__ == "__main__":
    main()
