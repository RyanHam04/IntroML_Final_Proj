from data import dataloader
from preprocess import preprocess

from sklearn.metrics import accuracy_score


def baseline(data):
    "Runs a baseline Logistic Regression Model"
    from model.logreg import Config, build_pipeline

    cfg = Config()
    lr = build_pipeline("logreg_base_config", cfg)
    lr.fit(data["X_train"], data["y_train"])
    iterations = lr.named_steps["model"].n_iter_[0]
    yhat = lr.predict(data["X_val"])

    print("------------BASELINE LOGREG-------------")
    print(f"Iterations needed to converge: {iterations}")
    print(data["y_val"].value_counts())
    print(f"Accuracy: {accuracy_score(data['y_val'], yhat)}")


def xgb(data):
    from model.xgb import Config, build_pipeline

    """
    We probably should give each run a unique name? 
    Regardless, I added the `run` attribute    
    """

    cfg = Config()
    xgb = build_pipeline("xgb_default_idk", cfg)
    xgb.fit(data["X_train"], data["y_train"])
    yhat = xgb.predict(data["X_val"])

    print(f"-----------XGB {xgb.run}-------------")
    print(data["y_val"].value_counts())
    print(f"Accuracy: {accuracy_score(data['y_val'], yhat)}")


def main():
    df = dataloader.load_dataset()
    data = preprocess.split_data(df, 0.7, 0.2, 0.1)
    baseline(data)
    xgb(data)


if __name__ == "__main__":
    main()
