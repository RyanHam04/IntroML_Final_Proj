from data import dataloader
from preprocess import preprocess

from sklearn.metrics import accuracy_score


def baseline(data):
    from model.logreg import Model, Config

    cfg1 = Config()
    lr = Model(cfg=cfg1)
    lr.fit(data["X_train"], data["y_train"])
    yhat = lr.predict(data["X_val"])
    print(f"Acc: {accuracy_score(data['y_val'], yhat)}")


def main():
    df = dataloader.load_dataset()
    data = preprocess.split_data(df, 0.7, 0.2, 0.1)
    baseline(data)


if __name__ == "__main__":
    main()
