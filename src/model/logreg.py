import pandas as pd
from sklearn.linear_model import LogisticRegression
from dataclasses import dataclass


@dataclass
class Config:
    """
    Contains Model Hyperparameters
    Defaults are just placeholders
    """

    max_iter: int = 5000
    class_weight: str = "balanced"  # As labels are imbalanced (2:1)


class Model:
    def __init__(self, cfg):
        self.cfg: Config = cfg
        self.model: LogisticRegression = self.build_model()

    def build_model(self) -> LogisticRegression:
        return LogisticRegression(**vars(self.cfg))

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)


if __name__ == "__main__":
    conf1 = Config()
    Model(cfg=conf1)
