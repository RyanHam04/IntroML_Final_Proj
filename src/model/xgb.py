import pandas as pd
from xgboost import XGBClassifier
from dataclasses import dataclass


@dataclass
class Config:
    """
    Contains Model Hyperparams
    Defaults are just placeholders
    """

    max_depth: int = 1
    min_child_weight: int = 1
    gamma: int = 1
    learning_rate: float = 0.5
    n_estimators: int = 1
    max_cat_threshold: float = 0.1
    objective: str = "binary:logistic"


class Model:
    def __init__(self, cfg):
        self.cfg: Config = cfg
        self.model: XGBClassifier = self.build_model()

    def build_model(self) -> XGBClassifier:
        return XGBClassifier(**vars(self.cfg))

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)


if __name__ == "__main__":
    conf1 = Config()
