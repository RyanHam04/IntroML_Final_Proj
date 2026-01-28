from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class Config:
    """
    Contains Model Hyperparameters
    Defaults are just placeholders
    """

    max_depth: int = 1
    min_child_weight: int = 1
    gamma: float = 1
    learning_rate: float = 0.5
    n_estimators: int = 1
    objective: str = "binary:logistic"


def build_pipeline(id: str, cfg: Config):
    p = Pipeline(
        steps=[("scaler", StandardScaler()), ("model", XGBClassifier(**vars(cfg)))]
    )
    p.id = id
    return p


if __name__ == "__main__":
    pass
