from sklearn.linear_model import LogisticRegression
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class Config:
    """
    Contains Model Hyperparameters
    Defaults are just placeholders
    """

    max_iter: int = 100
    class_weight: str = "balanced"  # As labels are imbalanced (2:1)


def build_pipeline(id: str, cfg: Config):
    p = Pipeline(
        steps=[("scaler", StandardScaler()), ("model", LogisticRegression(**vars(cfg)))]
    )
    p.name = id
    return p


if __name__ == "__main__":
    pass
