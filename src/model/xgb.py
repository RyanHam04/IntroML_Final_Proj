from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline


def build_pipeline():
    p = Pipeline(steps=[("model", XGBClassifier())])
    return p


if __name__ == "__main__":
    pass
