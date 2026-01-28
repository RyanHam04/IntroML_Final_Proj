from dataclasses import dataclass
from math import log
from model import xgb, logreg


@dataclass(frozen=True)
class Experiment:
    id: str  # Name of the experiment
    model: str  # logreg | xgb
    config: object  # Model configuration (max_depth etc.)
    k: int  # Amount of folds in K-fold


EXPERIMENTS = [
    Experiment(id="baseline-lr", model="logreg", config=logreg.Config(), k=5),
    Experiment(id="xgb-k=10", model="xgb", config=xgb.Config(), k=10),
]
