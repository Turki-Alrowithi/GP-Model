"""Model training and evaluation wrappers."""

from gpmodel.training.evaluator import EvalResult, evaluate
from gpmodel.training.trainer import TrainResult, train

__all__ = ["EvalResult", "TrainResult", "evaluate", "train"]
