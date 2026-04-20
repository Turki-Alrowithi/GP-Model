"""Model training and evaluation wrappers."""

from gpmodel.training.evaluator import EvalResult, evaluate
from gpmodel.training.merge import ClassMap, MergeStats, load_class_map, merge_datasets
from gpmodel.training.trainer import TrainResult, train

__all__ = [
    "ClassMap",
    "EvalResult",
    "MergeStats",
    "TrainResult",
    "evaluate",
    "load_class_map",
    "merge_datasets",
    "train",
]
