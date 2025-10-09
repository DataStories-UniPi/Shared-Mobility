import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Protocol

from .models import TaskType


class Estimator(Protocol):
    """Protocol for estimator types"""

    def fit(self, X, y, **kwargs): ...

    def predict(self, X, **kwargs): ...


class BaseEvaluator(ABC):
    """Abstract base class for model evaluation components"""

    def __init__(self, task_type: TaskType, target_col: str):
        """Initialize evaluator"""
        self.task_type = task_type
        self.target_col = target_col

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        """Abstract method for evaluation"""
        raise NotImplementedError


@contextmanager
def timer():
    """Context manager for precise timing"""
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start
