from collections.abc import Callable
from typing import Any, ClassVar, Dict, Optional, cast

from loguru import logger
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

from config.constants import RANDOM_SEED
from utils.models import EstimatorType, TaskType, Trainable

from .multioutput import MultiOutputXGBClassifier, MultiOutputXGBRegressor


class ModelFactory:
    """
    Facilitate the construction of regression and classification models
    with optional multi-output support.

    This class provides functionality to resolve and build models from either
    aliases, callables, or existing instances. It supports wrapping models with
    MultiOutputRegressor/MultiOutputClassifier when dealing with multi-output tasks.
    """

    __estimator_registry: ClassVar[Dict[str, Dict[str, Callable]]] = {
        "regression": {
            "xgb": lambda params: XGBRegressor(n_jobs=-1, random_state=RANDOM_SEED, **params),
            "rf": lambda params: RandomForestRegressor(
                criterion="absolute_error", n_jobs=-1, random_state=RANDOM_SEED, **params
            ),
        },
        "classification": {
            "xgb": lambda params: XGBClassifier(
                n_jobs=-1, random_state=RANDOM_SEED, **params
            ),
            "rf": lambda params: RandomForestClassifier(
                criterion="gini", n_jobs=-1, random_state=RANDOM_SEED, **params
            ),
        },
    }

    def __init__(
        self,
        task: TaskType = TaskType.REGRESSION,
        num_output: int = 1,
        verbose: bool = False,
        per_output: bool = False,
    ):
        """
        Initialize the ModelFactory class

        Args:
            task (default="regression"): The type of task
            num_output (default=1): The number of outputs required for prediction.
            verbose (default=False): Whether to enable verbose logging. Defaults to False.
            per_output (default=False): Whether to create a separate model for each output.
        """
        self.task = task
        self.verbose = verbose

        self.needs_multi_output_ = num_output > 1 and per_output

        logger.debug(f"Factory initialized with {task=}, {num_output=}")

    def _validate_estimator_name(self, estimator: str) -> None:
        """
        Validate that the estimator name exists in the registry.

        Args:
            estimator : The name of the estimator to validate.

        Raises:
            ValueError: If the estimator name is not found in the registry.
        """

        if estimator not in self.__estimator_registry[self.task]:
            available_models = list(self.__estimator_registry[self.task].keys())
            raise ValueError(
                f"Unknown estimator alias '{estimator}'. Available models: {available_models}"
            )

    def _resolve_estimator_from_name(
        self,
        estimator: str,
        estimator_params: Dict[str, Any],
    ) -> Trainable:
        """
        Resolve and instantiate an estimator from its name.

        Args:
            estimator : The name of the estimator to resolve.
            estimator_params : Parameters to pass to the estimator constructor.

        Returns:
            The instantiated estimator.
        """

        if self.verbose:
            logger.debug(f"Resolving '{estimator}' with params")

        # Cast the callable to avoid type errors
        factory_fn = cast(
            Callable[..., Trainable],
            self.__estimator_registry[self.task][estimator],
        )
        return factory_fn(estimator_params)

    def _resolve_estimator_from_callable(
        self,
        estimator: Callable,
        estimator_params: Dict[str, Any],
    ) -> Trainable:
        """
        Resolve and instantiate an estimator from a callable.

        Args:
            estimator : The callable to use for creating the estimator.
            estimator_params : Parameters to pass to the estimator constructor.

        Returns:
            The instantiated estimator.
        """
        if self.verbose:
            logger.info("Resolving estimator from provided callable.")
        return estimator(estimator_params)

    def _resolve_estimator_from_instance(
        self,
        estimator: BaseEstimator,
    ) -> Trainable:
        """
        Resolve and clone an estimator from an existing instance.

        Args:
            estimator : The estimator instance to clone.

        Returns:
            The cloned estimator.
        """
        return clone(estimator)

    def _resolve_estimator(
        self,
        estimator: EstimatorType,
        estimator_params: Optional[Dict[str, Any]] = None,
    ) -> Trainable:
        """
        Resolve and instantiate the estimator based on its type.

        Args:
            estimator (EstimatorType): Estimator alias (str), callable, or instance.
            estimator_params (Optional[Dict[str, Any]], optional): Parameters to pass
                to the estimator constructor. Defaults to None.

        Returns:
            Union[RegressorMixin, ClassifierMixin]: The resolved estimator.

        Raises:
            ValueError: If the estimator is invalid.
        """
        estimator_params = estimator_params or {}

        if isinstance(estimator, str):
            self._validate_estimator_name(estimator)
            return self._resolve_estimator_from_name(estimator, estimator_params)

        if callable(estimator):
            return self._resolve_estimator_from_callable(estimator, estimator_params)

        if isinstance(estimator, BaseEstimator):
            return self._resolve_estimator_from_instance(estimator)

        raise ValueError(
            "Invalid estimator. Must be a string alias, callable, or BaseEstimator instance."
        )

    def _wrap_with_multi_output_if_needed(self, model: Trainable) -> Trainable:
        """
        Wrap the model with a multi-output wrapper if needed.

        Args:
            model : The model to potentially wrap.

        Returns:
            The model, possibly wrapped in a multi-output wrapper.
        """
        if not self.needs_multi_output_:
            return model

        # Check if the model is already a multi-output model
        model_name = model.__class__.__name__.lower()
        is_already_multi_output = any(x in model_name for x in ["multoutput", "multioutput"])

        if is_already_multi_output:
            if self.verbose:
                logger.debug("Model already supports multi-output. Using as-is.")
            return model

        # Determine the appropriate wrapper class
        if isinstance(model, RegressorMixin):
            wrapper_class = MultiOutputXGBRegressor
        else:  # ClassifierMixin
            wrapper_class = MultiOutputXGBClassifier

        if self.verbose:
            logger.info(f"Wrapping model with {wrapper_class.__name__}")

        # Cast the model to BaseEstimator to satisfy type checking
        return wrapper_class(cast(BaseEstimator, model))

    def build_model(
        self,
        estimator: EstimatorType = "rf",
        estimator_params: Optional[Dict[str, Any]] = None,
    ) -> Trainable:
        """
        Build a model component, wrapping it in a multi-output wrapper if necessary.

        Args:
            estimator (optional, default="rf"): Estimator alias (str), callable, or instance.
            estimator_params (optional, default=None): Parameters to pass to the
                estimator constructor

        Returns:
            ModelThe built model, possibly wrapped in a multi-output wrapper.
        """
        estimator_params = estimator_params or {}
        model = self._resolve_estimator(estimator, estimator_params)

        if not hasattr(model, "fit") or not hasattr(model, "predict"):
            raise TypeError("Estimator must implement 'fit' and 'predict' methods")

        return self._wrap_with_multi_output_if_needed(model)
