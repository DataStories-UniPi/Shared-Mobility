from collections.abc import Iterable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context, check_is_fitted
from sklearn.utils._param_validation import HasMethods
from sklego.preprocessing import RepeatingBasisFunction


class RBFTransformer(TransformerMixin, BaseEstimator):
    """
    Repeating Basis Function transformer for time-series features.

    This transformer applies a RepeatingBasisFunction independently to each feature in the
    input data. The transformer takes in a list of feature names, a list of number of periods
    for each feature's basis functions, and a list of (min, max) input ranges for each
    feature's basis functions.
    The transformer is compatible with scikit-learn ColumnTransformer.

    Parameters
    ----------
    features : list[str]
        List of feature names to transform.
    n_periods : list[int]
        List of number of periods for each feature's basis functions.
    input_ranges : list[tuple[int, int]]
        List of (min, max) input ranges for each feature's basis functions.
    def __init__(
        self,
        features: list[str],
        n_periods: list[int],
        input_ranges: list[tuple[int, int]],
    ):
        if not (len(features) == len(n_periods) == len(input_ranges)):
            raise ValueError("all properties must have the same length")

    Attributes
    ----------
    config_ : dict[str, dict]
        Mapping of feature names to dictionaries containing
        {"n_periods": int, "input_range": tuple[int, int]}.
    _rbfs : dict[str, RepeatingBasisFunction]
        Mapping of feature names to RepeatingBasisFunction objects.
        self.features = features
        self.n_periods = n_periods
        self.input_ranges = input_ranges

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data.
    transform(X)
        Apply the RepeatingBasisFunction transformation to the data.
    get_feature_names_out(input_features=None)
        Get the feature names out.
        self.config_: dict[str, dict] = {
            feature: {"n_periods": period, "input_range": input_range}
            for feature, period, input_range in zip(features, n_periods, input_ranges)
        }

    Raises
    ------
    ValueError
        If the lengths of `features`, `n_periods`, and `input_ranges` do not match.
        If a column is not found in the config.
        If no fitted transformer is found for a column.
        self._rbfs: dict[str, RepeatingBasisFunction] = {}

    Notes
    -----
    This class assumes that the input data is a pandas DataFrame with named columns.
    The ordering of the features is important, meaning that the feature names in the
    input data must match the feature names in the config.

    """

    _parameter_constraints: dict = {
        "features": [HasMethods(["__iter__"])],  # type: ignore
        "n_periods": [HasMethods(["__iter__"])],  # type: ignore
        "input_ranges": [HasMethods(["__iter__"])],  # type: ignore
    }

    def __init__(
        self,
        features: Iterable[str],
        n_periods: Iterable[int],
        input_ranges: list[tuple[int, int]],
    ):
        if not (len(features) == len(n_periods) == len(input_ranges)):
            raise ValueError("all properties must have the same length")

        self.features = features
        self.n_periods = n_periods
        self.input_ranges = input_ranges

        self.config_: dict[str, dict] = {
            feature: {"n_periods": period, "input_range": input_range}
            for feature, period, input_range in zip(
                features,
                n_periods,
                input_ranges,
                strict=True,
            )
        }

        self._rbfs: dict[str, RepeatingBasisFunction] = {}

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        X = self._validate_data(X, accept_sparse=True, cast_to_ndarray=False)

        feature_names_out_ = []
        self.input_features_ = X.columns  # store input feature names

        for col in X.columns:
            if col not in self.config_:
                continue

            conf = self.config_[col]
            n_periods = conf["n_periods"]

            feature_names_out_.extend([f"{col}_rbf_{i + 1}" for i in range(n_periods)])

            rbf = RepeatingBasisFunction(
                n_periods=conf["n_periods"],
                column=0,
                input_range=conf["input_range"],
            )

            rbf.fit(X[[col]].values)
            self._rbfs[col] = rbf

        self.feature_names_out_ = np.array(feature_names_out_)
        self._is_fitted = True

        return self

    def transform(self, X):
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse=True, cast_to_ndarray=False)

        transformed = []

        for col in X.columns:
            if col not in self._rbfs:
                continue

            transformed.append(self._rbfs[col].transform(X[[col]].values))

        return np.hstack(transformed)

    def get_feature_names_out(self, input_features=None):
        """Return the output feature names."""
        check_is_fitted(self)

        if input_features is not None:
            # Optional: support partial input_features remapping
            raise NotImplementedError("Custom input features not supported yet. Use default.")

        return self.feature_names_out_
