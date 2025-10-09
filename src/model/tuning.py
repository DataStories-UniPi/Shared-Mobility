from collections.abc import Callable

import numpy as np
import pandas as pd
from loguru import logger
from optuna import TrialPruned
from optuna.distributions import FloatDistribution, IntDistribution
from optuna.study import Study
from optuna.trial import Trial
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from components import ModelFactory
from config.constants import GROUP_COLUMN, TARGET_COLUMN
from utils.models import TaskType

# Initialize the default model builder
model_builder = ModelFactory(task=TaskType.REGRESSION, num_output=len([TARGET_COLUMN]))


def objective(
    trial: Trial,
    X,
    y,
    model_builder: Callable = model_builder.build_model,
    **model_kwargs,
) -> float:
    """Create an objective function to optimize for Optuna hyperparameter tuning.

    This function takes a Trial object and the training and test data as input,
    and returns the mean absolute percentage error (MAPE) of the predictions
    made by the model with the hyperparameters suggested by the Trial object.

    The hyperparameter search space is defined by the `params` dictionary, which
    contains the possible values for each hyperparameter. The values are
    sampled from the given ranges using the `suggest_int` and `suggest_float`
    methods of the Trial object.

    The function first sets the hyperparameters of the model using the
    `set_params` method, then fits the model to the training data using the
    `fit` method, makes predictions on the test data using the `predict`
    method, and finally computes the MAPE using the `mean_absolute_percentage_error`
    function.

    The function returns the sMAPE as the objective value to be minimized by
    Optuna.

    Args:
        trial : The Trial object used to sample hyperparameters.
        X_train, y_train : The training data feature and target values.
        X_test, y_test : The test data feature and target values.
        model_builder (default=`ModelBuilder`): The function used to build the model.
        model_kwargs : Additional keyword arguments to pass to the model builder.

    Returns:
        The error of the predictions made by the model with the hyperparameters suggested
            by the Trial object.

    """

    # Define hyperparameter search space with domain-justified bounds
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }

    # Use TimeSeriesSplit to avoid look-ahead bias
    ts_cv = TimeSeriesSplit(n_splits=5)
    rmse_scores = []

    for fold, (train_idx, val_idx) in enumerate(ts_cv.split(X), start=1):
        logger.debug(f"Split [{fold}/5]...")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Handle edge case: very small folds
        if len(X_val) == 0:
            logger.warning(f"Fold {fold} has empty validation set. Skipping.")
            continue

        model = model_builder(**model_kwargs)
        model.set_params(**params)

        try:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            y_pred = model.predict(X_val)

            # Enforce positive integer values to get realist demand values
            y_pred = np.maximum(y_pred, 0)
            y_pred = np.round(y_pred).astype(int)

            y_pred = pd.DataFrame(y_pred, index=y_val.index, columns=y_val.columns)

            # Calculate error per district
            group_rmse = []
            for group in y_val.index.get_level_values(GROUP_COLUMN).unique():
                mask = y_pred.index.get_level_values(GROUP_COLUMN) == group
                y_val_group = y_val.loc[mask]
                y_pred_group = y_pred.loc[mask]

                group_rmse.append(root_mean_squared_error(y_val_group, y_pred_group))

            avg_trial_rmse = np.median(group_rmse)
            rmse_scores.append(avg_trial_rmse)

            # Pruning: intermediate reporting for early stopping
            trial.report(avg_trial_rmse, step=fold)
            if trial.should_prune():
                raise TrialPruned()

        except Exception as e:
            logger.error(f"Trial {trial.number}, Fold {fold}: {str(e)}")
            raise TrialPruned()  # Prune on failure

    mean_rmse = np.mean(rmse_scores)

    return np.mean(mean_rmse)


def get_bounds(study: Study) -> pd.DataFrame:
    """Get the hyperparameter bounds for the next optimization cycle.

    This function extracts the hyperparameter distributions from the first trial
    in the given Optuna study and returns a DataFrame with the hyperparameters
    and their adjusted bounds.

    Parameters
    ----------
    study : optuna.study.Study
        The study object that contains the optimization results.

    Returns
    -------
    pd.DataFrame
        DataFrame with the hyperparameter distributions
        and their adjusted bounds for the next optimization cycle.

    """
    # Extract the hyperparameter distributions from the first trial
    sample_trial = study.trials[0].distributions

    param_bounds = [
        [
            param,
            dist.low,
            dist.high,
            dist.step if hasattr(dist, "step") else 0,  # Handle categorical distributions
            dist.__class__.__name__,
        ]
        for param, dist in sample_trial.items()
        if isinstance(dist, FloatDistribution | IntDistribution)
    ]

    return pd.DataFrame(
        param_bounds,
        columns=["param", "min", "max", "step", "distribution"],
    )


def create_adaptive_search_space(
    study: Study,
    lower_bound: float = 0.25,
    upper_bound: float = 0.75,
    scale: float = 0.5,
) -> dict:
    """
    Adjust the hyperparameter space distributions based on the best parameters so far.

    The new distributions are created by adjusting the range of the
    hyperparameters based on the position of the best parameters
    in the original range. If the best parameter is near the lower or
    upper bound, the range is adjusted accordingly. Otherwise, the range
    is centered around the best parameter.

    Parameters
    ----------
    study : optuna.study.Study
        The study object that contains the optimization results.
    lower_bound : float, defaults to 0.25
        The lower bound of the range that the best parameter should
        be in for the range to be adjusted.
    upper_bound : float, defaults to 0.75
        The upper bound of the range that the best parameter should
        be in for the range to be adjusted.
    scale : float, defaults to 0.5
        The factor by which to adjust the range.

    Returns
    -------
    dict
        A dictionary with the new hyperparameter distributions.

    """
    distributions = study.trials[0].distributions
    best_params = study.best_params
    new_distributions = {}

    if any([lower_bound > 1, upper_bound > 1]):
        raise ValueError("lower_bound and upper_bound must be in the range [0,1]")

    for param_name, dist in distributions.items():
        if isinstance(dist, FloatDistribution | IntDistribution):
            original_width = dist.high - dist.low
            best_value = best_params[param_name]

            # Calculate position in range [0,1]
            best_scaled = (
                (best_value - dist.low) / original_width if original_width > 0 else 0.5
            )

            # Adjust range differently based on position
            if best_scaled < lower_bound:
                new_low = dist.low
                new_high = best_value + original_width * scale
            elif best_scaled > upper_bound:
                new_low = best_value - original_width * scale
                new_high = dist.high
            else:
                new_low = best_value - original_width * scale / 2
                new_high = best_value + original_width * scale / 2

            # Create appropriate distribution
            if isinstance(dist, IntDistribution):
                new_distributions[param_name] = IntDistribution(
                    low=int(new_low), high=int(new_high), step=dist.step
                )
            else:
                new_distributions[param_name] = FloatDistribution(
                    low=float(new_low),
                    high=float(new_high),
                    step=dist.step / 2 if dist.step is not None else dist.step,
                    log=hasattr(dist, "log") and dist.log,
                )
        else:
            new_distributions[param_name] = dist  # Keep categorical distributions unchanged

    return new_distributions
