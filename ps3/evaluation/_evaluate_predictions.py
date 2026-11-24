from typing import Dict, Optional

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import (
    auc,
    mean_absolute_error,
    mean_tweedie_deviance,
    root_mean_squared_error,
)


def compute_bias(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: Optional[ArrayLike] = None,
) -> float:
    """Compute the bias of predictions relative to the exposure-adjusted mean.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Actual values, for example the pure premium.
    y_pred : array-like of shape (n_samples,)
        Predicted values.
    sample_weight : array-like of shape (n_samples,), optional
        Weights, for example the exposure. If None, all observations are weighted
        equally.

    Returns
    -------
    float
        Weighted mean of ``y_pred`` minus weighted mean of ``y_true``.

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, or the weights sum to zero.
    """
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)

    if actual.shape != predicted.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, "
            f"got {actual.shape} and {predicted.shape}"
        )
    if actual.size == 0:
        raise ValueError("y_true and y_pred must not be empty")

    if sample_weight is None:
        weights = np.ones_like(actual)
    else:
        weights = np.asarray(sample_weight, dtype=float)
        if weights.shape != actual.shape:
            raise ValueError(
                f"sample_weight must have the same shape as y_true, "
                f"got {weights.shape} and {actual.shape}"
            )
    if weights.sum() == 0:
        raise ValueError("sample_weight must not sum to zero")

    actual_mean = np.average(actual, weights=weights)
    predicted_mean = np.average(predicted, weights=weights)
    return float(predicted_mean - actual_mean)


def compute_deviance(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: Optional[ArrayLike] = None,
    power: float = 1.5,
) -> float:
    """Compute the weighted Tweedie deviance per unit of weight.

    This is the loss used to compare the models: the sum of the weighted unit
    deviances divided by the sum of the weights, so it is comparable between data
    sets of different size. Lower is better, and 0 means a perfect prediction.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Actual values, for example the pure premium.
    y_pred : array-like of shape (n_samples,)
        Predicted values. Must be strictly positive for ``power`` >= 1.
    sample_weight : array-like of shape (n_samples,), optional
        Weights, for example the exposure. If None, all observations are weighted
        equally.
    power : float, optional
        Power of the Tweedie distribution, by default 1.5. For example 0 is
        normal, 1 is Poisson, 2 is Gamma.

    Returns
    -------
    float
        Weighted Tweedie deviance divided by the sum of the weights.
    """
    return float(
        mean_tweedie_deviance(y_true, y_pred, sample_weight=sample_weight, power=power)
    )


def compute_rmse_mae(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: Optional[ArrayLike] = None,
) -> Dict[str, float]:
    """Compute the (weighted) root mean squared error and mean absolute error.

    With weights ``w``::

        rmse = sqrt(sum(w * (y_true - y_pred) ** 2) / sum(w))
        mae = sum(w * abs(y_true - y_pred)) / sum(w)

    Both are in the units of the target (for example the pure premium) and lower is
    better. The RMSE penalises large errors more than the MAE.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Actual values, for example the pure premium.
    y_pred : array-like of shape (n_samples,)
        Predicted values.
    sample_weight : array-like of shape (n_samples,), optional
        Weights, for example the exposure. If None, all observations are weighted
        equally.

    Returns
    -------
    dict
        ``{"rmse": ..., "mae": ...}``, both floats.
    """
    return {
        "rmse": float(
            root_mean_squared_error(y_true, y_pred, sample_weight=sample_weight)
        ),
        "mae": float(mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)),
    }


def compute_gini(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: Optional[ArrayLike] = None,
) -> float:
    """Compute the Gini coefficient of the ranking of the predictions.

    Observations are ordered by
    increasing prediction (from the safest to the riskiest). The Lorenz curve is
    the cumulated claim amount (``y_true * sample_weight``) as a fraction of the
    total, plotted against the fraction of observations. The Gini coefficient is
    ``1 - 2 * AUC`` of that curve. A model with no ranking power has a Gini of 0,
    and the better the model ranks the riskiest observations last, the higher it is.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Actual values, for example the pure premium.
    y_pred : array-like of shape (n_samples,)
        Predicted values, used only to order the observations.
    sample_weight : array-like of shape (n_samples,), optional
        Weights, for example the exposure. If None, all observations are weighted
        equally.

    Returns
    -------
    float
        The Gini coefficient.

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, or the weighted actual values
        sum to zero.
    """
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)

    if actual.shape != predicted.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, "
            f"got {actual.shape} and {predicted.shape}"
        )
    if actual.size == 0:
        raise ValueError("y_true and y_pred must not be empty")

    if sample_weight is None:
        weights = np.ones_like(actual)
    else:
        weights = np.asarray(sample_weight, dtype=float)
        if weights.shape != actual.shape:
            raise ValueError(
                f"sample_weight must have the same shape as y_true, "
                f"got {weights.shape} and {actual.shape}"
            )

    # Order the observations by increasing predicted risk
    ranking = np.argsort(predicted)
    cumulated_claim_amount = np.cumsum(actual[ranking] * weights[ranking])
    if cumulated_claim_amount[-1] == 0:
        raise ValueError("The weighted y_true must not sum to zero")
    cumulated_claim_amount /= cumulated_claim_amount[-1]
    cumulated_samples = np.linspace(0, 1, len(cumulated_claim_amount))

    return float(1 - 2 * auc(cumulated_samples, cumulated_claim_amount))


def evaluate_predictions(
    df: pd.DataFrame,
    outcome_column: str,
    preds_column: str,
    exposure_column: Optional[str] = None,
    model_name: Optional[str] = None,
) -> pd.DataFrame:
    """Evaluate the predictions of a model with all the metrics.

    Calls ``compute_bias``, ``compute_deviance``, ``compute_rmse_mae`` and
    ``compute_gini`` on the actual values and the model's predictions, and collects
    the results in a data frame with one row per metric.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing the actual values, the predictions and, optionally, the
        weights. For example the test set with the predictions added as a column.
    outcome_column : str
        Name of the column with the actual values, for example "PurePremium".
    preds_column : str
        Name of the column with the model's predictions, for example "pp_t_glm1".
    exposure_column : str, optional
        Name of the column with the weights, for example "Exposure". If None, all
        observations are weighted equally.
    model_name : str, optional
        Name for the model's column in the result. By default ``preds_column``.

    Returns
    -------
    pd.DataFrame
        One column with the model's results, indexed by the metric names "bias",
        "deviance", "rmse", "mae" and "gini". Results for several models can be
        combined with ``pd.concat(..., axis=1)``.

    Raises
    ------
    ValueError
        If one of the columns is not in ``df``.
    """
    columns = [outcome_column, preds_column]
    if exposure_column is not None:
        columns.append(exposure_column)
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Columns not found in df: {missing_columns}")

    y_true = df[outcome_column]
    y_pred = df[preds_column]
    weights = None if exposure_column is None else df[exposure_column]

    metrics = {
        "bias": compute_bias(y_true, y_pred, weights),
        "deviance": compute_deviance(y_true, y_pred, weights),
        **compute_rmse_mae(y_true, y_pred, weights),
        "gini": compute_gini(y_true, y_pred, weights),
    }
    return pd.DataFrame({model_name or preds_column: metrics}, index=list(metrics))
