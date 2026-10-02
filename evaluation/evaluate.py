import numpy as np
from sklearn.metrics import mean_squared_error


def rmse(actual, predicted):
    return float(np.sqrt(mean_squared_error(actual, predicted)))


def peak_nonpeak_rmse(actual, predicted, threshold):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    peak_mask = actual > threshold
    nonpeak_mask = actual <= threshold

    result = {
        "overall_rmse": rmse(actual, predicted),
        "peak_rmse": None,
        "nonpeak_rmse": None,
        "peak_count": int(peak_mask.sum()),
        "nonpeak_count": int(nonpeak_mask.sum()),
    }

    if peak_mask.any():
        result["peak_rmse"] = rmse(actual[peak_mask], predicted[peak_mask])

    if nonpeak_mask.any():
        result["nonpeak_rmse"] = rmse(
            actual[nonpeak_mask],
            predicted[nonpeak_mask],
        )

    return result
