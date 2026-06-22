import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

__all__ = ["platt_scale", "isotonic_calibration", "calibration_curve"]


def _fit_platt(probs: np.ndarray, actuals: np.ndarray):
    log_odds = np.clip(np.log(probs / (1 - probs + 1e-15)), -10, 10)
    lr = LogisticRegression(C=1e10, solver="lbfgs")
    lr.fit(log_odds.reshape(-1, 1), actuals.astype(int))
    return lr


def platt_scale(probs: np.ndarray, actuals: np.ndarray) -> np.ndarray:
    """Platt scaling: fit logistic regression on log-odds, return calibrated probs."""
    lr = _fit_platt(probs, actuals)
    log_odds = np.clip(np.log(probs / (1 - probs + 1e-15)), -10, 10)
    return lr.predict_proba(log_odds.reshape(-1, 1))[:, 1]


def isotonic_calibration(probs: np.ndarray, actuals: np.ndarray) -> np.ndarray:
    """Isotonic regression calibration (non-parametric)."""
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(probs, actuals)
    return iso.transform(probs)


def calibration_curve(probs: np.ndarray, actuals: np.ndarray,
                      n_bins: int = 10) -> pd.DataFrame:
    probs = np.asarray(probs).flatten()
    actuals = np.asarray(actuals).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    data = []
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if not mask.any():
            continue
        data.append({
            "bin": i,
            "bin_center": (bins[i] + bins[i + 1]) / 2,
            "n": mask.sum(),
            "mean_predicted": probs[mask].mean(),
            "fraction_positives": actuals[mask].mean(),
        })
    return pd.DataFrame(data)
