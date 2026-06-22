import numpy as np
import pandas as pd

__all__ = ["brier_score", "log_loss", "accuracy", "calibration_error", "classification_report"]


def brier_score(probs: np.ndarray, actuals: np.ndarray) -> float:
    """Brier score for multi-class (N, K) or binary (N,) predictions."""
    probs = np.asarray(probs)
    actuals = np.asarray(actuals)
    if probs.ndim == 1:
        actuals_bin = (actuals == 1).astype(float)
        return float(np.mean((probs - actuals_bin) ** 2))
    else:
        n_classes = probs.shape[1]
        actuals_ohe = np.eye(n_classes)[actuals.astype(int)]
        return float(np.mean(np.sum((probs - actuals_ohe) ** 2, axis=1)))


def log_loss(probs: np.ndarray, actuals: np.ndarray, eps: float = 1e-15) -> float:
    probs = np.clip(np.asarray(probs), eps, 1 - eps)
    if probs.ndim == 1:
        actuals_bin = np.asarray(actuals).astype(float)
        return float(-np.mean(actuals_bin * np.log(probs) + (1 - actuals_bin) * np.log(1 - probs)))
    else:
        actuals_ohe = np.eye(probs.shape[1])[np.asarray(actuals).astype(int)]
        return float(-np.mean(np.sum(actuals_ohe * np.log(probs), axis=1)))


def accuracy(predicted: np.ndarray, actuals: np.ndarray) -> float:
    return float(np.mean(np.asarray(predicted) == np.asarray(actuals)))


def calibration_error(probs: np.ndarray, actuals: np.ndarray,
                      n_bins: int = 10) -> float:
    """Expected calibration error (ECE). Binary only."""
    probs = np.asarray(probs).flatten()
    actuals = np.asarray(actuals).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if not mask.any():
            continue
        bin_acc = actuals[mask].mean()
        bin_conf = probs[mask].mean()
        ece += np.abs(bin_acc - bin_conf) * mask.sum()
    return float(ece / len(probs))


def classification_report(df: pd.DataFrame, model, home_col: str = "home_team",
                          away_col: str = "away_team", result_col: str = "result") -> dict:
    correct = 0
    results = []
    for _, row in df.iterrows():
        if row[home_col] not in model.team_idx or row[away_col] not in model.team_idx:
            continue
        pred = model.predict(row[home_col], row[away_col])
        probs = {"H": pred["prob_home"], "D": pred["prob_draw"], "A": pred["prob_away"]}
        predicted = max(probs, key=probs.get)
        actual = row[result_col]
        if predicted == actual:
            correct += 1
        results.append({
            "predicted": predicted,
            "actual": actual,
            "correct": predicted == actual,
            "prob_H": pred["prob_home"],
            "prob_D": pred["prob_draw"],
            "prob_A": pred["prob_away"],
        })
    results_df = pd.DataFrame(results)
    acc = correct / len(results_df) * 100 if len(results_df) > 0 else 0.0
    prob_array = results_df[["prob_H", "prob_D", "prob_A"]].values
    actual_map = {"H": 0, "D": 1, "A": 2}
    actual_array = np.array([actual_map[r] for r in results_df["actual"]])
    return {
        "accuracy": acc,
        "n": len(results_df),
        "n_correct": correct,
        "brier": brier_score(prob_array, actual_array),
        "log_loss": log_loss(prob_array, actual_array),
    }
