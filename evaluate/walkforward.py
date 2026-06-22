import numpy as np
import pandas as pd
from models.base import PoissonBase
from betting.engine import backtest
from betting.sizing import FractionalKelly

__all__ = ["walkforward_cv", "optimize_decay"]


def walkforward_cv(model: PoissonBase, df: pd.DataFrame,
                   n_initial_seasons: int = 2,
                   decay: float = 0.0, sizing=None,
                   min_ev: float = 0.05, max_odds: float = 4.0) -> list:
    seasons = sorted(df["season"].unique())
    results = []

    for i in range(n_initial_seasons, len(seasons)):
        train = seasons[:i]
        test = [seasons[i]]
        result = backtest(
            model, df, train, test,
            decay=decay, sizing=sizing,
            min_ev=min_ev, max_odds=max_odds,
        )
        if result:
            result["train_seasons"] = train
            result["test_seasons"] = test
            results.append(result)

    if results:
        avg_roi = np.mean([r["roi"] for r in results])
        avg_win_rate = np.mean([r["win_rate"] for r in results])
        total_bets = sum(r["n_bets"] for r in results)
        print(f"\n=== WALK-FORWARD CV (decay={decay}) ===")
        print(f"Fold: {len(results)} temporadas de test")
        print(f"Total apuestas: {total_bets}")
        print(f"ROI promedio: {avg_roi:.2f}%")
        print(f"Win rate promedio: {avg_win_rate:.1f}%")

    return results


def optimize_decay(model: PoissonBase, df: pd.DataFrame,
                   decay_values: list = None,
                   n_initial_seasons: int = 2,
                   min_ev: float = 0.05, max_odds: float = 4.0,
                   sizing=None) -> dict:
    if decay_values is None:
        decay_values = [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01]

    sizing = sizing or FractionalKelly()
    best = {"decay": None, "roi": -float("inf")}
    scores = []

    for decay in decay_values:
        results = walkforward_cv(
            model, df, n_initial_seasons=n_initial_seasons,
            decay=decay, sizing=sizing,
            min_ev=min_ev, max_odds=max_odds,
        )
        if not results:
            continue
        avg_roi = np.mean([r["roi"] for r in results])
        total_bets = sum(r["n_bets"] for r in results)
        scores.append({"decay": decay, "avg_roi": avg_roi, "total_bets": total_bets})
        if avg_roi > best["roi"]:
            best = {"decay": decay, "roi": avg_roi, "total_bets": total_bets}

    print(f"\n=== OPTIMIZACIÓN DE DECAY ===")
    for s in sorted(scores, key=lambda x: -x["avg_roi"]):
        marker = " <<<" if s["decay"] == best["decay"] else ""
        print(f"  decay={s['decay']:.3f}  ROI={s['avg_roi']:.2f}%  bets={s['total_bets']}{marker}")

    print(f"\nMejor decay: {best['decay']} (ROI: {best['roi']:.2f}%)")
    return best
