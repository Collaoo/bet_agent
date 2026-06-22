"""
test_models.py
Test de efectividad y ROI con calibración de probabilidades.

Uso:
    python test_models.py
    python test_models.py --quick
"""

import argparse
import time

import pandas as pd
import numpy as np

from models import PoissonModel, PoissonDecay, DixonColesModel, load_data
from betting import (
    FractionalKelly, FlatStaking, BettingEngine,
    betting_metrics, drawdown_analysis,
)
from evaluate.metrics import classification_report


def split_calibrate(df, cal_pct=0.3):
    """Split df into train and calibration sets chronologically."""
    n = len(df)
    split_idx = int(n * (1 - cal_pct))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def runner(desc, model, train_df, cal_df, test_df, sizing,
           min_ev=0.05, max_odds=4.0):
    print(f"  {desc:<62} ", end="", flush=True)
    t0 = time.time()

    model.fit(train_df, decay=model._decay if hasattr(model, '_decay') else 0.0)
    model.calibrate(cal_df, verbose=False)

    engine = BettingEngine(model, sizing=sizing, min_ev=min_ev, max_odds=max_odds)
    if hasattr(engine.sizing, "reset"):
        engine.sizing.reset()
    bets_df = engine.run(train_df, test_df, decay=0.0)
    elapsed = time.time() - t0

    if bets_df.empty:
        print(f"  0 bets  ⏱ {elapsed:.1f}s")
        return {"desc": desc, "n_bets": 0, "roi": 0.0, "bankroll": 1000, "bankroll_end": 1000}

    m = betting_metrics(bets_df)
    dd = drawdown_analysis(bets_df)

    print(f"{m['n_bets']:>3} bets  |  "
          f"Win {m['win_rate']:>5.1f}%  |  "
          f"ROI {m['roi']:>+7.2f}%  |  "
          f"Sharpe {m['sharpe_ratio']:>+6.3f}  |  "
          f"pf {m['profit_factor']:>5.2f}  |  "
          f"⏱ {elapsed:.1f}s")

    m["desc"] = desc
    m["max_dd"] = dd["max_drawdown_pct"]
    m["bankroll_end"] = bets_df["bankroll"].iloc[-1] if len(bets_df) else 1000.0
    return m


def run_backtest_suite(df, quick=False):
    seasons = sorted(df["season"].unique())
    if quick:
        train_seasons = seasons[:6]
        test_seasons = [seasons[6]]
    else:
        train_seasons = seasons[:-1]
        test_seasons = [seasons[-1]]

    raw_train = df[df["season"].isin(train_seasons)]
    test_df = df[df["season"].isin(test_seasons)]

    # Split train into train/calibrate
    train_df, cal_df = split_calibrate(raw_train, cal_pct=0.3)

    print(f"\n{'=' * 72}")
    print(f"  BACKTEST CON CALIBRACION — {len(df)} partidos")
    print(f"{'=' * 72}")
    print(f"  Train: {train_seasons} ({len(train_df)} part)")
    print(f"  Cal:    {len(cal_df)} part (30%% del train)")
    print(f"  Test:   {test_seasons} ({len(test_df)} part)\n")

    model_configs = [
        ("Poisson",      PoissonModel,      0.0),
        ("PoissonDecay", PoissonDecay,      0.004),
        ("DixonColes",   DixonColesModel,   0.002),
    ]

    sizing_list = [
        ("Kelly 1/4",  FractionalKelly(fraction=0.25, max_stake_pct=0.03)),
        ("Flat 2%",    FlatStaking(stake_pct=0.02)),
    ]

    min_ev_values = [0.05, 0.10, 0.15, 0.20] if not quick else [0.05, 0.10]

    results = []
    fit_time = 0.0

    for model_name, ModelClass, decay in model_configs:
        for min_ev in min_ev_values:
            for sizing_name, sizing in sizing_list:
                t0 = time.time()
                model = ModelClass()
                model._decay = decay
                m = runner(
                    f"{model_name} decay={decay} | min_ev={min_ev:.2f} | {sizing_name}",
                    model, train_df, cal_df, test_df,
                    sizing=sizing, min_ev=min_ev,
                )
                ft = time.time() - t0
                fit_time += ft
                m["model"] = model_name
                m["sizing"] = sizing_name
                m["min_ev"] = min_ev
                results.append(m)

    # ── Reporte ──
    print(f"\n\n{'=' * 72}")
    print(f"  REPORTE COMPARATIVO  (fit total: {fit_time:.0f}s)")
    print(f"{'=' * 72}")

    report = pd.DataFrame(results)
    report = report[report["n_bets"] > 0].copy()
    if report.empty:
        print("  No se encontraron apuestas con valor en el periodo de test.")
        return report
    report = report.sort_values("roi", ascending=False).reset_index(drop=True)
    report.index = report.index + 1

    cols = ["desc", "n_bets", "win_rate", "roi", "sharpe_ratio", "profit_factor", "max_dd"]
    display = report[cols].copy()
    display.columns = ["Config", "Bets", "Win%", "ROI%", "Sharpe", "ProfitF", "MaxDD%"]
    print(f"\n{display.to_string()}\n")

    if len(report):
        best = report.iloc[0]
        print(f"  🏆  Mejor ROI%: {best['roi']:+.2f}% → {best['desc']}")
        best_s = report.loc[report["sharpe_ratio"].idxmax()]
        print(f"  🏆  Mejor Sharpe: {best_s['sharpe_ratio']:+.3f} → {best_s['desc']}")

    report.to_csv("data/test_results_calibrated.csv", index=False)
    print(f"\n  Reporte guardado en data/test_results_calibrated.csv")
    return report


def run_accuracy_test(df):
    """Accuracy, Brier, LogLoss con y sin calibración."""
    seasons = sorted(df["season"].unique())
    raw_train = df[df["season"] != seasons[-1]]
    test_df = df[df["season"] == seasons[-1]]
    train_df, cal_df = split_calibrate(raw_train, cal_pct=0.3)

    print(f"\n\n{'=' * 72}")
    print(f"  PRECISION CON CALIBRACION")
    print(f"{'=' * 72}")
    print(f"  Train: {len(train_df)} | Cal: {len(cal_df)} | Test: {len(test_df)}\n")

    configs = [
        ("Poisson",      PoissonModel,      0.0),
        ("PoissonDecay", PoissonDecay,      0.004),
        ("DixonColes",   DixonColesModel,   0.002),
    ]

    rows = []

    for name, ModelClass, decay in configs:
        model = ModelClass()
        model._decay = decay
        t0 = time.time()

        model.fit(train_df, decay=decay)
        r_raw = classification_report(test_df, model)

        model.calibrate(cal_df)
        r_cal = classification_report(test_df, model)

        ft = time.time() - t0

        rows.append({
            "Modelo": name,
            "Acc_raw": round(r_raw["accuracy"], 1),
            "Acc_cal": round(r_cal["accuracy"], 1),
            "Brier_raw": round(r_raw["brier"], 4),
            "Brier_cal": round(r_cal["brier"], 4),
            "LL_raw": round(r_raw["log_loss"], 4),
            "LL_cal": round(r_cal["log_loss"], 4),
            "T": round(model._temperature, 3) if model._temperature else 1.0,
        })

        print(f"  {name:<14}  "
              f"Acc: {r_raw['accuracy']:.1f}% → {r_cal['accuracy']:.1f}%  "
              f"Brier: {r_raw['brier']:.4f} → {r_cal['brier']:.4f}  "
              f"T={model._temperature:.3f}")
        print(f"  {'':>14}  "
              f"LL: {r_raw['log_loss']:.4f} → {r_cal['log_loss']:.4f}  "
              f"N={r_cal['n']}  fit: {ft:.1f}s\n")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    print("Cargando datos...")
    df = load_data()
    print(f"  {len(df)} partidos | {', '.join(df['season'].unique())}")

    acc_df = run_accuracy_test(df)

    report_df = run_backtest_suite(df, quick=args.quick)

    print(f"\n\n{'=' * 72}")
    print(f"  RESUMEN FINAL")
    print(f"{'=' * 72}\n")
    print(acc_df.to_string(index=False))
    print()

    if len(report_df):
        best = report_df.iloc[0]
        print(f"  Mejor config:")
        print(f"    {best['desc']}")
        print(f"    ROI: {best['roi']:+.2f}% | Sharpe: {best['sharpe_ratio']:.3f} | "
              f"Bets: {best['n_bets']} | Win: {best['win_rate']:.1f}%")
