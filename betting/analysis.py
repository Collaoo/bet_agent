import pandas as pd
import numpy as np

__all__ = [
    "betting_metrics", "equity_curve", "drawdown_analysis",
    "market_breakdown", "sharpe_ratio", "sortino_ratio",
]


def betting_metrics(bets_df: pd.DataFrame) -> dict:
    if bets_df.empty:
        return {}
    total_staked = bets_df["stake"].sum()
    total_pnl = bets_df["pnl"].sum()
    n_bets = len(bets_df)
    n_wins = bets_df["won"].sum()
    n_losses = n_bets - n_wins
    win_rate = bets_df["won"].mean() * 100
    roi = total_pnl / total_staked * 100 if total_staked > 0 else 0.0
    avg_odds = bets_df["odds"].mean()
    avg_stake = bets_df["stake"].mean()
    avg_pnl = bets_df["pnl"].mean()
    total_won = bets_df[bets_df["won"]]["pnl"].sum()
    total_lost = abs(bets_df[~bets_df["won"]]["pnl"].sum())
    profit_factor = total_won / total_lost if total_lost > 0 else float("inf")
    net_profit = bets_df["bankroll"].iloc[-1] - 1000.0 if "bankroll" in bets_df.columns else total_pnl

    returns = bets_df["pnl"] / bets_df["stake"]
    daily = bets_df.groupby("date")["pnl"].sum() if "date" in bets_df.columns else returns
    sharpe = sharpe_ratio(daily) if isinstance(daily, pd.Series) else 0.0
    sortino = sortino_ratio(daily) if isinstance(daily, pd.Series) else 0.0

    return {
        "n_bets": n_bets,
        "n_wins": int(n_wins),
        "n_losses": int(n_losses),
        "win_rate": round(win_rate, 2),
        "roi": round(roi, 2),
        "total_staked": round(total_staked, 2),
        "total_pnl": round(total_pnl, 2),
        "net_profit": round(net_profit, 2),
        "avg_odds": round(avg_odds, 3),
        "avg_stake": round(avg_stake, 2),
        "avg_pnl": round(avg_pnl, 2),
        "profit_factor": round(profit_factor, 3),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
    }


def equity_curve(bets_df: pd.DataFrame) -> pd.DataFrame:
    if bets_df.empty or "bankroll" not in bets_df.columns:
        return pd.DataFrame()
    return bets_df[["date", "bankroll"]].copy()


def drawdown_analysis(bets_df: pd.DataFrame) -> dict:
    if bets_df.empty or "bankroll" not in bets_df.columns:
        return {}
    equity = bets_df["bankroll"].values
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak * 100
    max_dd = dd.min()
    max_dd_idx = dd.argmin()
    dd_duration = 0
    for i in range(len(dd)):
        if dd[i] < 0:
            dd_duration += 1
        else:
            dd_duration = 0
    return {
        "max_drawdown_pct": round(float(max_dd), 2),
        "max_drawdown_at": int(max_dd_idx),
        "max_drawdown_periods": dd_duration,
        "avg_drawdown_pct": round(float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0, 2),
    }


def market_breakdown(bets_df: pd.DataFrame) -> pd.DataFrame:
    if bets_df.empty:
        return pd.DataFrame()
    groups = bets_df.groupby("market")
    data = []
    for market, grp in groups:
        staked = grp["stake"].sum()
        pnl = grp["pnl"].sum()
        data.append({
            "market": market,
            "bets": len(grp),
            "win_rate": grp["won"].mean() * 100,
            "roi": pnl / staked * 100 if staked > 0 else 0.0,
            "pnl": round(pnl, 2),
            "avg_odds": grp["odds"].mean(),
        })
    return pd.DataFrame(data).sort_values("pnl", ascending=False).reset_index(drop=True)


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float((returns.mean() - rf) / returns.std() * np.sqrt(252))


def sortino_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float((returns.mean() - rf) / downside.std() * np.sqrt(252))
