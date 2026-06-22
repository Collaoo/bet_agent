import pandas as pd
import numpy as np
import re
from models.base import PoissonBase
from betting.markets import over_under_probs, btts_probs, expected_value
from betting.sizing import SizingStrategy, FractionalKelly

__all__ = ["BettingEngine", "backtest", "walkforward_backtest"]


def _find_odds_cols(df: pd.DataFrame) -> dict:
    """Scan DataFrame columns and group odds by market.
    Returns: {'h': [(odds, bookmaker), ...], 'd': [...], 'a': [...]}
    """
    groups = {"h": [], "d": [], "a": []}
    for col in df.columns:
        col_lower = col.lower().strip()
        # Match patterns: odds_h, odds_B365_h, odds_Pinnacle_h, etc.
        m = re.match(r"odds_(?:([a-z0-9]+)_)?([hda])$", col_lower)
        if m:
            bookmaker = m.group(1) or "default"
            market = m.group(2)
            groups.setdefault(market, []).append((col, bookmaker))
    return groups


def _get_best_odds(row: pd.Series, market: str,
                   odds_cols: dict) -> tuple:
    """Get best odds for a market across all bookmakers.
    Returns (best_odds, bookmaker_name)."""
    candidates = []
    for col, bookmaker in odds_cols.get(market.lower(), []):
        val = row.get(col, 0)
        if pd.notna(val) and val > 0:
            candidates.append((float(val), bookmaker))
    if not candidates:
        return (0.0, "")
    best = max(candidates, key=lambda x: x[0])
    return best


class BettingEngine:

    def __init__(self, model: PoissonBase,
                 sizing: SizingStrategy = None,
                 min_ev: float = 0.05,
                 max_odds: float = 4.0,
                 use_best_odds: bool = True):
        self.model = model
        self.sizing = sizing or FractionalKelly()
        self.min_ev = min_ev
        self.max_odds = max_odds
        self.use_best_odds = use_best_odds
        self._odds_cols = {}

    def _resolve_odds(self, row: pd.Series, market: str) -> tuple:
        if self.use_best_odds and self._odds_cols:
            return _get_best_odds(row, market, self._odds_cols)
        col = f"odds_{market.lower()}"
        val = row.get(col, 0)
        return (float(val) if pd.notna(val) and val > 0 else 0.0, "")

    def _place_bets(self, row: pd.Series, bankroll: float) -> list:
        bets = []
        pred = self.model.predict_probs(row["home_team"], row["away_team"])
        sm = pred.pop("score_matrix")

        market_keys = [
            ("H",  pred["prob_H"]),
            ("D",  pred["prob_D"]),
            ("A",  pred["prob_A"]),
        ]

        ou = over_under_probs(sm)
        market_keys.append(("over",  ou["over"]))
        market_keys.append(("under", ou["under"]))

        btts = btts_probs(sm)
        market_keys.append(("btts_yes", btts["btts_yes"]))
        market_keys.append(("btts_no",  btts["btts_no"]))

        for market, prob in market_keys:
            odds, bookmaker = self._resolve_odds(row, market)
            if odds <= 0 or odds > self.max_odds:
                continue
            ev = expected_value(prob, odds)
            if ev < self.min_ev:
                continue
            stake = self.sizing.compute_stake(prob, odds, bankroll,
                                              model_prob=prob)
            if stake <= 0:
                continue

            won = self._check_result(row, market)
            pnl = stake * (odds - 1) if won else -stake

            bet = {
                "date":      str(row.get("date", "")),
                "match":     f"{row['home_team']} vs {row['away_team']}",
                "market":    market,
                "our_prob":  round(prob, 4),
                "odds":      odds,
                "impl_prob": round(1 / odds, 4),
                "ev":        round(ev, 4),
                "stake":     round(stake, 2),
                "won":       won,
                "pnl":       round(pnl, 2),
            }
            if bookmaker:
                bet["bookmaker"] = bookmaker
            bets.append(bet)

            if hasattr(self.sizing, "record_result"):
                self.sizing.record_result(won)

        return bets

    def _check_result(self, row: pd.Series, market: str) -> bool:
        result = row.get("result", "")
        if market in ("H", "D", "A"):
            return result == market
        hg, ag = int(row["home_goals"]), int(row["away_goals"])
        if market == "over":
            return hg + ag >= 3
        elif market == "under":
            return hg + ag <= 2
        elif market == "btts_yes":
            return hg > 0 and ag > 0
        elif market == "btts_no":
            return hg == 0 or ag == 0
        return False

    def run(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
            decay: float = 0.0) -> pd.DataFrame:
        if hasattr(self.sizing, "reset"):
            self.sizing.reset()

        if not self.model.fitted:
            self.model.fit(train_df, decay=decay)

        self._odds_cols = _find_odds_cols(test_df) if self.use_best_odds else {}
        bankroll = 1000.0
        all_bets = []

        for _, row in test_df.iterrows():
            if (row["home_team"] not in self.model.team_idx or
                    row["away_team"] not in self.model.team_idx):
                continue

            bets = self._place_bets(row, bankroll)
            for b in bets:
                bankroll += b["pnl"]
                b["bankroll"] = round(bankroll, 2)
            all_bets.extend(bets)

        return pd.DataFrame(all_bets)


def backtest(model, df, train_seasons, test_seasons, decay=0.0,
             sizing=None, min_ev=0.05, max_odds=4.0,
             use_best_odds=True) -> dict:
    engine = BettingEngine(model, sizing=sizing, min_ev=min_ev,
                           max_odds=max_odds, use_best_odds=use_best_odds)
    train_df = df[df["season"].isin(train_seasons)].copy()
    test_df = df[df["season"].isin(test_seasons)].copy()
    bets_df = engine.run(train_df, test_df, decay=decay)

    if bets_df.empty:
        print("Sin apuestas con valor en el periodo de test.")
        return {}

    total_staked = bets_df["stake"].sum()
    total_pnl = bets_df["pnl"].sum()
    roi = total_pnl / total_staked * 100 if total_staked > 0 else 0.0
    win_rate = bets_df["won"].mean() * 100
    n_bets = len(bets_df)
    final_bankroll = bets_df["bankroll"].iloc[-1] if len(bets_df) else 1000.0

    n_bookmakers = len(set(
        b for b in bets_df.get("bookmaker", []) if b
    )) if "bookmaker" in bets_df.columns else 1

    print(f"\n=== BACKTEST ({type(model).__name__}) ===")
    print(f"Train: {train_seasons} | Test: {test_seasons}")
    print(f"Decay: {decay} | min_ev: {min_ev} | max_odds: {max_odds}")
    print(f"Bookmakers: {n_bookmakers} | Best odds: {use_best_odds}")
    print(f"{'─' * 50}")
    print(f"Apuestas    : {n_bets}")
    print(f"Win rate    : {win_rate:.1f}%")
    print(f"P&L         : ${total_pnl:+.2f}")
    print(f"ROI         : {roi:+.2f}%")
    print(f"Bankroll    : ${final_bankroll:.2f}  (inicio: $1000)")

    for m in bets_df["market"].unique():
        sub = bets_df[bets_df["market"] == m]
        m_roi = sub["pnl"].sum() / sub["stake"].sum() * 100
        print(f"  {m}: {len(sub)} apuestas | ROI {m_roi:+.2f}% | "
              f"win rate {sub['won'].mean()*100:.1f}%")

    return {
        "bets_df": bets_df,
        "roi": roi,
        "bankroll": final_bankroll,
        "n_bets": n_bets,
        "win_rate": win_rate,
    }


def walkforward_backtest(model, df, n_initial_seasons: int = 2,
                         decay=0.0, sizing=None,
                         min_ev=0.05, max_odds=4.0) -> list:
    seasons = sorted(df["season"].unique())
    results = []

    for i in range(n_initial_seasons, len(seasons)):
        train_seasons = seasons[:i]
        test_seasons = [seasons[i]]
        result = backtest(
            model, df, train_seasons, test_seasons,
            decay=decay, sizing=sizing, min_ev=min_ev, max_odds=max_odds,
        )
        if result:
            result["train_seasons"] = train_seasons
            result["test_seasons"] = test_seasons
            results.append(result)

    return results
