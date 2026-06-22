"""
model_with_odds.py
Modelo de Poisson con decay temporal + detección de value bets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import poisson
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")


def load_data(path: str = "data/bundesliga_with_odds.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["FTHG", "FTAG", "B365H", "B365D", "B365A"])
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    return df.sort_values("Date").reset_index(drop=True)


class PoissonModel:

    def __init__(self):
        self.teams          = []
        self.team_idx       = {}
        self.attack_params  = None
        self.defense_params = None
        self.home_adv       = None
        self.fitted         = False

    def fit(self, df: pd.DataFrame, decay: float = 0.004):
        """decay: peso exponencial por dia hacia atras.
        0.004 = partido de hace 6 meses pesa ~40% menos."""
        self.teams    = sorted(df["HomeTeam"].unique().tolist())
        self.team_idx = {t: i for i, t in enumerate(self.teams)}
        n             = len(self.teams)

        max_date = df["Date"].max()
        days_ago = (max_date - df["Date"]).dt.days
        weights  = np.exp(-decay * days_ago)

        def neg_log_likelihood(params):
            atk  = params[:n]
            dfc  = params[n:2*n]
            hadv = params[-1]
            ll   = 0.0
            for i, (_, row) in enumerate(df.iterrows()):
                hi = self.team_idx[row["HomeTeam"]]
                ai = self.team_idx[row["AwayTeam"]]
                lh = np.exp(hadv + atk[hi] - dfc[ai])
                la = np.exp(atk[ai] - dfc[hi])
                w  = weights.iloc[i]
                ll += w * poisson.logpmf(row["FTHG"], lh)
                ll += w * poisson.logpmf(row["FTAG"], la)
            return -ll

        x0          = np.zeros(2 * n + 1)
        x0[-1]      = 0.1
        constraints = {"type": "eq", "fun": lambda p: np.sum(p[:n])}

        result = minimize(neg_log_likelihood, x0, method="SLSQP",
                          constraints=constraints,
                          options={"maxiter": 1000, "ftol": 1e-10})

        self.attack_params  = result.x[:n]
        self.defense_params = result.x[n:2*n]
        self.home_adv       = result.x[-1]
        self.fitted         = True

        print(f"Modelo entrenado | equipos: {n} | partidos: {len(df)} | "
              f"decay: {decay} | convergió: {result.success}")

    def predict(self, home: str, away: str, max_goals: int = 8) -> dict:
        hi = self.team_idx[home]
        ai = self.team_idx[away]
        lh = np.exp(self.home_adv + self.attack_params[hi] - self.defense_params[ai])
        la = np.exp(self.attack_params[ai] - self.defense_params[hi])
        sm = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                sm[i, j] = poisson.pmf(i, lh) * poisson.pmf(j, la)
        return {
            "prob_H": float(np.sum(np.tril(sm, -1))),
            "prob_D": float(np.sum(np.diag(sm))),
            "prob_A": float(np.sum(np.triu(sm, 1))),
        }

    def team_strengths(self) -> pd.DataFrame:
        return pd.DataFrame({
            "team":    self.teams,
            "attack":  np.exp(self.attack_params),
            "defense": np.exp(-self.defense_params),
        }).sort_values("attack", ascending=False).reset_index(drop=True)


def expected_value(prob: float, odds: float) -> float:
    return prob * odds - 1


def kelly_stake(prob: float, odds: float, fraction: float = 0.25) -> float:
    edge = prob * odds - 1
    if edge <= 0:
        return 0.0
    return (edge / (odds - 1)) * fraction


def backtest(df, model, train_seasons, test_seasons,
             decay=0.004, min_ev=0.10,
             kelly_fraction=0.25, max_stake_pct=0.03,
             max_odds=4.0):

    train_df = df[df["season"].isin(train_seasons)].copy()
    test_df  = df[df["season"].isin(test_seasons)].copy()

    model.fit(train_df, decay=decay)

    bankroll = 1000.0
    bets     = []

    for _, row in test_df.iterrows():
        if row["HomeTeam"] not in model.team_idx or row["AwayTeam"] not in model.team_idx:
            continue

        pred    = model.predict(row["HomeTeam"], row["AwayTeam"])
        markets = [
            ("H", pred["prob_H"], row["B365H"]),
            ("A", pred["prob_A"], row["B365A"]),
        ]

        for market, prob, odds in markets:
            if odds > max_odds:
                continue
            ev = expected_value(prob, odds)
            if ev < min_ev:
                continue

            stake_pct = min(kelly_stake(prob, odds, kelly_fraction), max_stake_pct)
            stake     = bankroll * stake_pct
            won       = row["FTR"] == market
            pnl       = stake * (odds - 1) if won else -stake
            bankroll  += pnl

            bets.append({
                "date":      row["Date"].strftime("%d/%m/%Y"),
                "match":     f"{row['HomeTeam']} vs {row['AwayTeam']}",
                "market":    market,
                "our_prob":  round(prob, 4),
                "odds":      odds,
                "impl_prob": round(1 / odds, 4),
                "ev":        round(ev, 4),
                "stake":     round(stake, 2),
                "won":       won,
                "pnl":       round(pnl, 2),
                "bankroll":  round(bankroll, 2),
            })

    bets_df = pd.DataFrame(bets)

    if bets_df.empty:
        print("Sin apuestas con valor en el periodo de test.")
        return {}

    total_staked = bets_df["stake"].sum()
    total_pnl    = bets_df["pnl"].sum()
    roi          = total_pnl / total_staked * 100
    win_rate     = bets_df["won"].mean() * 100

    print(f"\n=== BACKTEST ===")
    print(f"Train: {train_seasons} | Test: {test_seasons}")
    print(f"Decay: {decay} | min_ev: {min_ev} | max_odds: {max_odds}")
    print(f"─────────────────────────────────")
    print(f"Apuestas    : {len(bets_df)}")
    print(f"Win rate    : {win_rate:.1f}%")
    print(f"P&L         : ${total_pnl:+.2f}")
    print(f"ROI         : {roi:+.2f}%")
    print(f"Bankroll    : ${bankroll:.2f}  (inicio: $1000)")

    for m in ["H", "A"]:
        sub = bets_df[bets_df["market"] == m]
        if len(sub):
            m_roi = sub["pnl"].sum() / sub["stake"].sum() * 100
            print(f"  {m}: {len(sub)} apuestas | ROI {m_roi:+.2f}% | "
                  f"win rate {sub['won'].mean()*100:.1f}%")

    return {"bets_df": bets_df, "roi": roi, "bankroll": bankroll}


if __name__ == "__main__":
    df = load_data()
    print(f"Datos: {len(df)} partidos | temporadas: {df['season'].unique()}\n")

    for decay in [0.002, 0.004, 0.006]:
        model = PoissonModel()
        backtest(
            df, model,
            train_seasons  = ["2022/23"],
            test_seasons   = ["2023/24"],
            decay          = decay,
            min_ev         = 0.10,
            kelly_fraction = 0.25,
            max_stake_pct  = 0.03,
            max_odds       = 4.0,
        )
        print()
