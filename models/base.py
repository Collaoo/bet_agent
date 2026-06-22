import numpy as np
import pandas as pd
import json
from pathlib import Path
from scipy.stats import poisson
from scipy.optimize import minimize, minimize_scalar
import warnings
warnings.filterwarnings("ignore")

BOOKIE_PREFIXES = ["B365", "BW", "IW", "PS", "WH", "VC",
                   "SB", "SJ", "SY", "GB", "BS", "PP",
                   "PIN", "PSC", "Pinnacle", "Betfair", "BF",
                   "BetCR", "Fa", "VCS", "Interwetten",
                   "Unibet", "UB", "Marathon", "MR"]

def _normalize_odds_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and normalize multi-bookmaker odds columns."""
    rename = {}
    for col in df.columns:
        for prefix in BOOKIE_PREFIXES:
            if col == f"{prefix}H":
                rename[col] = f"odds_{prefix}_h"
            elif col == f"{prefix}D":
                rename[col] = f"odds_{prefix}_d"
            elif col == f"{prefix}A":
                rename[col] = f"odds_{prefix}_a"
    for col in list(df.columns):
        low = col.lower().replace(" ", "_")
        if low.startswith("odds_") and col != low:
            rename[col] = low
    return df.rename(columns=rename)

COLUMN_MAP = {
    "FTHG": "home_goals", "FTAG": "away_goals", "FTR": "result",
    "HomeTeam": "home_team", "AwayTeam": "away_team",
    "home_team": "home_team", "away_team": "away_team",
    "home_goals": "home_goals", "away_goals": "away_goals",
    "result": "result", "Date": "date", "date": "date",
    "season": "season",
    "B365H": "odds_h", "B365D": "odds_d", "B365A": "odds_a",
}

def load_data(path: str = "data/historical_data_bundesliga.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={c: COLUMN_MAP.get(c, c) for c in df.columns})
    df = _normalize_odds_cols(df)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["home_goals", "away_goals"])
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    return df.sort_values("date").reset_index(drop=True)


class PoissonBase:

    def __init__(self):
        self.teams = []
        self.team_idx = {}
        self.attack_params = None
        self.defense_params = None
        self.home_adv = None
        self.fitted = False
        self._temperature = None

    # ── Extension points ──
    def _n_extra(self):
        return 1

    def _extra_init(self):
        return [0.1]

    def _extra_bounds(self):
        return [(None, None)]

    def _score_prob(self, hg, ag, lh, la):
        return poisson.pmf(hg, lh) * poisson.pmf(ag, la)

    def _log_likelihood_contrib(self, hg, ag, lh, la):
        return poisson.logpmf(hg, lh) + poisson.logpmf(ag, la)

    # ── Fitting ──
    def _parse_params(self, params):
        n = len(self.teams)
        atk = params[:n]
        dfc = params[n:2*n]
        self.home_adv = params[2*n]
        return atk, dfc, [self.home_adv]

    def _neg_log_likelihood(self, params):
        atk, dfc, _ = self._parse_params(params)
        lh = np.exp(self.home_adv + atk[self._home_idx] - dfc[self._away_idx])
        la = np.exp(atk[self._away_idx] - dfc[self._home_idx])
        ll = np.sum(self._weights * self._log_likelihood_contrib(
            self._home_goals, self._away_goals, lh, la))
        return -ll

    def fit(self, df: pd.DataFrame, decay: float = 0.0, verbose: bool = True) -> "PoissonBase":
        self.teams = sorted(set(df["home_team"].unique()) | set(df["away_team"].unique()))
        self.team_idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)

        self._home_idx = df["home_team"].map(self.team_idx).values.astype(np.int32)
        self._away_idx = df["away_team"].map(self.team_idx).values.astype(np.int32)
        self._home_goals = df["home_goals"].values.astype(np.int32)
        self._away_goals = df["away_goals"].values.astype(np.int32)

        max_date = df["date"].max()
        days_ago = (max_date - df["date"]).dt.days
        self._weights = np.exp(-decay * days_ago).values

        x0 = np.concatenate([np.zeros(2 * n), self._extra_init()])
        bounds = [(None, None)] * (2 * n) + self._extra_bounds()
        eq_constraint = {"type": "eq", "fun": lambda p: np.sum(p[:n])}

        result = minimize(
            self._neg_log_likelihood, x0,
            method="SLSQP",
            bounds=bounds,
            constraints=eq_constraint,
            options={"maxiter": 1000, "ftol": 1e-8},
        )

        self.attack_params = result.x[:n]
        self.defense_params = result.x[n:2*n]
        self.home_adv = result.x[2*n]
        self.fitted = True

        if verbose:
            print(f"{type(self).__name__} entrenado | equipos: {n} | partidos: {len(df)} | "
                  f"decay: {decay} | convergió: {result.success} | iters: {result.nit}")
        return self

    # ── Calibration ──
    def _calibrate_matrix(self, sm: np.ndarray) -> np.ndarray:
        if self._temperature is None or abs(self._temperature - 1.0) < 1e-6:
            return sm
        sm_cal = sm ** (1.0 / self._temperature)
        total = sm_cal.sum()
        if total > 0:
            sm_cal = sm_cal / total
        return sm_cal

    def calibrate(self, df: pd.DataFrame, max_goals: int = 8,
                  verbose: bool = True) -> "PoissonBase":
        if not self.fitted:
            raise ValueError("Fit the model before calibrating.")

        def _nll(T):
            ll = 0.0
            for _, row in df.iterrows():
                if row["home_team"] not in self.team_idx or row["away_team"] not in self.team_idx:
                    continue
                _, _, sm = self._build_raw_score_matrix(row["home_team"], row["away_team"], max_goals)
                sm_cal = sm ** (1.0 / max(T, 0.01))
                sm_cal = sm_cal / sm_cal.sum()
                hg, ag = int(row["home_goals"]), int(row["away_goals"])
                if hg <= max_goals and ag <= max_goals:
                    ll += np.log(max(sm_cal[hg, ag], 1e-15))
            return -ll

        result = minimize_scalar(_nll, bounds=(0.05, 10.0), method="bounded")
        self._temperature = float(result.x)
        if verbose:
            print(f"  Temperatura optima: {self._temperature:.4f} | NLL: {result.fun:.4f}")
        return self

    # ── Predict ──
    def _build_raw_score_matrix(self, home: str, away: str, max_goals: int = 8):
        hi = self.team_idx[home]
        ai = self.team_idx[away]
        lh = float(np.exp(self.home_adv + self.attack_params[hi] - self.defense_params[ai]))
        la = float(np.exp(self.attack_params[ai] - self.defense_params[hi]))
        hg_arr = np.arange(max_goals + 1, dtype=int)
        ag_arr = np.arange(max_goals + 1, dtype=int)
        hg_grid, ag_grid = np.meshgrid(hg_arr, ag_arr, indexing="ij")
        sm = self._score_prob(hg_grid, ag_grid, lh, la)
        total = sm.sum()
        if total > 0:
            sm = sm / total
        return lh, la, sm

    def _build_score_matrix(self, home: str, away: str, max_goals: int = 8):
        lh, la, sm = self._build_raw_score_matrix(home, away, max_goals)
        sm = self._calibrate_matrix(sm)
        return lh, la, sm

    def predict(self, home: str, away: str, max_goals: int = 8) -> dict:
        lh, la, sm = self._build_score_matrix(home, away, max_goals)
        prob_h = float(np.sum(np.tril(sm, -1)))
        prob_d = float(np.sum(np.diag(sm)))
        prob_a = float(np.sum(np.triu(sm, 1)))

        scores = sorted(
            [(i, j, sm[i, j]) for i in range(max_goals+1) for j in range(max_goals+1)],
            key=lambda x: -x[2],
        )

        return {
            "home": home, "away": away,
            "lambda_home": round(lh, 3), "lambda_away": round(la, 3),
            "prob_home": round(prob_h, 4), "prob_draw": round(prob_d, 4), "prob_away": round(prob_a, 4),
            "fair_odds_h": round(1 / prob_h, 2) if prob_h > 0 else float("inf"),
            "fair_odds_d": round(1 / prob_d, 2) if prob_d > 0 else float("inf"),
            "fair_odds_a": round(1 / prob_a, 2) if prob_a > 0 else float("inf"),
            "top_scores": [(i, j, round(p * 100, 2)) for i, j, p in scores[:5]],
            "score_matrix": sm,
        }

    def predict_probs(self, home: str, away: str, max_goals: int = 8) -> dict:
        lh, la, sm = self._build_score_matrix(home, away, max_goals)
        return {
            "prob_H": float(np.sum(np.tril(sm, -1))),
            "prob_D": float(np.sum(np.diag(sm))),
            "prob_A": float(np.sum(np.triu(sm, 1))),
            "lambda_home": lh, "lambda_away": la,
            "score_matrix": sm,
        }

    def team_strengths(self) -> pd.DataFrame:
        return pd.DataFrame({
            "team": self.teams,
            "attack": np.exp(self.attack_params),
            "defense": np.exp(-self.defense_params),
        }).sort_values("attack", ascending=False).reset_index(drop=True)

    def save(self, path: str = "models/poisson_model.json"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "teams": self.teams,
            "attack_params": self.attack_params.tolist(),
            "defense_params": self.defense_params.tolist(),
            "home_adv": float(self.home_adv),
        }
        if self._temperature is not None:
            data["temperature"] = float(self._temperature)
        json.dump(data, open(path, "w"), indent=2)

    def load(self, path: str = "models/poisson_model.json"):
        d = json.load(open(path))
        self.teams = d["teams"]
        self.team_idx = {t: i for i, t in enumerate(self.teams)}
        self.attack_params = np.array(d["attack_params"])
        self.defense_params = np.array(d["defense_params"])
        self.home_adv = d["home_adv"]
        self._temperature = d.get("temperature")
        self.fitted = True
        print(f"Modelo cargado desde {path}")
        return self
