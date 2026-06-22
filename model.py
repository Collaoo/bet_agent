"""
model.py
Modelo de Poisson entrenado con datos reales de Bundesliga.
Uso:
    python model.py
    python model.py --predict "FC Bayern München" "Borussia Dortmund"
"""

import pandas as pd
import numpy as np
import argparse
import json
from pathlib import Path
from scipy.stats import poisson
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")


def load_data(path: str = "data/bundesliga_matches.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=["home_goals", "away_goals"])
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    return df.sort_values("date").reset_index(drop=True)


class PoissonModel:

    def __init__(self):
        self.teams          = []
        self.team_idx       = {}
        self.attack_params  = None
        self.defense_params = None
        self.home_adv       = None
        self.fitted         = False

    def fit(self, df: pd.DataFrame):
        self.teams    = sorted(df["home_team"].unique().tolist())
        self.team_idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)

        def neg_log_likelihood(params):
            atk  = params[:n]
            dfc  = params[n:2*n]
            hadv = params[-1]
            ll   = 0.0
            for _, row in df.iterrows():
                hi = self.team_idx[row["home_team"]]
                ai = self.team_idx[row["away_team"]]
                lh = np.exp(hadv + atk[hi] - dfc[ai])
                la = np.exp(atk[ai] - dfc[hi])
                ll += poisson.logpmf(int(row["home_goals"]), lh)
                ll += poisson.logpmf(int(row["away_goals"]), la)
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

        print(f"Modelo entrenado | equipos: {n} | partidos: {len(df)} | convergió: {result.success}")

    def predict(self, home: str, away: str, max_goals: int = 8) -> dict:
        assert self.fitted, "Llama fit() primero"

        hi = self.team_idx[home]
        ai = self.team_idx[away]

        lh = np.exp(self.home_adv + self.attack_params[hi] - self.defense_params[ai])
        la = np.exp(self.attack_params[ai] - self.defense_params[hi])

        sm = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                sm[i, j] = poisson.pmf(i, lh) * poisson.pmf(j, la)

        prob_h = float(np.sum(np.tril(sm, -1)))
        prob_d = float(np.sum(np.diag(sm)))
        prob_a = float(np.sum(np.triu(sm, 1)))

        scores = sorted(
            [(i, j, sm[i, j]) for i in range(max_goals+1) for j in range(max_goals+1)],
            key=lambda x: -x[2]
        )

        return {
            "home":        home,
            "away":        away,
            "lambda_home": round(lh, 3),
            "lambda_away": round(la, 3),
            "prob_home":   round(prob_h, 4),
            "prob_draw":   round(prob_d, 4),
            "prob_away":   round(prob_a, 4),
            "fair_odds_h": round(1 / prob_h, 2),
            "fair_odds_d": round(1 / prob_d, 2),
            "fair_odds_a": round(1 / prob_a, 2),
            "top_scores":  [(i, j, round(p * 100, 2)) for i, j, p in scores[:5]],
        }

    def team_strengths(self) -> pd.DataFrame:
        return pd.DataFrame({
            "team":    self.teams,
            "attack":  np.exp(self.attack_params),
            "defense": np.exp(-self.defense_params),
        }).sort_values("attack", ascending=False).reset_index(drop=True)

    def save(self, path: str = "models/poisson_model.json"):
        Path(path).parent.mkdir(exist_ok=True)
        json.dump({
            "teams":          self.teams,
            "attack_params":  self.attack_params.tolist(),
            "defense_params": self.defense_params.tolist(),
            "home_adv":       float(self.home_adv),
        }, open(path, "w"), indent=2)
        print(f"Modelo guardado en {path}")

    def load(self, path: str = "models/poisson_model.json"):
        d = json.load(open(path))
        self.teams          = d["teams"]
        self.team_idx       = {t: i for i, t in enumerate(self.teams)}
        self.attack_params  = np.array(d["attack_params"])
        self.defense_params = np.array(d["defense_params"])
        self.home_adv       = d["home_adv"]
        self.fitted         = True
        print(f"Modelo cargado desde {path}")


def evaluate(df: pd.DataFrame, model: PoissonModel) -> dict:
    """Entrena con 2023, evalúa en 2024."""
    train = df[df["season"] == 2023]
    test  = df[df["season"] == 2024]

    model.fit(train)

    correct = 0
    results = []

    for _, row in test.iterrows():
        if row["home_team"] not in model.team_idx or row["away_team"] not in model.team_idx:
            continue

        pred      = model.predict(row["home_team"], row["away_team"])
        probs     = {"H": pred["prob_home"], "D": pred["prob_draw"], "A": pred["prob_away"]}
        predicted = max(probs, key=probs.get)
        actual    = row["result"]

        if predicted == actual:
            correct += 1

        results.append({
            "date":      row["date"],
            "match":     f"{row['home_team']} vs {row['away_team']}",
            "predicted": predicted,
            "actual":    actual,
            "correct":   predicted == actual,
            "prob_H":    pred["prob_home"],
            "prob_D":    pred["prob_draw"],
            "prob_A":    pred["prob_away"],
        })

    results_df = pd.DataFrame(results)
    accuracy   = correct / len(results_df) * 100

    print(f"\n=== EVALUACIÓN (test: temporada 2024) ===")
    print(f"Partidos evaluados : {len(results_df)}")
    print(f"Correctos          : {correct}")
    print(f"Accuracy           : {accuracy:.1f}%")
    print(f"Baseline aleatorio : 33.3% | típico modelos fútbol: 50-55%")

    for r in ["H", "D", "A"]:
        sub = results_df[results_df["predicted"] == r]
        if len(sub):
            acc_r = sub["correct"].mean() * 100
            print(f"  Predice {r}: {acc_r:.1f}% correcto ({len(sub)} casos)")

    return {"accuracy": accuracy, "results_df": results_df}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", nargs=2, metavar=("HOME", "AWAY"))
    args = parser.parse_args()

    df = load_data()
    print(f"Datos: {len(df)} partidos | temporadas: {sorted(df['season'].unique())}\n")

    model = PoissonModel()

    if args.predict:
        model_path = "models/poisson_model.json"
        if Path(model_path).exists():
            model.load(model_path)
        else:
            model.fit(df)
            model.save(model_path)

        home, away = args.predict
        pred = model.predict(home, away)

        print(f"\n{'='*52}")
        print(f"  {pred['home']} vs {pred['away']}")
        print(f"{'='*52}")
        print(f"  Goles esperados : {pred['lambda_home']} - {pred['lambda_away']}")
        print(f"  Probabilidades  : H {pred['prob_home']*100:.1f}%  D {pred['prob_draw']*100:.1f}%  A {pred['prob_away']*100:.1f}%")
        print(f"  Cuotas justas   : {pred['fair_odds_h']}  /  {pred['fair_odds_d']}  /  {pred['fair_odds_a']}")
        print(f"\n  Marcadores más probables:")
        for hg, ag, p in pred["top_scores"]:
            tag = "Local" if hg > ag else ("Empate" if hg == ag else "Visitante")
            print(f"    {hg}-{ag}  {p:.1f}%  ({tag})")

    else:
        eval_results = evaluate(df, model)
        model.save()

        print("\n=== FUERZA DE EQUIPOS ===")
        print(f"  Ventaja local: {np.exp(model.home_adv):.3f}x\n")
        strengths = model.team_strengths()
        print(f"  {'Equipo':<30} {'Ataque':>8} {'Defensa':>8}")
        print(f"  {'-'*48}")
        for _, row in strengths.iterrows():
            print(f"  {row['team']:<30} {row['attack']:>8.3f} {row['defense']:>8.3f}")
