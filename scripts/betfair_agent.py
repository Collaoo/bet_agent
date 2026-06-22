"""
betfair_agent.py
Agente automático de value betting para Betfair Exchange.
1. Entrena modelo con datos históricos
2. Obtiene odds de Betfair (via Betfair API o The Odds API)
3. Calcula valor esperado para cada partido
4. Coloca apuestas automáticamente si EV > min_ev

Uso:
    # Probar conexión y buscar value bets (simulación)
    python scripts/betfair_agent.py --simulate

    # Ejecución real (requiere credenciales Betfair)
    python scripts/betfair_agent.py --live

    # Especificar credenciales
    python scripts/betfair_agent.py --app-key KEY --username USER --password PASS
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from models import DixonColesModel, load_data
from betting.betfair import BetfairClient, BetfairAPIError

# ── Config ──
MIN_EV = 0.10
MAX_ODDS = 4.0
STAKE_FRACTION = 0.02  # 2% de bankroll por apuesta
BANKROLL = 1000.0

# Equivalencias de nombres de equipos entre football-data.co.uk y Betfair
TEAM_NAME_MAP = {
    "M'gladbach": "Borussia Mönchengladbach",
    "Ein Frankfurt": "Eintracht Frankfurt",
    "FC Koln": "1. FC Köln",
    "Mainz": "1. FSV Mainz 05",
    "St Pauli": "FC St. Pauli",
    "Nurnberg": "1. FC Nürnberg",
    "Ein Frankfurt": "Eintracht Frankfurt",
    "Greuther Furth": "SpVgg Greuther Fürth",
    "Hertha": "Hertha BSC",
    "Leverkusen": "Bayer Leverkusen",
    "M'gladbach": "Borussia Mönchengladbach",
    "Munich 1860": "TSV 1860 München",
    "Paderborn": "SC Paderborn 07",
    "Wolfsburg": "VfL Wolfsburg",
    "Erzgebirge Aue": "FC Erzgebirge Aue",
    "Cottbus": "FC Energie Cottbus",
    "Bielefeld": "Arminia Bielefeld",
    "Karlsruhe": "Karlsruher SC",
    "Aachen": "Alemannia Aachen",
    "Duisburg": "MSV Duisburg",
    "Kaiserslautern": "1. FC Kaiserslautern",
    "Bochum": "VfL Bochum",
    "Stuttgart": "VfB Stuttgart",
    "Hamburg": "HSV",
    "Freiburg": "SC Freiburg",
}


def normalize_team(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)


def find_betfair_bundesliga(client: BetfairClient) -> list:
    """Find current Bundesliga MATCH_ODDS markets from Betfair."""
    print("Buscando mercados de Bundesliga en Betfair...")
    markets = client.find_bundesliga_markets(max_results=200)
    print(f"  Encontrados {len(markets)} mercados")
    return markets





def simulate_pinnacle_proxy(model) -> None:
    """Backtest usando solo odds de Pinnacle como proxy de Betfair."""
    import re
    df = load_data()
    seasons = sorted(df["season"].unique())

    train_df = df[df["season"].isin(seasons[:-1])].copy()
    cal_df = train_df.iloc[int(len(train_df) * 0.7):].copy()
    train_df = train_df.iloc[:int(len(train_df) * 0.7)].copy()
    test_df = df[df["season"] == seasons[-1]].copy()

    # Pinnacle columns
    ps_cols = {"h": [], "d": [], "a": []}
    for c in test_df.columns:
        m = re.match(r"odds_(psc?)_([hda])$", c.lower())
        if m:
            ps_cols[m.group(2)].append((c, m.group(1)))

    bankroll = BANKROLL
    bets = []
    for _, row in test_df.iterrows():
        if row["home_team"] not in model.team_idx or row["away_team"] not in model.team_idx:
            continue
        pred = model.predict_probs(row["home_team"], row["away_team"])

        for mkt, prob in [("H", pred["prob_H"]), ("D", pred["prob_D"]), ("A", pred["prob_A"])]:
            best_odds = 0
            for col, _ in ps_cols.get(mkt.lower(), []):
                v = row.get(col, 0)
                if pd.notna(v) and v > 0 and v > best_odds:
                    best_odds = v
            if best_odds <= 0 or best_odds > MAX_ODDS:
                continue
            ev = prob * best_odds - 1
            if ev < MIN_EV:
                continue

            won = row["result"] == mkt
            stake = bankroll * STAKE_FRACTION
            pnl = stake * (best_odds - 1) if won else -stake
            bankroll += pnl
            bets.append({
                "match": f"{row['home_team']} vs {row['away_team']}",
                "market": mkt, "prob": prob, "odds": best_odds,
                "ev": ev, "won": won, "stake": stake, "pnl": pnl,
            })

    df_bets = pd.DataFrame(bets)
    if df_bets.empty:
        print("  Sin value bets en este periodo.")
        return

    roi = df_bets.pnl.sum() / df_bets.stake.sum() * 100
    win_rate = df_bets.won.mean() * 100

    print(f"\n  {'=' * 50}")
    print(f"  PROXY BETFAIR (Pinnacle odds) — {len(df_bets)} apuestas")
    print(f"  {'=' * 50}")
    print(f"  Bankroll final: ${bankroll:.2f} (inicio: ${BANKROLL})")
    print(f"  ROI: {roi:+.2f}% | Win rate: {win_rate:.1f}% | "
          f"P&L: ${df_bets.pnl.sum():.2f}\n")

    # Top 10 bets
    print("  Top apuestas por EV:")
    for _, b in df_bets.sort_values("ev", ascending=False).head(10).iterrows():
        print(f"  {b['match']:35} | {b['market']} | EV={b['ev']:+.2%} | "
              f"odds={b['odds']:.2f} | {'✓' if b['won'] else '✗'} | "
              f"pnl=${b['pnl']:+.1f}")

    if os.environ.get("BETFAIR_APP_KEY") or os.environ.get("ODDS_API_KEY"):
        print("\n  💡 Credenciales detectadas. Usá --live para apuestas reales.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate", action="store_true", default=True,
                        help="Modo simulación (default)")
    parser.add_argument("--live", action="store_true",
                        help="Modo real: coloca apuestas en Betfair")
    parser.add_argument("--app-key", default=os.getenv("BETFAIR_APP_KEY", ""))
    parser.add_argument("--username", default=os.getenv("BETFAIR_USERNAME", ""))
    parser.add_argument("--password", default=os.getenv("BETFAIR_PASSWORD", ""))
    args = parser.parse_args()

    # Cargar modelo
    print("Cargando modelo Dixon-Coles...")
    df = load_data()
    seasons = sorted(df["season"].unique())
    train_df = df[df["season"].isin(seasons[:-1])].copy()
    cal_df = train_df.iloc[int(len(train_df) * 0.7):].copy()
    train_df = train_df.iloc[:int(len(train_df) * 0.7)].copy()

    model = DixonColesModel()
    model.fit(train_df, decay=0.002)
    model.calibrate(cal_df, verbose=False)
    print(f"  Modelo listo | {len(model.teams)} equipos\n")

    if args.live and args.app_key:
        client = BetfairClient(
            app_key=args.app_key,
            username=args.username,
            password=args.password,
        )
        try:
            client.login()
            print("Conectado a Betfair Exchange\n")
        except BetfairAPIError as e:
            print(f"Error de login: {e}")
            print("Usando modo simulación...\n")
            simulate_pinnacle_proxy(model)
            return

        try:
            markets = find_betfair_bundesliga(client)
        except BetfairAPIError:
            print("  Error API (sin credenciales completas). Usando proxy...")
            simulate_pinnacle_proxy(model)
            return

        if not markets:
            print("  No hay mercados activos. Usando proxy...")
            simulate_pinnacle_proxy(model)
            return

        value_bets = []
        for m in markets:
            event = m.get("event", {})
            home, away = event.get("homeTeam", ""), event.get("awayTeam", "")
            if not home or not away:
                continue
            hn, an = normalize_team(home), normalize_team(away)
            if hn not in model.team_idx or an not in model.team_idx:
                continue
            try:
                raw = client.get_match_odds(m["marketId"])
                odds = client.parse_match_odds(raw, home, away)
            except Exception:
                continue
            pred = model.predict_probs(hn, an)
            for mk, pk, ok in [("H", "prob_H", "odds_h"),
                               ("D", "prob_D", "odds_d"),
                               ("A", "prob_A", "odds_a")]:
                prob, odds_val = pred[pk], odds[ok]
                if odds_val <= 0 or odds_val > MAX_ODDS:
                    continue
                ev = prob * odds_val - 1
                if ev < MIN_EV:
                    continue
                sid = client.get_selection_id(m, home if mk == "H" else ("Draw" if mk == "D" else away))
                value_bets.append({
                    "market_id": m["marketId"],
                    "selection_id": sid, "side": "BACK",
                    "price": odds_val, "size": round(BANKROLL * STAKE_FRACTION, 2),
                    "ev": ev, "match": f"{home} vs {away}", "market": mk,
                })
        if not value_bets:
            print("  Sin value bets hoy.")
            return
        print(f"\n  Colocando {len(value_bets)} apuestas...")
        for vb in sorted(value_bets, key=lambda x: -x["ev"]):
            try:
                result = client.place_bet(
                    vb["market_id"], vb["selection_id"],
                    vb["side"], vb["price"], vb["size"],
                )
                status = result.get("status", "")
                print(f"  {vb['match']:35} | {vb['market']} | "
                      f"odds={vb['price']:.2f} | EV={vb['ev']:.1%} | "
                      f"${vb['size']:.1f} | {status}")
            except BetfairAPIError as e:
                print(f"  ❌ {vb['match']}: {e}")
    else:
        simulate_pinnacle_proxy(model)


if __name__ == "__main__":
    main()
