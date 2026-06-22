"""
live_betting.py
Busca value bets en vivo usando The Odds API + modelo entrenado.

Uso:
    export ODDS_API_KEY="tu_key"
    python scripts/live_betting.py --model models/best_model.json
    python scripts/live_betting.py --sport soccer_epl --min-ev 0.10
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import PoissonModel, PoissonDecay, DixonColesModel, load_data
from data.odds_api import OddsAPIClient
from betting.sizing import FractionalKelly


def load_model(model_path: str = None):
    if model_path and Path(model_path).exists():
        # Detect model type from filename
        name = Path(model_path).stem
        if "dixon" in name.lower():
            model = DixonColesModel()
        else:
            model = PoissonModel()
        model.load(model_path)
        return model

    print("Entrenando modelo con datos históricos...")
    df = load_data()
    model = PoissonDecay().fit(df, decay=0.004)
    model.calibrate(df, verbose=False)
    model.save("models/live_model.json")
    return model


def main():
    parser = argparse.ArgumentParser(description="Live value betting")
    parser.add_argument("--model", default=None, help="Ruta al modelo guardado")
    parser.add_argument("--sport", default="soccer_germany_bundesliga",
                        help="Sport key (default: soccer_germany_bundesliga)")
    parser.add_argument("--min-ev", type=float, default=0.05,
                        help="Minimo expected value")
    parser.add_argument("--max-odds", type=float, default=5.0,
                        help="Maximo odds permitido")
    parser.add_argument("--bankroll", type=float, default=1000,
                        help="Bankroll actual")
    parser.add_argument("--kelly", type=float, default=0.25,
                        help="Fraccion de Kelly")
    parser.add_argument("--regions", default="uk,eu",
                        help="Regiones de odds (uk,eu,us,au)")
    args = parser.parse_args()

    model = load_model(args.model)
    kelly = FractionalKelly(fraction=args.kelly, max_stake_pct=0.03)

    client = OddsAPIClient()

    print(f"\n{'=' * 60}")
    print(f"  BUSCANDO VALUE BETS — {args.sport}")
    print(f"  Modelo: {type(model).__name__}")
    print(f"  Min EV: {args.min_ev} | Max odds: {args.max_odds}")
    print(f"  Kelly: {args.kelly} | Bankroll: ${args.bankroll:.0f}")
    print(f"{'=' * 60}")

    events = client.fetch_odds(
        sport=args.sport,
        regions=args.regions,
        markets="h2h",
    )

    if not events:
        print("\n⚠️  No se pudieron obtener odds. Verificá:")
        print("   1. ODDS_API_KEY configurada")
        print(f"   2. Deporte '{args.sport}' existe")
        print("   3. Conexión a internet")
        return

    print(f"\n  Partidos encontrados: {len(events)}")

    bets = client.value_bets(model, events, min_ev=args.min_ev,
                             max_odds=args.max_odds)

    if not bets:
        print("\n  No se encontraron value bets.")
        return

    print(f"\n  Value bets encontrados: {len(bets)}\n")
    print(f"  {'Match':<40} {'Mkt':>3} {'Prob':>6} {'Odds':>5} "
          f"{'EV':>7} {'Casa':<12}")
    print(f"  {'-'*40} {'-'*3} {'-'*6} {'-'*5} "
          f"{'-'*7} {'-'*12}")

    for b in bets:
        print(f"  {b['match']:<40} {b['market']:>3} "
              f"{b['our_prob']*100:>5.1f}% "
              f"{b['odds']:>5.2f} "
              f"{b['ev']*100:>+6.1f}% "
              f"{b.get('bookmaker', '-'):<12}")

    total_ev = sum(b["ev"] for b in bets)
    print(f"\n  Total expected value: {total_ev*100:+.1f}%")
    print(f"  Kelly stake sugerido: "
          f"${args.bankroll * args.kelly / len(bets):.2f} por bet")

    print(f"\n  Para apostar, ejecutá con:")
    print(f"  --bankroll {args.bankroll} --kelly {args.kelly}")


if __name__ == "__main__":
    main()
