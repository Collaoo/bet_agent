"""
download_data.py
Descarga datos históricos de football-data.co.uk con odds de múltiples casas.
Incluye Bundesliga y opcionalmente otras ligas.

Uso:
    python scripts/download_data.py
    python scripts/download_data.py --leagues D1 D2  # Bundesliga 1 y 2
    python scripts/download_data.py --start 2010      # desde 2010/11
"""

import argparse
import sys
from pathlib import Path
from urllib.request import urlopen
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_URL = "https://www.football-data.co.uk/mmz4281"

LEAGUES = {
    "D1": "bundesliga",
    "D2": "bundesliga_2",
    "E0": "epl",
    "E1": "championship",
    "I1": "serie_a",
    "SP1": "la_liga",
    "F1": "ligue_1",
    "N1": "eredivisie",
    "P1": "primeira_liga",
    "B1": "jupiler_pro_league",
}


def season_codes(start: int, end: int) -> list:
    codes = []
    for year in range(start, end):
        short_start = year % 100
        short_end = (year + 1) % 100
        codes.append(f"{short_start:02d}{short_end:02d}")
    return codes


def download_season(league_code: str, season_code: str) -> pd.DataFrame:
    url = f"{BASE_URL}/{season_code}/{league_code}.csv"
    try:
        df = pd.read_csv(url)
        df["season"] = f"{int(season_code[:2])+2000}/{(int(season_code[:2])+2001) % 100:02d}"
        return df
    except Exception as e:
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leagues", nargs="+", default=["D1"],
                        choices=list(LEAGUES.keys()),
                        help="Codigos de liga (default: D1)")
    parser.add_argument("--start", type=int, default=2000,
                        help="Temporada inicial (default: 2000)")
    parser.add_argument("--end", type=int, default=2026,
                        help="Temporada final (default: 2026)")
    parser.add_argument("--output", default="data/historical_data.csv",
                        help="Archivo de salida")
    args = parser.parse_args()

    codes = season_codes(args.start, args.end)
    all_dfs = []

    print(f"Descargando {len(args.leagues)} ligas, {len(codes)} temporadas...")
    print(f"  Ligas: {', '.join(f'{c} ({LEAGUES[c]})' for c in args.leagues)}")

    for league in args.leagues:
        league_name = LEAGUES[league]
        total = 0
        for sc in codes:
            df = download_season(league, sc)
            if not df.empty:
                df["league"] = league_name
                all_dfs.append(df)
                total += len(df)
                print(f"  {league_name} {sc}: {len(df)} partidos")
        print(f"  → {league_name}: {total} partidos total")

    if not all_dfs:
        print("No se descargaron datos.")
        return

    df = pd.concat(all_dfs, ignore_index=True)

    # Columnas esenciales
    keep = ["league", "season", "Date", "HomeTeam", "AwayTeam",
            "FTHG", "FTAG", "FTR"]

    # Detectar columnas de odds (terminan en H, D, A)
    odds_cols = sorted(set(
        col for col in df.columns
        if col[-1] in ("H", "D", "A")
        and col not in ("FTHG", "FTHG", "HTHG", "HTAG", "FTH", "FTA")
        and col[:-1] in ("B365", "BW", "IW", "PS", "WH", "VC",
                         "SB", "SJ", "SY", "GB", "BS", "PP",
                         "B365C", "BWC", "IWC", "PSC", "WHC", "VCC",
                         "Max", "MaxC", "Avg", "AvgC")
    ))
    keep.extend(odds_cols)

    df = df[keep].copy()
    df = df.dropna(subset=["FTHG", "FTAG", "FTR"])
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    df.to_csv(args.output, index=False)

    print(f"\nTotal: {len(df)} partidos guardados en {args.output}")
    print(f"Columnas: {list(df.columns)}")
    print(f"Bookmakers detectados: {len(odds_cols) // 3}")


if __name__ == "__main__":
    main()
