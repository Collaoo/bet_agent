"""
odds_api.py
Cliente para The Odds API (the-odds-api.com).
Obtiene odds en vivo de 30+ casas de apuestas.

Uso:
    from data.odds_api import OddsAPIClient

    client = OddsAPIClient(api_key="tu_key")
    events = client.fetch_odds()
    best = client.get_best_odds(events)
"""

import os
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError

BASE_URL = "https://api.the-odds-api.com/v4"

SPORTS = {
    "soccer_germany_bundesliga": "Bundesliga",
    "soccer_epl": "Premier League",
    "soccer_spain_la_liga": "La Liga",
    "soccer_italy_serie_a": "Serie A",
    "soccer_france_ligue_one": "Ligue 1",
}

REGIONS = ["uk", "eu", "us", "au"]
MARKETS = ["h2h", "totals"]


class OddsAPIClient:

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("ODDS_API_KEY", "")
        if not self.api_key:
            print("⚠️  ODDS_API_KEY no configurada. "
                  "Obtené una gratis en https://the-odds-api.com")
        self._requests_used = 0

    def _get(self, path: str, params: dict = None) -> dict:
        if not self.api_key:
            return {}
        params_str = "&".join(f"{k}={v}" for k, v in (params or {}).items())
        url = f"{BASE_URL}/{path}?apiKey={self.api_key}&{params_str}"
        req = Request(url, headers={"User-Agent": "bet_agent/1.0"})
        try:
            with urlopen(req, timeout=15) as resp:
                self._requests_used += 1
                return json.loads(resp.read())
        except HTTPError as e:
            body = e.read().decode()
            print(f"⚠️  API error {e.code}: {body}")
            return {}

    def get_usage(self) -> dict:
        data = self._get("sports")
        if isinstance(data, list):
            return {"sports_available": len(data), "requests_used": self._requests_used}
        return data

    def fetch_odds(self, sport: str = "soccer_germany_bundesliga",
                   regions: str = "uk,eu",
                   markets: str = "h2h,totals",
                   bookmakers: str = None) -> list:
        params = {
            "regions": regions,
            "markets": markets,
        }
        if bookmakers:
            params["bookmakers"] = bookmakers
        data = self._get(f"sports/{sport}/odds", params)
        if not data:
            return []
        return data if isinstance(data, list) else []

    def get_best_odds(self, events: list, market: str = "h2h") -> list:
        results = []
        for event in events:
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            commence = event.get("commence_time", "")

            best = {}
            for bm in event.get("bookmakers", []):
                for m in bm.get("markets", []):
                    if m["key"] != market:
                        continue
                    for outcome in m["outcomes"]:
                        name = outcome["name"]
                        price = outcome["price"]
                        bm_key = bm["key"]
                        if name not in best or price > best[name]["price"]:
                            best[name] = {"price": price, "bookmaker": bm_key}

            if "h2h" in market and home and away:
                h_out = best.get(home, {})
                d_out = best.get("Draw", {})
                a_out = best.get(away, {})
                if h_out and d_out and a_out:
                    results.append({
                        "home_team": home,
                        "away_team": away,
                        "commence_time": commence,
                        "best_h": round(h_out["price"], 2),
                        "best_d": round(d_out["price"], 2),
                        "best_a": round(a_out["price"], 2),
                        "bookmaker_h": h_out["bookmaker"],
                        "bookmaker_d": d_out["bookmaker"],
                        "bookmaker_a": a_out["bookmaker"],
                    })
            elif "totals" in market:
                for outcome in best:
                    ov = best[outcome]
                    results.append({
                        "match": f"{home} vs {away}",
                        "point": outcome.get("point", ""),
                        "over_price": ov.get("price", ""),
                        "under_price": "",
                    })

        return results

    def best_odds_to_row(self, best_odds: dict) -> dict:
        return {
            "home_team": best_odds["home_team"],
            "away_team": best_odds["away_team"],
            "date": best_odds.get("commence_time", ""),
            "odds_h": best_odds["best_h"],
            "odds_d": best_odds["best_d"],
            "odds_a": best_odds["best_a"],
            "odds_h_bookmaker": best_odds["bookmaker_h"],
            "odds_d_bookmaker": best_odds["bookmaker_d"],
            "odds_a_bookmaker": best_odds["bookmaker_a"],
        }

    def value_bets(self, model, events: list, min_ev: float = 0.05,
                   max_odds: float = 4.0) -> list:
        best_odds_list = self.get_best_odds(events, market="h2h")
        bets = []

        for odds_row in best_odds_list:
            home, away = odds_row["home_team"], odds_row["away_team"]
            if home not in model.team_idx or away not in model.team_idx:
                continue

            pred = model.predict_probs(home, away)
            row = self.best_odds_to_row(odds_row)

            for market_key, prob_key, odds_key in [
                ("H", "prob_H", "odds_h"),
                ("D", "prob_D", "odds_d"),
                ("A", "prob_A", "odds_a"),
            ]:
                prob = pred[prob_key]
                odds = row[odds_key]
                if odds <= 0 or odds > max_odds:
                    continue
                ev = prob * odds - 1
                if ev < min_ev:
                    continue
                bets.append({
                    "date": row.get("date", ""),
                    "match": f"{home} vs {away}",
                    "market": market_key,
                    "our_prob": round(prob, 4),
                    "odds": odds,
                    "impl_prob": round(1 / odds, 4),
                    "ev": round(ev, 4),
                    "bookmaker": row.get(f"{odds_key}_bookmaker", ""),
                })

        return sorted(bets, key=lambda x: -x["ev"])
