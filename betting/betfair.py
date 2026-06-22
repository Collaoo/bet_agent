"""
betfair.py
Cliente para la API-NG de Betfair Exchange.
Login, consulta de mercados, odds y colocación de apuestas.

Uso:
    from betting.betfair import BetfairClient
    client = BetfairClient(app_key="...", session_token="...")
    events = client.list_events("soccer")
    market = client.get_market_catalogue(event_id="...")
    book = client.get_market_book(market_id="...")
    result = client.place_bet(market_id="...", side="BACK", price=2.5, size=10)
"""

import json
import time
import hashlib
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from urllib.error import HTTPError

IDENTITY_URL = "https://identitysso.betfair.com/api/login"
API_URL = "https://api.betfair.com/exchange/betting/json-rpc/v1"
NAV_URL = "https://api.betfair.com/exchange/betting/rest/v1/navigation/menu.json"

# Mapeo de nombres de liga a event type IDs de Betfair
SPORT_IDS = {
    "soccer": 1,
    "tennis": 2,
    "horse_racing": 7,
}


class BetfairAPIError(Exception):
    pass


class BetfairClient:

    def __init__(self, app_key: str = "", session_token: str = "",
                 username: str = "", password: str = ""):
        self.app_key = app_key
        self.session_token = session_token
        self.username = username
        self.password = password
        self._certs = None

    # ── Auth ──

    def login(self) -> str:
        """Login with username/password, returns session token."""
        if not self.username or not self.password:
            raise BetfairAPIError("Username and password required for login.")
        body = f"username={self.username}&password={self.password}"
        req = Request(
            IDENTITY_URL, data=body.encode(),
            headers={
                "X-Application": self.app_key,
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
        )
        try:
            with urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
        except HTTPError as e:
            raise BetfairAPIError(f"Login HTTP {e.code}: {e.read().decode()}")

        if data.get("status") == "SUCCESS":
            self.session_token = data["token"]
            return self.session_token
        raise BetfairAPIError(f"Login failed: {data.get('error')}")

    def _headers(self) -> dict:
        h = {"X-Application": self.app_key, "Accept": "application/json"}
        if self.session_token:
            h["X-Authentication"] = self.session_token
        return h

    def _rpc(self, method: str, params: dict = None) -> dict:
        body = json.dumps({
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": 1,
        }).encode()
        req = Request(API_URL, data=body, headers=self._headers())
        try:
            with urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
        except HTTPError as e:
            raise BetfairAPIError(f"RPC HTTP {e.code}: {e.read().decode()}")

        if "error" in data:
            raise BetfairAPIError(data["error"].get("message", str(data["error"])))
        return data.get("result", {})

    # ── Navigation / Events ──

    def get_navigation(self) -> list:
        """Get full event navigation tree."""
        req = Request(NAV_URL, headers=self._headers())
        with urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())

    def list_events(self, sport: str = "soccer") -> list:
        """List all active events for a sport."""
        result = self._rpc("SportsAPING/v1.0/listEvents", {
            "filter": {"eventTypeIds": [SPORT_IDS.get(sport, 1)]},
        })
        return result

    def list_competitions(self, sport: str = "soccer") -> list:
        result = self._rpc("SportsAPING/v1.0/listCompetitions", {
            "filter": {"eventTypeIds": [SPORT_IDS.get(sport, 1)]},
        })
        return result

    # ── Markets ──

    def list_market_catalogue(
        self, event_ids: list = None,
        competition_ids: list = None,
        max_results: int = 50,
    ) -> list:
        """List markets for given events or competitions."""
        filt = {"eventTypeIds": [1]}
        if event_ids:
            filt["eventIds"] = event_ids
        if competition_ids:
            filt["competitionIds"] = competition_ids

        result = self._rpc("SportsAPING/v1.0/listMarketCatalogue", {
            "filter": filt,
            "maxResults": max_results,
            "marketProjection": [
                "COMPETITION", "EVENT", "MARKET_DESCRIPTION",
                "RUNNER_DESCRIPTION", "RUNNER_METADATA",
            ],
        })
        return result

    def get_market_book(self, market_ids: list,
                        price_projection: str = "EX_BEST_OFFERS") -> list:
        """Get current prices for markets."""
        result = self._rpc("SportsAPING/v1.0/listMarketBook", {
            "marketIds": market_ids,
            "priceProjection": {
                "priceData": ["EX_BEST_OFFERS", "EX_TRADED_VOLUME"],
            },
        })
        return result

    # ── Prices helper ──

    def get_match_odds(self, market_id: str) -> dict:
        """Get BACK prices for MATCH_ODDS market.
        Returns: {'home': {'price': x, 'size': y}, 'draw': ..., 'away': ...}
        """
        book = self.get_market_book([market_id])
        if not book:
            return {}

        runners = book[0].get("runners", [])
        result = {}
        for r in runners:
            selection_id = r["selectionId"]
            name = r.get("runnerName", str(selection_id))
            back = r.get("ex", {}).get("availableToBack", [])
            if back:
                best = max(back, key=lambda x: x["price"])
                result[name] = {"price": best["price"], "size": best["size"]}
            else:
                result[name] = {"price": 0, "size": 0}
        return result

    def parse_match_odds(self, raw: dict, home_team: str, away_team: str) -> dict:
        """Convert raw match odds to home/draw/away format."""
        h = raw.get(home_team, {}).get("price", 0)
        d = raw.get("Draw", {}).get("price", 0)
        a = raw.get(away_team, {}).get("price", 0)
        return {"odds_h": h, "odds_d": d, "odds_a": a}

    # ── Place Bets ──

    def place_bet(self, market_id: str, selection_id: int,
                  side: str, price: float, size: float,
                  order_type: str = "LIMIT",
                  persistence: str = "LAPSE") -> dict:
        """Place a single bet on Betfair.
        side: "BACK" (bet to win) or "LAY" (bet to lose)
        """
        result = self._rpc("SportsAPING/v1.0/placeOrders", {
            "marketId": market_id,
            "instructions": [{
                "selectionId": selection_id,
                "side": side,
                "orderType": order_type,
                "limitOrder": {
                    "price": price,
                    "size": size,
                    "persistenceType": persistence,
                },
            }],
        })
        return result

    def place_bets_batch(self, instructions: list) -> dict:
        """Place multiple bets in one request.
        instructions: list of dicts with marketId, selectionId, side, price, size
        """
        result = self._rpc("SportsAPING/v1.0/placeOrders", {
            "marketId": instructions[0]["marketId"],
            "instructions": [
                {
                    "selectionId": i["selectionId"],
                    "side": i["side"],
                    "orderType": "LIMIT",
                    "limitOrder": {
                        "price": i["price"],
                        "size": i["size"],
                        "persistenceType": i.get("persistence", "LAPSE"),
                    },
                }
                for i in instructions
            ],
        })
        return result

    # ── Account ──

    def get_account_funds(self) -> dict:
        result = self._rpc("AccountAPING/v1.0/getAccountFunds", {})
        return result

    def list_cleared_orders(self, from_date: str = None,
                            to_date: str = None) -> list:
        params = {"betStatus": "SETTLED"}
        if from_date:
            params["fromDate"] = from_date
        if to_date:
            params["toDate"] = to_date
        result = self._rpc("AccountAPING/v1.0/listClearedOrders", params)
        return result.get("clearedOrders", [])

    # ── Utils ──

    def find_bundesliga_markets(self, max_results: int = 100) -> list:
        """Find MATCH_ODDS markets for current Bundesliga matches."""
        catalog = self.list_market_catalogue(max_results=max_results)
        markets = []
        for m in catalog:
            desc = m.get("description", {})
            if desc.get("marketType") == "MATCH_ODDS" and "Bundesliga" in str(m.get("competition", {}).get("name", "")):
                markets.append(m)
        return markets

    def get_selection_id(self, market: dict, runner_name: str) -> int:
        """Get selection ID for a runner by name."""
        for r in market.get("runners", []):
            if r.get("runnerName", "").lower() == runner_name.lower():
                return r["selectionId"]
        return 0
