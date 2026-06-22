from betting.betfair import BetfairClient

client = BetfairClient(
    app_key="89QyugmMoH3Y8BA3",
    username="cristobal.collao95@gmail.com",
    password="Notevasameter1..",
)
token = client.login()
print("Token:", token[:30] + "...")

# Probar listar eventos
events = client.list_events("soccer")
print(f"Eventos activos: {len(events)}")

# Buscar mercados de Bundesliga
markets = client.find_bundesliga_markets(max_results=50)
print(f"Mercados Bundesliga encontrados: {len(markets)}")
for m in markets[:5]:
    event = m.get("event", {})
    print(f"  {event.get('name', '?')} - marketId: {m['marketId']}")

# Si hay mercados, mostrar odds del primero
if markets:
    mkt = markets[0]
    odds = client.get_match_odds(mkt["marketId"])
    print(f"\nOdds para {mkt.get('event',{}).get('name','?')}:")
    for name, data in odds.items():
        if data["price"] > 0:
            print(f"  {name}: {data['price']} (size: {data['size']})")