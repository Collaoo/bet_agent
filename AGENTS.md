# bet_agent — Agente de Value Betting para Betfair

## Objetivo
Sistema de apuestas automáticas que predice resultados de fútbol usando modelos Poisson y encuentra value bets en Betfair Exchange.

## Modelos (models/)
- **`base.py`** — `PoissonBase`: clase base con fit vectorizado, calibración por temperatura, normalización multi-bookmaker
- **`poisson.py`** — `PoissonModel`: Poisson estándar
- **`decay.py`** — `PoissonDecay`: con decay temporal (equipos recientes pesan más)
- **`dixon_coles.py`** — `DixonColesModel`: corrige correlación en resultados de pocos goles (0-0, 1-0)
- **`load_data()`** — por defecto usa `data/historical_data_bundesliga.csv`

## Betting Engine (betting/)
- **`engine.py`** — `BettingEngine`: itera partidos, compara probs del modelo vs odds de múltiples bookmakers, filtra por EV, coloca apuesta
- **`markets.py`** — Cálculo de O/U 2.5, BTTS, hándicap asiático, fair odds
- **`sizing.py`** — FractionalKelly, FlatStaking, ConfidenceBased, AdaptiveDrawdown
- **`analysis.py`** — Sharpe, Sortino, drawdown, profit factor
- **`betfair.py`** — Cliente Betfair API-NG: login, listMarketCatalogue, get_match_odds, placeOrders

## Evaluación (evaluate/)
- **`metrics.py`** — Brier score, log-loss, accuracy, ECE
- **`calibration.py`** — Platt scaling, isotonic, calibration_curve
- **`walkforward.py`** — Walk-forward cross validation, optimize_decay

## Data
- **`data/historical_data_bundesliga.csv`** — 13,158 partidos (Bundesliga 1 y 2, 2004–2026) con odds de 11 bookmakers
- **`data/odds_api.py`** — Cliente para The Odds API (API key: `492758f63369838302d7d8a9ded849e5`)

## Scripts
- **`scripts/download_data.py`** — Descarga datos históricos de football-data.co.uk
- **`scripts/betfair_agent.py`** — Agente automático: entrena modelo, consulta Betfair, apuesta si EV > 10%
- **`scripts/live_betting.py`** — Value betting con The Odds API

## Resultados del Backtest (temporada 2025/26)

### Multi-bookmaker (mejor odds entre 11 casas)
| Modelo | Config | ROI | Sharpe | Bets |
|--------|--------|:---:|:------:|:----:|
| DixonColes | Kelly 1/4, EV≥15% | **+5.53%** | 1.01 | 239 |
| DixonColes | Kelly 1/4, EV≥10% | **+5.38%** | 1.16 | 292 |
| DixonColes | Flat 2%, EV≥15% | +5.46% | 1.00 | 239 |
| PoissonDecay | Flat 2%, EV≥15% | +2.35% | 0.52 | 287 |
| Poisson | Flat 2%, EV≥15% | +1.00% | 0.17 | 223 |

### Proxy Betfair (solo Pinnacle — bookmaker sharp)
| Config | ROI | Win% | Bets |
|--------|:---:|:----:|:----:|
| DixonColes + Kelly 1/4, EV≥10% | **+17.83%** | 45.5% | 123 |

### Por bookmaker (quién aporta el edge)
| Bookmaker | ROI | Bets |
|-----------|:---:|:----:|
| Pinnacle (PS) | +10.58% | 45 |
| BetWay (BW) | +7.05% | 65 |
| Bet365 (B365) | +4.15% | 127 |
| Pinnacle Closing (PSC) | +1.82% | 55 |


## Cómo ejecutar

```bash
# Simulación con proxy Pinnacle (no necesita credenciales)
python scripts/betfair_agent.py

# Modo real (cuando haya temporada activa)

# Backtest completo
python test_models.py

# Backtest rápido
python test_models.py --quick
```

## Bugs corregidos
1. **`betting/engine.py`** — `_get_best_odds()` y `_resolve_odds()` usaban market en uppercase (`"H"`) pero las columnas están en lowercase (`"h"`)

## Pendientes
- [ ] Obtener Live App Key de Betfair (developer.betfair.com > cambiar a tipo Live)
- [ ] Agregar más mercados (O/U 2.5, hándicap asiático) al betfair_agent
- [ ] Guardar historial de apuestas en SQLite
- [ ] Notificaciones (Telegram/email) cuando se coloca una apuesta
- [ ] Scheduler automático (cron cada N minutos)
- [ ] Subir a GitHub (con .gitignore para credenciales)
