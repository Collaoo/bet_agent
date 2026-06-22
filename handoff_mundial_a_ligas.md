# Handoff: Mundial 2026 → Ligas Europeas (Premier + Bundesliga)

## 1. Estado Actual del Proyecto

### Código existente (`bet_agent/`)
| Archivo | Propósito |
|:--------|:-----------|
| `models/base.py` | `PoissonBase` — fit vectorizado, calibración por temperatura, predict |
| `models/poisson.py` | `PoissonModel` — Poisson estándar |
| `models/decay.py` | `PoissonDecay` — con temporal decay |
| `models/dixon_coles.py` | `DixonColesModel` — corrige correlación 0-0, 1-0 |
| `betting/engine.py` | `BettingEngine` — backtest multi-bookmaker, walk-forward |
| `betting/sizing.py` | `FractionalKelly`, `FlatStaking`, `ConfidenceBased`, `AdaptiveDrawdown` |
| `betting/analysis.py` | Sharpe, Sortino, drawdown, profit factor |
| `betting/markets.py` | O/U 2.5, BTTS, fair odds |
| `evaluate/metrics.py` | Brier, log-loss, accuracy, ECE |
| `evaluate/calibration.py` | Platt scaling, isotonic, calibration_curve |
| `evaluate/walkforward.py` | Walk-forward CV, optimize_decay |
| `test_models.py` | Suit de backtest completa (3 modelos × 2 sizings × 4 min_ev) |
| `data/wc_2026_xg.csv` | 17 partidos del Mundial 2026 con xG (xGscore.io scraping) |
| `data/historical_data_bundesliga.csv` | 13,158 partidos Bundesliga 1 y 2 (2004–2026), 11 bookmakers |
| `data/odds_api.py` | Cliente The Odds API (key: `492758f63369838302d7d8a9ded849e5`) |

### Resultados del backtest Bundesliga (temporada 2025/26)
| Modelo | Config | ROI | Sharpe | Bets |
|--------|--------|:---:|:------:|:----:|
| DixonColes | Kelly 1/4, EV≥15% | **+5.53%** | 1.01 | 239 |
| DixonColes | Kelly 1/4, EV≥10% | **+5.38%** | 1.16 | 292 |
| PoissonDecay | Flat 2%, EV≥15% | +2.35% | 0.52 | 287 |
| Poisson | Flat 2%, EV≥15% | +1.00% | 0.17 | 223 |

## 2. Arquitectura del Mundial 2026 (plan discutido, NO implementado)

Para el Mundial 2026 se diseñó un sistema de **dos ratings** que se suma al pipeline existente:

### Componentes nuevos (pendientes de crear)

```
scripts/predictor.py           ← predict_probs(), compute_kelly(), normalize(), platt_calibrate()
scripts/elo_model.py           ← compute_elo() genérico con name normalization
scripts/fetch_xg.py            ← xGscore.io scraping via BeautifulSoup
scripts/calibrar_xg_rating.py  ← walk-forward calibration de alpha
scripts/calibrar_platt.py      ← entrenamiento Platt scaling (multinomial logit)
scripts/calibrar_temperature.py ← temperature scaling (descartado, T≈1.0)
scripts/comparar_modelos.py    ← live monitoring Elo vs Elo+xG
scripts/tabla_xg.py            ← tabla Coolbet con Elo+xG
scripts/tabla_final.py         ← tabla definitiva con Kelly en vivo

models/xg_rating.py            ← clase XGRating (K=60, update por share de xG)
models/platt_calibrator.pkl    ← calibrator Platt entrenado
models/elo_prior_dixon_coles.json ← modelo EPDC (solo usado para el dict elo_ratings)

data/international_results_raw.csv ← ~49k partidos internacionales martj42 (1872–2026)
data/wc_2026_xg.csv            ← YA EXISTE (17 partidos con xG)
```

### Pipeline completo (como se planeó)

```
1. Entrenar Elo baseline
   compute_elo() sobre international_results_raw.csv (K global)
   → dict elo_ratings {team: rating}

2. Extraer xG de cada partido del Mundial
   fetch_xg.py → wc_2026_xg.csv (17 partidos)
   → xG Rating arranca en 1500, K_xg=60, update por share de xG

3. Combinar ratings con alpha=0.10
   elo_rating * (1-alpha) + xg_rating * alpha
   → score final para Poisson(λ=2.5)

4. Calibrar con Platt scaling
   Features: [log(ph/pd), log(pa/pd)], C=0.1
   → Guardar en platt_calibrator.pkl

5. Calcular Kelly vs odds de Coolbet
   Kelly 1/10, EV≥1%, bank $20,000 fijo
```

### Decisiones técnicas clave del Mundial
- **λ=2.5 global**: óptimo en 406 partidos históricos de Mundial (Brier 0.5680). No necesita temperature scaling.
- **alpha=0.10**: se calibró walk-forward sobre 17 partidos. Con solo 1 match/equipo, el xGR tiene poca señal; alpha subirá tras matchday 3.
- **Platt marginal pero aceptado**: LogLoss CV 0.9678 vs raw 0.9641 (empeora). Sin embargo reduce sobreconfianza en favoritos extremos (England 84%→81%).
- **xG Rating K=60**: porque el xG es menos volátil que goles reales. K Elo normal = 30 para internacionales.
- **Kelly 1/10**: conservador porque el modelo no está probado. En Bundesliga se usó Kelly 1/4 con buenos resultados.
- **Descartado Dixon-Coles attack/defense**: los parámetros tienen sesgo por eliminatorias disparejas. Elo puro + xG Rating es más robusto.
- **Descartado RL/LinUCB para Mundial**: 64 partidos cada 4 años es insuficiente para entrenar.

## 3. Plan para Ligas (Premier + Bundesliga)

### Datos
| Fuente | Cobertura | Uso |
|:-------|:----------|:----|
| football-data.co.uk | Premier 2000–2026, Bundesliga 2000–2026 | Odds históricos para backtesting |
| Understat | xG por partido (2014–2026, 5 ligas top) | xG rating por liga |
| The Odds API | 500 req/mes gratis | Live odds de Coolbet |

### Cronología de entrenamiento
| Split | Periodo | Partidos (Premier + Bundesliga) |
|:-----:|:-------:|:-------------------------------:|
| **Train** | 2014–2023 | ~5,400 partidos |
| **Validate** | 2024 | ~760 partidos |
| **Test** | 2025 | ~760 partidos |

### Modelos a entrenar (en orden de prioridad)

1. **Elo baseline** (K calibrado por liga)
   - compute_elo() sobre train + validate
   - K optimizado por walk-forward (Brier en validate)
   - Baseline mínimo para comparar modelos más complejos

2. **Poisson con λ calibrado por liga**
   - λ Premier ≠ λ Bundesliga (Premier más goleadora ~2.7, Bundesliga ~2.5)
   - Recalibrar λ cada temporada en walk-forward

3. **xG Rating con K calibrado**
   - Understat requiere mapeo de nombres de equipos
   - xGR por liga (no global como en el Mundial)
   - K calibrado walk-forward

4. **Combinación Elo + xG con alpha calibrado**
   - Igual que el pipeline del Mundial
   - alpha liga-específico (quizás Bundesliga necesite más xG que Premier)

5. **Dixon-Coles con Elo priors**
   - Ya implementado en `models/dixon_coles.py`
   - Entrenar con decay (el mejor decay se calibra en walk-forward)
   - La ventaja sobre Elo puro es que modela ataque/defensa por separado

6. **LinUCB (Contextual Bandit)** — solo si los modelos superan la validación
   - Viabilidad confirmada: ~6,000 partidos train es suficiente para RL lineal
   - Modela la decisión de apostar/no-apostar como un bandit
   - Features: diff_elo, diff_xgr, odds, home_adv, form_5, days_rest
   - Diferencia clave: explora estratégicamente en vez de apostar todo con EV>umbral

### Estrategia de stake
- Kelly 1/4 (igual que en Bundesliga, que dio Sharpe >1.0)
- EV mínimo: calibrar en validate (empezar con ≥10%)
- Escalado: bankroll fijo vs variable (decidir tras backtest)

### Backtesting
- football-data.co.uk tiene odds históricos completos con múltiples bookmakers
- `BettingEngine.run()` funciona directamente con estos datos
- El pipeline ya está implementado en `test_models.py`
- Evaluación: Brier, LogLoss, ROI, Sharpe, Sortino, Max Drawdown, Profit Factor

### Evaluación de calibración
- Platt scaling: si las probabilidades raw ya están bien calibradas (Brier <0.18), Platt no suma
- Temperature scaling: calibrar por liga (cada liga puede tener T distinta)
- ECE (Expected Calibration Error): métrica principal para calibración

## 4. Arquitectura de Exportación

### `core/` — paquete reusable para ligas
```
bet_agent/
  core/
    __init__.py
    elo.py            ← compute_elo() genérico (extraído de scripts/elo_model.py)
    xg_rating.py      ← XGRating class (de models/xg_rating.py)
    predictor.py      ← predict_probs(), compute_kelly(), normalize(), platt_calibrate()
    calibrator.pkl    ← Platt calibrator exportable
```

### Código que se reemplaza / adapta
| Mundial 2026 | Ligas |
|:-------------|:------|
| xGscore.io scraper | Understat (xG por temporada) |
| international_results_raw.csv | football-data.co.uk (Premier + Bundesliga) |
| Team name mapping WC | Mapping liga-específico |
| λ=2.5 global | λ calibrado por liga |
| K_xg=60 global | K_xg calibrado por liga |
| Coolbet only | multi-bookmaker (11 bookmakers disponibles) |
| Kelly 1/10 | Kelly 1/4 (probado en Bundesliga) |
| $20k fijo | $1000 fijo para backtest |

## 5. Preguntas Pendientes para la Fase Ligas

1. **xG**: Understat tiene xG desde 2014 para 5 ligas. Si queremos Bundesliga desde 2004, tendremos ~10 años sin xG. ¿Hacer Elo-only para 2004–2014 y agregar xG desde 2014?
2. **Frecuencia de re-entrenamiento**: ¿cada temporada completa o cada matchday?
3. **Betfair live**: ¿seguir con Betfair API o solo The Odds API (Coolbet)?
4. **LinUCB**: ¿implementarlo desde el inicio o solo como mejora si los modelos lineales no superan ROI 5% en validate?
5. **Alertas**: ¿Telegram/email cuando hay apuesta con EV>umbral?

## 6. Notas Técnicas

- Python 3.14 — `UnicodeEncodeError` al imprimir `≈`, `Δ`, `←` en PowerShell. Usar `.encode('utf-8', errors='replace')` o evitar estos caracteres en prints.
- The Odds API: 500 req/mes gratis. Para live odds de 2 ligas (~20 partidos/semana) alcanza justo. Si se agregan más ligas, toca pagar o rotar APIs.
- Understat requiere manejo de JavaScript ligero (a veces tienen Cloudflare). Alternativa: FBref con Selenium.
- El código de `bet_agent/` está listo para backtest con football-data.co.uk. `test_models.py` ya carga el CSV, entrena, calibra, y reporta ROI/Sharpe.
