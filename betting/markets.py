import numpy as np

__all__ = [
    "over_under_probs", "btts_probs", "asian_handicap_probs",
    "fair_odds", "expected_value", "implied_probability",
]


def over_under_probs(sm: np.ndarray, threshold: float = 2.5) -> dict:
    under = 0.0
    over = 0.0
    for i in range(sm.shape[0]):
        for j in range(sm.shape[1]):
            total = i + j
            if total <= threshold:
                under += sm[i, j]
            else:
                over += sm[i, j]
    return {"over": float(over), "under": float(under)}


def btts_probs(sm: np.ndarray) -> dict:
    yes = 0.0
    no = 0.0
    for i in range(sm.shape[0]):
        for j in range(sm.shape[1]):
            if i > 0 and j > 0:
                yes += sm[i, j]
            else:
                no += sm[i, j]
    return {"btts_yes": float(yes), "btts_no": float(no)}


def asian_handicap_probs(sm: np.ndarray, handicap: float = -0.5) -> dict:
    home_goals = np.arange(sm.shape[0])
    away_goals = np.arange(sm.shape[1])
    goal_diff = home_goals[:, None] - away_goals[None, :]

    if handicap == 0:
        home_win = np.sum(sm[goal_diff > 0])
        push = np.sum(sm[goal_diff == 0])
        away_win = np.sum(sm[goal_diff < 0])
    elif handicap > 0:
        adj = goal_diff - handicap
        home_win = np.sum(sm[adj > 0])
        push = np.sum(sm[adj == 0])
        away_win = np.sum(sm[adj < 0])
    else:
        adj = goal_diff + abs(handicap)
        away_adj = -goal_diff + abs(handicap)
        home_win = np.sum(sm[adj > 0])
        push = np.sum(sm[adj == 0])
        away_win = np.sum(sm[adj < 0])

    return {
        "home_win": float(home_win),
        "push": float(push),
        "away_win": float(away_win),
    }


def fair_odds(prob: float) -> float:
    return 1.0 / prob if prob > 0 else float("inf")


def expected_value(prob: float, odds: float) -> float:
    return prob * odds - 1


def implied_probability(odds: float) -> float:
    return 1.0 / odds
