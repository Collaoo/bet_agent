import abc

__all__ = [
    "SizingStrategy", "FractionalKelly", "FlatStaking",
    "ConfidenceBased", "AdaptiveDrawdown",
]


class SizingStrategy(abc.ABC):

    @abc.abstractmethod
    def compute_stake(self, prob: float, odds: float, bankroll: float,
                      model_prob: float = None) -> float:
        pass

    def __repr__(self):
        return f"{type(self).__name__}()"


class FractionalKelly(SizingStrategy):

    def __init__(self, fraction: float = 0.25, max_stake_pct: float = 0.03,
                 min_edge: float = 0.0):
        self.fraction = fraction
        self.max_stake_pct = max_stake_pct
        self.min_edge = min_edge

    def compute_stake(self, prob: float, odds: float, bankroll: float,
                      model_prob: float = None) -> float:
        edge = prob * odds - 1
        if edge <= self.min_edge:
            return 0.0
        stake_pct = (edge / (odds - 1)) * self.fraction
        stake_pct = min(stake_pct, self.max_stake_pct)
        return bankroll * stake_pct

    def __repr__(self):
        return (f"FractionalKelly(fraction={self.fraction}, "
                f"max_stake={self.max_stake_pct}, min_edge={self.min_edge})")


class FlatStaking(SizingStrategy):

    def __init__(self, stake_pct: float = 0.02, min_edge: float = 0.05):
        self.stake_pct = stake_pct
        self.min_edge = min_edge

    def compute_stake(self, prob: float, odds: float, bankroll: float,
                      model_prob: float = None) -> float:
        edge = prob * odds - 1
        if edge <= self.min_edge:
            return 0.0
        return bankroll * self.stake_pct

    def __repr__(self):
        return f"FlatStaking(stake_pct={self.stake_pct}, min_edge={self.min_edge})"


class ConfidenceBased(SizingStrategy):

    def __init__(self, base_fraction: float = 0.25, max_stake_pct: float = 0.05,
                 min_edge: float = 0.02):
        self.base_fraction = base_fraction
        self.max_stake_pct = max_stake_pct
        self.min_edge = min_edge

    def compute_stake(self, prob: float, odds: float, bankroll: float,
                      model_prob: float = None) -> float:
        edge = prob * odds - 1
        if edge <= self.min_edge:
            return 0.0
        base_kelly = edge / (odds - 1)
        confidence = abs(prob - 1 / odds) if model_prob is None else abs(model_prob - 1 / odds)
        mult = min(confidence * 5, 1.0)
        stake_pct = base_kelly * self.base_fraction * mult
        stake_pct = min(stake_pct, self.max_stake_pct)
        return bankroll * stake_pct

    def __repr__(self):
        return (f"ConfidenceBased(base_fraction={self.base_fraction}, "
                f"max_stake={self.max_stake_pct})")


class AdaptiveDrawdown(SizingStrategy):

    def __init__(self, base_strategy: SizingStrategy = None,
                 recovery_factor: float = 0.5, max_reduce: float = 0.8):
        self.base = base_strategy or FractionalKelly()
        self.recovery_factor = recovery_factor
        self.max_reduce = max_reduce
        self._consecutive_losses = 0

    def compute_stake(self, prob: float, odds: float, bankroll: float,
                      model_prob: float = None) -> float:
        base_stake = self.base.compute_stake(prob, odds, bankroll, model_prob)
        reduction = min(self._consecutive_losses * self.recovery_factor * 0.1, self.max_reduce)
        return base_stake * (1 - reduction)

    def record_result(self, won: bool):
        if won:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1

    def reset(self):
        self._consecutive_losses = 0

    def __repr__(self):
        return (f"AdaptiveDrawdown(base={self.base}, "
                f"recovery_factor={self.recovery_factor})")
