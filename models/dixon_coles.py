import numpy as np
import pandas as pd
import json
from pathlib import Path
from scipy.stats import poisson
from models.base import PoissonBase, load_data


def dc_correction(hg, ag, lh, la, rho):
    hg = np.asarray(hg)
    ag = np.asarray(ag)
    lh = np.broadcast_to(np.asarray(lh), hg.shape)
    la = np.broadcast_to(np.asarray(la), hg.shape)
    tau = np.ones_like(hg, dtype=float)
    m00 = (hg == 0) & (ag == 0)
    m10 = (hg == 1) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m11 = (hg == 1) & (ag == 1)
    tau[m00] = 1 - lh[m00] * la[m00] * rho
    tau[m10] = 1 + la[m10] * rho
    tau[m01] = 1 + lh[m01] * rho
    tau[m11] = 1 - rho
    return np.clip(tau, 1e-10, None)


class DixonColesModel(PoissonBase):

    def __init__(self):
        super().__init__()
        self.rho = None

    def _n_extra(self):
        return 2

    def _extra_init(self):
        return [0.1, -0.1]

    def _extra_bounds(self):
        return [(None, None), (-0.99, 0.99)]

    def _parse_params(self, params):
        n = len(self.teams)
        atk = params[:n]
        dfc = params[n:2*n]
        self.home_adv = params[-2]
        self.rho = params[-1]
        return atk, dfc, [self.home_adv, self.rho]

    def _score_prob(self, hg, ag, lh, la):
        p = poisson.pmf(hg, lh) * poisson.pmf(ag, la)
        return p * dc_correction(hg, ag, lh, la, self.rho)

    def _log_likelihood_contrib(self, hg, ag, lh, la):
        ll = poisson.logpmf(hg, lh) + poisson.logpmf(ag, la)
        log_tau = np.zeros_like(hg, dtype=float)
        m00 = (hg == 0) & (ag == 0)
        m10 = (hg == 1) & (ag == 0)
        m01 = (hg == 0) & (ag == 1)
        m11 = (hg == 1) & (ag == 1)
        log_tau[m00] = np.log(1 - lh[m00] * la[m00] * self.rho)
        log_tau[m10] = np.log(1 + la[m10] * self.rho)
        log_tau[m01] = np.log(1 + lh[m01] * self.rho)
        log_tau[m11] = np.log(1 - self.rho)
        log_tau = np.clip(log_tau, -20, None)
        return ll + log_tau

    def fit(self, df, decay=0.002, verbose=True):
        return super().fit(df, decay=decay, verbose=verbose)

    def save(self, path: str = "models/dixon_coles_model.json"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "teams": self.teams,
            "attack_params": self.attack_params.tolist(),
            "defense_params": self.defense_params.tolist(),
            "home_adv": float(self.home_adv),
            "rho": float(self.rho),
        }
        if self._temperature is not None:
            data["temperature"] = float(self._temperature)
        json.dump(data, open(path, "w"), indent=2)

    def load(self, path: str = "models/dixon_coles_model.json"):
        d = json.load(open(path))
        self.teams = d["teams"]
        self.team_idx = {t: i for i, t in enumerate(self.teams)}
        self.attack_params = np.array(d["attack_params"])
        self.defense_params = np.array(d["defense_params"])
        self.home_adv = d["home_adv"]
        self.rho = d["rho"]
        self._temperature = d.get("temperature")
        self.fitted = True
        print(f"Dixon-Coles cargado desde {path}")
        return self
