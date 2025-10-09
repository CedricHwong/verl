# Copyright 2025 Kingdee
# Licensed under the Apache License, Version 2.0

from __future__ import annotations
import numpy as np


class SuccessRateEstimator:
    """
    Online success-rate estimator per uid:
      - maintains EMA counts of successes / failures
      - exposes `estimate(uids) -> p_hat` = (alpha + succ_ema) / (alpha + beta + succ_ema + fail_ema)

    Typical usage in the trainer:
      1) p_hat = est.estimate(uids)
      2) allocation = knapsack(p_hat, ...)
      3) after reward computed, est.update_batch(uids_expanded, successes)
    """

    def __init__(self, ema: float = 0.7, alpha: float = 1.0, beta: float = 1.0):
        assert 0.0 < ema < 1.0, "ema must be in (0,1)"
        self.ema = float(ema)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self._succ_fail_ema: dict[str, tuple[float, float]] = {}  # uid -> (succ_ema, fail_ema)

    def update_batch(self, uids: list[str], successes: np.ndarray) -> None:
        """
        Args:
          uids      : list[str], length = number of responses in this step
          successes : np.ndarray of {0,1}, same length
        """
        assert len(uids) == len(successes)
        # aggregate per uid for this step
        step_aggr: dict[str, tuple[int, int]] = {}
        for u, s in zip(uids, successes):
            s = int(s > 0)
            a, b = step_aggr.get(u, (0, 0))
            if s: a += 1
            else: b += 1
            step_aggr[u] = (a, b)

        # EMA update
        for u, (a, b) in step_aggr.items():
            sa, sb = self._succ_fail_ema.get(u, (0.0, 0.0))
            sa = self.ema * sa + (1.0 - self.ema) * float(a)
            sb = self.ema * sb + (1.0 - self.ema) * float(b)
            self._succ_fail_ema[u] = (sa, sb)

    def estimate(self, uids: list[str]) -> np.ndarray:
        """Return p_hat for each uid in the order given."""
        p = np.zeros(len(uids), dtype=np.float64)
        for i, u in enumerate(uids):
            sa, sb = self._succ_fail_ema.get(u, (0.0, 0.0))
            p[i] = (self.alpha + sa) / (self.alpha + self.beta + sa + sb + 1e-9)
        return p