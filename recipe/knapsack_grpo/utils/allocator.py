# Copyright 2025 Kingdee
# Licensed under the Apache License, Version 2.0

from __future__ import annotations
import numpy as np


def _prob_nonzero_grad(N: int, p: float) -> float:
    """
    P(there exists both success and failure among N draws)
    = 1 - p^N - (1-p)^N
    """
    if p <= 0.0 or p >= 1.0 or N <= 0:
        return 0.0
    return 1.0 - (p ** N) - ((1.0 - p) ** N)


def _info_gain(p: float) -> float:
    """
    Heuristic from the paper: p * (1-p)^2  (peak ~= 1/3)
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return p * (1.0 - p) * (1.0 - p)


def _value_table(p: float, n_low: int, n_up: int) -> np.ndarray:
    ks = np.arange(n_low, n_up + 1, dtype=np.int32)
    return np.array([_prob_nonzero_grad(int(k), p) * _info_gain(p) for k in ks], dtype=np.float64)


def _knapsack_dp(p: np.ndarray, total_budget: int, n_low: int, n_up: int) -> np.ndarray:
    """
    DP over per-sample integer budget k in [n_low, n_up], cost=1 per unit.
    Complexity: O(M * B * (n_up - n_low + 1)) — workable for B up to a few thousands.

    Returns:
      alloc: np.ndarray (M,), integer counts summing to total_budget
    """
    M = int(p.shape[0])
    base = np.full(M, n_low, dtype=np.int32)
    B = total_budget - int(base.sum())
    if B < 0:
        take = max(0, total_budget // M)
        return np.full(M, take, dtype=np.int32)

    vt = [_value_table(float(pi), n_low, n_up) for pi in p]  # each length = (n_up-n_low+1)

    dp = np.full((M + 1, B + 1), -1e30, dtype=np.float64)
    dp[0, 0] = 0.0
    choice = np.full((M, B + 1), -1, dtype=np.int32)

    for i in range(1, M + 1):
        vi = vt[i - 1]
        max_extra = min(B, n_up - n_low)
        for b in range(B + 1):
            best, best_k = dp[i - 1, b], 0
            for k_extra in range(0, min(b, max_extra) + 1):
                cand = dp[i - 1, b - k_extra]
                if cand <= -1e29:
                    continue
                inc = (vi[k_extra] - vi[0]) if k_extra > 0 else 0.0
                val = cand + inc
                if val > best:
                    best, best_k = val, k_extra
            dp[i, b] = best
            choice[i - 1, b] = best_k

    # backtrack
    alloc = base.copy()
    b = B
    for i in range(M, 0, -1):
        k_extra = choice[i - 1, b]
        if k_extra < 0:
            k_extra = 0
        alloc[i - 1] += k_extra
        b -= k_extra
    return alloc


def allocate_with_extremes(
    p: np.ndarray,
    total_budget: int,
    n_low: int,
    n_up: int,
    hard_fallback_share: float = 1.0,
    easy_min_cover: bool = True,
) -> np.ndarray:
    """
    Handle extreme p≈0 / p≈1 gracefully:
      - p≈1: at least n_low (so we still log some 'easy' examples if desired)
      - p≈0: distribute remaining budget evenly (or by heuristic in future)

    Then run DP on mid region.
    """
    assert total_budget >= 0
    assert 1 <= n_low <= n_up

    eps = 1e-8
    M = len(p)
    alloc = np.zeros(M, dtype=np.int32)

    idx_hard = p <= eps
    idx_easy = p >= 1.0 - eps
    idx_mid = (~idx_hard) & (~idx_easy)

    if easy_min_cover and idx_easy.any():
        alloc[idx_easy] = n_low

    budget_mid = total_budget - int(alloc.sum())
    if budget_mid < 0:
        # scale down; should be rare
        scale = total_budget / max(1, int(alloc.sum()))
        return np.floor(alloc.astype(float) * scale).astype(np.int32)

    if idx_mid.any():
        alloc_mid = _knapsack_dp(p[idx_mid], budget_mid, n_low, n_up)
        alloc[idx_mid] = alloc_mid

    used = int(alloc.sum())
    remain = total_budget - used

    if remain > 0 and idx_hard.any():
        give = np.zeros(idx_hard.sum(), dtype=np.int32)
        # simple even split; future: weight by expected token length, etc.
        give[:] = remain // len(give)
        give[: (remain % len(give))] += 1
        alloc[idx_hard] = np.clip(alloc[idx_hard] + give, n_low, n_up)

    # final sanity
    diff = total_budget - int(alloc.sum())
    if diff != 0 and M > 0:
        # fix by adjusting first few entries within bounds
        sgn = 1 if diff > 0 else -1
        diff = abs(diff)
        for i in range(M):
            if diff == 0:
                break
            can = (n_up - alloc[i]) if sgn > 0 else (alloc[i] - n_low)
            take = min(can, diff)
            alloc[i] += sgn * take
            diff -= take

    return alloc