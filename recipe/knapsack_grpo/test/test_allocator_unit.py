# recipe/knapsack_grpo/tests/test_allocator_unit.py
import numpy as np
from itertools import product

from recipe.knapsack_grpo.utils.allocator import allocate_with_extremes as alloc
from recipe.knapsack_grpo.utils.allocator import _value_table as vtab  # 内部函数用于目标值对比


def _total_value(p_vec, Ns, n_low, n_up):
    val = 0.0
    for i in range(len(p_vec)):
        val += vtab(float(p_vec[i]), n_low, n_up)[Ns[i] - n_low]
    return float(val)


def test_budget_and_bounds_basic():
    p = np.array([0.2, 0.5, 0.8], dtype=float)
    B, n_low, n_up = 18, 2, 10
    Ns = alloc(p, total_budget=B, n_low=n_low, n_up=n_up)
    assert int(Ns.sum()) == B
    assert (Ns >= n_low).all() and (Ns <= n_up).all()


def test_matches_bruteforce_small():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, size=4).astype(float)
    B, n_low, n_up = 12, 2, 6

    # 我们的分配
    Ns = alloc(p, total_budget=B, n_low=n_low, n_up=n_up)

    # 小规模穷举作为“金标准”
    best_val = -1e18
    best = None
    for Ns2 in product(range(n_low, n_up + 1), repeat=len(p)):
        if sum(Ns2) != B:
            continue
        val = _total_value(p, Ns2, n_low, n_up)
        if val > best_val:
            best_val, best = val, np.array(Ns2, dtype=int)

    assert best is not None
    # 允许存在多个最优解；只比较目标值
    v1 = _total_value(p, Ns, n_low, n_up)
    v2 = _total_value(p, best, n_low, n_up)
    assert abs(v1 - v2) < 1e-9