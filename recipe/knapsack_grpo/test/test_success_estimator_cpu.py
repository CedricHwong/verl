# recipe/knapsack_grpo/tests/test_success_estimator_cpu.py
import numpy as np
from recipe.knapsack_grpo.utils.success_estimator import SuccessRateEstimator


def test_ema_estimator_converges():
    rng = np.random.default_rng(42)
    est = SuccessRateEstimator(ema=0.6, alpha=1.0, beta=1.0)

    # 三个 uid，真实成功率各不相同
    uids = ["a", "b", "c"]
    p_true = {"a": 0.2, "b": 0.5, "c": 0.8}

    # 模拟 200 步，每步每个 uid 生成 4 次（聚合后再 EMA）
    for _ in range(200):
        step_uids = []
        succs = []
        for u in uids:
            # 4 次伯努利试验
            s = rng.binomial(1, p_true[u], size=4)
            step_uids.extend([u] * 4)
            succs.extend(s.tolist())
        est.update_batch(step_uids, np.array(succs, dtype=int))

    p_hat = est.estimate(uids)
    # 宽松判定：误差 < 0.1
    assert abs(p_hat[0] - p_true["a"]) < 0.1
    assert abs(p_hat[1] - p_true["b"]) < 0.1
    assert abs(p_hat[2] - p_true["c"]) < 0.1