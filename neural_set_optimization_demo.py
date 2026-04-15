# -*- coding: utf-8 -*-
"""
两阶段设施选址 + Neural Set Optimization 式延拓（排序延拓 Φ、次梯度、舍入、简单粒子搜索）
内层：容量约束 CFLP —— 给定 S，解线性分配 min sum_ij d_i c_ij x_ij
      s.t. sum_{j in S} x_ij = 1, sum_i d_i x_ij <= u_j, x_ij >= 0
外层：max_S F(S),  F(S) = -sum_{j in S} f_j - T(S)，不可行时 F(S)=empty_f

内层 LP 默认使用 Gurobi（gurobipy）；若无许可证/未安装，可设 solver="scipy" 使用 HiGHS。
安装：pip install gurobipy（需从 Gurobi 官网申请学术/商业许可证并配置）
"""

from __future__ import annotations

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB

    _HAS_GUROBI = True
except ImportError:
    _HAS_GUROBI = False

from scipy.optimize import linprog


class CapacitatedFacilityOracle:
    """给定设施子集 S，用 LP 求最小运输成本 T(S)；F(S) = -固定成本 - T(S)。"""

    def __init__(
        self,
        f: np.ndarray,
        C: np.ndarray,
        d: np.ndarray,
        u: np.ndarray,
        *,
        solver: str = "gurobi",
    ):
        """
        f: (n,) 各设施固定开通成本
        C: (m, n) 单位运价 c_ij
        d: (m,) 客户需求
        u: (n,) 各设施容量（同一总量纲下与 d_i x_ij 可比，通常为「可承接需求量」上限）
        """
        self.f = np.asarray(f, dtype=float).reshape(-1)
        self.C = np.asarray(C, dtype=float)
        self.d = np.asarray(d, dtype=float).reshape(-1)
        self.u = np.asarray(u, dtype=float).reshape(-1)
        self.n = self.f.size
        self.m = self.d.size
        if self.C.shape != (self.m, self.n):
            raise ValueError(f"C 形状应为 ({self.m}, {self.n}), 得到 {self.C.shape}")
        if self.u.size != self.n:
            raise ValueError("u 长度应为 n")
        sol = solver.lower().strip()
        if sol == "gurobi" and not _HAS_GUROBI:
            raise ImportError(
                "solver='gurobi' 需要安装 gurobipy 并配置 Gurobi 许可证。"
                "可改用 solver='scipy'，或执行: pip install gurobipy"
            )
        if sol not in ("gurobi", "scipy"):
            raise ValueError("solver 应为 'gurobi' 或 'scipy'")
        self._solver = sol
        worst_transport = float(np.sum(self.d) * np.max(self.C))
        self.empty_f = -(np.sum(self.f) + worst_transport + 1.0)
        # 同一子集 S 在 Φ/次梯度/粒子中会被反复求值，缓存 T(S)
        self._t_cache: dict[frozenset, float] = {}

    def transport_cost_lp(self, S: set[int] | list[int]) -> float:
        """
        T(S) = min_{x>=0} sum_{i,j in S} d_i c_ij x_ij
        s.t. 对每个 i: sum_{j in S} x_ij = 1
             对每个 j in S: sum_i d_i x_ij <= u_j
        不可行时返回 inf。
        """
        if not S:
            return np.inf
        key = frozenset(S)
        if key in self._t_cache:
            return self._t_cache[key]
        if self._solver == "gurobi":
            val = self._transport_cost_gurobi(S)
        else:
            val = self._transport_cost_scipy(S)
        self._t_cache[key] = val
        return val

    def _transport_cost_gurobi(self, S: set[int] | list[int]) -> float:
        """内层 LP：Gurobi 求解。"""
        J = sorted(S)
        m, lenJ = self.m, len(J)
        model = gp.Model()
        model.setParam("OutputFlag", 0)
        model.setParam("LogToConsole", 0)
        x = model.addVars(m, lenJ, lb=0.0, vtype=GRB.CONTINUOUS, name="x")
        obj = gp.quicksum(
            self.d[i] * self.C[i, J[jj]] * x[i, jj] for i in range(m) for jj in range(lenJ)
        )
        model.setObjective(obj, GRB.MINIMIZE)
        for i in range(m):
            model.addConstr(gp.quicksum(x[i, jj] for jj in range(lenJ)) == 1.0, name=f"demand_{i}")
        for jj, j in enumerate(J):
            model.addConstr(
                gp.quicksum(self.d[i] * x[i, jj] for i in range(m)) <= self.u[j],
                name=f"cap_{j}",
            )
        model.optimize()
        if model.Status == GRB.OPTIMAL:
            val = float(model.ObjVal)
        elif model.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
            val = np.inf
        else:
            val = np.inf
        model.dispose()
        return val

    def _transport_cost_scipy(self, S: set[int] | list[int]) -> float:
        """内层 LP：SciPy HiGHS（无 Gurobi 时使用）。"""
        J = sorted(S)
        m, lenJ = self.m, len(J)
        n_var = m * lenJ
        c = np.zeros(n_var)
        for jj, j in enumerate(J):
            for i in range(m):
                c[jj * m + i] = self.d[i] * self.C[i, j]

        A_eq = np.zeros((m, n_var))
        b_eq = np.ones(m)
        for i in range(m):
            for jj in range(lenJ):
                A_eq[i, jj * m + i] = 1.0

        A_ub = np.zeros((lenJ, n_var))
        b_ub = np.zeros(lenJ)
        for jj, j in enumerate(J):
            for i in range(m):
                A_ub[jj, jj * m + i] = self.d[i]
            b_ub[jj] = self.u[j]

        bounds = [(0.0, None)] * n_var
        res = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )
        if not res.success:
            return np.inf
        return float(res.fun)

    def F(self, S: set[int] | list[int] | None) -> float:
        """F(S) = - sum_{j in S} f_j - T(S)。空集或内层不可行返回 self.empty_f。"""
        if not S:
            return self.empty_f
        S = set(S)
        fixed = float(sum(self.f[j] for j in S))
        t = self.transport_cost_lp(S)
        if np.isinf(t):
            return self.empty_f
        return -fixed - t


def permutation_by_descending_scores(x: np.ndarray) -> np.ndarray:
    """返回 pi，使得 x[pi[0]] >= x[pi[1]] >= ..."""
    x = np.asarray(x, dtype=float).reshape(-1)
    return np.argsort(-x)


def chain_subsets(pi: np.ndarray, k: int) -> set[int]:
    """S_k = {pi[0], ..., pi[k-1]}，k>=1"""
    return set(pi[:k].tolist())


def phi_extension(oracle: CapacitatedFacilityOracle, x: np.ndarray) -> float:
    """
    Φ(x) = sum_{i=1}^n (x_pi(i) - x_pi(i+1)) F(S_i),  x_pi(n+1)=0
    """
    x = np.clip(np.asarray(x, dtype=float).reshape(-1), 0.0, 1.0)
    n = oracle.n
    if x.size != n:
        raise ValueError("x 长度应为 n")
    pi = permutation_by_descending_scores(x)
    x_perm = x[pi]
    x_next = np.append(x_perm[1:], 0.0)

    total = 0.0
    for i in range(n):
        S_i = chain_subsets(pi, i + 1)
        w = x_perm[i] - x_next[i]
        total += w * oracle.F(S_i)
    return float(total)


def subgradient_g(oracle: CapacitatedFacilityOracle, x: np.ndarray) -> np.ndarray:
    """g_pi(k) = F(S_k) - F(S_{k-1})，映射回 g[j]。"""
    x = np.clip(np.asarray(x, dtype=float).reshape(-1), 0.0, 1.0)
    n = oracle.n
    pi = permutation_by_descending_scores(x)
    g = np.zeros(n)
    F_prev = oracle.F(set())
    for i in range(n):
        S_i = chain_subsets(pi, i + 1)
        Fi = oracle.F(S_i)
        g[pi[i]] = Fi - F_prev
        F_prev = Fi
    return g


def rounding_argmax_chain(
    oracle: CapacitatedFacilityOracle, x: np.ndarray
) -> tuple[set[int], float, int]:
    """R(x) in argmax_{k=0..n} F(S_k)。"""
    x = np.clip(np.asarray(x, dtype=float).reshape(-1), 0.0, 1.0)
    pi = permutation_by_descending_scores(x)
    best_S: set[int] = set()
    best_val = oracle.F(set())
    best_k = 0
    for k in range(1, oracle.n + 1):
        S_k = chain_subsets(pi, k)
        v = oracle.F(S_k)
        if v > best_val:
            best_val = v
            best_S = S_k
            best_k = k
    return best_S, best_val, best_k


def brute_force_best(oracle: CapacitatedFacilityOracle) -> tuple[set[int], float]:
    """枚举所有非空子集（n<=20 可用）。"""
    n = oracle.n
    best_S: set[int] = set()
    best = -np.inf
    for mask in range(1, 1 << n):
        S = {j for j in range(n) if (mask >> j) & 1}
        v = oracle.F(S)
        if v > best:
            best = v
            best_S = set(S)
    return best_S, float(best)


def particle_step(
    oracle: CapacitatedFacilityOracle,
    particles: np.ndarray,
    tau: float,
    eta: float,
    lam: float,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """soft-min 权重 + 对 Φ 的次梯度上升。"""
    M, n = particles.shape
    phis = np.array([phi_extension(oracle, particles[m]) for m in range(M)])
    tau = max(tau, 1e-8)
    z = -phis / tau
    z -= np.max(z)
    w = np.exp(z)
    w /= w.sum()
    new = np.zeros_like(particles)
    for m in range(M):
        g = subgradient_g(oracle, particles[m])
        noise = np.sqrt(max(eta, 1e-12)) * sigma * rng.standard_normal(n) if sigma > 0 else 0.0
        step = eta * w[m] * g - eta * lam * particles[m] + noise
        new[m] = np.clip(particles[m] + step, 0.0, 1.0)
    return new


def run_particle_search(
    oracle: CapacitatedFacilityOracle,
    M: int = 8,
    n_iter: int = 200,
    eta: float = 0.15,
    tau: float = 0.5,
    lam: float = 0.01,
    sigma: float = 0.02,
    seed: int = 0,
) -> tuple[np.ndarray, set[int], float]:
    rng = np.random.default_rng(seed)
    n = oracle.n
    particles = rng.uniform(0.2, 0.8, size=(M, n))
    for _ in range(n_iter):
        particles = particle_step(oracle, particles, tau, eta, lam, sigma, rng)
        tau *= 0.995

    best_round_val = -np.inf
    best_S: set[int] = set()
    best_x = particles[0].copy()
    for m in range(M):
        S, v, _ = rounding_argmax_chain(oracle, particles[m])
        if v > best_round_val:
            best_round_val = v
            best_S = S
            best_x = particles[m].copy()
    return best_x, best_S, best_round_val


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------
def main():
    # 4 设施，3 客户；容量设为 2：单设施无法吞下总需求 3，至少需 2 个设施
    f = np.array([10.0, 12.0, 11.0, 9.0])
    C = np.array(
        [
            [2.0, 5.0, 4.0, 3.0],
            [6.0, 3.0, 5.0, 4.0],
            [4.0, 6.0, 2.0, 5.0],
        ]
    )
    d = np.array([1.0, 1.0, 1.0])
    u = np.array([2.0, 2.0, 2.0, 2.0])

    solver = "gurobi" if _HAS_GUROBI else "scipy"
    oracle = CapacitatedFacilityOracle(f, C, d, u, solver=solver)

    print("=== 两阶段设施选址（容量约束 LP）===")
    print(f"内层求解器: {solver}" + ("（未安装 gurobipy 时自动使用 scipy）" if solver == "scipy" else ""))
    print("f =", f)
    print("u =", u)
    print("C =\n", C)
    print("d =", d)
    print("总需求 =", d.sum(), "；单设施容量 = 2 => 至少 2 个设施才可能可行")
    print()

    print("=== 暴力枚举最优子集 ===")
    S_star, v_star = brute_force_best(oracle)
    print(f"最优 S* = {sorted(S_star)},  F(S*) = {v_star:.6f}")

    print("\n=== 排序延拓 Φ(x) 与次梯度（随机 x）===")
    rng = np.random.default_rng(42)
    x_test = rng.uniform(0, 1, size=oracle.n)
    phi = phi_extension(oracle, x_test)
    g = subgradient_g(oracle, x_test)
    print(f"x = {np.round(x_test, 4)}")
    print(f"Φ(x) = {phi:.6f}")
    print(f"g (subgradient) = {np.round(g, 4)}")

    print("\n=== 舍入 R(x) ===")
    S_r, v_r, k_r = rounding_argmax_chain(oracle, x_test)
    print(f"R(x) = {sorted(S_r)},  F = {v_r:.6f},  链上索引 k = {k_r}")

    print("\n=== 粒子搜索（启发式，多随机种子取最优）===")
    v_b = -np.inf
    S_b: set[int] = set()
    for sd in (7, 13, 42, 99):
        _, S_try, v_try = run_particle_search(
            oracle, M=16, n_iter=250, seed=sd, eta=0.2, sigma=0.03
        )
        if v_try > v_b:
            v_b, S_b = v_try, S_try
    print(f"舍入最优 F = {v_b:.6f},  S = {sorted(S_b)}")
    print(f"与暴力最优差距: {v_star - v_b:.6f}（启发式不保证全局最优）")
    if v_star > v_b + 1e-3:
        print("提示：可增大 n_iter / M 或调整 eta、sigma 继续搜索。")

    assert v_star >= v_b - 1e-2, "舍入不应优于暴力枚举（数值误差外）"

    print("\n[测试通过]")


if __name__ == "__main__":
    main()
