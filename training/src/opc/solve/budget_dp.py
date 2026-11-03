"""Recombining per-component frontiers into a global one (WS1-4).

``F(i, b) = max_{0<=j<=b} [ F(i-1, b-j) + f_{K_i}(j) ]``, plus the kernel's additive constant.
Cost ``O(|components| * k^2)``.

Why this is exact, and why the handoff's T4 worry is empty: under (L) the intersection value is
constant on each connected component of the accepted overlap graph, and **two distinct components
can never carry the same value**.  (If ``A-B`` and ``C-D`` are edges in different components and
all four contain ``h``, the cross pairs do not overlap, so they are nested; chasing the four
inclusions forces either ``C&D = D`` or ``A subset-of C subset-of B``, each of which contradicts
one of the two edges being an overlap.)  So the reticulations a solution uses partition cleanly
across components, ``sum_K j_K`` is the true count, and the budget is additive exactly as the
weight is.

Note ``f_K`` is **not** required to be concave -- this is a general (max,+) convolution and a
greedy exchange over marginal gains would be wrong.  It is also non-decreasing in ``j`` only when
each point is a true optimum; under a time cap the points are incumbents, so a decrease is a
contamination signal rather than a bug in the DP (B1's T6).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["GlobalSolution", "recombine"]


@dataclass
class GlobalSolution:
    """The best accepted set over all components at each total budget ``0..k``."""

    kernel_weight: float
    objective: np.ndarray  # float[k+1]
    allocation: list[list[int]]  # allocation[b][i] = budget given to component i
    any_capped: bool

    def best(self, k: int) -> tuple[float, list[int]]:
        b = min(k, len(self.objective) - 1)
        return float(self.objective[b]), self.allocation[b]


def recombine(frontiers, kernel_weight: float, k: int) -> GlobalSolution:
    """``frontiers[i][j]`` is ``f_{K_i}(j)``; missing budgets fall back to the largest available.

    A component whose own ``j_max`` is below ``k`` simply cannot use more -- a reticulation needs
    two clusters to nominate its hybrid, so ``j_max = floor(V/2)`` -- and its frontier is flat
    beyond that point.
    """
    n_comp = len(frontiers)
    neg = float("-inf")
    dp = np.full((n_comp + 1, k + 1), neg)
    choice = np.zeros((n_comp + 1, k + 1), dtype=np.int32)
    dp[0, :] = 0.0

    capped = False
    for i, f in enumerate(frontiers, start=1):
        vals = [s.objective for s in f]
        capped |= any(not s.optimal for s in f)
        top = len(vals) - 1
        for b in range(k + 1):
            best, arg = neg, 0
            for j in range(0, min(b, top) + 1):
                prev = dp[i - 1, b - j]
                if prev == neg:
                    continue
                cand = prev + vals[j]
                if cand > best:
                    best, arg = cand, j
            # Budget above this component's ceiling is simply unused by it.
            if b > top and dp[i - 1, b - top] != neg:
                cand = dp[i - 1, b - top] + vals[top]
                if cand > best:
                    best, arg = cand, top
            dp[i, b] = best
            choice[i, b] = arg

    allocation = []
    for b in range(k + 1):
        alloc = [0] * n_comp
        rem = b
        for i in range(n_comp, 0, -1):
            j = int(choice[i, rem])
            alloc[i - 1] = j
            rem -= j
        allocation.append(alloc)
    return GlobalSolution(
        kernel_weight=float(kernel_weight),
        objective=dp[n_comp] + float(kernel_weight),
        allocation=allocation,
        any_capped=bool(capped),
    )
