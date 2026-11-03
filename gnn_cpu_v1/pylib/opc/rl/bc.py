"""W4 -- behaviour cloning and the two feature diagnostics.  ``RL_PLAN.md`` §5.

§5 is explicit that the supervised arm is **not** a parallel branch: it trains *the encoder weights
the RL policy uses*, and every line of it is on the RL critical path.  Three deliverables:

* **W4.1 -- oracle behaviour cloning.**  Order ``C(N_true) & C_tau`` canonically and train
  ``pi`` by cross-entropy on that action sequence under the env's real legality masks.  Output is
  the RL initialisation, ``bc_init.pt``.  §5's argument is sample efficiency -- a random-init
  policy would spend most of its budget rediscovering greedy.  A65/A69 add a second, independent
  reason: the reward's cost is set by how *diverse* the policy's terminal states are -- 6.7 s per
  reward for a random policy against 1.6 s for a near-greedy one -- so an untrained policy is also
  the expensive one to evaluate.
* **W4.2 -- the pointwise head.**  ``h_A -> P(A in C(N_true))`` on labels that are free.  **G-R3**
  is its AUC minus support-ranking's AUC: if the features cannot beat sorting by support, no policy
  over those features will either.
* **G-R2** -- cloning *greedy* rather than the oracle, which asks whether the architecture can
  represent the strongest baseline at all.  ``--target greedy``.

**Any prefix of a feasible cluster system is feasible**, so the oracle order is always legal and
never needs a repair step: (L) and (N3O) are downward-closed (a sub-family of a laminar-compatible
family is one), and the budget counts *distinct* opened classes, which only grows.  ``w0_baseline``
measured ``true_feasible`` True on 60 of 60 cells with ``j_true == k`` exactly, and the same holds
at tau = 2 and k = 4 over the 192-cell sweep.  :func:`oracle_actions` still asserts it rather than
assuming it -- a silently skipped oracle action would train the policy on a *different* target and
show up only as a mysteriously low G-R2.
"""

from __future__ import annotations

import numpy as np
import torch

from opc import bitset
from opc.rl.policy import dynamic_features

__all__ = ["oracle_actions", "clone_loss", "set_clone_loss", "swap_clone_loss",
           "auc", "support_auc"]


def oracle_actions(inst, env, target: str = "oracle") -> np.ndarray:
    """The action sequence to imitate, in the order the policy will be asked to produce it.

    ``oracle``  -- ``C(N_true) & C_tau``, **descending support, ties by descending size**.  Support
    order is the canonical one because it is the order the incumbent baseline works in, so the
    residual RL has to learn is the *difference* between the two rather than the whole selection.
    ``greedy``  -- greedy-by-support's own selection.  **G-R2.**

    **The kernel is removed, and forgetting to remove it is a silent corruption of the target.**
    ``ClusterEnv.reset`` accepts every degree-0 cluster into ``S_0`` unconditionally -- that is
    ``SearchState``'s own convention, and it is sound (an isolated vertex crosses nothing, so no
    constraint can ever refuse it).  Those clusters are therefore *in the selection* and are not
    *actions*: the mask is ``~in_s``, so replaying one is illegal.  Measured before this filter:
    **228 illegal actions** over 96 cloned cells, about 4.8% of the sequence, every one of them a
    kernel row.  The kernel is median 0 and max 5 at tau = 2, which is exactly the size of defect
    that survives a smoke test.

    ``scripts/w4_bc.py`` fails on any residual illegal action rather than skipping it, because a
    skipped action means the policy was trained on a *different* sequence from the one G-R2 then
    scores it against.
    """
    env.reset()
    preaccepted = env.state.in_s.copy()
    if target == "greedy":
        from opc.rl.env import greedy_support_rollout
        rows = greedy_support_rollout(env)
        # **In the order greedy *accepted* them, which is not the order it returns them.**
        # ``SearchState.selection`` is ``np.flatnonzero(in_s)`` -- ascending cluster index, an
        # artefact of the bitmap.  Greedy accepts in descending support with a stable tie-break
        # (``greedy_support_rollout`` picks ``argmin(rank)`` over the legal set), so re-sorting by
        # descending support reproduces the acceptance order exactly.  Cloning the index order
        # instead asks the policy to produce a sequence that *contradicts* the greedy-indicator
        # feature it is given: measured, that costs G-R2 0.945 against 1.000.
        order = np.argsort(-np.asarray(inst.gs.support, dtype=np.float64), kind="stable")
        rank = np.empty(inst.n_clusters, dtype=np.int64)
        rank[order] = np.arange(inst.n_clusters)
        rows = np.asarray(rows, dtype=np.int64)
        rows = rows[np.argsort(rank[rows], kind="stable")]
    elif target == "oracle":
        rows = np.asarray(inst.truth_rows, dtype=np.int64)
        if rows.size:
            support = np.asarray(inst.gs.support, dtype=np.float64)[rows]
            sizes = bitset.popcounts(np.ascontiguousarray(inst.gs.words[rows])).astype(np.int64)
            rows = rows[np.lexsort((-sizes, -support))]   # primary -support, secondary -size
    else:
        raise ValueError(f"unknown clone target {target!r}")
    rows = np.asarray(rows, dtype=np.int64)
    return rows[~preaccepted[rows]] if rows.size else rows


def _step_logits(policy, enc, inst, env, mask, deg, *, device, dynamic_extra=None):
    """``pi(. | s_t)``'s masked logits at the env's current state.

    Factored out rather than spelled inline in each caller.  :func:`swap_clone_loss`'s docstring
    records why: "a second copy of that arithmetic is precisely how a clone target and a rollout
    come to disagree about what state the policy is standing at", and there are now two clone
    targets over this MDP.
    """
    support = np.asarray(inst.gs.support, dtype=np.float64)
    hidden = policy.cfg.hidden
    n_legal = int(mask.sum())
    return policy.logits(
        enc,
        accepted_sum=(enc.h[env.selection].sum(0, keepdim=True)
                      if env.selection.size else torch.zeros(1, hidden, device=device)),
        accepted_n=torch.tensor([float(env.selection.size)], device=device),
        last_action=torch.tensor([env.last_action], dtype=torch.long, device=device),
        scalars=torch.tensor([[(env.budget - env.state.n_hybrids) / max(1, env.budget),
                               env.budget / inst.n, env.t / inst.n,
                               n_legal / max(1, inst.n_clusters)]],
                             dtype=torch.float32, device=device),
        dynamic=torch.as_tensor(
            dynamic_features(support, mask, deg, int(env.selection.size),
                             extra=dynamic_extra),
            device=device).unsqueeze(0),
        mask=torch.as_tensor(mask, device=device).unsqueeze(0),
    )[0]


def clone_loss(policy, enc, inst, env, actions, *, device, dynamic_extra=None):
    """Cross-entropy of ``pi(. | s_t)`` against ``actions[t]``, stepping the real env.

    Returns ``(loss, n_steps, n_illegal)``.  ``n_illegal`` must be zero -- see the module
    docstring -- and the caller is expected to fail on it rather than average it away.

    **This target is order-sensitive and the objective is not**, which is what
    :func:`set_clone_loss` exists to test rather than to assert.
    """
    env.reset()
    mask, deg = env.legal_mask(with_features=True)
    losses, n_illegal = [], 0
    for a in np.asarray(actions, dtype=np.int64).tolist():
        if not mask[a]:
            n_illegal += 1
            continue
        lg = _step_logits(policy, enc, inst, env, mask, deg, device=device,
                              dynamic_extra=dynamic_extra)
        losses.append(-torch.log_softmax(lg, dim=-1)[a])
        obs, _r, done, _i = env.step(a)
        mask, deg = obs["legal"], obs["deg_sel"]
        if done:
            break
    if not losses:
        return torch.zeros((), device=device), 0, n_illegal
    return torch.stack(losses).mean(), len(losses), n_illegal


def set_clone_loss(policy, enc, inst, env, actions, *, device, dynamic_extra=None):
    """``-log sum_{i in T_t} pi(i | s_t)`` -- *any* legal true cluster is a correct answer.

    **The objective is a set and the canonical order is an artefact of writing it down.**  Any
    prefix of a feasible cluster system is feasible (the module docstring's own argument), so many
    action sequences realise ``C(N_true)`` and only the terminal set is ever scored.
    :func:`clone_loss` nevertheless penalises the policy for preferring a true cluster that comes
    later in ``oracle_actions``' descending-support order -- a cost paid for nothing, and one that
    grows with how many true clusters are simultaneously legal.

    This is the partial-label form of the same target: put mass anywhere in ``T_t``, the set of
    currently legal clusters of ``C(N_true)``.  It is exactly the policy A91 measured hitting the
    exact-recovery ceiling ("prefer any legal true cluster"), which :func:`clone_loss`'s target
    is not.

    The env is still stepped along ``actions`` -- teacher forcing on the canonical order -- so the
    **state distribution is identical** to :func:`clone_loss`'s and the loss is the single variable
    between them.

    Returns ``(loss, n_steps, n_illegal)``.
    """
    truth = np.zeros(int(inst.n_clusters), dtype=bool)
    truth[np.asarray(inst.truth_rows, dtype=np.int64)] = True
    env.reset()
    mask, deg = env.legal_mask(with_features=True)
    losses, n_illegal = [], 0
    for a in np.asarray(actions, dtype=np.int64).tolist():
        if not mask[a]:
            n_illegal += 1
            continue
        legal_true = np.flatnonzero(mask & truth)
        if legal_true.size:
            lg = _step_logits(policy, enc, inst, env, mask, deg, device=device,
                              dynamic_extra=dynamic_extra)
            lp = torch.log_softmax(lg, dim=-1)[
                torch.as_tensor(legal_true, dtype=torch.long, device=device)]
            losses.append(-torch.logsumexp(lp, dim=0))
        obs, _r, done, _i = env.step(a)
        mask, deg = obs["legal"], obs["deg_sel"]
        if done:
            break
    if not losses:
        return torch.zeros((), device=device), 0, n_illegal
    return torch.stack(losses).mean(), len(losses), n_illegal


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank AUC, ties averaged.  Written out rather than pulled in so it has no sklearn dependency.

    The tie handling is not cosmetic here: **51% of the tau = 0 ground set has support exactly 1**
    (A29) and the tau = 2 set is far from tie-free either, so a support ranking is heavily tied and
    an AUC that broke ties arbitrarily would flatter or punish the baseline at random.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="stable")
    ranks = np.empty(s.size, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1, dtype=np.float64)
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def support_auc(inst) -> float:
    """**G-R3's comparator**: how well support alone separates ``C(N_true)`` from the rest."""
    y = np.zeros(inst.n_clusters, dtype=bool)
    y[np.asarray(inst.truth_rows, dtype=np.int64)] = True
    return auc(np.asarray(inst.gs.support, dtype=np.float64), y)


# ---------------------------------------------------------------------------- W8: the swap MDP


def swap_clone_loss(policy, enc, env, actions, *, device):
    """Cross-entropy of ``pi(. | S_t)`` against ``actions[t]``, stepping a :class:`SwapEnv`.

    The same statement as :func:`clone_loss` over the amended MDP, and it goes through the *same*
    decoder call the sampler uses (``rollout._swap_step_logits``) rather than re-deriving the
    context -- ``clone_loss`` above spells its four scalars out inline, and a second copy of that
    arithmetic is precisely how a clone target and a rollout come to disagree about what state the
    policy is standing at.

    Returns ``(loss, n_steps, n_illegal)``.  ``n_illegal`` must be zero: the target comes from
    ``swap.oracle_swap_actions``, which generates it by stepping this env, so a non-zero count
    means the replay diverged from the generation and the driver is expected to **fail** on it
    rather than average it away (A72's first defect cost G-R2 0.939 against 1.000 by being
    averaged away).
    """
    from opc.rl.rollout import _swap_step_logits  # noqa: PLC0415  -- avoids a module cycle

    obs = env.reset()
    losses, n_illegal = [], 0
    for a in [int(x) for x in actions]:
        if not obs.legal[a]:
            n_illegal += 1
            continue
        lg = _swap_step_logits(policy, enc, obs, device)
        losses.append(-torch.log_softmax(lg, dim=-1)[a])
        obs, _done = env.step(a)
    if not losses:
        return torch.zeros((), device=device), 0, n_illegal
    return torch.stack(losses).mean(), len(losses), n_illegal
