"""W5's sampling half: K rollouts of one instance, with their log-probs and entropies.

``RL_PLAN.md`` §4 samples ``K`` rollouts per instance and normalises the terminal reward within the
group.  This module produces the group; :mod:`opc.rl.reward` prices it and :mod:`opc.rl.train`
turns the pair into a gradient.

**The rollouts are sequential, not lock-stepped, and that is a measurement rather than laziness.**
Lock-stepping needs ``K`` live :class:`~opc.rl.env.ClusterEnv` states per instance, and a
``SearchState`` carries one ``heuristics._State`` per component with that component's CSR --
3.6 MB at a tau = 2 cell, so ``B x K`` of them is hundreds of megabytes to rebuild whenever the
instance sample changes.  What it would buy is decoder *launches*: 400 batched calls a step instead
of 6,400 small ones, about one second.  But the step is dominated by the environment either way --
a rollout is 28 ms at tau = 2 (measured), so 128 rollouts is 3.6 s whichever order they run in --
so lock-stepping would trade a large, thrashing memory pool for ~20% of a term that is not the
bottleneck.  One env per instance, reset in place (which is what makes reset 57 ms rather than
423 ms), reused by every rollout in the group.

**The backward is still batched.**  Sequential *sampling* does not mean sequential *learning*: the
per-step log-probs accumulate into one graph across the whole group and the whole instance batch,
and ``train`` calls backward once.

**POMO-style distinct first actions are on by default, and A69 is why.**  After behaviour cloning
the policy sits in the greedy basin, and that basin is narrow: dropping four of greedy's clusters
and re-completing greedily returned **48 identical networks in 48 draws**, and even at
``epsilon = 0.3`` the median draw differed from greedy by two clusters.  A group of ``K``
i.i.d. samples from a near-deterministic policy is a group of ``K`` copies, every advantage is
exactly zero, and the run stalls while looking healthy.  Forcing the ``K`` first actions to be
distinct -- drawn without replacement from ``pi(. | s_0)``, so they are still the policy's own
preferences -- makes the group non-degenerate by construction.

*The one approximation this makes, stated:* the log-probability credited to a first action is the
plain ``pi(a | s_0)``, not the without-replacement (Plackett-Luce) correction.  The bias is
confined to step 0 of ~50, and ``--pomo-grad exclude`` drops the first action from the objective
entirely for anyone who would rather have no bias than a small one.  ``n_distinct`` is logged every
step either way, because a collapsed group is the failure this is guarding against and it must be
visible rather than inferred.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from opc.rl.policy import dynamic_features

__all__ = ["Rollouts", "rollout_group"]


@dataclass
class Rollouts:
    """One group: ``K`` terminal selections of one instance, with the graph to learn from."""

    selections: list           # K arrays of ground-set rows
    logp: torch.Tensor         # (K,) sum_t log pi(a_t | s_t) -- carries grad
    #: ``(K,)`` **sum** over steps of ``H[pi(. | s_t)]``, not the mean, and the pairing is the
    #: point: ``logp`` is a sum over ~50 steps and is of order 350, so an entropy term averaged
    #: over steps (order 5) enters the loss two orders of magnitude smaller than the term it is
    #: meant to regularise, and RL_PLAN §4's schedule (beta 0.02 -> 0.001) would do nothing at all.
    #: Summing both makes ``beta`` the per-step trade-off it reads as.  ``entropy_per_step`` is the
    #: quantity to *log*, because that is the one that is comparable across episode lengths.
    entropy: torch.Tensor      # (K,) sum over steps of H[pi(. | s_t)] -- carries grad
    lengths: np.ndarray        # (K,) episode lengths
    n_legal_mean: float        # mean legal-set size over all visited states
    n_distinct: int            # distinct terminal selections -- the collapse diagnostic

    def __len__(self) -> int:
        return len(self.selections)


def _scalars(env, inst, n_legal: int, device) -> torch.Tensor:
    """The four context scalars of §3: ``k_remaining``, ``k/n``, ``t/n``, legal-set share."""
    return torch.tensor(
        [[(env.budget - env.state.n_hybrids) / max(1, env.budget),
          env.budget / inst.n,
          env.t / inst.n,
          n_legal / max(1, inst.n_clusters)]],
        dtype=torch.float32, device=device)


def rollout_group(policy, enc, inst, env, k_rollouts: int, *, device, rng,
                  greedy: bool = False, pomo: bool = True, pomo_grad: str = "include",
                  temperature: float = 1.0, dynamic_extra=None) -> Rollouts:
    """Sample ``k_rollouts`` episodes on ``inst``, all sharing the encoding ``enc``."""
    support = np.asarray(inst.gs.support, dtype=np.float64)
    hidden = policy.cfg.hidden

    first_actions = None
    if pomo and k_rollouts > 1 and not greedy:
        env.reset()
        mask0, deg0 = env.legal_mask(with_features=True)
        n_legal0 = int(mask0.sum())
        if n_legal0 >= k_rollouts:
            lg = policy.logits(
                enc,
                accepted_sum=torch.zeros(1, hidden, device=device),
                accepted_n=torch.zeros(1, device=device),
                last_action=torch.full((1,), -1, dtype=torch.long, device=device),
                scalars=_scalars(env, inst, n_legal0, device),
                dynamic=torch.as_tensor(
                    dynamic_features(support, mask0, deg0, 0, extra=dynamic_extra),
                    device=device).unsqueeze(0),
                mask=torch.as_tensor(mask0, device=device).unsqueeze(0),
            )
            with torch.no_grad():
                probs = torch.softmax(lg[0] / temperature, dim=-1)
            first_actions = torch.multinomial(probs, k_rollouts, replacement=False).tolist()

    logps, ents, sels, lens = [], [], [], []
    legal_seen: list[int] = []
    for r in range(k_rollouts):
        env.reset()
        mask, deg = env.legal_mask(with_features=True)
        step_logp, step_ent, t = [], [], 0
        acc_sum = torch.zeros(1, hidden, device=device)
        while mask.any():
            n_legal = int(mask.sum())
            legal_seen.append(n_legal)
            lg = policy.logits(
                enc,
                # Maintained incrementally: the mean of the accepted embeddings is a running sum,
                # so the context costs O(1) a step rather than O(|S_t|), and -- more to the point
                # -- indexing a CUDA tensor with ``env.selection`` (numpy) would copy the index
                # host-to-device 6,400 times a gradient step.
                accepted_sum=acc_sum,
                accepted_n=torch.tensor([float(env.selection.size)], device=device),
                last_action=torch.tensor([env.last_action], dtype=torch.long, device=device),
                scalars=_scalars(env, inst, n_legal, device),
                dynamic=torch.as_tensor(
                    dynamic_features(support, mask, deg, int(env.selection.size),
                                     extra=dynamic_extra),
                    device=device).unsqueeze(0),
                mask=torch.as_tensor(mask, device=device).unsqueeze(0),
            )[0]
            logp = torch.log_softmax(lg / temperature, dim=-1)
            forced = (t == 0 and first_actions is not None)
            if greedy:
                a = int(torch.argmax(logp))
            elif forced:
                a = int(first_actions[r])
            else:
                a = int(torch.multinomial(logp.detach().exp(), 1))
            # ``-inf`` logits contribute 0 * -inf = nan to the entropy, so the sum is taken over
            # the legal set only.  A nan entropy silently poisons the whole loss.
            p = logp.exp()
            keep = torch.isfinite(logp)
            if not (forced and pomo_grad == "exclude"):
                step_logp.append(logp[a])
            step_ent.append(-(p[keep] * logp[keep]).sum())
            obs, _reward, done, _info = env.step(a)
            acc_sum = acc_sum + enc.h[a].unsqueeze(0)
            mask, deg = obs["legal"], obs["deg_sel"]
            t += 1
            if done:
                break
        sels.append(env.selection.copy())
        lens.append(t)
        logps.append(torch.stack(step_logp).sum() if step_logp
                     else torch.zeros((), device=device))
        ents.append(torch.stack(step_ent).sum() if step_ent else torch.zeros((), device=device))

    return Rollouts(
        selections=sels, logp=torch.stack(logps), entropy=torch.stack(ents),
        lengths=np.asarray(lens, dtype=np.int64),
        n_legal_mean=float(np.mean(legal_seen)) if legal_seen else 0.0,
        n_distinct=len({tuple(sorted(s.tolist())) for s in sels}),
    )


# ---------------------------------------------------------------------------- W8: the swap MDP


@dataclass
class SwapRollouts:
    """One group of swap walks: ``K`` trajectories of ``H + 1`` maximal cluster systems each.

    **Per-step log-probs and entropies, not sums.**  ``Rollouts`` above sums both over the episode
    because v1's return is terminal and every step of a rollout carries the same advantage.  W8's
    dense credit gives step ``t`` its *own* advantage (the return-to-go of the record increments),
    so the sum has to be taken after the advantage rather than before it, and taking it here would
    throw away exactly the structure the amendment exists to exploit.  ``--credit flat`` sums these
    itself and reproduces v1's shape.

    ``selections`` is ``K x (H + 1)``: **the start state is included**, because it is the state the
    record starts at and ``MPL(greedy)`` is what every improvement is measured against.  It is also
    the same array for all ``K`` rollouts, so it costs one cache hit and not ``K`` fits.
    """

    selections: list           # K lists of (H + 1) ground-set row arrays
    logp: torch.Tensor         # (K, H) log pi(a_t | s_t) -- carries grad
    entropy: torch.Tensor      # (K, H) H[pi(. | s_t)]    -- carries grad
    actions: np.ndarray        # (K, H)
    horizon: int
    n_legal_mean: float
    #: Distinct **first swaps** actually taken.  POMO forces these apart; if this is below ``K``
    #: the forcing did not fire (too few legal actions) and the group may be degenerate.
    n_distinct_first: int
    #: Distinct systems over the whole ``K x (H + 1)`` pool -- the real diversity number now, since
    #: the pool is ``K x H`` rather than ``K`` (the horizon is a pool-size knob).
    n_distinct_states: int
    #: Steps that landed on a system already visited *within their own trajectory*.  The tabu ring
    #: is what this measures; a rate that does not fall when ``--tabu`` is raised means the cycles
    #: are longer than the tenure.
    n_revisits: int

    def __len__(self) -> int:
        return len(self.selections)


def _swap_step_logits(policy, enc, obs, device):
    """One decoder call at a swap state.  ``|S_t|`` changes wholesale, so the context is gathered.

    ``rollout_group`` maintains ``accepted_sum`` incrementally because v1 only ever *adds*.  A swap
    displaces a median of 3-4 incumbents, so there is no running sum to maintain -- the mean of the
    accepted embeddings is re-gathered from ``enc.h``.  It is an index-select of ~50 rows out of
    ``(|C|, 128)``, which is cheaper than the mask this replaced.
    """
    sel = torch.as_tensor(obs.selection, dtype=torch.long, device=device)
    acc = enc.h.index_select(0, sel).sum(0, keepdim=True)
    return policy.logits(
        enc,
        accepted_sum=acc,
        accepted_n=torch.tensor([float(obs.selection.size)], device=device),
        last_action=torch.tensor([obs.last_action], dtype=torch.long, device=device),
        scalars=torch.as_tensor(obs.scalars, device=device).unsqueeze(0),
        dynamic=torch.as_tensor(obs.dynamic, device=device).unsqueeze(0),
        mask=torch.as_tensor(obs.legal, device=device).unsqueeze(0),
    )[0]


def swap_rollout_group(policy, enc, env, k_rollouts: int, horizon: int, *, device,
                       greedy: bool = False, pomo: bool = True, pomo_grad: str = "include",
                       temperature: float = 1.0, seed: int | None = None) -> SwapRollouts:
    """``K`` swap walks of ``H`` steps on one instance, all sharing the encoding ``enc``.

    **``horizon`` is a caller's argument, never the policy's.**  Nothing passed to
    :meth:`opc.rl.policy.Policy.logits` here is a function of ``t`` or of ``H`` --
    ``opc.rl.swap.swap_scalars`` replaced v1's ``t/n`` with ``swap_distance``, which is a property
    of the state.  That is what makes the walk a stationary Markov chain and therefore what makes
    the ``H = 24`` trajectory the *prefix* of the ``H = 96`` one at a fixed seed.  Pinned by
    ``tests/test_rl_swap.py::test_prefix_property``, because it is the whole basis for treating
    ``H`` as a free inference parameter and it would be silently lost the moment anything here
    started reading ``env.t``.

    **POMO is more natural here than in v1, not less.**  Every rollout starts at the *same* state
    -- greedy -- so ``K`` i.i.d. samples from a behaviour-cloned policy would be ``K`` copies with
    certainty rather than merely in practice (A69 measured 48 of 48 identical draws in v1's much
    weaker version of this). Forcing the first swaps apart makes the group non-degenerate by
    construction, and "K different first swaps from greedy" is exactly the pool A76 §12.2's search
    was sampling by hand.

    **``seed`` gives each rollout its own generator, and that is what makes the prefix property
    hold for the whole group rather than for rollout 0 alone.**  The rollouts are sampled
    sequentially, so on the global RNG stream rollout ``r`` starts at whatever position rollout
    ``r - 1`` left it -- which is a function of ``H``.  The first draft of this function was
    written that way and ``test_prefix_property`` caught it: rollout 0 matched at two horizons and
    rollouts 1 and 2 did not, for a reason that has nothing to do with the policy.  A per-rollout
    generator seeded ``seed * 1_000_003 + r`` makes every rollout's stream a function of ``(seed,
    r)`` only.  ``seed=None`` keeps the global stream, which is what training wants.
    """
    def _gen(offset: int):
        if seed is None:
            return None
        g = torch.Generator(device=torch.device(device))
        g.manual_seed(int(seed) * 1_000_003 + offset)
        return g

    first_actions = None
    if pomo and k_rollouts > 1 and not greedy:
        obs0 = env.reset()
        if int(obs0.legal.sum()) >= k_rollouts:
            # ``no_grad`` around the whole call: this decode only *chooses* the K first swaps, and
            # each of them is credited again inside its own rollout at step 0.  Keeping it in the
            # graph would retain a second ``(1, |C|)`` score per instance for no gradient.
            with torch.no_grad():
                lg0 = _swap_step_logits(policy, enc, obs0, device)
                probs = torch.softmax(lg0 / temperature, dim=-1)
            first_actions = torch.multinomial(probs, k_rollouts, replacement=False,
                                              generator=_gen(-1)).tolist()

    logps, ents, sels, acts = [], [], [], []
    legal_seen: list[int] = []
    revisits = 0
    for r in range(k_rollouts):
        gen = _gen(r)
        obs = env.reset()
        traj, step_logp, step_ent, taken = [obs.selection.copy()], [], [], []
        for t in range(int(horizon)):
            legal_seen.append(int(obs.legal.sum()))
            lg = _swap_step_logits(policy, enc, obs, device)
            logp = torch.log_softmax(lg / temperature, dim=-1)
            forced = (t == 0 and first_actions is not None)
            if greedy:
                a = int(torch.argmax(logp))
            elif forced:
                a = int(first_actions[r])
            else:
                a = int(torch.multinomial(logp.detach().exp(), 1, generator=gen))
            # ``-inf`` logits contribute ``0 * -inf = nan`` to the entropy, so the sum is taken
            # over the legal set only -- a nan entropy silently poisons the whole loss.
            p, keep = logp.exp(), torch.isfinite(logp)
            step_ent.append(-(p[keep] * logp[keep]).sum())
            # ``pomo_grad="exclude"`` drops the biased first term.  Unlike v1 it cannot simply be
            # omitted from a sum: dense credit indexes by step, so the term is **zeroed** in place
            # and the shape is kept.
            lp = logp[a]
            step_logp.append(torch.zeros((), device=device) * lp
                             if (forced and pomo_grad == "exclude") else lp)
            obs, _done = env.step(a)
            traj.append(obs.selection.copy())
            taken.append(a)
        revisits += int(env.n_revisits)
        sels.append(traj)
        acts.append(taken)
        logps.append(torch.stack(step_logp) if step_logp else torch.zeros(0, device=device))
        ents.append(torch.stack(step_ent) if step_ent else torch.zeros(0, device=device))

    distinct = {env._fingerprint(s) for traj in sels for s in traj}  # noqa: SLF001
    return SwapRollouts(
        selections=sels, logp=torch.stack(logps), entropy=torch.stack(ents),
        actions=np.asarray(acts, dtype=np.int64).reshape(k_rollouts, -1),
        horizon=int(horizon),
        n_legal_mean=float(np.mean(legal_seen)) if legal_seen else 0.0,
        n_distinct_first=len({a[0] for a in acts if a}),
        n_distinct_states=len(distinct), n_revisits=revisits)
