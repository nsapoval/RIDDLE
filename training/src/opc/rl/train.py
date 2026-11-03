"""W5 -- group-normalised REINFORCE over cluster-system construction.  ``RL_PLAN.md`` §4.

    for each batch of B instances:
        encode each instance once
        sample K rollouts per instance (K distinct first actions, POMO-style)
        evaluate all B*K terminal states in one batched reward call
        A_i = (R_i - mu_group) / (sigma_group + eps)
        loss = - sum_i A_i sum_t log pi(a_t | s_t) - beta * H[pi]

**Why this and not PPO** is §4's argument and it stands: a terminal-only reward over a 50-step
horizon makes ``V(s_t)`` hard to fit, the K rollouts *are* the diverse pool generate-then-select
wants (which is G-R5), and group normalisation removes a per-instance MPL scale that varies by
thousands of log units across cells without a learned normaliser.

**Two things this module does that §4 does not say, and both are consequences of measurement.**

*The instance batch is sampled at one ``n``.*  Every network in one batched MPL fit must share a
triple index.  Sampling ``B`` instances at a single ``n`` is what lets the whole gradient step be
one fit, which matters because the evaluator's cost tracks *pooled shapes* rather than rows (A65):
eight groups pooled is far cheaper than eight pooled fits, and the CUDA graph is captured once
rather than eight times.

*The advantage can be a rank.*  ``--advantage rank`` replaces the z-score with the group's rank,
centred and scaled.  A group-normalised advantage **is** a ranking, so if the reward's truncation
is heterogeneous (G-R1b measured Kendall tau 0.93 between a 300- and a 1,200-iteration fit, A69),
a z-score propagates the magnitude error while a rank propagates only the ordering error -- which
is the part G-R1d measures directly.  The z-score stays the default because it is §4's, and the
pair is an ablation rather than a preference.

**The greedy baseline rides along for free.**  greedy-by-support's selection is deterministic per
instance, so scoring it through the same :class:`~opc.rl.reward.MPLReward` is a cache miss once and
a hit forever after.  ``reward_vs_greedy`` is therefore in every log row at no cost, and it is the
quantity **G-R4** is stated in.  It is *reported*, never fed to the advantage: making greedy the
baseline would be a different algorithm from §4's.
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from opc.rl import instances as I
from opc.rl import policy as P
from opc.rl import reward as R
from opc.rl.env import ClusterEnv, greedy_support_rollout
from opc.rl.rollout import rollout_group, swap_rollout_group
from opc.rl.swap import SwapEnv

__all__ = ["TrainConfig", "SwapTrainConfig", "InstancePool", "Trainer", "SwapTrainer",
           "advantage", "stepwise_advantage", "returns_to_go", "split_by_rep", "select_split"]


def split_by_rep(keys, train: int = 15, val: int = 18) -> dict:
    """``{"train": [...], "val": [...], "test": [...]}``, split on the **base topology**.

    Cells that share a ``rep`` derive from one base tree -- A2's blocked design makes ``r``, ``cu``
    and ``gamma`` within-topology contrasts on purpose -- so a split that put
    ``n50_rep3_r1_cu0.5_g0.5_small`` in train and ``n50_rep3_r2_cu0.5_g0.5_small`` in test would be
    testing on a re-parameterisation of a training tree.  The default is replicates 0-14 train,
    15-17 validation, 18-19 test: five of twenty topologies never seen during training.

    **The cut is an index on ``rep``, not a fraction, and it raises rather than mis-splitting.**
    The defaults are the S-grids' twenty replicates; on a hundred they give 15 train / 3 val / 82
    test and on the b-grids' successors 15 / 3 / 882, *silently*, which is the hazard
    ``w10_gnn.py`` documents for itself and which ``w4_bc``, ``w5_train``, ``w8_swap`` and
    ``w6_eval`` all still call into with the defaults.  So a rep range that runs past ``2 * val``
    is refused: it can only mean the cut belongs to a different grid than the keys do.  The guard
    is a no-op on every phase the defaults were written for (s20/s30/s50 stop at rep 19).
    """
    out: dict[str, list] = {"train": [], "val": [], "test": []}
    reps = [int(key.split("_rep")[1].split("_")[0]) for key in keys]
    if reps and max(reps) + 1 >= 2 * val:
        raise ValueError(
            f"split_by_rep: keys run to rep {max(reps)} but the validation cut is {val}, which "
            f"would put {sum(r >= val for r in reps)} of {len(reps)} cells in test.  Pass "
            f"--split-train/--split-val for this grid (the b-grids use 70/80; the x-grids 700/800)."
        )
    for key, rep in zip(keys, reps):
        out["train" if rep < train else "val" if rep < val else "test"].append(key)
    return out


def select_split(keys, split: str, train: int = 15, val: int = 18, *, phase: str = "") -> list:
    """:func:`split_by_rep`, plus ``split="all"`` -- every key, uncut.

    An **external** dataset is never trained on, so there is no leakage to prevent and no cut to
    make.  Saying that explicitly is better than the alternatives, both of which are wrong in a way
    nothing reports: :func:`split_by_rep` is *right* to refuse a rep range it cannot cut, so
    weakening its guard would break the grids it was written for; and there is in fact no
    ``(train, val)`` pair that puts 50 replicates in one split -- ``0..49`` needs ``val >= 26`` to
    pass the guard, and then ``test`` is 24 cells while ``--split train`` silently returns all 50
    under a name that says the opposite.

    ``phase`` is checked rather than trusted: ``"all"`` on a grid this repo trains on would
    evaluate on training cells, and that is the one way this convenience could produce a wrong
    number instead of an error.
    """
    if split != "all":
        return split_by_rep(keys, train, val)[split]
    from opc import grid  # local: opc.grid imports nothing from opc.rl, and this keeps it that way
    if phase in grid.PHASES:
        raise ValueError(
            f"--split all is refused on {phase!r}: it is a grid this repo trains on, so 'all' "
            f"would evaluate on training cells.  Use --split test.")
    return list(keys)


@dataclass
class TrainConfig:
    batch: int = 8                 # B: instances per gradient step
    rollouts: int = 16             # K: rollouts per instance
    lr: float = 1e-4
    grad_clip: float = 1.0
    entropy_start: float = 0.02    # §4's beta schedule
    entropy_end: float = 0.001
    entropy_steps: int = 2000
    advantage: str = "z"           # "z" | "rank"
    reward: str = "f1"             # "f1" | "mpl"
    pomo: bool = True
    pomo_grad: str = "include"
    temperature: float = 1.0
    steps: int = 1000
    checkpoint_every: int = 250
    seed: int = 0
    pool_size: int = 32            # instances held in memory
    curriculum: dict = field(default_factory=dict)   # phase -> sampling weight


def advantage(rewards: np.ndarray, kind: str, eps: float = 1e-8) -> np.ndarray:
    """``(K,)`` group-normalised advantage.

    ``rank`` maps the group to ``[-1, 1]`` by order alone.  Ties -- which are common, because two
    rollouts can reach the same selection and therefore exactly the same reward -- get the average
    rank, so tied rollouts receive identical advantages and cannot push each other apart.
    """
    r = np.asarray(rewards, dtype=np.float64)
    if r.size < 2:
        return np.zeros_like(r)
    if kind == "rank":
        order = np.argsort(np.argsort(r, kind="stable"), kind="stable").astype(np.float64)
        # average ranks over ties, so equal rewards give equal advantage
        for v in np.unique(r):
            m = r == v
            if m.sum() > 1:
                order[m] = order[m].mean()
        return 2.0 * order / (r.size - 1) - 1.0
    return (r - r.mean()) / (r.std() + eps)


class InstancePool:
    """LRU over the on-disk instance cache, holding the device-side graph beside each instance.

    Loading is a pickle read; ``graph_tensors`` is a gather plus a ``bincount`` over ``2|E|``.
    Neither is expensive, but at ``B`` instances a step and ~1,000 steps both would be paid
    thousands of times for a pool of ~960 cells that fits in memory several times over.
    """

    def __init__(self, keys_by_phase: dict, results: Path, tau: int, device, size: int = 32,
                 swap_tabu: int | None = None, swap_top_m: int | None = None,
                 lazy_env: bool = False, pair_graph: bool = False):
        self.keys_by_phase = keys_by_phase
        self.results = Path(results)
        self.tau = int(tau)
        self.device = device
        self.size = int(size)
        #: When set, each entry also carries a :class:`~opc.rl.swap.SwapEnv` built **on the same
        #: ``ClusterEnv``** -- so the construction env, its greedy selection and the swap walk's
        #: start state are one object and cannot drift apart.  ``None`` keeps v1's entries exactly.
        self.swap_tabu = swap_tabu
        self.swap_top_m = swap_top_m
        #: Build ``env`` and ``greedy`` on **first access** rather than on the pool miss.
        #:
        #: This is a *memory* switch, not a speed one, and it is what makes a 1,600-instance pool
        #: possible.  :class:`~opc.rl.env.ClusterEnv` builds a
        #: :class:`opc.solve.heuristics._State` per component whose ``_csr`` upcasts the overlap
        #: graph to **int64** -- a second copy of it at 16 B per directed edge against
        #: :class:`~opc.rl.env.OverlapCSR`'s 8 B.  Measured resident cost per cell: 8.7 MiB of
        #: instance arrays **+ 10.5 MiB** of env at n = 25, and 18.7 **+ 20.5** at n = 50.  At the
        #: 1,600 cells W10's 10x training set needs, eager construction is ~45.6 GiB against ~46 GB
        #: free, and A89 is the record of what an OOM does to a pool here: it **hangs**, silently.
        #:
        #: Under a pointwise BCE loss the *train* cells never touch either -- only the validation
        #: decode does -- so W10 opts in and pays for ~200 envs instead of ~1,600.  **The default is
        #: off**, because ``w4_bc``, ``w5_train``, ``w8_swap`` and ``w11_seq`` all read
        #: ``entry["env"]``/``entry["greedy"]`` on train cells and must keep getting today's entry
        #: unchanged.  ``swap_tabu`` forces eager construction regardless: :class:`~opc.rl.swap.SwapEnv`
        #: takes the env as a constructor argument and runs its own greedy rollout on it.
        self.lazy_env = bool(lazy_env) and swap_tabu is None
        #: Build the graph with :class:`~opc.rl.policy.PairConv`'s extras (clusters, hash table,
        #: containment).  Off by default, so every other caller's entry is unchanged.
        self.pair_graph = bool(pair_graph)
        self._lru: OrderedDict = OrderedDict()

    def get(self, phase: str, key: str, arm: str = ""):
        """One instance's entry.  ``arm`` selects the gene trees it was built from.

        **The LRU key is the triple**, because a mixed-arm run holds the *same* ``(phase, key)``
        twice with a different candidate set, overlap graph and node-feature matrix each time --
        ``Cell.key`` carries no arm, exactly as it carries no phase.  ``arm=""`` is the simulated
        trees and reproduces every landed caller's behaviour byte for byte.
        """
        hit = self._lru.get((phase, key, arm))
        if hit is not None:
            self._lru.move_to_end((phase, key, arm))
            return hit
        inst = I.load(I.cache_dir(self.results, phase, self.tau, arm) / f"{key}.pkl", arm=arm)
        g = ((P.graph_tensors(inst.graph, inst.gs.support, self.device, words=inst.gs.words,
                              n_taxa=int(inst.n), m=float(inst.m))
              if self.pair_graph else P.graph_tensors(inst.graph, inst.gs.support, self.device))
             if self.device is not None else None)
        entry = {"inst": inst, "graph": g,
                 "x_nodes": torch.as_tensor(inst.x_nodes, device=self.device),
                 "x_global": torch.as_tensor(inst.x_global, device=self.device)}
        if self.lazy_env:
            entry["env"] = entry["greedy"] = None
        else:
            env = ClusterEnv(inst.gs, budget=inst.k, graph=inst.graph)
            entry["env"], entry["greedy"] = env, greedy_support_rollout(env)
            if self.swap_tabu is not None:
                entry["swap"] = SwapEnv(inst.gs, budget=inst.k, graph=inst.graph,
                                        tabu=self.swap_tabu, top_m=self.swap_top_m, env=env)
        self._lru[(phase, key, arm)] = entry
        while len(self._lru) > self.size:
            self._lru.popitem(last=False)
        return entry

    def env_of(self, phase: str, key: str, arm: str = ""):
        """The entry's :class:`~opc.rl.env.ClusterEnv`, built and memoised on first access.

        The accessor exists because a dict cannot hold a lazy value transparently.  Under
        ``lazy_env=False`` it is a plain read of what ``get`` already built, so every existing
        caller may switch to it without changing behaviour.
        """
        e = self.get(phase, key, arm)
        if e["env"] is None:
            e["env"] = ClusterEnv(e["inst"].gs, budget=e["inst"].k, graph=e["inst"].graph)
        return e["env"]

    def greedy_of(self, phase: str, key: str, arm: str = ""):
        """The entry's greedy-by-support selection, built and memoised on first access."""
        e = self.get(phase, key, arm)
        if e["greedy"] is None:
            e["greedy"] = greedy_support_rollout(self.env_of(phase, key, arm))
        return e["greedy"]


class Trainer:
    def __init__(self, policy: P.Policy, pool: InstancePool, cfg: TrainConfig, *,
                 device, reward_fn=None, run_dir: Path | None = None):
        self.policy = policy
        self.pool = pool
        self.cfg = cfg
        self.device = device
        self.reward_fn = reward_fn
        self.run_dir = Path(run_dir) if run_dir else None
        self.opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)
        self.rng = np.random.default_rng(cfg.seed)
        self.step_i = 0
        self._phases = list(cfg.curriculum) or list(pool.keys_by_phase)
        w = np.asarray([cfg.curriculum.get(p, 1.0) for p in self._phases], dtype=float)
        self._weights = w / w.sum()

    # ------------------------------------------------------------------ schedule

    def beta(self) -> float:
        f = min(1.0, self.step_i / max(1, self.cfg.entropy_steps))
        return self.cfg.entropy_start + f * (self.cfg.entropy_end - self.cfg.entropy_start)

    def sample_instances(self):
        """``B`` instances, **all at one ``n``** -- see the module docstring."""
        phase = self._phases[int(self.rng.choice(len(self._phases), p=self._weights))]
        keys = self.pool.keys_by_phase[phase]
        take = self.rng.choice(len(keys), size=min(self.cfg.batch, len(keys)), replace=False)
        return phase, [self.pool.get(phase, keys[int(i)]) for i in take]

    # ------------------------------------------------------------------ one step

    def step(self) -> dict:
        cfg = self.cfg
        t_step = time.perf_counter()
        phase, entries = self.sample_instances()

        t0 = time.perf_counter()
        encs = [self.policy.encode(e["x_nodes"], e["x_global"], e["graph"]) for e in entries]
        t_encode = time.perf_counter() - t0

        t0 = time.perf_counter()
        groups = [rollout_group(self.policy, enc, e["inst"], e["env"], cfg.rollouts,
                                device=self.device, rng=self.rng, pomo=cfg.pomo,
                                pomo_grad=cfg.pomo_grad, temperature=cfg.temperature)
                  for enc, e in zip(encs, entries, strict=True)]
        t_rollout = time.perf_counter() - t0

        # The greedy selection is appended to every group so its reward comes back in the same
        # call.  It is a cache hit after the first time it is seen, and it never enters the
        # advantage -- only ``reward_vs_greedy``.
        t0 = time.perf_counter()
        items = [(e["inst"], [*g.selections, e["greedy"]])
                 for e, g in zip(entries, groups, strict=True)]
        rewards, rstats = self._reward(items)
        t_reward = time.perf_counter() - t0

        advs, logps, ents = [], [], []
        r_mean, r_greedy, r_best = [], [], []
        f1_mean, f1_max, f1_sel, f1_greedy = [], [], [], []
        n_floor = 0
        for e, g, vals in zip(entries, groups, rewards, strict=True):
            group_r, greedy_r = vals[:-1], vals[-1]
            n_floor += int((np.asarray(group_r) <= R.UNREALISABLE / 2.0).sum())
            advs.append(advantage(R.winsorise_unrealisable(group_r), cfg.advantage))
            logps.append(g.logp)
            ents.append(g.entropy)
            r_mean.append(float(np.mean(group_r)))
            r_best.append(float(np.max(group_r)))
            r_greedy.append(float(greedy_r))
            # **Three cluster-F1s per group, and they are three different questions.**
            #   `f1_mean`  -- what an average rollout recovers.
            #   `f1_max`   -- the pool's *ceiling*: is a good network being generated at all?  This
            #                 is G-R5's quantity, and it is free here because F1 costs no MPL fit.
            #   `f1_sel`   -- the F1 of the rollout the **reward** picks, which is what
            #                 generate-then-select would actually deliver.  Under `--reward f1`
            #                 this equals `f1_max` by construction, which makes it a self-check;
            #                 under `--reward mpl` the gap between them is exactly how much of the
            #                 pool's quality the objective fails to select.
            per = [e["inst"].cluster_f1(sel) for sel in g.selections]
            f1_mean.append(float(np.mean(per)))
            f1_max.append(float(np.max(per)))
            f1_sel.append(float(per[int(np.argmax(group_r))]))
            f1_greedy.append(float(e["inst"].cluster_f1(e["greedy"])))

        adv = torch.as_tensor(np.concatenate(advs), dtype=torch.float32, device=self.device)
        logp = torch.cat(logps)
        ent = torch.cat(ents)
        beta = self.beta()
        pg_term = -(adv * logp).mean()
        ent_term = -beta * ent.mean()
        loss = pg_term + ent_term

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.grad_clip)
        self.opt.step()
        self.step_i += 1

        row = {
            "step": self.step_i, "phase": phase, "reward": cfg.reward,
            "n": entries[0]["inst"].n, "beta": beta, "loss": float(loss.detach()),
            "grad_norm": float(gnorm),
            "loss_pg": float(pg_term.detach()), "loss_entropy": float(ent_term.detach()),
            # Logged per step, so it is comparable across episode lengths and against ln|legal|;
            # the loss uses the sum (see ``Rollouts.entropy``).
            "entropy": float((ent.detach() / torch.as_tensor(
                np.concatenate([g.lengths for g in groups]).clip(1),
                dtype=ent.dtype, device=ent.device)).mean()),
            "adv_spread": float(np.std(np.concatenate(advs))),
            "reward_mean": float(np.mean(r_mean)), "reward_best": float(np.mean(r_best)),
            "reward_greedy": float(np.mean(r_greedy)),
            "reward_vs_greedy": float(np.mean(np.asarray(r_mean) - np.asarray(r_greedy))),
            "best_vs_greedy": float(np.mean(np.asarray(r_best) - np.asarray(r_greedy))),
            "n_distinct_mean": float(np.mean([g.n_distinct for g in groups])),
            "n_unrealisable": n_floor,
            "episode_len_mean": float(np.mean([g.lengths.mean() for g in groups])),
            "legal_mean": float(np.mean([g.n_legal_mean for g in groups])),
            "cluster_f1_mean": float(np.mean(f1_mean)),
            "cluster_f1_max": float(np.mean(f1_max)),
            "cluster_f1_selected": float(np.mean(f1_sel)),
            "cluster_f1_greedy": float(np.mean(f1_greedy)),
            "f1_max_vs_greedy": float(np.mean(np.asarray(f1_max) - np.asarray(f1_greedy))),
            "f1_selected_vs_greedy": float(np.mean(np.asarray(f1_sel) - np.asarray(f1_greedy))),
            "seconds_encode": t_encode, "seconds_rollout": t_rollout,
            "seconds_reward": t_reward, "seconds_step": time.perf_counter() - t_step,
        }
        if rstats is not None:
            row.update({f"mpl_{k}": v for k, v in asdict(rstats).items()})
        return row

    def _reward(self, items):
        if self.cfg.reward == "f1":
            return [R.f1_rewards(inst, sels) for inst, sels in items], None
        return self.reward_fn.batch(items)

    # ------------------------------------------------------------------ checkpoints

    def save(self, tag: str) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        out = self.run_dir / f"{tag}.pt"
        torch.save({"step": self.step_i, "policy": self.policy.state_dict(),
                    "optimiser": self.opt.state_dict(),
                    "policy_config": asdict(self.policy.cfg),
                    "train_config": asdict(self.cfg)}, out)
        (self.run_dir / "config.json").write_text(json.dumps(
            {"policy": asdict(self.policy.cfg), "train": asdict(self.cfg)}, indent=1))
        return out


# ------------------------------------------------------------------------ W8: the swap MDP


@dataclass
class SwapTrainConfig(TrainConfig):
    """:class:`TrainConfig` plus the four things the swap MDP adds.

    ``batch`` and ``rollouts`` default **down** from v1's 8 x 16, and the arithmetic is the reason
    rather than caution.  The best-seen return needs an MPL value at every visited state, so a step
    presents ``B x K x (H + 1)`` networks instead of ``B x K`` -- ~450 at ``B = 2, K = 4`` and mean
    ``H = 56``.  Measured at **352 ms per state** on an easy cell and **546 ms** on a hard one at
    100 x 2 (`probe_swap_cost` P1), that is **~205 s a step**; the landed 300 x 4 schedule prices
    the same step at ~25 min, which is 57 gradient steps in 24 GPU-hours against the ~234 arm
    ``mpl`` needed in v1.  It is also why ``K`` may be this small at all: the candidate pool is
    ``K x H``, not ``K``.

    *Pooling does not scale the horizon away, and how far it goes depends on the cell.*  Marginal
    pooled trinet shapes are **1.86 per fitted network** over realistic (epsilon-greedy) walks on
    an easy cell and **5.1** on a hard one -- and **5.1 for a uniform walk on that same easy
    cell**, against 5.0 for 25 *unrelated* single swaps.  A uniform-walk benchmark therefore
    overstates a real rollout's cost by 2.7x, which is A69's lesson about measuring on the wrong
    action distribution, running the other way.
    """

    batch: int = 2
    rollouts: int = 4
    #: ``H`` is drawn per gradient step and held **fixed within it**, so one group is normalised at
    #: one horizon.  Randomising it is what puts states at ``t > 48`` in the training distribution;
    #: the policy itself never sees ``H`` (nothing in ``opc.rl.swap`` is a function of ``t``), so
    #: this widens the visited-state distribution rather than conditioning the policy on anything.
    horizon_min: int = 16
    horizon_max: int = 96
    tabu: int = 8
    #: Support cap on the action set.  ``None`` is the default and the reason is in
    #: :data:`opc.rl.swap.SwapEnv.DEFAULT_TOP_M`: it concentrates the improving swaps 3-19x and
    #: makes 8.7% of missed truth clusters unreachable at ``top_m = 256``.
    top_m: int | None = None
    #: ``dense`` credits step ``t`` with the return-to-go of the record increments; ``flat`` gives
    #: every step of a rollout the one advantage of ``max_t M_t``.  ``sum_t r_t == max M_t - M_0``
    #: exactly (``reward.record_increments``), so the two optimise the **same** return and differ
    #: only in where credit lands -- which is the fault A76 §12.5 located in v1's MDP.
    credit: str = "dense"
    #: Log units below which a new record is fit noise rather than progress.  A76 §12.6 measured
    #: 65 log units of movement from batch composition alone and §12's unresolved steps at a median
    #: ``|dMPL|`` of 136, against the resolved items' 43,231.
    record_deadband: float = 500.0
    #: ``< 1`` front-loads: a record set early is worth more than the same record set late, which
    #: is what makes a *short* inference horizon good rather than merely a prefix.  Left at 1.0 --
    #: the horizon is randomised instead, and the two are alternative answers to one question.
    discount: float = 1.0
    mpl_chunk: int = 128


def returns_to_go(rewards: np.ndarray, discount: float = 1.0) -> np.ndarray:
    """``G[t] = sum_{s >= t} discount^(s-t) r[s]``, per row."""
    r = np.asarray(rewards, dtype=np.float64)
    out = np.zeros_like(r)
    acc = np.zeros(r.shape[0], dtype=np.float64)
    for t in range(r.shape[1] - 1, -1, -1):
        acc = r[:, t] + discount * acc
        out[:, t] = acc
    return out


def stepwise_advantage(returns: np.ndarray, kind: str = "z", eps: float = 1e-8) -> np.ndarray:
    """``(K, H)`` -> group-normalised advantage, **column by column**.

    All ``K`` rollouts of a group are at step ``t`` simultaneously and started from the *same*
    state -- greedy -- so the other ``K - 1`` returns-to-go at step ``t`` are a proper baseline for
    rollout ``i``'s.  Normalising down the whole matrix instead would let the shrinking scale of a
    return-to-go (it has fewer steps left to accumulate) masquerade as a signal about the action.

    A column where every rollout got the same return -- common, because a converged walk earns
    nothing anywhere -- has zero spread and yields zero advantage, which is correct: there is
    nothing to learn there.  ``Trainer`` logs how often that happens rather than letting a step
    that taught nothing look like a step that did.
    """
    g = np.asarray(returns, dtype=np.float64)
    if g.shape[0] < 2:
        return np.zeros_like(g)
    if kind == "rank":
        return np.stack([advantage(g[:, t], "rank") for t in range(g.shape[1])], axis=1)
    mu = g.mean(axis=0, keepdims=True)
    sd = g.std(axis=0, keepdims=True)
    return (g - mu) / (sd + eps)


class SwapTrainer:
    """GRPO over swap walks.  Same loop as :class:`Trainer`; three things are different.

    1. **The episode starts at the answer's incumbent.**  ``S_0`` is greedy-by-support, so
       ``M_0 = MPL(greedy)`` and every number in a log row is already stated against the baseline
       G-R4 is denominated in -- ``reward_greedy`` is not a separate cache lookup, it is column 0
       of the trajectory.
    2. **The return is the best state seen, not the last.**  Measured on six real cells with the
       oracle target, the best state is **not** the final one on two of them (F1 0.939 final
       against 0.959 best, and 0.788 against 0.827) and the argmax sits at step 2-11 of 24-88.  So
       "walk for ``H`` and report the best seen" is not a convenience over a stopping rule; on a
       third of cells it is worth more than the stopping rule could have found.
    3. **The horizon is drawn per step.**  See :class:`SwapTrainConfig`.

    The loss terms are **per-step means** over ``(rollout, step)``, not sums.  That is what keeps
    ``beta`` comparable across a randomised ``H``: v1 summed both over the episode because its
    horizon was fixed by the mask, and summing here would make the entropy bonus six times heavier
    at ``H = 96`` than at ``H = 16`` for no stated reason.
    """

    def __init__(self, policy: P.Policy, pool: InstancePool, cfg: SwapTrainConfig, *,
                 device, reward_fn=None, run_dir: Path | None = None):
        self.policy = policy
        self.pool = pool
        self.cfg = cfg
        self.device = device
        self.reward_fn = reward_fn
        self.run_dir = Path(run_dir) if run_dir else None
        self.opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)
        self.rng = np.random.default_rng(cfg.seed)
        self.step_i = 0
        self._phases = list(cfg.curriculum) or list(pool.keys_by_phase)
        w = np.asarray([cfg.curriculum.get(p, 1.0) for p in self._phases], dtype=float)
        self._weights = w / w.sum()

    beta = Trainer.beta
    sample_instances = Trainer.sample_instances
    save = Trainer.save

    def horizon(self) -> int:
        return int(self.rng.integers(self.cfg.horizon_min, self.cfg.horizon_max + 1))

    def step(self) -> dict:
        cfg = self.cfg
        t_step = time.perf_counter()
        horizon = self.horizon()
        phase, entries = self.sample_instances()

        t0 = time.perf_counter()
        encs = [self.policy.encode(e["x_nodes"], e["x_global"], e["graph"]) for e in entries]
        t_encode = time.perf_counter() - t0

        t0 = time.perf_counter()
        groups = [swap_rollout_group(self.policy, enc, e["swap"], cfg.rollouts, horizon,
                                     device=self.device, pomo=cfg.pomo, pomo_grad=cfg.pomo_grad,
                                     temperature=cfg.temperature)
                  for enc, e in zip(encs, entries, strict=True)]
        t_rollout = time.perf_counter() - t0

        # One reward call for the whole step: every trajectory of every instance, flattened.  The
        # K trajectories share their first state (greedy) and the dedup is on the *closed* system,
        # so that is one fit and not K -- and it is a cache hit on every step after the first.
        t0 = time.perf_counter()
        items = [(e["inst"], [s for traj in g.selections for s in traj])
                 for e, g in zip(entries, groups, strict=True)]
        flat, rstats = self._reward(items)
        t_reward = time.perf_counter() - t0

        advs, logps, ents = [], [], []
        row_stats: list[dict] = []
        n_null = 0
        for e, g, vals in zip(entries, groups, flat, strict=True):
            inst = e["inst"]
            m = np.asarray(vals, dtype=np.float64).reshape(cfg.rollouts, horizon + 1)
            inc = np.stack([R.record_increments(m[i], cfg.record_deadband)
                            for i in range(cfg.rollouts)])
            gains = m.max(axis=1) - m[:, 0]
            if cfg.credit == "flat":
                a = advantage(gains, cfg.advantage)
                advs.append(np.repeat(a[:, None], horizon, axis=1))
            else:
                advs.append(stepwise_advantage(returns_to_go(inc, cfg.discount), cfg.advantage))
            n_null += int((advs[-1].std(axis=0) == 0).sum())
            logps.append(g.logp)
            ents.append(g.entropy)

            # **Three cluster F1s, and they are three different questions** -- v1's trio, restated
            # for a pool that is ``K x (H + 1)`` rather than ``K``.
            #   `f1_selected` -- the F1 of the state the *reward* picks out of the pool, which is
            #                    what a deployed generate-then-select would actually return.
            #   `f1_ceiling`  -- the best F1 anywhere in the pool: is a good system being visited
            #                    at all?  Free, because F1 costs no fit.
            #   `f1_greedy`   -- the start state, i.e. column 0.
            best_flat = int(np.argmax(np.asarray(vals)))
            sels = [s for traj in g.selections for s in traj]
            f1_all = np.asarray([inst.cluster_f1(s) for s in sels])
            row_stats.append({
                "mpl_greedy": float(m[0, 0]),
                "best_vs_greedy": float((m.max(axis=1) - m[:, 0]).max()),
                "mean_vs_greedy": float(gains.mean()),
                # Where in the walk the record was set.  If this piles up at ``H`` the horizon is
                # binding; if it piles up early, reporting the *final* state would have lost it.
                "best_step": float(np.mean(np.argmax(m, axis=1))),
                "best_is_last": float(np.mean(np.argmax(m, axis=1) == horizon)),
                "f1_selected": float(f1_all[best_flat]),
                "f1_ceiling": float(f1_all.max()),
                "f1_greedy": float(inst.cluster_f1(g.selections[0][0])),
                # ``sum_t r_t <= max_t M_t - M_0``; the gap is what the deadband swallowed.
                "deadband_leak": float((gains - inc.sum(axis=1)).mean()),
                "revisits": float(g.n_revisits) / max(1, cfg.rollouts * horizon),
                "distinct_states": float(g.n_distinct_states),
                "distinct_first": float(g.n_distinct_first),
                "legal_mean": float(g.n_legal_mean),
            })

        adv = torch.as_tensor(np.concatenate([a.reshape(-1) for a in advs]),
                              dtype=torch.float32, device=self.device)
        logp = torch.cat([x.reshape(-1) for x in logps])
        ent = torch.cat([x.reshape(-1) for x in ents])
        beta = self.beta()
        pg_term = -(adv * logp).mean()
        ent_term = -beta * ent.mean()
        loss = pg_term + ent_term

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.grad_clip)
        self.opt.step()
        self.step_i += 1

        agg = {k: float(np.mean([r[k] for r in row_stats])) for k in row_stats[0]}
        row = {
            "step": self.step_i, "phase": phase, "reward": cfg.reward, "credit": cfg.credit,
            "horizon": horizon, "n": entries[0]["inst"].n, "beta": beta,
            "loss": float(loss.detach()), "grad_norm": float(gnorm),
            "loss_pg": float(pg_term.detach()), "loss_entropy": float(ent_term.detach()),
            # Per step, so it is comparable across a randomised horizon and against ln|legal|.
            "entropy": float(ent.detach().mean()),
            "adv_spread": float(np.std(np.concatenate([a.reshape(-1) for a in advs]))),
            "n_null_columns": n_null,
            "f1_selected_vs_greedy": agg["f1_selected"] - agg["f1_greedy"],
            "f1_ceiling_vs_greedy": agg["f1_ceiling"] - agg["f1_greedy"],
            "seconds_encode": t_encode, "seconds_rollout": t_rollout,
            "seconds_reward": t_reward, "seconds_step": time.perf_counter() - t_step,
            **agg,
        }
        if rstats is not None:
            row.update({f"mpl_{k}": v for k, v in asdict(rstats).items()})
        return row

    def _reward(self, items):
        if self.cfg.reward == "f1":
            return [R.f1_rewards(inst, sels) for inst, sels in items], None
        return self.reward_fn.batch(items)
