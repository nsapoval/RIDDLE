"""F3's missing half: search over *network* space with edge moves, for comparison.

``followup-plan-v0.3.md`` §F3 opens with the sentence the search tier never tested --

    Tests the premise behind the pivot: that searching cluster-system space under (L)/(N3O) is
    materially better than searching network space with edge moves.

-- and `results/FOLLOWUP_v0.3_search.md` ran only the cluster-system half.  §0.1 asserts that
move-based annealing over network space fails at 18 taxa, but cites a PhyloNet timing rather than
reproducing it here, so the comparison the pivot rests on has never been made on this grid.  This
module makes it.

**The move set.**  NNI on the tree part, plus the reticulation moves ``opc.mpl.perturb`` already
validates -- relocate, delete, add.  That is the standard neighbourhood for level-1 network search
and it is deliberately *not* restricted by the tau = 2 candidate set: the whole point of the
comparison is that cluster-system search gets a data-derived ground set and network search does
not.

**Times are a scaffold, not data.**  Any topological move breaks the monotonicity of ``net.time``,
and repairing it edge by edge is where a subtle bug would live.  Instead :func:`retime` rebuilds
every time from the node's own cluster as ``n_taxa - |c(v)|`` -- exactly the convention
``closure.network_from_hasse`` uses for a cluster-derived network.  So both halves of the
comparison
carry the same (meaningless) placeholder times and both get their real lengths from the same
``Fit`` transfer and checkpoint re-optimisation.  Nothing about the comparison turns on times.

**Every proposal is validated before it is scored.**  ``is_galled`` and ``level <= 1`` are re-
derived
from the moved network, and the leaf set is checked against the original; a move that fails returns
``None`` and costs no evaluation.  A network search that silently left the model class would be the
same genre of defect as A55 and A56, and this is where it would enter.
"""

from __future__ import annotations

import numpy as np

from opc import bitset
from opc.mpl import perturb
from opc.simulate import networks as N

__all__ = ["retime", "words_of", "cluster_f1", "nni", "propose", "MOVES"]

#: The proposal kinds, and the weight each is drawn with.  Tree moves dominate because the tree
#: part
#: carries ~49 of the ~50 clusters; the reticulation moves exist so ``k`` can change and drift.
MOVES = ("nni", "nni", "nni", "relocate", "delete-add")


def retime(net: N.Network) -> N.Network:
    """Rebuild ``net.time`` as ``n_taxa - |c(v)|``, in place, and return it.

    Monotone along every edge by construction, because a child's cluster is contained in its
    parent's.  This is ``closure.network_from_hasse``'s convention, which is what makes a
    network-space candidate and a cluster-space candidate carry the same placeholder parameters.
    """
    # ``hardwired_clusters`` keys only the *reachable* nodes: a Network carries dead ids left by
    # reticulation placement and degree-2 suppression.  Unreachable nodes never appear in
    # ``Parameters.of``, which walks the same postorder, so leaving their times alone is safe.
    below = N.hardwired_clusters(net)
    for v, c in below.items():
        net.time[v] = float(net.n_taxa - len(c))
    return net


def words_of(net: N.Network) -> np.ndarray:
    """``uint64[n_nodes, W]``: the cluster each node stands for, packed.

    The key that ``followup.search.Fit`` transfers parameters on is a *(child cluster, parent
    cluster)* pair, and it reads those clusters out of an array indexed by node id.  For a
    cluster-derived network that array is the closed word list; for an arbitrary network it is
    this.
    Same keys, so a fit taken on one is transferable to the other -- which is what lets both halves
    of the comparison share one ``Scorer``.
    """
    below = N.hardwired_clusters(net)
    empty = bitset.from_indices([], net.n_taxa)
    return np.stack([bitset.from_indices(sorted(below[v]), net.n_taxa) if v in below else empty
                     for v in range(net.n_nodes)])


def cluster_f1(candidate: N.Network, truth: N.Network) -> dict:
    """Cluster precision/recall/F1 between two networks' nontrivial clustering systems.

    ``opc.solve.truth.align`` cannot be used here: it scores *rows of a ground set*, and a
    network-space candidate is under no obligation to have its clusters in one.  This compares the
    systems directly, which is what ``align`` does underneath, and reduces to the same number when
    the candidate's clusters happen to be in the ground set.
    """
    a, b = N.nontrivial_clusters(candidate), N.nontrivial_clusters(truth)
    correct = len(a & b)
    prec = correct / len(a) if a else 0.0
    rec = correct / len(b) if b else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"align_n_selected": len(a), "align_n_true": len(b), "align_n_correct": correct,
            "align_precision": prec, "align_recall": rec, "align_f1": f1,
            "align_rho_selected": candidate.n_retics, "align_rho_true": truth.n_retics,
            "align_cluster_distance": len(a ^ b)}


def _copy(net: N.Network) -> N.Network:
    import copy as _c
    return _c.deepcopy(net)


def nni(net: N.Network, rng: np.random.Generator, *, max_tries: int = 40) -> N.Network | None:
    """One nearest-neighbour interchange on the tree part; ``None`` if no legal one was found.

    Picks an internal tree edge ``(u, v)`` with both endpoints internal and neither a hybrid or a
    donor, then swaps one of ``v``'s tree children with ``v``'s sibling under ``u``.  Nodes that a
    reticulation cycle passes through are excluded up front rather than repaired afterwards, and
    the
    result is still validated -- excluding them is an optimisation, not the guarantee.
    """
    touched: set[int] = set()
    for r in net.retics:
        touched |= N.cycle_vertices(net, r)
    for _ in range(max_tries):
        cand = [v for v in range(net.n_nodes)
                if not net.is_leaf(v) and net.tree_parent[v] >= 0
                and v not in touched and net.tree_parent[v] not in touched
                and len(net.tree_children[v]) == 2
                and len(net.tree_children[net.tree_parent[v]]) == 2]
        if not cand:
            return None
        v = int(cand[rng.integers(len(cand))])
        u = net.tree_parent[v]
        sib = [w for w in net.tree_children[u] if w != v]
        if not sib:
            continue
        w = int(sib[0])
        a = int(net.tree_children[v][int(rng.integers(2))])
        out = _copy(net)
        out.tree_children[u] = [x if x != w else a for x in out.tree_children[u]]
        out.tree_children[v] = [x if x != a else w for x in out.tree_children[v]]
        out.tree_parent[a], out.tree_parent[w] = u, v
        retime(out)
        if _valid(out, net):
            return out
    return None


def _valid(out: N.Network, ref: N.Network) -> bool:
    """The moved network is still a level-1 galled network on the same leaves."""
    try:
        if set(out.leaf_taxon.values()) != set(ref.leaf_taxon.values()):
            return False
        below = N.hardwired_clusters(out)
        if out.root not in below or len(below[out.root]) != out.n_taxa:
            return False
        return N.is_galled(out) and N.level(out) <= 1
    except Exception:  # noqa: BLE001
        return False


def propose(net: N.Network, rng: np.random.Generator, k: int,
            **kw) -> tuple[str, N.Network] | None:
    """One neighbour of ``net`` under the move set; ``None`` if the draw produced nothing legal."""
    kind = MOVES[int(rng.integers(len(MOVES)))]
    if kind == "nni":
        out = nni(net, rng)
        return ("nni", out) if out is not None else None

    # **A reticulation whose hybrid is a leaf cannot be moved by ``perturb``.**  A hybrid species
    # with one extant descendant is the common biological case and it is exactly what the *regular*
    # version of a network produces when a hybrid's cluster is a singleton -- so a cluster-derived
    # start is full of them.  ``likelihood.Parameters.of`` handles it explicitly; ``perturb`` does
    # not, because E0b only ever calls it on generator output, where a hybrid is never a leaf.
    # ``resolve_one_reticulation`` then tries to suppress a degree-2 node that has no child and
    # raises.  Those reticulations are skipped here rather than repaired: the move set is what is
    # under test, not ``perturb``.
    movable = [i for i, r in enumerate(net.retics) if not net.is_leaf(r.hybrid)]
    try:
        if kind == "relocate" and movable:
            i = int(movable[rng.integers(len(movable))])
            out = perturb.relocate_hybrid(net, i, rng, **kw)
            return ("relocate", retime(out)) if out is not None else None
        if movable:  # delete-add: drop one reticulation and place a fresh one
            base = perturb.delete_reticulation(net, int(movable[rng.integers(len(movable))]))
            out = perturb.add_reticulation(base, rng, **kw) if base.n_retics < k else base
            return ("delete-add", retime(out)) if out is not None else None
        if net.n_retics < k:
            out = perturb.add_reticulation(_copy(net), rng, **kw)
            return ("add", retime(out)) if out is not None else None
    except (ValueError, KeyError, RuntimeError):
        return None       # a move the machinery cannot perform is an illegal move, not a crash
    return None


def hill_climb(start, scorer, truth, k, *, budget, rng, on_checkpoint,
               checkpoints=(0, 10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000),
               refit_every: int = 5, **kw) -> list[dict]:
    """Steepest-ascent-by-sampling over the move set, with the same protocol as the cluster search.

    One proposal per evaluation, accepted if it scores better -- the network-space analogue of
    ``followup.search._anneal`` at zero temperature.  Hill climbing rather than annealing because
    §3.4 measured annealing as the weakest of the cluster-space methods, and the comparison should
    not be lost to a temperature schedule.

    Everything that makes the two halves comparable is shared: the same ``Scorer``, so evaluations
    are counted the same way; the same ``Fit`` transfer keyed on cluster pairs; the same geometric
    checkpoints, each a **full-triple** re-optimisation.
    """
    cur = retime(_copy(start))
    cur_val = scorer.score_network(cur, words_of(cur))
    scorer.n_evals += 1
    best, best_val = cur, cur_val
    records: list[dict] = []
    targets = [int(c) for c in checkpoints if c <= budget]
    if budget not in targets:
        targets.append(int(budget))
    ti = 0
    last_fit_at = 0
    accepted = rejected = illegal = 0

    def emit(at):
        row = {"checkpoint": int(at), "n_evals": int(scorer.n_evals),
               "n_accepted": accepted, "n_rejected": rejected, "n_illegal": illegal,
               **on_checkpoint(int(scorer.n_evals), best)}
        records.append(row)

    while ti < len(targets) and scorer.n_evals >= targets[ti]:
        emit(targets[ti])
        ti += 1
    while scorer.n_evals < budget:
        pr = propose(cur, rng, k, **kw)
        if pr is None:
            illegal += 1
            continue
        _, cand = pr
        val = scorer.score_network(cand, words_of(cand))
        scorer.n_evals += 1
        if val > cur_val:
            cur, cur_val, accepted = cand, val, accepted + 1
            if val > best_val:
                best, best_val = cand, val
                if refit_every > 0 and scorer.n_evals - last_fit_at >= refit_every * 100:
                    scorer.refit_network(best, words_of(best))
                    last_fit_at = scorer.n_evals
        else:
            rejected += 1
        while ti < len(targets) and scorer.n_evals >= targets[ti]:
            emit(targets[ti])
            ti += 1
    while ti < len(targets):
        emit(targets[ti])
        ti += 1
    return records
