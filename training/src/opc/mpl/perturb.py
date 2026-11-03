"""Near-miss networks: the alternatives E0b scores the truth against, and E2a's extra edge.

``exp-plan-n50-mpl-v0.1`` §E0b asks whether MPL separates the truth from networks that are *nearly*
right -- not from random ones, which would be a much weaker test.  Three neighbours, plus E2a's:

``delete_reticulation``  drop one reticulation, keeping the topology otherwise intact (k -> k-1).
``relocate_hybrid``      keep k, but move one reticulation's hybrid head to a different edge alive
                         at the same instant.  The hardest of the three, and the one that tests
                         whether MPL localises a reticulation rather than merely counting them.
``underlying_tree``      resolve every reticulation the same way (k -> 0).
``add_reticulation``     E2a: one more edge on the true topology (k -> k+1), where ``sup MPL`` must
                         be non-decreasing because gamma -> 0 embeds the smaller model.

Everything here goes through the generator's own placement machinery rather than editing edges by
hand, so a perturbed network satisfies exactly the constraints a *generated* one does -- level-1,
no shortcuts, and the cycle-size band if one is asked for.  Reusing
:func:`opc.simulate.networks._add_reticulation` needs ``consumed`` restored first, since
:func:`~opc.simulate.networks.with_reticulations` clears it; that is what
:func:`~opc.simulate.networks.recompute_consumed` is for.
"""

from __future__ import annotations

import numpy as np

from opc.mpl.msnc3 import resolve_one_reticulation
from opc.simulate import networks as N

__all__ = [
    "delete_reticulation",
    "relocate_hybrid",
    "underlying_tree",
    "add_reticulation",
    "embed_parameters",
    "neighbours",
]


def delete_reticulation(net: N.Network, index: int) -> N.Network:
    """Drop one reticulation by keeping the hybrid's *tree* parent -- the horizontal edge goes."""
    out = resolve_one_reticulation(net, index, take_donor=False)
    N.recompute_consumed(out)
    return out


def underlying_tree(net: N.Network) -> N.Network:
    return N.displayed_tree(net, take_donor=False)


def add_reticulation(net: N.Network, rng: np.random.Generator, *,
                     cycle_size_range=(3, None), vertex_disjoint: bool = True,
                     max_tries: int = 400) -> N.Network | None:
    """One more reticulation on this topology.  ``None`` if no legal placement was found."""
    out = net.copy()
    N.recompute_consumed(out)
    ok = N._add_reticulation(  # noqa: SLF001
        out, rng, max_tries=max_tries,
        vertex_disjoint=vertex_disjoint, cycle_size_range=cycle_size_range,
    )
    return out if ok else None


def relocate_hybrid(net: N.Network, index: int, rng: np.random.Generator, *,
                    cycle_size_range=(3, None), vertex_disjoint: bool = True,
                    max_tries: int = 400) -> N.Network | None:
    """Same underlying tree, same ``k``, **one reticulation redrawn**.

    **A documented deviation from the plan**, which asks for "the true topology with one
    reticulation head relocated to a random compatible edge" -- the donor held fixed and only the
    head moved.  This deletes the reticulation and draws a fresh one by the generator's own rule,
    so the donor moves too.  Two consequences, and they point opposite ways:

    * the alternative is a network the grid could have produced, satisfying every constraint a
      generated one does, which a hand-edited head-move would not automatically be;
    * it is a **weaker** near miss than the plan intends -- further from the truth -- so E0b's
      separation against it is easier than the plan's version, and the number should be read as an
      upper bound on how well the criterion localises a reticulation.

    Tightening this to a true head-move means placing with the donor edge fixed, which the
    generator's pair-based rule does not currently express.  Recorded in ``HANDOFF.md``.

    The redraw can land on the original placement, so it is compared on the *cluster* of the new
    hybrid.  ``None`` if no distinct legal placement was found.
    """
    original = frozenset(N.hardwired_clusters(net)[net.retics[index].hybrid])
    stripped = delete_reticulation(net, index)
    for _ in range(8):
        out = add_reticulation(
            stripped, rng, cycle_size_range=cycle_size_range,
            vertex_disjoint=vertex_disjoint, max_tries=max_tries,
        )
        if out is None:
            return None
        new = frozenset(N.hardwired_clusters(out)[out.retics[-1].hybrid])
        if new != original:
            return out
    return None


def embed_parameters(small_params, big_model, bigger: N.Network):
    """The (k+1) parameter vector that reproduces a fitted (k) model exactly, at ``gamma_new = 0``.

    E2a's whole argument is that ``sup MPL`` cannot fall when a reticulation is added, because the
    smaller model is a *point* of the larger parameter space.  Writing that point down is not
    optional: handing the optimiser the larger network's own generating lengths instead measures
    how far the optimiser can climb from an arbitrary start, and the two differ by thousands of log
    units -- measured, a median deficit of 3.9e3, which is more than enough to manufacture
    violations that are not there.

    :func:`add_reticulation` splits exactly two tree edges, at the donor and at the hybrid, and
    leaves every other node id alone.  So the mapping is: copy every shared edge's fitted length,
    and divide each split edge's fitted length between its two halves in the proportion the
    generating lengths had.  At ``gamma_new = 0`` the hybrid takes its tree parent, so each pair of
    halves sits in series and only their sum matters -- which makes the proportional split exact
    rather than merely reasonable.
    """
    ret = bigger.retics[-1]
    donor, hybrid = ret.donor, ret.hybrid
    (c1,) = bigger.tree_children[donor]
    (c2,) = bigger.tree_children[hybrid]
    p1, p2 = bigger.tree_parent[donor], bigger.tree_parent[hybrid]

    fitted = dict(zip(small_params.edges, small_params.lengths, strict=True))
    out = big_model.params
    lengths = np.array(out.lengths, dtype=float)
    for i, (child, parent) in enumerate(out.edges):
        if (child, parent) in fitted:
            lengths[i] = fitted[(child, parent)]
    for split_node, child, parent in ((donor, c1, p1), (hybrid, c2, p2)):
        total = fitted.get((child, parent))
        if total is None:
            continue  # the split edge was pendant in the smaller model, so it has no parameter
        upper, lower = (split_node, parent), (child, split_node)
        iu, il = out.edges.index(upper), out.edges.index(lower)
        gen = out.lengths[iu] + out.lengths[il]
        share = out.lengths[iu] / gen if gen > 0 else 0.5
        lengths[iu], lengths[il] = total * share, total * (1.0 - share)

    gammas = np.array(out.gammas, dtype=float)
    fitted_g = dict(zip(small_params.hybrids, small_params.gammas, strict=True))
    for i, h in enumerate(out.hybrids):
        gammas[i] = 0.0 if h == hybrid else fitted_g.get(h, gammas[i])
    return out.with_vector(np.concatenate([lengths, gammas]))


def neighbours(net: N.Network, rng: np.random.Generator, **kwargs) -> dict[str, N.Network]:
    """E0b's alternative set: ``{deleted, relocated, tree}``, whichever are available.

    At ``k = 0`` only the tree exists, and it *is* the truth, so E0b has nothing to separate --
    those instances are reported but excluded from the threshold, which is why the plan calls
    ``k = 0`` the Than-Nakhleh sanity cell rather than a test of the criterion.
    """
    out: dict[str, N.Network] = {}
    if net.n_retics:
        which = int(rng.integers(net.n_retics))
        out["deleted"] = delete_reticulation(net, which)
        moved = relocate_hybrid(net, which, rng, **kwargs)
        if moved is not None:
            out["relocated"] = moved
        out["tree"] = underlying_tree(net)
    return out
