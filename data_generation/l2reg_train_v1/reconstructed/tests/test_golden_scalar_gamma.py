"""Golden fingerprints of everything that reads the scalar ``Network.gamma`` (A62).

Per-reticulation inheritance probabilities (``Network.gammas``) were added for the level-2
benchmark.  With them unset, every existing output must stay byte-identical -- the published
artifacts are append-only.  The hashes in ``data/golden_scalar_gamma.json`` were written *before*
that change, from the code at ``ab6824a``, by::

    python tests/test_golden_scalar_gamma.py --write

and this test re-derives them.  The networks come from ``build_base`` seeds (the ones
``test_networks.py`` uses), not from ``results/``, so it needs no cache.  Every reader of the
scalar is covered: the eNewick writer, the metadata sidecar, the MSNC oracle, the optimiser's
parameter vector, and both triple-probability paths in ``msnc3``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from opc.mpl import likelihood as L
from opc.mpl import msnc3
from opc.simulate import msnc_oracle as oracle
from opc.simulate import networks as N

GOLDEN = Path(__file__).parent / "data" / "golden_scalar_gamma.json"
SHAPES = [(10, 2, 110), (25, 3, 125), (50, 5, 150)]
TRIPLES = [(0, 1, 2), (1, 4, 7), (2, 5, 9)]
# 0.5 is the tie that sends the #H copy to the donor edge; 0.7 makes the donor edge major.
GAMMAS = (0.1, 0.5, 0.7)


def _h(x) -> str:
    return hashlib.sha256(x if isinstance(x, bytes) else x.encode()).hexdigest()


def fingerprints() -> dict:
    out = {}
    for n, r_max, seed in SHAPES:
        base = N.build_base(n, r_max, seed=seed)
        for r in sorted({0, 1, r_max}):
            for cu in (0.25, 1.0):
                for g in GAMMAS:
                    net = N.with_gamma(N.scaled(N.with_reticulations(base, r), cu), g)
                    out[f"n{n}_r{r}_cu{cu}_g{g}"] = {
                        "enewick": _h(N.to_enewick(net)),
                        "metadata": _h(json.dumps(N.metadata(net), sort_keys=True)),
                        "level": N.level(net),
                        "oracle": _h("\n".join(oracle.simulate_gene_trees(net, 100, seed=3))),
                        "params": _h(L.Parameters.of(net).vector.tobytes()),
                        "msnc3": _h(b"".join(msnc3.triple_probabilities(net, t).tobytes()
                                             for t in TRIPLES)),
                        "mixture": _h(b"".join(msnc3.triple_probabilities_mixture(net, t).tobytes()
                                               for t in TRIPLES)),
                    }
    return out


def test_scalar_gamma_outputs_are_unchanged():
    want = json.loads(GOLDEN.read_text())
    got = fingerprints()
    # Not vacuous: the fixture reaches reticulate networks and the gamma = 0.5 tie.
    assert any(k.split("_")[1] != "r0" and k.endswith("_g0.5") for k in got)
    assert set(got) == set(want)
    changed = {k: [f for f in got[k] if got[k][f] != want[k][f]] for k in got}
    changed = {k: v for k, v in changed.items() if v}
    assert not changed, f"scalar-gamma outputs changed: {changed}"


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit("usage: python tests/test_golden_scalar_gamma.py --write")
    GOLDEN.parent.mkdir(exist_ok=True)
    GOLDEN.write_text(json.dumps(fingerprints(), indent=1, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN}")
