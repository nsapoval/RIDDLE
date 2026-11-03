#!/usr/bin/env python3
"""fungi: which file tree a run reads and writes -- the landed run, or a variant of it.

Standard library only, so it imports under both ``gnn_cpu_v1/.venv`` and phylo-env.  Every ``fz_*``
script takes its ``DATA`` / ``ANALYSIS`` from here, and nothing else decides them.

* ``FZ_VARIANT`` unset -- the landed run: ``fungi/data``, ``fungi/analysis``.  Nothing a landed script
  writes changes (``tag()`` is empty, ``banner()`` prints nothing).
* ``FZ_VARIANT=collapsed`` -- ``ROOT = fungi/collapsed``: ``DATA = ROOT/data`` (the gene trees with their
  zero-length edges contracted, by ``fz_collapse.py``), ``ANALYSIS = ROOT/analysis``.  ``RAW_DATA`` and
  ``LANDED`` still name the landed inputs and outputs, for the scripts that read them on purpose.
* ``FZ_ROOT=<dir>`` (with ``FZ_VARIANT``) moves the variant root, for plumbing smokes in a scratch dir.
* ``FZ_BETA=train`` (with or without ``FZ_VARIANT``) -- the same run under another z+tc beta:
  ``ANALYSIS = ROOT/analysis_trainfit``, beside the ``ROOT/analysis`` it derives from (``PARENT``), and the
  only calibration pool is ``train`` (``nni_ensemble/runs_trainfit/calib/train__z6.json``).  Stage B6
  onward is recomputed there; what the beta cannot change is linked or copied from ``PARENT`` by
  ``fz_beta.py``, which writes the stamp ``ANALYSIS/BETA.json`` every other script requires.

A variant refuses to start without ``ROOT/VARIANT.json``, the stamp ``fz_collapse.py`` writes (it is
the one script exempt), and ``verify_inputs()`` re-hashes the collapsed tree files against that stamp.

    python fungi/scripts/fz_paths.py data|analysis|root     # one path, for zsh
    python fungi/scripts/fz_paths.py --manifest OUT          # sha256 of every file under fungi/analysis
    python fungi/scripts/fz_paths.py --verify-landed OUT     # ... and later, that nothing changed
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FUNGI = HERE.parent
REPO = FUNGI.parent
RAW_DATA = FUNGI / "data"
LANDED = FUNGI / "analysis"
VARIANTS = ("collapsed",)
#: ``FZ_BETA`` -> the analysis tree's suffix.  Unset: the published betas, pools sim (headline) and netcs.
BETAS = {"train": "trainfit"}

VARIANT = os.environ.get("FZ_VARIANT", "").strip()
if VARIANT and VARIANT not in VARIANTS:
    raise SystemExit(f"FZ_VARIANT={VARIANT!r}: expected one of {VARIANTS} or unset")
if os.environ.get("FZ_ROOT") and not VARIANT:
    raise SystemExit("FZ_ROOT is set without FZ_VARIANT: a scratch root is only for a variant")
BETA = os.environ.get("FZ_BETA", "").strip()
if BETA and BETA not in BETAS:
    raise SystemExit(f"FZ_BETA={BETA!r}: expected one of {tuple(BETAS)} or unset")
ROOT = (Path(os.environ["FZ_ROOT"]).resolve() if os.environ.get("FZ_ROOT") else FUNGI / VARIANT) \
    if VARIANT else FUNGI
DATA = ROOT / "data"
PARENT = ROOT / "analysis"
ANALYSIS = ROOT / f"analysis_{BETAS[BETA]}" if BETA else PARENT
STAMP = ROOT / "VARIANT.json"
BETA_STAMP = ANALYSIS / "BETA.json"
TREES = "genetrees.rooted.tre"
#: The landed run under this beta: what a variant is compared against (``fz_compare``).
LANDED_RUN = FUNGI / f"analysis_{BETAS[BETA]}" if BETA else LANDED
#: The calibration pools Stage B6 writes, the headline first, and so the MAP arms ``ztc-<pool>``.
MAP_POOLS = (BETA,) if BETA else ("sim", "netcs")
HEADLINE_POOL = MAP_POOLS[0]
POOL_LABEL = {"sim": "sim", "netcs": "netcs", "train": "train-fit"}

_EXEMPT = ("fz_collapse.py", "fz_paths.py")
_ME = Path(sys.argv[0] if sys.argv else "").name
if VARIANT and not STAMP.exists() and _ME not in _EXEMPT:
    raise SystemExit(f"FZ_VARIANT={VARIANT}: {STAMP} is missing -- run fz_collapse.py first")
if BETA and not BETA_STAMP.exists() and _ME not in ("fz_beta.py", "fz_paths.py"):
    raise SystemExit(f"FZ_BETA={BETA}: {BETA_STAMP} is missing -- run fz_beta.py --stage first")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stamp() -> dict:
    return json.loads(STAMP.read_text()) if VARIANT else {}


def beta_tag() -> dict:
    """Fields a beta run adds to the JSON it writes; empty without ``FZ_BETA``."""
    if not BETA:
        return {}
    return {"beta": BETA, "beta_analysis": str(ANALYSIS), "beta_stamp_sha256": sha256(BETA_STAMP)}


def tag() -> dict:
    """Fields a variant adds to the JSON it writes; empty for the landed run, so its outputs keep their bytes."""
    if not VARIANT:
        return beta_tag()
    return {"variant": VARIANT, "variant_root": str(ROOT), "variant_stamp_sha256": sha256(STAMP),
            **beta_tag()}


def banner(script: str = "") -> None:
    name = script or Path(sys.argv[0]).name
    if BETA:
        print(f"[fz_paths] {name}: variant {VARIANT or 'landed'}, beta {BETA}, analysis {ANALYSIS}",
              file=sys.stderr, flush=True)
    elif VARIANT:
        print(f"[fz_paths] {name}: variant {VARIANT}, root {ROOT}", file=sys.stderr, flush=True)


def trees_file(scheme: str) -> Path:
    return DATA / scheme / TREES


def verify_inputs(schemes=("short", "medium")) -> None:
    """A variant's tree files are the ones its stamp was written for (no-op for the landed run)."""
    if not VARIANT:
        return
    st = stamp()
    for s in schemes:
        want = st["schemes"][s]["dst_sha256"]
        got = sha256(trees_file(s))
        if got != want:
            raise SystemExit(f"{trees_file(s)}: sha256 {got[:12]} is not the stamp's {want[:12]}")
        if sha256(RAW_DATA / s / TREES) != st["schemes"][s]["src_sha256"]:
            raise SystemExit(f"{RAW_DATA / s / TREES}: the raw input changed since the stamp")


def manifest(root: Path = LANDED) -> dict[str, str]:
    return {str(p.relative_to(FUNGI)): sha256(p) for p in sorted(Path(root).rglob("*")) if p.is_file()}


def main() -> int:
    a = sys.argv[1:]
    if a and a[0] in ("data", "analysis", "root"):
        print({"data": DATA, "analysis": ANALYSIS, "root": ROOT}[a[0]])
        return 0
    if len(a) == 2 and a[0] == "--manifest":
        m = manifest()
        Path(a[1]).write_text(json.dumps(m, indent=0) + "\n")
        print(f"[fz_paths] {len(m)} files under {LANDED} -> {a[1]}")
        return 0
    if len(a) == 2 and a[0] == "--verify-landed":
        want = json.loads(Path(a[1]).read_text())
        got = manifest()
        changed = sorted(k for k in want if got.get(k) != want[k])
        added = sorted(set(got) - set(want))
        for k in changed:
            print(f"[fz_paths] CHANGED or MISSING {k}", file=sys.stderr)
        for k in added:
            print(f"[fz_paths] ADDED {k}", file=sys.stderr)
        print(f"[fz_paths] landed: {len(want)} files, {len(changed)} changed/missing, {len(added)} added")
        return 1 if changed or added else 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
