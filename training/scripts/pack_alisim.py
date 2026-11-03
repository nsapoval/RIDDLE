#!/usr/bin/env python3
"""Build the AliSim hand-off package: per-network gene tree Newicks, ready to move to a CPU box.

The package is self-contained -- it carries the trees, the true networks they came from, a
manifest, a verbatim copy of the three ``opc.simulate`` modules the arm needs, and the driver that
runs it.  On the far side it needs Python 3.9+ and an IQ-TREE **>= 2.2.0** binary (AliSim
was added in 2.2.0; an older one fails every locus with `Invalid "--alisim" option.`),
and nothing else.

**True gene trees are what ships, in coalescent units.**  Not the grafted, rescaled,
AliSim-ready ones, even though those are the same size gzipped -- the graft is deterministic
(:func:`opc.simulate.outgroup.graft_file` has no RNG in it), so shipping both would be two copies
of one object with no rule saying which is authoritative.  ``run_estimate.py --stage graft``
materialises the AliSim-ready form in one command, and ``--stage run`` does it internally anyway.

What is *not* free to move to the far side is the outgroup's divergence time ``T_out``: it is
``(1 + alpha) * p99(root heights)`` **over the whole cell**, so a cell must travel whole or its
outgroup moves.  Every cell here is all 1,000 of its loci for that reason.

    .venv/bin/python scripts/pack_alisim.py --out handoff/alisim_gt_v1 --jobs 12

Selection follows the request it was built for -- the x-grids in full and the b-grids' 20-cell
test splits -- but the package *carries* the b-grid train and validation splits too, marked
``requested=0`` in the manifest.  They cost 370 MB to include and a second round trip to leave
out; ``run_estimate.py --all`` is what reaches them.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opc import grid                       # noqa: E402
from opc.simulate import seqsim            # noqa: E402

#: phase -> (split_train, split_val).  The cuts every driver on that grid was run with; anything
#: else silently re-splits (``opc.rl.train.split_by_rep`` documents the hazard and now raises).
SPLITS = {"x25": (700, 800), "x50": (700, 800),
          "b25": (70, 80), "b50": (70, 80), "b100": (70, 80), "b250": (70, 80),
          # v2: the second extended draw, and the n = 75 rung.
          "y25": (700, 800), "y50": (700, 800), "x75": (700, 800), "b75": (70, 80)}

#: phase -> which of its cells this package asks the far side to **run**.  ``"all"``, ``"test"``,
#: or explicit ``(lo, hi)`` rep windows.
#:
#: **A table, because the rule is not a function of the phase name.**  This was the inline
#: ``int(phase.startswith("x") or split == "test")``, and under it ``x75`` -- an x-phase -- would
#: have requested all 900 of its cells where only 400 have instances: **430 CPU-h against 199**,
#: with nothing failing and the surplus only visible as a number in a README table.  The y-phases
#: would have gone the other way and requested their test splits alone.
#:
#: Every cell of a listed phase still *ships*; ``requested`` only sets what ``run_estimate.py``
#: takes by default, and ``--all`` reaches the rest.  An unlisted phase **raises**, for
#: ``paths.tier_of``'s and ``arms.arm_dirs``' reason: the alternative is a package that silently
#: asks for the wrong work.
REQUESTED: dict[str, object] = {
    "x25": "all", "x50": "all",
    "b25": "test", "b50": "test", "b100": "test", "b250": "test",
    "y25": "all", "y50": "all",
    "x75": ((0, 300), (700, 800)),   # the 300 train + 100 val cells that get instances
    "b75": "test",
}


def is_requested(phase: str, rep: int, split: str) -> bool:
    """Does the package ask the far side to run this cell?  Raises on an unlisted phase."""
    try:
        rule = REQUESTED[phase]
    except KeyError:
        raise SystemExit(
            f"pack_alisim: no REQUESTED rule for phase {phase!r}.  Add one rather than letting "
            f"the packer guess from the name -- see the table's docstring."
        ) from None
    if rule == "all":
        return True
    if rule == "test":
        return split == "test"
    return any(lo <= rep < hi for lo, hi in rule)

#: The modules ``run_estimate.py`` imports, copied byte for byte.  All are pure stdlib;
#: ``seqsim.plan_jobs`` is the only thing in them that touches ``opc.grid`` and the runner does
#: not call it.
#:
#: **``arms.py`` is here because the list went stale and the package shipped broken.**  W14 added
#: ``from opc import arms as _arms`` to ``seqsim.py`` at *module* level, long after `alisim_gt_v1`
#: was built -- and nothing connected the two, so `alisim_gt_v2` was packed, hashed and declared
#: complete while `run_estimate.py` could not get past its own import line on the far side.  The
#: comment above was part of the failure: it asserted the closure ("all three are pure stdlib")
#: rather than checking it.  :func:`check_vendored` checks it now.
VENDOR = ("arms.py", "simulate/outgroup.py", "simulate/phylip.py", "simulate/seqsim.py")


def check_vendored(pylib_root: Path) -> None:
    """Import the vendored tree the way the far side will, in a **subprocess with a clean path**.

    Not a static scan of ``from opc ...`` lines: a scan has to model conditional and function-level
    imports, and the thing that actually matters is whether ``run_estimate.py``'s first real
    statement works on a box that has this directory and nothing else.  So it is executed.

    **``-S`` and the ``__file__`` assertion are both load-bearing, and the first draft had
    neither.**  This repo installs ``opc`` *editable* (``__editable__.opc-0.1.0.pth`` -> ``src/``),
    and ``-I`` does not disable the virtualenv's own site-packages -- only the *user* one.  So a
    probe that merely imports ``opc.simulate.seqsim`` resolves it to ``src/`` and passes no matter
    what is in ``pylib``: it is vacuous, and it passed on a package whose ``arms.py`` had been
    deleted.  ``-S`` drops site-packages so the editable install is gone (and, as a bonus, a
    vendored module that quietly grew a third-party import now fails here rather than there), and
    the location check says the module under test is the one that shipped.
    """
    import subprocess
    root = Path(pylib_root).resolve()
    probe = ("import sys; sys.path.insert(0, %r);"
             "from opc.simulate import seqsim, outgroup, phylip;"
             "from opc import arms;"
             "seqsim.SeqModel();"
             "assert seqsim.__file__.startswith(%r), 'imported ' + seqsim.__file__;"
             "assert arms.__file__.startswith(%r), 'imported ' + arms.__file__;"
             "print('ok')" % (str(root), str(root), str(root)))
    r = subprocess.run([sys.executable, "-I", "-S", "-c", probe],
                       capture_output=True, text=True, cwd=str(root.parent))
    if r.returncode != 0 or "ok" not in r.stdout:
        raise SystemExit(
            "pack_alisim: the vendored package does not import on its own.\n"
            f"{r.stderr.strip()}\n"
            "Add the missing module to VENDOR -- the far side has this directory and nothing else."
        )


def _digest_and_count(args) -> tuple[str, str, int, int]:
    """``(key, sha256, bytes, n_trees)`` -- the count is what makes the manifest a check."""
    key, src = args
    h = hashlib.sha256()
    with open(src, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    with gzip.open(src, "rt") as fh:
        n = sum(1 for ln in fh if ln.strip())
    return key, h.hexdigest(), src.stat().st_size, n


def _copy(args) -> int:
    src, dst = args
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return dst.stat().st_size


def collect(results: Path, phases) -> list[dict]:
    rows = []
    for phase in phases:
        cells = grid.enumerate_cells(**grid.PHASES[phase])
        train, val = SPLITS[phase]
        root = results / phase
        for c in cells:
            gt = root / "genetrees_julia" / f"{c.key}.nwk.gz"
            enw = root / "enewick" / f"{c.key}.enewick"
            for p in (gt, enw):
                if not p.exists():
                    raise SystemExit(f"{phase}: missing {p}")
            split = "train" if c.rep < train else "val" if c.rep < val else "test"
            rows.append({
                "phase": phase, "cell_key": c.key, "base_key": c.base_key,
                "n_taxa": c.n, "k": c.r, "cu": c.cu,
                "gamma_range": f"{c.gamma_range[0]}-{c.gamma_range[1]}" if c.gamma_range else "",
                "stratum": c.stratum, "rep": c.rep, "split": split,
                # What the request asks the far side to run.  See REQUESTED -- it is a table
                # rather than a predicate on the phase name, and it raises on an unlisted phase.
                "requested": int(is_requested(phase, c.rep, split)),
                "n_loci": c.m_max, "master_seed": c.master_seed,
                "genetrees": f"trees/{phase}/{c.key}.nwk.gz",
                "network": f"networks/{phase}/{c.key}.enewick",
                "_gt_src": gt, "_enw_src": enw,
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("handoff/alisim_gt_v1"))
    ap.add_argument("--phases", default="x25,x50,b25,b50,b100,b250")
    ap.add_argument("--bp", type=int, default=500, help="the bp the README's cost table is for")
    ap.add_argument("--jobs", "-j", type=int, default=12)
    ap.add_argument("--tar", action="store_true", help="also write <out>.tar beside the directory")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.jobs >= 100:
        print("refusing --jobs >= 100 (RLIMIT_NPROC pool deadlock)")
        return 2

    repo = Path(__file__).resolve().parents[1]
    phases = args.phases.split(",")
    t0 = time.perf_counter()
    rows = collect(args.results, phases)
    print(f"{len(rows)} cells over {len(phases)} phases")

    model = seqsim.SeqModel()
    for ph in phases:
        sel = [r for r in rows if r["phase"] == ph and r["requested"]]
        s = sum(r["n_loci"] * seqsim.seconds_per_locus(r["n_taxa"], args.bp) for r in sel)
        print(f"  {ph:>5}: {len(sel):>4} requested of {sum(r['phase'] == ph for r in rows):>4}"
              f"   {s * seqsim.CONCURRENCY_PENALTY / 3600:8.1f} CPU-h at {args.bp} bp")
    if args.dry_run:
        return 0

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    # --- checksum + tree count, which is also the only real check that a cell is intact ---------
    print("hashing and counting trees ...", flush=True)
    with Pool(args.jobs) as pool:
        for key, sha, nbytes, n in pool.imap_unordered(
                _digest_and_count, [(r["cell_key"] + "|" + r["phase"], r["_gt_src"]) for r in rows],
                chunksize=8):
            k, ph = key.split("|")
            r = next(x for x in rows if x["cell_key"] == k and x["phase"] == ph)
            r["sha256"] = sha
            r["bytes"] = nbytes
            if n != r["n_loci"]:
                raise SystemExit(f"{ph}/{k}: {n} trees on disk, manifest says {r['n_loci']}")
    print(f"  all {len(rows)} cells carry their full {rows[0]['n_loci']} loci")

    # --- payload -------------------------------------------------------------------------------
    print("copying trees and networks ...", flush=True)
    pairs = [(r["_gt_src"], out / r["genetrees"]) for r in rows]
    pairs += [(r["_enw_src"], out / r["network"]) for r in rows]
    with Pool(args.jobs) as pool:
        total = sum(pool.imap_unordered(_copy, pairs, chunksize=8))
    print(f"  {total / 1e9:.2f} GB in {len(pairs)} files")

    # --- the vendored library, byte for byte ----------------------------------------------------
    pylib = out / "pylib" / "opc"
    (pylib / "simulate").mkdir(parents=True, exist_ok=True)
    (pylib / "__init__.py").write_text("")
    (pylib / "simulate" / "__init__.py").write_text("")
    for rel in VENDOR:
        shutil.copyfile(repo / "src" / "opc" / rel, pylib / rel)
    shutil.copyfile(repo / "scripts" / "handoff" / "run_estimate.py", out / "run_estimate.py")
    (out / "run_estimate.py").chmod(0o755)
    # Before SHA256SUMS, so a package that cannot import is never hashed and declared complete.
    check_vendored(out / "pylib")
    print("  vendored package imports standalone")

    # --- manifest -------------------------------------------------------------------------------
    cols = ["phase", "cell_key", "base_key", "n_taxa", "k", "cu", "gamma_range", "stratum",
            "rep", "split", "requested", "n_loci", "master_seed", "genetrees", "network",
            "bytes", "sha256"]
    with open(out / "MANIFEST.tsv", "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in sorted(rows, key=lambda r: (r["n_taxa"], r["phase"], r["rep"])):
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")

    # A handful of seeds computed *here*, which run_estimate.py re-derives and refuses to start on
    # a mismatch.  The repo's cross-validation convention, applied to the one thing in the package
    # that is recomputed rather than shipped.
    seed_check = []
    for ph in phases:
        r = next(x for x in rows if x["phase"] == ph)
        for bp in (500, 1000):
            a = grid.chain_seed(r["master_seed"], "alisim", r["cell_key"], bp) % (2 ** 31)
            i = grid.chain_seed(r["master_seed"], "iqtree", r["cell_key"], bp) % (2 ** 31)
            seed_check.append({"phase": ph, "cell_key": r["cell_key"], "bp": bp,
                               "master_seed": r["master_seed"],
                               "seed_alisim": a, "seed_iqtree": i})

    meta = {
        "package": out.name,
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "built_from": {"repo": str(repo), "commit": _git(repo, "rev-parse", "HEAD"),
                       "describe": _git(repo, "log", "-1", "--pretty=%s")},
        "phases": phases,
        "splits": {p: {"train": SPLITS[p][0], "val": SPLITS[p][1]} for p in phases},
        "n_cells": len(rows),
        "n_cells_requested": sum(r["requested"] for r in rows),
        "n_loci_per_cell": rows[0]["n_loci"],
        "default_bp": args.bp,
        "model": model.as_dict(),
        "vendored": list(VENDOR),
        "seed_check": seed_check,
        "cost_cpu_hours": {
            ph: round(sum(r["n_loci"] * seqsim.seconds_per_locus(r["n_taxa"], args.bp)
                          for r in rows if r["phase"] == ph and r["requested"])
                      * seqsim.CONCURRENCY_PENALTY / 3600, 1) for ph in phases},
    }
    (out / "manifest.json").write_text(json.dumps(meta, indent=1) + "\n")
    (out / "model.json").write_text(json.dumps(model.as_dict(), indent=1) + "\n")

    write_readme(out, rows, meta, args)

    # --- checksums over everything but the checksum file ---------------------------------------
    print("writing SHA256SUMS ...", flush=True)
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    with Pool(args.jobs) as pool:
        digests = pool.map(_sha, files, chunksize=8)
    with open(out / "SHA256SUMS", "w") as fh:
        for p, d in zip(files, digests):
            fh.write(f"{d}  {p.relative_to(out)}\n")

    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"\npackage: {out}  ({size / 1e9:.2f} GB, {len(files) + 1} files, "
          f"{time.perf_counter() - t0:.0f} s)")
    if args.tar:
        tar = out.with_suffix(out.suffix + ".tar")
        print(f"tarring -> {tar} (no compression: the payload is already gzip)")
        shutil.make_archive(str(out), "tar", root_dir=str(out.parent), base_dir=out.name)
    print(f"\nmove it with:\n  rsync -a --info=progress2 {out}/ USER@BOX:/path/{out.name}/")
    print(f"then on the box:\n  cd /path/{out.name} && python3 run_estimate.py "
          f"--iqtree /path/to/iqtree2 --jobs 32 --dry-run")
    return 0


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git(repo: Path, *a) -> str:
    import subprocess
    try:
        return subprocess.run(["git", "-C", str(repo), *a], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is nice to have, not a reason to fail
        return "unknown"


def write_readme(out: Path, rows, meta, args) -> None:
    tmpl = (Path(__file__).resolve().parent / "handoff" / "README.md").read_text()
    lines = ["| phase | n | k | cells | requested | loci (requested) | CPU-h @ %d bp |"
             % args.bp,
             "|---|---|---|---|---|---|---|"]
    for ph in meta["phases"]:
        sub = [r for r in rows if r["phase"] == ph]
        req = [r for r in sub if r["requested"]]
        ks = sorted({r["k"] for r in sub})
        lines.append(f"| `{ph}` | {sub[0]['n_taxa']} | {','.join(map(str, ks))} | {len(sub)} | "
                     f"{len(req)} | {sum(r['n_loci'] for r in req):,} | "
                     f"{meta['cost_cpu_hours'][ph]:.1f} |")
    tot = sum(meta["cost_cpu_hours"].values())
    lines.append(f"| **total** | | | **{len(rows)}** | **{meta['n_cells_requested']}** | "
                 f"**{sum(r['n_loci'] for r in rows if r['requested']):,}** | **{tot:.1f}** |")
    body = tmpl.replace("<!-- INVENTORY -->", "\n".join(lines))
    body = body.replace("<!-- BUILT -->", meta["built"]).replace("<!-- COMMIT -->",
                                                                 meta["built_from"]["commit"][:12])
    m = meta["model"]
    body = body.replace("<!-- MODEL -->",
                        f"`{m['sim_model']}` + indels `{m['indel_ins']:g},{m['indel_del']:g}` "
                        f"size `{m['indel_size']}`, branch scale `{m['branch_scale']:g}` "
                        f"(theta = {m['theta']:g}), inferred under `{m['inference_model']}`")
    (out / "README.md").write_text(body)


if __name__ == "__main__":
    raise SystemExit(main())
