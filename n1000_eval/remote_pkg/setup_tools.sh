#!/usr/bin/env bash
# Build the two resolvers exactly as pinned on the Mac (MANIFEST.json "tools"):
#   TREE-QMC 4.1.5 @ e577f8c + treeqmc-e577f8c.patch   (NetCS = tree-qmc --network)
#   BROOQS @ 17b25b2 in a Python 3.12 venv
# into ./tools/.  ~15-20 min, mostly R packages.  Skip this if the box already has both at those
# pins: export TREE_QMC=/path/to/tree-qmc BROOQS=/path/to/brooqs (plus R_HOME / LD_LIBRARY_PATH
# for that tree-qmc's R) and go straight to `$PY run_remote.py --check` (any Python >= 3.9).
#
# The recipe is tob_qmc_v1/setup.sh's, which built TREE-QMC on this box before; README.md there
# lists the four things that blocked that build.  Everything, caches included, stays under this
# directory: /home on the shared box is full.
set -euo pipefail
cd "$(dirname "$0")"
HERE="$PWD"
TOOLS="$HERE/tools"
PREFIX="$TOOLS/env"
mkdir -p "$TOOLS"
export CONDA_PKGS_DIRS="$TOOLS/conda-pkgs" PIP_CACHE_DIR="$TOOLS/pip-cache"
export R_LIBS_USER="$PREFIX/lib/R/library" XDG_CACHE_HOME="$TOOLS/cache"

command -v conda >/dev/null || { echo "conda not found (see the header of this script)"; exit 1; }

# ---- One env, one solve (2026-10-08).  Earlier versions of this script created the env and then
# patched it in place -- GCC 13, then a sysroot, then GCC 15 + sysroot -- and the last of those
# layered pins ended in a conda conflict dump.  Now the env is created ONCE with everything pinned
# up front, and afterwards only *checked*: nothing below installs into it.
#   * sysroot_linux-64: the glibc headers.  conda can otherwise resolve 2.12 (CentOS 6), which
#     predates timespec_get (the `'timespec_get' has not been declared` failure).  2.28 when the
#     host glibc has it, else 2.17: binaries built against a sysroot need at least that glibc.
#   * compilers: unpinned, so the solver picks a gcc/gxx consistent with libstdcxx -- which the
#     prebuilt r-rinside links against (the `__cxa_call_terminate@CXXABI_1.3.15` failure).
# An env from an earlier run of this script carries those layered pins: the checks below detect
# that and tell you to remove it.  TQMC_ENV=/path/to/a/working/env reuses an existing R env
# instead (it needs Rscript + Rcpp + RInside + MSCquartets and a conda compiler toolchain).
HOST_GLIBC=$(ldd --version 2>&1 | head -1 | grep -oE '[0-9]+\.[0-9]+' | tail -1 || true)
HOST_GLIBC=${HOST_GLIBC:-0.0}
vge () { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }   # $1 >= $2
if vge "$HOST_GLIBC" 2.28; then WANT=2.28
elif vge "$HOST_GLIBC" 2.17; then WANT=2.17
else echo "[setup] host glibc '$HOST_GLIBC' < 2.17 (or unreadable: '$(ldd --version 2>&1 | head -1)')"
     echo "        no conda-forge sysroot can serve this host; stop and report this line"; exit 1
fi
if [ -n "${TQMC_ENV:-}" ]; then
  PREFIX="$TQMC_ENV"
  echo "[setup] reusing env $PREFIX (TQMC_ENV)"
elif [ ! -x "$PREFIX/bin/Rscript" ]; then
  echo "[setup] creating $PREFIX in one solve: R 4.4, toolchain, sysroot $WANT, Python 3.12"
  conda create -y -p "$PREFIX" -c conda-forge --override-channels \
      r-base=4.4 r-rcpp r-rinside r-ape r-igraph r-phangorn r-quadprog r-fastmatch r-rdpack \
      compilers "sysroot_linux-64=$WANT" make cmake git python=3.12
fi
export PATH="$PREFIX/bin:$PATH"

# ---- Checks only.  Any failure: stop, print what to send back, install nothing.
pkg_ver () { conda list -p "$PREFIX" "$1" 2>/dev/null | awk -v p="$1" '$1==p{print $2}'; }
major () { echo "${1%%.*}"; }
SYSROOT=$(pkg_ver sysroot_linux-64); RT=$(pkg_ver libstdcxx); RT=${RT:-$(pkg_ver libstdcxx-ng)}
GXX=$(pkg_ver gxx_linux-64)
echo "[setup] host glibc $HOST_GLIBC | env: sysroot_linux-64 $SYSROOT, libstdcxx $RT, gxx_linux-64 $GXX"
SYSROOT_TIME_H="$PREFIX/x86_64-conda-linux-gnu/sysroot/usr/include/time.h"
fail () {
  echo "[setup] FAILED: $1"
  echo "        env packages:"; conda list -p "$PREFIX" | grep -E "^(sysroot|kernel-headers|gcc|gxx|libstdcxx|libgcc|r-base |r-rinside)" | sed 's/^/          /'
  if [ -z "${TQMC_ENV:-}" ]; then
    echo "        This env is from an earlier run of this script (layered pins) or solved badly."
    echo "        Remove ONLY it and re-run:   rm -rf $PREFIX && bash setup_tools.sh"
  fi
  exit 1
}
grep -q timespec_get "$SYSROOT_TIME_H" 2>/dev/null || fail "sysroot $SYSROOT has no timespec_get in time.h"
vge "$HOST_GLIBC" "$SYSROOT" || fail "sysroot $SYSROOT is newer than host glibc $HOST_GLIBC"
[ -n "$GXX" ] || fail "no gxx_linux-64 in the env"
# Not equality: conda-forge's libstdcxx runtime routinely runs a major ahead of gxx (2026-10-08:
# libstdcxx 16.2 with gxx 15.3), which is fine -- a newer runtime serves older-compiled code.  What
# broke before was a compiler too OLD for RInside: it needs CXXABI_1.3.15, i.e. GCC >= 14.  The
# RInside link probe below is the actual verdict; this is the fast pre-check.
[ "$(major "$GXX")" -ge 14 ] || fail "gxx $GXX < 14: cannot link the prebuilt RInside (CXXABI_1.3.15)"
[ -z "$RT" ] || [ "$(major "$RT")" -ge "$(major "$GXX")" ] || fail "libstdcxx $RT older than gxx $GXX"

echo "[setup] sysroot OK: sysroot_linux-64 $SYSROOT declares timespec_get;" \
     "$("$PREFIX/bin/x86_64-conda-linux-gnu-c++" --version | head -1)"
# A one-line C++17 compile of <ctime>: the exact thing that failed, before R spends minutes on it.
printf '#include <ctime>\nint main(){std::timespec t; return std::timespec_get(&t, TIME_UTC) ? 0 : 1;}\n' > "$TOOLS/ctime_probe.cpp"
"$PREFIX/bin/x86_64-conda-linux-gnu-c++" -std=gnu++17 "$TOOLS/ctime_probe.cpp" -o "$TOOLS/ctime_probe" \
  && "$TOOLS/ctime_probe" && echo "[setup] <ctime> probe compiles and runs" \
  || fail "<ctime> probe did not compile or run"
# The link that failed before: an executable against RInside, so ld checks RInside's own symbol
# versions (CXXABI_*) against this compiler's libstdc++.
RINSIDE_LIB="$PREFIX/lib/R/library/RInside/lib"
printf 'int main(){return 0;}\n' > "$TOOLS/rinside_probe.cpp"
"$PREFIX/bin/x86_64-conda-linux-gnu-c++" "$TOOLS/rinside_probe.cpp" -o "$TOOLS/rinside_probe" \
    -Wl,--no-as-needed -L"$RINSIDE_LIB" -lRInside -L"$PREFIX/lib/R/lib" -lR \
    -Wl,-rpath-link,"$PREFIX/lib" -Wl,-rpath-link,"$PREFIX/lib/R/lib" \
  && echo "[setup] RInside link probe OK" || fail "linking against RInside failed (see ld output above)"

Rscript --vanilla -e '
  if (!requireNamespace("MSCquartets", quietly=TRUE))
    install.packages("MSCquartets", repos="https://cloud.r-project.org", Ncpus=4)
  cat("MSCquartets", as.character(packageVersion("MSCquartets")), "\n")'

# ---- TREE-QMC at the Mac's pin, with the Mac's patch (C++17, no VLA, libstdc++'s random_shuffle
#      body inlined -- on glibc that is the same draw sequence std::random_shuffle makes)
QMC_SRC="$TOOLS/TREE-QMC"
[ -d "$QMC_SRC/.git" ] || git clone https://github.com/molloy-lab/TREE-QMC.git "$QMC_SRC"
git -C "$QMC_SRC" config core.autocrlf false        # tree_of_blobs.cpp is CRLF; the patch keeps it
git -C "$QMC_SRC" checkout -q e577f8ce468d9f7567721ee723074ff686baf703
# Idempotent, and verified: the working tree's diff must BE the shipped patch, byte for byte
# (sha256 in MANIFEST.json).  A clean tree gets the patch; anything else is reset first -- this
# also repairs a clone left unpatched by the 2026-10-07 "corrupt patch" failure.
want=$(python3 -c "import json; print(json.load(open('$HERE/MANIFEST.json'))['tools']['tree_qmc_patch_sha256'])")
have=$(git -C "$QMC_SRC" diff | sha256sum | cut -d' ' -f1)
if [ "$have" != "$want" ]; then
  git -C "$QMC_SRC" checkout -q -- .
  git -C "$QMC_SRC" apply "$HERE/treeqmc-e577f8c.patch"
  have=$(git -C "$QMC_SRC" diff | sha256sum | cut -d' ' -f1)
  [ "$have" = "$want" ] || { echo "[setup] patched TREE-QMC diff $have != manifest $want"; exit 1; }
  rm -rf "$QMC_SRC/build"                           # never reuse a build of other sources
fi
echo "[setup] TREE-QMC e577f8c, patch verified ($want)"
( cd "$TOOLS/TREE-QMC/external/MQLib" && make -j8 )
mkdir -p "$TOOLS/TREE-QMC/build"
( cd "$TOOLS/TREE-QMC/build" \
  && cmake .. -DCMAKE_BUILD_TYPE=Release -DRSCRIPT_BIN="$PREFIX/bin/Rscript" \
  && cmake --build . -j8 )

# ---- BROOQS at the Mac's pin
if [ ! -d "$TOOLS/BROOQS" ]; then
  git clone https://github.com/shayesteh99/BROOQS.git "$TOOLS/BROOQS"
  git -C "$TOOLS/BROOQS" checkout 17b25b2e06e5c8ef5d3e884360c0a6b69cd0a400
fi
[ -x "$TOOLS/.venv-brooqs/bin/brooqs" ] || {
  PY312="$PREFIX/bin/python3.12"; [ -x "$PY312" ] || PY312=$(command -v python3.12) \
    || { echo "[setup] no python3.12 (the Mac's BROOQS runs on 3.12)"; exit 1; }
  "$PY312" -m venv "$TOOLS/.venv-brooqs"
  # The Mac's versions; brooqs pulls the rest.
  "$TOOLS/.venv-brooqs/bin/pip" install numpy==2.5.3 treeswift==1.1.47 "$TOOLS/BROOQS"
}

cat > "$HERE/env.sh" <<EOF
# source this before run_remote.py: tree-qmc embeds R and must find this R at run time
export R_HOME="$PREFIX/lib/R"
export LD_LIBRARY_PATH="$PREFIX/lib/R/lib:$PREFIX/lib:\${LD_LIBRARY_PATH:-}"
export TREE_QMC="$TOOLS/TREE-QMC/build/tree-qmc"
export BROOQS="$TOOLS/.venv-brooqs/bin/brooqs"
# run_remote.py needs Python >= 3.9; the system python3 here is older (3.6 on RHEL 8)
export PY="$PREFIX/bin/python3"
EOF
. "$HERE/env.sh"
echo "[setup] tree-qmc: $("$TREE_QMC" --version 2>&1 | head -1)"
echo "[setup] brooqs:   $("$BROOQS" -h >/dev/null 2>&1 && echo ok)"
echo "[setup] done -- now:  source env.sh && \$PY run_remote.py --check   (\$PY = $PREFIX/bin/python3)"
