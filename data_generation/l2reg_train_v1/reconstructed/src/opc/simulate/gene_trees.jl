#!/usr/bin/env julia
"""
Gene trees under the network multispecies coalescent, via PhyloCoalSimulations.

    julia --project=julia src/opc/simulate/gene_trees.jl manifest.json

The manifest is a JSON array of jobs, each ``{enewick, out, n_loci, seed, n_taxa, r, cu}`` plus
optional ``level`` (default ``r > 0 ? 1 : 0``) and ``rescale``.  One
process handles many jobs so the ~15 s package load is amortised; the Python driver shards the
manifest across processes.

Everything before `simulatecoalescent` is a gate.  The ones that matter:

* **gamma.**  `readnewick` leaves `gamma = -1.0` *silently, with no warning*, when both partners
  of a hybrid lack it, and the failure only surfaces later inside `simulatecoalescent`.  Checked
  immediately after the read.
* **round trip.**  `writenewick` -> `readnewick` must preserve the hardwired-cluster row set and
  the gamma multiset.  Catches PhyloNetworks issue #44 (gamma exactly 0.5 corrupting `ismajor`)
  and any writer bug, in one cheap check.
* **time consistency and equal leaf heights**, on a hand-rolled min/max root-to-node traversal
  with a *relative* tolerance.  A birth-death network is time-consistent and ultrametric by
  construction, so any violation means the branch lengths were corrupted in transit -- this is
  the check that would catch SiPhyNetwork issue #8 (~0.875 added to hybrid edge lengths) once the
  BDH panel feeds this same entry point.
"""

using CodecZlib
using JSON3
using PhyloCoalSimulations
using PhyloNetworks
using StableRNGs

const LEN_TOL = 1e-9      # anything more negative than this is an error, not float noise
const REL_TOL = 1e-6      # relative tolerance for time consistency / ultrametricity

"""Clamp float-noise negatives to zero; anything genuinely negative is an error."""
function sanitize_lengths!(net)
    clamped = 0
    for e in net.edge
        isfinite(e.length) || error("edge $(e.number) has non-finite length $(e.length)")
        if e.length < 0.0
            e.length < -LEN_TOL && error("edge $(e.number) has negative length $(e.length)")
            e.length = 0.0
            clamped += 1
        end
    end
    clamped
end

"""Every hybrid edge must carry a usable gamma, and partners must sum to 1."""
function check_gammas(net)
    for e in net.edge
        e.hybrid || continue
        e.gamma == -1.0 && error("hybrid edge $(e.number) has no gamma (readnewick is silent)")
        (0.0 < e.gamma < 1.0) || error("hybrid edge $(e.number) has gamma $(e.gamma)")
    end
    for h in net.hybrid
        parents = [e for e in h.edge if getchild(e) === h]
        length(parents) == 2 || error("hybrid node $(h.number) has $(length(parents)) parents")
        s = sum(e.gamma for e in parents)
        abs(s - 1.0) < 1e-9 || error("hybrid node $(h.number) gammas sum to $s")
    end
end

"""Min and max root-to-node path length, over the whole DAG."""
function node_height_bounds(net)
    directedges!(net)
    preorder!(net)
    root = getroot(net)
    lo = Dict{Int,Float64}(root.number => 0.0)
    hi = Dict{Int,Float64}(root.number => 0.0)
    for nd in net.vec_node
        nd === root && continue
        ins = [e for e in nd.edge if getchild(e) === nd]
        isempty(ins) && error("non-root node $(nd.number) has no parent edge")
        lo[nd.number] = minimum(lo[getparent(e).number] + e.length for e in ins)
        hi[nd.number] = maximum(hi[getparent(e).number] + e.length for e in ins)
    end
    lo, hi
end

function check_time_consistent_and_ultrametric(net, label)
    lo, hi = node_height_bounds(net)
    scale = maximum(values(hi))
    scale > 0 || error("$label: network has zero height")
    for nd in net.node
        spread = hi[nd.number] - lo[nd.number]
        spread / scale < REL_TOL ||
            error("$label: node $(nd.number) is time-inconsistent (spread $spread, height $scale)")
    end
    leaf_h = [hi[nd.number] for nd in net.node if nd.leaf]
    (maximum(leaf_h) - minimum(leaf_h)) / scale < REL_TOL ||
        error("$label: not ultrametric (leaf heights span " *
              "$(maximum(leaf_h) - minimum(leaf_h)), height $scale)")
    scale
end

"""Sorted hardwired-cluster rows, as a comparable canonical form."""
canonical_clusters(net) = sort([join(r, "") for r in eachrow(hardwiredclusters(net, tiplabels(net)))])

function check_roundtrip(net)
    before = canonical_clusters(net)
    gb = sort([e.gamma for e in net.edge if e.hybrid])
    again = readnewick(writenewick(net))
    after = canonical_clusters(again)
    before == after || error("newick round trip changed the hardwired clusters")
    ga = sort([e.gamma for e in again.edge if e.hybrid])
    (length(gb) == length(ga) && all(abs.(gb .- ga) .< 1e-9)) ||
        error("newick round trip changed the gammas: $gb -> $ga")
end

"""Median length of the non-pendant tree edges -- the quantity the ILS axis is defined on."""
function median_internal_length(net)
    xs = Float64[]
    for e in net.edge
        e.hybrid && continue
        c = getchild(e)
        c.leaf && continue
        push!(xs, e.length)
    end
    isempty(xs) && return NaN
    sort!(xs)
    n = length(xs)
    isodd(n) ? xs[(n + 1) ÷ 2] : (xs[n ÷ 2] + xs[n ÷ 2 + 1]) / 2
end

function run_job(job)
    net = readnewick(read(String(job.enewick), String))

    n_clamped = sanitize_lengths!(net)
    check_gammas(net)

    length(tiplabels(net)) == job.n_taxa ||
        error("expected $(job.n_taxa) taxa, got $(length(tiplabels(net)))")
    length(unique(tiplabels(net))) == job.n_taxa || error("duplicate taxon labels")
    net.numhybrids == job.r || error("expected r=$(job.r), got $(net.numhybrids)")
    lvl = getlevel(net)
    # `level` is optional so that every manifest written before the level-2 benchmark existed --
    # all of them level-1 by construction -- keeps its meaning.  That benchmark passes it.
    want = get(job, :level, job.r > 0 ? 1 : 0)
    lvl == want || error("expected level $want, got $lvl")

    # Check *before* rescaling too: SiPhyNetwork's branch-length bug (issue #8) adds a constant
    # in absolute birth-death time units, so it is largest here and the blame is unambiguous.
    check_time_consistent_and_ultrametric(net, "as read")
    check_roundtrip(net)

    if get(job, :rescale, false)
        # The BDH panel arrives in absolute birth-death time; the constrained generator already
        # arrives in coalescent units.  Uniform scaling preserves time consistency either way.
        med0 = median_internal_length(net)
        (isfinite(med0) && med0 > 0) || error("cannot rescale: median internal branch is $med0")
        f = job.cu / med0
        for e in net.edge
            e.length *= f
        end
        check_time_consistent_and_ultrametric(net, "after rescale")
    end

    med = median_internal_length(net)
    abs(med - job.cu) < 1e-6 * max(job.cu, 1.0) ||
        error("median internal branch $med != target cu $(job.cu)")

    rng = StableRNG(job.seed % typemax(UInt32))
    open(GzipCompressorStream, String(job.out), "w") do io
        # Chunk so peak memory stays flat regardless of n_loci.
        remaining = job.n_loci
        while remaining > 0
            k = min(remaining, 500)
            for t in simulatecoalescent(rng, net, k, 1)
                println(io, writenewick(t; round = true, digits = 8))
            end
            remaining -= k
        end
    end
    (clamped = n_clamped, level = lvl, isgalled = isgalled(net),
     istreechild = first(istreechild(net)), median_internal = med)
end

function main()
    length(ARGS) == 1 || error("usage: gene_trees.jl <manifest.json>")
    jobs = JSON3.read(read(ARGS[1], String))
    failures = 0
    for job in jobs
        try
            info = run_job(job)
            println("ok $(job.out) level=$(info.level) galled=$(info.isgalled) " *
                    "treechild=$(info.istreechild) clamped=$(info.clamped)")
        catch err
            failures += 1
            println("FAILED $(job.out): $(sprint(showerror, err))")
        end
    end
    exit(failures == 0 ? 0 : 1)
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end
