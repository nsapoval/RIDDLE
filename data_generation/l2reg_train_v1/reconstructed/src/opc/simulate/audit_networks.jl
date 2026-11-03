#!/usr/bin/env julia
"""
An independent structural audit of written eNewick files (A62).

    julia --project=julia src/opc/simulate/audit_networks.jl paths.json out.json

`paths.json` is a JSON array of eNewick paths.  PhyloNetworks parses each file, and this script
reports what *it* sees.  Nothing here shares code with the Python writer, so agreement means the
file says what the generator meant:

* `clusters` -- the distinct hardwired clusters (descendant-leaf set of every node);
* `hybrids` -- per hybrid node: its cluster, and for each parent edge its gamma, whether it is
  major, and the *parent's* cluster.  So the `#H` copy is checked to sit on the right edge, not
  just the gamma multiset;
* `level` (`getlevel`), per-blob minor-edge counts from `biconnectedcomponents`,
  `hashybridladder`, `istreechild`, `isgalled`;
* `girth_N` / `girth_R` -- the shortest undirected cycle as parsed, and after contracting every
  hybrid->child edge, by a BFS written here.
"""

using JSON3
using PhyloNetworks

function node_clusters(net)
    directedges!(net)
    preorder!(net)
    cl = Dict{Int,Vector{String}}()
    for nd in reverse(net.vec_node)
        if nd.leaf
            cl[nd.number] = [nd.name]
        else
            s = Set{String}()
            for c in getchildren(nd)
                union!(s, cl[c.number])
            end
            cl[nd.number] = sort!(collect(s))
        end
    end
    cl
end

"""Shortest undirected cycle, BFS from every vertex; parallel edges count as a 2-cycle."""
function girth(edges::Vector{Tuple{Int,Int}})
    adj = Dict{Int,Vector{Tuple{Int,Int}}}()
    for (k, (a, b)) in enumerate(edges)
        push!(get!(adj, a, Tuple{Int,Int}[]), (b, k))
        push!(get!(adj, b, Tuple{Int,Int}[]), (a, k))
    end
    best = typemax(Int)
    for s in keys(adj)
        dist = Dict(s => 0)
        via = Dict(s => 0)
        queue = [s]
        head = 1
        while head <= length(queue)
            x = queue[head]
            head += 1
            for (y, k) in adj[x]
                k == via[x] && continue
                if !haskey(dist, y)
                    dist[y] = dist[x] + 1
                    via[y] = k
                    push!(queue, y)
                else
                    best = min(best, dist[x] + dist[y] + 1)
                end
            end
        end
    end
    best == typemax(Int) ? 0 : best
end

function audit(path)
    net = readnewick(read(path, String))
    cl = node_clusters(net)
    hybrids = Dict{String,Any}()
    for h in net.hybrid
        pars = [e for e in h.edge if getchild(e) === h]
        hybrids[h.name] = Dict(
            "cluster" => cl[h.number],
            "parents" => [Dict("gamma" => e.gamma, "ismajor" => e.ismajor,
                               "cluster" => cl[getparent(e).number]) for e in pars],
        )
    end
    edges_N = [(getparent(e).number, getchild(e).number) for e in net.edge]
    # R: each hybrid absorbs its (single) child; the contracted edge disappears.
    rep = Dict{Int,Int}()
    for h in net.hybrid
        kids = getchildren(h)
        length(kids) == 1 || error("hybrid $(h.name) has $(length(kids)) children")
        rep[kids[1].number] = h.number
    end
    r(x) = get(rep, x, x)
    edges_R = [(r(a), r(b)) for (a, b) in edges_N if r(a) != r(b)]
    blobs = biconnectedcomponents(net, true)
    Dict(
        "path" => path,
        "taxa" => sort(tiplabels(net)),
        "numhybrids" => net.numhybrids,
        "clusters" => sort(unique(collect(values(cl)))),
        "hybrids" => hybrids,
        "level" => getlevel(net),
        "blob_retics" => sort([count(e -> e.hybrid && !e.ismajor, b) for b in blobs]),
        "ladder" => hashybridladder(net),
        "treechild" => istreechild(net)[1],
        "galled" => isgalled(net),
        "girth_N" => girth(edges_N),
        "girth_R" => girth(edges_R),
    )
end

function main()
    length(ARGS) == 2 || error("usage: audit_networks.jl <paths.json> <out.json>")
    rows = Any[]
    for path in JSON3.read(read(ARGS[1], String))
        try
            push!(rows, audit(String(path)))
        catch err
            push!(rows, Dict("path" => String(path), "error" => sprint(showerror, err)))
        end
    end
    open(ARGS[2], "w") do io
        JSON3.write(io, rows)
    end
    println("audited $(length(rows)) networks, $(count(r -> haskey(r, "error"), rows)) errors")
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end
