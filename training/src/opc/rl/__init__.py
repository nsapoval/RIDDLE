"""``RL_PLAN.md`` -- the reinforcement-learning stage.

Kept separate from :mod:`opc.solve` (the retained MW-GTCS baseline) and :mod:`opc.followup` (the
v0.3 follow-up) for the same reason those two are separate from each other: **importing this
package must not change what the landed apparatus computes.**  Nothing here is imported by
``opc.solve``, ``opc.followup`` or ``opc.mpl``; the dependency runs one way only.

The house rule that every fast kernel keeps a slow independent reference applies here with a
particular edge, because the fast kernel is now on a *different device* in a *different framework*:
:mod:`opc.rl.mpl_torch` is checked against :mod:`opc.mpl.likelihood` (numpy, shape-grouped), which
is itself checked against :mod:`opc.mpl.msnc3` (the readable per-triple DP), which is checked
against PhyloNet and against the MSNC simulator.  A torch port that agrees with the numpy path to
float64 round-off inherits that whole chain; one that does not has found something.
"""
