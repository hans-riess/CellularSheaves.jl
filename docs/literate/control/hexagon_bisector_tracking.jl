# # Hexagon Coordination: Tracking a Target Nobody Can See
#
# Six agents hold a regular hexagon around a moving target. Three of them —
# agents 1, 3 and 5 — can each measure exactly **one number**: the projection of the
# target onto the *angle bisector* at their own vertex. Agents 2, 4 and 6 measure nothing
# at all.
#
# No agent can locate the target. Recovering it, and telling the agents who cannot see it
# where to go, is a [`harmonic_extension`](@ref) over a coordination sheaf.
#
# This page builds that sheaf, checks it is well posed, watches it track, and then breaks
# it on purpose — because the way it fails is the best argument for the sheaf-theoretic
# formulation.
#
# ## The formation as an affine sheaf
#
# Agent `i` sits at angle ``\theta_i = 2\pi(i-1)/6`` and distance ``r`` from the formation
# centre, with nominal offset ``d_i = r(\cos\theta_i, \sin\theta_i)``. The formation is
# rigid: what we want to assert on each ring edge is a **fixed displacement**,
#
# ```math
# p_i - p_j = d_i - d_j
# ```
#
# which is an *affine* constraint, not a linear one, and so cannot be a linear sheaf's
# edge condition as it stands. The standard fix is homogeneous coordinates: each vertex
# stalk is ``[p; h] \in \mathbb{R}^3`` and the restriction map out of agent ``i`` is
#
# ```math
# T(d_i) = \begin{bmatrix} I_2 & -d_i \\ 0 & 1 \end{bmatrix},
# \qquad T(d_i)\begin{bmatrix} p \\ h \end{bmatrix} = \begin{bmatrix} p - h\,d_i \\ h \end{bmatrix}
# ```
#
# so edge ``(i,j)`` asserts ``p_i - h_i d_i = p_j - h_j d_j``. With ``h \equiv 1`` that is
# exactly the displacement constraint above, and the shared value ``p_i - d_i`` is the
# formation centre.
#
# ## What an observer sees
#
# The bisector at vertex ``i`` — the line making congruent angles to agents ``i-1`` and
# ``i+1`` — is, for a regular polygon, the radial line, with inward unit direction
# ``u_i = -d_i / r``. An observing agent measures only ``u_i^\top p_t``. Its observation
# edge therefore carries a **two-dimensional** stalk: one projection row, and one *gauge*
# row carrying ``h``.
#
# The gauge row is load-bearing rather than decorative, and it is the one place this
# construction can go quietly wrong — we will demonstrate that below.

using CellularSheaves
using CellularSheaves.Formations
using Graphs
using LinearAlgebra
using Plots
using Printf

# A single house style for every figure below, matching the other control examples.

default(framestyle = :box, grid = true, gridalpha = 0.18, gridstyle = :dot,
    titlefontsize = 10, guidefontsize = 9, legendfontsize = 8, tickfontsize = 8,
    markerstrokewidth = 0, size = (720, 380))

const RING = :steelblue
const OBSERVER = :crimson
const BISECTOR = :seagreen
const GHOST = :gray60

const NA, R = 6, 1.0
const TV = NA + 1

sheaf = build_projection_escort_ring(NA, TV, R; observers = [1, 3, 5])

offsets = [R .* [cos(2π * (i - 1) / NA), sin(2π * (i - 1) / NA)] for i in 1:NA]
directions = bisector_directions(NA)

@printf("vertex stalks      : %s\n", string(sheaf.vertex_stalks))
@printf("edges              : %d  (6 ring + 3 observation)\n", ne(sheaf.underlying_graph))
@printf("edge stalk dims    : %s\n", string(sort(collect(values(sheaf.edge_stalks)))))
@printf("coboundary size    : %s\n", string(size(coboundary_map(sheaf))))

# ## The escort formation is an exact global section
#
# Placing agent ``i`` at ``p_t + d_i`` satisfies every constraint simultaneously — the
# ring edges by construction, and each observation edge because
# ``u_i^\top(p_i - d_i) = u_i^\top p_t``. The Dirichlet energy is zero to machine precision.

section(p) = vcat([vcat(p .+ offsets[i], 1.0) for i in 1:NA]..., vcat(p, 1.0))

for p in ([0.0, 0.0], [0.4, -0.25], [-0.9, 0.6])
    @printf("‖δx‖ at target (%5.2f, %5.2f) = %.3e\n", p[1], p[2],
            norm(coboundary_map(sheaf) * section(p)))
end

# ## Three scalars determine the formation
#
# The ring is rigid, so once ``h = 1`` the free formation has exactly two degrees of
# freedom — its centre. Three observers supply three scalars, and because their bisectors
# sit 120° apart the fusion is *isotropic*:
#
# ```math
# \sum_{i \in \{1,3,5\}} u_i u_i^\top = \tfrac{3}{2} I_2
# ```
#
# so the estimate is perfectly conditioned in every direction. `harmonic_extension`
# recovers the escort formation exactly, and reports an empty null space.

fused = sum(directions[i] * directions[i]' for i in (1, 3, 5))
@printf("Σ uᵢuᵢᵀ = [%.3f %.3f; %.3f %.3f]\n", fused[1,1], fused[1,2], fused[2,1], fused[2,2])

target = [0.55, -0.30]
x, null_basis = harmonic_extension(sheaf, Dict(TV => vcat(target, 1.0)))
positions = [Vector(x)[3(i-1)+1:3(i-1)+2] for i in 1:NA]

@printf("undetermined directions : %d\n", size(null_basis, 2))
@printf("max error vs exact      : %.3e\n",
        maximum(norm(positions[i] - (target .+ offsets[i])) for i in 1:NA))

# The figure below is the whole construction in one picture. Note that agents 2, 4 and 6
# touch no observation edge whatsoever: their position is *entirely* the harmonic
# extension's doing.

function draw_formation(positions, target, observers; null_basis = zeros(0, 0), title = "")
    plt = plot(; aspect_ratio = :equal, xlims = (-2.6, 2.6), ylims = (-1.9, 1.9),
        title = title, legend = :topleft, size = (720, 470))

    if size(null_basis, 2) > 0
        centre = sum(positions) ./ length(positions)
        for k in 1:size(null_basis, 2)
            w = normalize(null_basis[1:2, k])
            plot!(plt, [centre[1] - 2.4w[1], centre[1] + 2.4w[1]],
                       [centre[2] - 2.4w[2], centre[2] + 2.4w[2]];
                label = k == 1 ? "undetermined direction" : "", color = OBSERVER,
                linewidth = 9, alpha = 0.22)
        end
    end

    for (n, i) in enumerate(observers)
        u, a = directions[i], positions[i]
        plot!(plt, [a[1] - 3u[1], a[1] + 3u[1]], [a[2] - 3u[2], a[2] + 3u[2]];
            label = n == 1 ? "bisector ℓᵢ" : "", color = BISECTOR, linestyle = :dash, linewidth = 1.2)
        foot = a .+ dot(target .- a, u) .* u
        plot!(plt, [target[1], foot[1]], [target[2], foot[2]];
            label = n == 1 ? "observed projection" : "", color = OBSERVER,
            linestyle = :dashdot, linewidth = 1.4)
        scatter!(plt, [foot[1]], [foot[2]]; label = "", color = OBSERVER, markersize = 5)
    end

    ring = [positions; [positions[1]]]
    plot!(plt, first.(ring), last.(ring); label = "ring edges (fixed displacement)",
        color = RING, linewidth = 2.2)
    scatter!(plt, first.(positions), last.(positions); label = "agents", color = RING, markersize = 9)
    for i in 1:NA
        annotate!(plt, positions[i][1], positions[i][2], text(string(i), 7, :white))
    end
    scatter!(plt, [target[1]], [target[2]]; label = "target", color = :black,
        marker = :star5, markersize = 11)
    return plt
end

draw_formation(positions, target, [1, 3, 5];
    title = "Three scalar readings, fused into a full 2-D estimate")

# ## Tracking a moving target
#
# The topology never changes, so the sheaf is built once and only the boundary data moves.
# Because `harmonic_extension` is linear in that data, feeding it the target's *velocity*
# returns the reference's velocity — a feedforward term that removes the lag a purely
# proportional controller would show.
#
# One detail decides whether this works: the homogeneous slot is `1.0` in the position
# problem and `0.0` in the rate problem. The affine gauge is a constant, so it has no rate;
# a `1.0` there would inject a spurious dilation into every agent's feedforward.

path(t) = [1.5cos(0.6t), 0.85sin(1.2t)]
rate(t) = [-0.9sin(0.6t), 1.02cos(1.2t)]

reference(t) = harmonic_extension(sheaf, Dict(TV => vcat(path(t), 1.0)))[1]
reference_rate(t) = harmonic_extension(sheaf, Dict(TV => vcat(rate(t), 0.0)))[1]

recovered_centre(t) = let q = Vector(reference(t))
    sum(q[3(i-1)+1:3(i-1)+2] .- offsets[i] for i in 1:NA) ./ NA
end

times = range(0, 12; length = 260)
centres = recovered_centre.(times)

plot(times, [c[1] for c in centres]; label = "recovered centre x", color = RING, linewidth = 2,
    xlabel = "time", ylabel = "position", title = "The fused estimate follows the target exactly")
plot!(times, [c[2] for c in centres]; label = "recovered centre y", color = OBSERVER, linewidth = 2)
plot!(times, [path(t)[1] for t in times]; label = "target x", color = :black,
    linestyle = :dash, linewidth = 1.2)
plot!(times, [path(t)[2] for t in times]; label = "target y", color = GHOST,
    linestyle = :dash, linewidth = 1.2)

# The feedforward term is exact, not an approximation, because the reduced Laplacian blocks
# do not depend on time. Comparing it against a finite difference of the position solve
# confirms it:

let t = 3.0, h = 1e-6
    analytic = Vector(reference_rate(t))[1:2]
    numeric = (Vector(reference(t + h))[1:2] - Vector(reference(t - h))[1:2]) ./ (2h)
    @printf("feedforward   : [%.6f, %.6f]\n", analytic[1], analytic[2])
    @printf("finite diff   : [%.6f, %.6f]\n", numeric[1], numeric[2])
    @printf("difference    : %.3e\n", norm(analytic - numeric))
end

# ## Removing observers, and what the sheaf says about it
#
# Two scalars are enough, provided the two bisectors are independent. Below, every observer
# subset of the hexagon, scored by the dimension of the resulting null space.

subsets = [[1, 3, 5], [1, 3], [3, 5], [1, 2], [1, 4], [3, 6], [3], [1]]
for observers in subsets
    s = build_projection_escort_ring(NA, TV, R; observers = observers)
    _, nb = harmonic_extension(s, Dict(TV => [0.55, -0.30, 1.0]))
    @printf("observers %-10s → %d scalar reading(s), %d undetermined direction(s)\n",
            string(observers), length(observers), size(nb, 2))
end

# Two results there are worth pausing on.
#
# `[1, 4]` and `[3, 6]` are *pairs* of observers that still leave the formation
# under-determined. Those vertices are diametrically opposite, so their bisectors are
# antiparallel — the second reading is the first one again, and buys nothing. Counting
# observers is not the same as counting information, and the sheaf knows the difference.
#
# A single observer leaves exactly one direction free, and it is the translation
# perpendicular to that agent's bisector:

solo = build_projection_escort_ring(NA, TV, R; observers = [3])
x_solo, null_solo = harmonic_extension(solo, Dict(TV => vcat(target, 1.0)))
positions_solo = [Vector(x_solo)[3(i-1)+1:3(i-1)+2] for i in 1:NA]
w = normalize(null_solo[1:2, 1])

@printf("null dimension          : %d\n", size(null_solo, 2))
@printf("‖w · u₃‖                : %.3e   (orthogonal to the bisector)\n", abs(dot(w, directions[3])))
@printf("uniform across agents?  : %.3e   (max deviation)\n",
        maximum(norm(null_solo[3(i-1)+1:3(i-1)+2, 1] - null_solo[1:2, 1]) for i in 1:NA))
@printf("Dirichlet energy        : %.3e   (still a valid solution)\n",
        norm(coboundary_map(solo) * Vector(x_solo)))

draw_formation(positions_solo, target, [3]; null_basis = null_solo,
    title = "One observer: the ring may slide along the bar at no energy cost")

# The ring sits off the target, yet the sheaf energy is zero. That configuration is a
# perfectly valid solution — it simply is not the only one. A least-squares solver would
# have returned one of these points with no indication that the others existed;
# `harmonic_extension` hands back the whole family.
#
# !!! note "The particular solution is not the minimum-norm one"
#     `harmonic_extension` returns *a* representative of the solution set together with a
#     basis for the indeterminate directions. In the rank-deficient case, do not test the
#     representative itself — test membership in `x_p + null_basis * c`.
#
# ## Why the observation edge needs a second row
#
# It is tempting to give the observation edge a one-dimensional stalk: the agent measures
# one number, so why carry a second row? Because ``T(d_i)`` scales each offset by ``h``,
# and a formation with ``h = 0`` has *every* displacement vector annihilated. The ring
# collapses to a point at zero energy.
#
# With only the projection row, ``h`` is pinned merely to a constant around the ring and
# that constant is free. Building the degenerate variant by hand shows the mode explicitly:

collapsible = EuclideanSheaf{Float64}(fill(3, TV))
for i in 1:NA
    j = i % NA + 1
    add_sheaf_edge!(collapsible, i, j, affine_translation_matrix(offsets[i]),
                    affine_translation_matrix(offsets[j]))
end
for i in (1, 3, 5)
    row = reshape(vcat(directions[i], 0.0), 1, 3)
    add_sheaf_edge!(collapsible, i, TV, row * affine_translation_matrix(offsets[i]), row)
end

_, collapse_null = harmonic_extension(collapsible, Dict(TV => vcat(target, 1.0)))
scales = [collapse_null[3i, 1] for i in 1:NA]
residuals = [norm(collapse_null[3(i-1)+1:3(i-1)+2, 1] - scales[i] .* offsets[i]) for i in 1:NA]

@printf("without the gauge row → %d undetermined direction(s)\n", size(collapse_null, 2))
@printf("  h equal across agents?      max deviation %.3e\n", maximum(abs.(scales .- scales[1])))
@printf("  mode is pᵢ = h·dᵢ exactly?  max residual  %.3e\n", maximum(residuals))
@printf("with the gauge row    → %d undetermined direction(s)\n", size(null_basis, 2))

# The null direction is exactly the uniform dilation ``p_i = h\,d_i``: the hexagon shrinking
# toward its own centre. Keeping the gauge row lets the target's boundary value
# ``[p_t; 1]`` propagate ``h = 1`` into the formation and removes the mode entirely.
#
# ## Where to go next
#
# `examples/hexagon_coordination/` runs this sheaf live in the browser, with the target on
# the arrow keys and the observer set on the number keys, so the degeneracies above can be
# switched on and off by hand. Target paths recorded there replay on the Georgia Tech
# Robotarium through the companion demo in `robotarium_python_simulator`.
