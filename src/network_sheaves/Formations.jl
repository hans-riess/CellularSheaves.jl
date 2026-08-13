module Formations

using LinearAlgebra
using ..EuclideanSheaves: EuclideanSheaf, add_sheaf_edge!
using ArgCheck: @argcheck

export se3_translation_matrix, se3_rotation_matrix, se3_affine_matrix, affine_translation_matrix, build_escort_topology, build_escort_ring, build_escort_clique, bisector_directions, build_projection_escort_ring

"""
    se3_translation_matrix(d::AbstractVector)

Returns the 4x4 homogeneous affine translation matrix.
"""
function se3_translation_matrix(d::AbstractVector)
    @assert length(d) == 3 "Translation vector must be 3-dimensional"
    [I(3) -d; 0 0 0 1]
end

"""
    affine_translation_matrix(d::AbstractVector)

Returns the `(n+1)x(n+1)` homogeneous affine translation matrix for a translation
vector `d` of arbitrary dimension `n`. Generalizes `se3_translation_matrix` (the
`n=3` case) to any stalk dimension, for non-SE(3) formations (e.g. planar agents).
"""
function affine_translation_matrix(d::AbstractVector)
    n = length(d)
    [I(n) -d; zeros(1, n) 1]
end

"""
    se3_rotation_matrix(R::AbstractMatrix)

Returns the 4x4 homogeneous affine rotation matrix.
"""
function se3_rotation_matrix(R::AbstractMatrix)
    @assert size(R) == (3, 3) "Rotation matrix must be 3x3"
    [R zeros(3); 0 0 0 1]
end

"""
    se3_rotation_matrix(; θx::Real=0.0, θy::Real=0.0, θz::Real=0.0)

Returns the 4x4 homogeneous affine rotation matrix corresponding to Euler angles (XYZ convention).
"""
function se3_rotation_matrix(; θx::Real=0.0, θy::Real=0.0, θz::Real=0.0)
    Rx = [1.0 0.0 0.0; 0.0 cos(θx) -sin(θx); 0.0 sin(θx) cos(θx)]
    Ry = [cos(θy) 0.0 sin(θy); 0.0 1.0 0.0; -sin(θy) 0.0 cos(θy)]
    Rz = [cos(θz) -sin(θz) 0.0; sin(θz) cos(θz) 0.0; 0.0 0.0 1.0]
    se3_rotation_matrix(Rz * Ry * Rx)
end

"""
    se3_affine_matrix(R::AbstractMatrix, d::AbstractVector)

Returns the 4x4 homogeneous affine matrix with rotation `R` and translation `d`.
"""
function se3_affine_matrix(R::AbstractMatrix, d::AbstractVector)
    @assert length(d) == 3 "Translation vector must be 3-dimensional"
    @assert size(R) == (3, 3) "Rotation matrix must be 3x3"
    [R -d; 0 0 0 1]
end

"""
    se3_affine_matrix(d::AbstractVector; θx::Real=0.0, θy::Real=0.0, θz::Real=0.0)

Returns the 4x4 homogeneous affine matrix using Euler angles and translation `d`.
"""
function se3_affine_matrix(d::AbstractVector; θx::Real=0.0, θy::Real=0.0, θz::Real=0.0)
    R_hom = se3_rotation_matrix(; θx=θx, θy=θy, θz=θz)
    R = R_hom[1:3, 1:3]
    se3_affine_matrix(R, d)
end

"""
    build_escort_topology(kind::Symbol, n_agents::Int, target_node::Int, radius::Float64;
                          observers=1:n_agents, D::Int=4, affine::Bool=true) -> EuclideanSheaf{Float64}

Constructs and returns a `EuclideanSheaf` (with stalk dimension `D`, default 4) for an `n_agents`
escort formation around `target_node`.

An escort formation bundles two concerns that are mathematically independent, and this function
keeps them so:

- **Geometry** — where each agent sits. Regardless of `kind`, agent `i` is placed at angle
  `2π(i-1)/n_agents` and distance `radius` in the plane spanned by the first two translation
  coordinates (this is exactly what `build_escort_ring` has always done).
- **Consensus topology** — which pairs of agents are directly wired together by a sheaf edge.
  `kind` selects this graph:

  - `:ring`   — cycle: agent `i` shares an edge with agent `i % n_agents + 1`. A `2`-agent ring
    is a degenerate 2-cycle (it produces two parallel edges between the same pair of agents), but
    the formation is still well-defined and rigid.
  - `:path`   — open chain: agent `i` shares an edge with `i + 1`, for `i in 1:n_agents-1`.
  - `:star`   — hub-and-spoke: agent `1` shares an edge with every agent `i in 2:n_agents`.
  - `:clique` — all-to-all: every pair `i < j` shares an edge.

Every edge constraint has the algebraic form "agent `i` = centre + `d_i`" — a shared centre plus a
per-agent translation offset — so the constraint system is globally realizable for *any connected*
choice of `kind`. Consequently, for all four topologies above, the resulting sheaf's space of
exact global sections is exactly `D`-dimensional, parameterized by the formation centre: the
formation is rigid no matter which agents happen to be directly wired together. This is the
property the hierarchical layered-control architecture depends on (see
`docs/issues/007-nested-layered-systems-design.md`, §3.2).

When `affine=true` (default), stalks use `D-1` homogeneous-affine translation coordinates plus one
homogeneous row (e.g. `D=4` for SE(3): 3D translation), and restriction maps are
`affine_translation_matrix` offsets — this recovers the original SE(3) escort ring for `D=4`. When
`affine=false`, stalks are `D` plain (non-homogeneous) Euclidean coordinates; a purely linear
restriction map cannot represent a translation at all, so every restriction map is the identity (a
pure consensus topology) and `radius` must be `0.0`.

`observers` names which agents (local indices `1:n_agents`) are pinned to `target_node`.
"""
function build_escort_topology(kind::Symbol, n_agents::Int, target_node::Int, radius::Float64;
                               observers=1:n_agents, D::Int=4, affine::Bool=true)
    @argcheck kind in (:ring, :path, :star, :clique) "kind must be one of :ring, :path, :star, :clique (got :$kind)"
    @argcheck affine || radius == 0.0 "Non-affine (linear) stalks cannot represent a nonzero radius offset; set affine=true or radius=0.0"
    @argcheck all(1 .<= o .<= n_agents for o in observers) "observers must be within 1:n_agents"
    min_agents = kind == :ring ? 2 : 1
    @argcheck n_agents >= min_agents "n_agents must be >= $min_agents for kind=:$kind"

    total_nodes = max(n_agents, target_node)
    sheaf = EuclideanSheaf{Float64}(fill(D, total_nodes))

    restriction_matrix = if affine
        trans_dim = D - 1
        # Translation offset for agent i in the world frame, placed in the plane
        # spanned by the first two translation coordinates (zero elsewhere)
        function angular_offset(i)
            angle = (i - 1) * 2π / n_agents
            d = zeros(trans_dim)
            trans_dim >= 1 && (d[1] = cos(angle) * radius)
            trans_dim >= 2 && (d[2] = sin(angle) * radius)
            return d
        end
        i -> affine_translation_matrix(angular_offset(i))
    else
        # Non-affine (plain linear) stalks cannot represent a translation at all,
        # so every restriction map is simply the identity (pure consensus).
        i -> Matrix{Float64}(I, D, D)
    end

    # Consensus edges: which agents are directly wired together, per `kind`.
    consensus_edges = if kind == :ring
        [(i, i % n_agents + 1) for i in 1:n_agents]
    elseif kind == :path
        [(i, i + 1) for i in 1:(n_agents - 1)]
    elseif kind == :star
        [(1, i) for i in 2:n_agents]
    else # :clique
        [(i, j) for i in 1:n_agents for j in (i + 1):n_agents]
    end

    for (i, j) in consensus_edges
        add_sheaf_edge!(sheaf, i, j, restriction_matrix(i), restriction_matrix(j))
    end

    # Pin observers to the target
    F_target = Matrix{Float64}(I, D, D)
    for i in observers
        add_sheaf_edge!(sheaf, i, target_node, restriction_matrix(i), F_target)
    end

    return sheaf
end

"""
    build_escort_ring(n_agents::Int, target_node::Int, radius::Float64; observers=1:n_agents, D::Int=4, affine::Bool=true)

Constructs and returns a `EuclideanSheaf` for an `n_agents` escort *ring* around `target_node` —
a thin wrapper around [`build_escort_topology`](@ref)`(:ring, ...)`. See that docstring for the
full explanation of the geometry/topology split, `D`, and `affine`.
"""
build_escort_ring(n_agents::Int, target_node::Int, radius::Float64; observers=1:n_agents, D::Int=4, affine::Bool=true) =
    build_escort_topology(:ring, n_agents, target_node, radius; observers, D, affine)

"""
    build_escort_clique(n_agents::Int, target_node::Int, radius::Float64; observers=1:n_agents, D::Int=4, affine::Bool=true)

Constructs and returns a `EuclideanSheaf` for an `n_agents` escort *clique* (all-to-all consensus)
around `target_node` — a thin wrapper around [`build_escort_topology`](@ref)`(:clique, ...)`. See
that docstring for the full explanation of the geometry/topology split, `D`, and `affine`.
"""
build_escort_clique(n_agents::Int, target_node::Int, radius::Float64; observers=1:n_agents, D::Int=4, affine::Bool=true) =
    build_escort_topology(:clique, n_agents, target_node, radius; observers, D, affine)

"""
    bisector_directions(n_agents::Int; trans_dim::Int=2) -> Vector{Vector{Float64}}

Inward unit vectors along the *angle bisector* at each vertex of a regular `n_agents`-gon.

The bisector at vertex `i` is the line through agent `i` making congruent angles to its two
ring neighbours `i-1` and `i+1`. For a regular polygon that line is the radial one, so the
bisector direction is simply

```math
u_i = -d_i / r, \\qquad d_i = r\\,(\\cos\\theta_i, \\sin\\theta_i), \\quad \\theta_i = 2\\pi(i-1)/n
```

pointing from agent `i` toward the centre. The result is independent of the radius. Vectors
are returned in `trans_dim` coordinates with the bisector living in the first two and zeros
elsewhere, matching the offset convention of [`build_escort_topology`](@ref).
"""
function bisector_directions(n_agents::Int; trans_dim::Int=2)
    @argcheck n_agents >= 3 "n_agents must be >= 3 (got $n_agents)"
    @argcheck trans_dim >= 2 "trans_dim must be >= 2 to carry a planar bisector (got $trans_dim)"
    map(1:n_agents) do i
        angle = (i - 1) * 2π / n_agents
        u = zeros(trans_dim)
        u[1] = -cos(angle)
        u[2] = -sin(angle)
        return u
    end
end

"""
    build_projection_escort_ring(n_agents::Int, target_node::Int, radius::Real;
                                 observers=1:2:n_agents, D::Int=3) -> EuclideanSheaf{Float64}

Build an affine escort ring in which observing agents see only a **one-dimensional
projection** of the target, rather than its full position.

This is the sheaf behind the hexagon coordination demo. Where
[`build_escort_ring`](@ref) pins each observer to the target with a full-rank identity map
— "agent `i` knows exactly where the target is" — this constructor replaces that pin with a
single scalar reading along the agent's own [`bisector_directions`](@ref) line:

```math
u_i^\\top (p_i - h_i d_i) = u_i^\\top p_t
```

No observer can localise the target on its own. Recovering the target's position, and
propagating it to the agents that observe *nothing*, is exactly the work done by
[`harmonic_extension`](@ref) — which is the point of the construction.

# Geometry

Agent `i` sits at angle `2π(i-1)/n_agents` and distance `radius` from the formation centre,
with nominal offset `d_i`. Stalks are homogeneous affine: a vertex cochain is `[p; h]` with
`p` the position in `D-1` translation coordinates and `h` the homogeneous coordinate. Ring
edges carry [`affine_translation_matrix`](@ref) offsets, so with `h = 1` edge `(i,j)`
asserts the fixed displacement `p_i - p_j = d_i - d_j`.

# The observation edge, and why its stalk is two-dimensional

Writing `S_i` for the `2 × D` matrix whose first row is `u_i` (padded with a zero in the
homogeneous column) and whose second row is `[0 … 0 1]`, the edge from agent `i` to
`target_node` has a **2-dimensional** stalk with restriction maps

```julia
S_i * affine_translation_matrix(d_i)    # agent side
S_i                                     # target side
```

The first row is the projection. The second — the *gauge* row — is load-bearing rather than
decorative. `affine_translation_matrix` scales each offset by `h`, so a formation with
`h = 0` has all its displacement vectors annihilated and costs zero Dirichlet energy: the
ring collapses to a point. Dropping the gauge row leaves `h` pinned only up to a constant
across the ring, and the resulting Laplacian is singular along precisely that uniform
dilation mode (the same degeneracy discussed in the rescaling-formation example). Keeping
it lets the target's boundary value `[p_t; 1]` propagate `h = 1` into the formation.

# Rank and degeneracy

The ring is rigid, so the free formation has two translational degrees of freedom (its
centre) once `h = 1`. Each observer contributes one scalar, so the extension is uniquely
determined as soon as two observers have linearly independent bisector directions. The
default `observers = 1:2:n_agents` picks every other agent; for a hexagon that is agents
`1, 3, 5`, whose directions sit 120° apart and satisfy `∑ u_i u_i^\\top = (3/2) I` — an
isotropic, perfectly conditioned fusion of the three readings.

Under-determined choices are legitimate inputs and are reported rather than rejected: a
single observer, or two diametrically opposite ones (whose bisectors are antiparallel),
leave the formation's translation along one direction genuinely undetermined.
[`harmonic_extension`](@ref) returns that direction as a null-space basis. Note its
particular solution is *a* representative of the solution set, not the minimum-norm one.

# Arguments

- `n_agents`: number of agents around the ring; vertices `1:n_agents`.
- `target_node`: vertex index of the target, pinned as boundary data.
- `radius`: ring radius. Keep this `O(1)`–`O(10)`; the rank tolerance used for null-space
  detection is relative to the Laplacian's spectrum, and a very large radius narrows the
  margin between a true null direction and a merely small eigenvalue.
- `observers`: agent indices (in `1:n_agents`) that observe the target.
- `D`: vertex stalk dimension — `D-1` translation coordinates plus one homogeneous
  coordinate. `D = 3` is planar; `D = 4` matches the SE(3) escort convention.
"""
function build_projection_escort_ring(n_agents::Int, target_node::Int, radius::Real;
                                      observers=1:2:n_agents, D::Int=3)
    @argcheck n_agents >= 3 "n_agents must be >= 3 for a non-degenerate ring (got $n_agents)"
    @argcheck D >= 3 "D must be >= 3: at least two translation coordinates plus a homogeneous one (got $D)"
    @argcheck radius > 0 "radius must be positive to define a bisector direction (got $radius)"
    @argcheck target_node > n_agents "target_node must be outside 1:n_agents (got $target_node)"
    @argcheck all(1 <= o <= n_agents for o in observers) "observers must be within 1:n_agents"
    @argcheck !isempty(observers) "at least one observer is required to pin the homogeneous coordinate"

    trans_dim = D - 1
    total_nodes = max(n_agents, target_node)
    sheaf = EuclideanSheaf{Float64}(fill(D, total_nodes))

    offsets = map(1:n_agents) do i
        angle = (i - 1) * 2π / n_agents
        d = zeros(trans_dim)
        d[1] = cos(angle) * radius
        d[2] = sin(angle) * radius
        return d
    end
    directions = bisector_directions(n_agents; trans_dim=trans_dim)
    frames = affine_translation_matrix.(offsets)

    for i in 1:n_agents
        j = i % n_agents + 1
        add_sheaf_edge!(sheaf, i, j, frames[i], frames[j])
    end

    for i in observers
        selector = zeros(2, D)
        selector[1, 1:trans_dim] = directions[i]
        selector[2, D] = 1.0
        add_sheaf_edge!(sheaf, i, target_node, selector * frames[i], selector)
    end

    return sheaf
end

end # module
