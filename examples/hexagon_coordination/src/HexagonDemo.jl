"""
    HexagonDemo

Simulation state for the interactive hexagon coordination demo.

Six agents hold a regular hexagon around a target that the user drives with the keyboard.
Each agent carries an **observation rank** that can be cycled live:

| rank | what the agent knows |
|---|---|
| 0 | nothing — no observation edge at all |
| 1 | one scalar, the target's projection onto its own angle bisector |
| 2 | the target's position in full (the classic escort pin) |

Every frame, a harmonic extension over the coordination sheaf fuses whatever readings exist
and hands each agent a position reference. The sheaf is built by
`build_projection_escort_ring`; see its docstring for the mathematics, including why every
observation edge carries a gauge row.

This module holds no I/O. `server.jl` drives it and streams `snapshot` out over a WebSocket.
"""
module HexagonDemo

using CellularSheaves
using LinearAlgebra

export DemoState, step!, snapshot, cycle_rank!, set_ranks!, toggle_all_ranks!,
       connect_agents!, disconnect_agents!, reset_edges!, toggle_ghosts!,
       add_node!, remove_node!, clear_nodes!, n_agents, MAX_AGENTS,
       adjust_gain!, cycle_feedforward!, reset!, toggle_recording!, save_track

const MAX_AGENTS = 6
const RADIUS = 0.55
const D = 3

# The agent count is whatever the user has drawn, so both it and the target's vertex index
# are derived from the current formation rather than fixed.
n_agents(state) = length(state.offsets)
target_vertex(state) = length(state.offsets) + 1
const MAX_RANK = D - 1

# The arena the browser draws, in the same units as the ring radius. Sized so the ring plus
# the target's reachable box stays comfortably inside the frame.
const ARENA_X = 4.0
const ARENA_Y = 2.5

const TARGET_ACCELERATION = 6.0     # units/s^2 while a key is held
const TARGET_DAMPING = 3.2          # 1/s, coasts to a stop on release
const TARGET_MAX_SPEED = 2.4

# Pursuit defaults. The target's terminal speed is TARGET_ACCELERATION / TARGET_DAMPING =
# 1.875, so a gain of 2.5 with no feedforward leaves a steady lag of about 0.75 units --
# roughly 1.4 ring radii. The ring trails, swings wide on direction changes, and closes in
# when the target slows. Raising `feedforward` to 1.0 cancels that lag exactly at any gain,
# which is why turning the gain down alone would not have produced any.
const DEFAULT_GAIN = 2.5
const DEFAULT_FEEDFORWARD = 0.0
const AGENT_MAX_SPEED = 2.2
const GAIN_LIMITS = (0.4, 12.0)
const FEEDFORWARD_STEPS = (0.0, 0.5, 1.0)

"""
    DemoState(; ranks=[1,0,1,0,1,0], radius=$RADIUS)

Mutable state of a running demo: the sheaf, the agents, the target, the controller gains,
and the recording buffer.
"""
mutable struct DemoState
    radius::Float64
    ranks::Vector{Int}
    restore_ranks::Vector{Int}
    edges::Vector{Tuple{Int,Int}}
    show_ghosts::Bool
    sheaf::EuclideanSheaf{Float64}
    offsets::Vector{Vector{Float64}}
    directions::Vector{Vector{Float64}}
    null_basis::Matrix{Float64}
    positions::Vector{Vector{Float64}}
    reference::Vector{Vector{Float64}}
    target::Vector{Float64}
    velocity::Vector{Float64}
    command::Vector{Float64}
    gain::Float64
    feedforward::Float64
    time::Float64
    recording::Bool
    track::Vector{NTuple{3,Float64}}
end

function DemoState(; ranks::AbstractVector{<:Integer}=[i % 2 == 1 ? 1 : 0 for i in 1:MAX_AGENTS],
                   radius::Real=RADIUS)
    state = DemoState(Float64(radius), Int[], Int[], Tuple{Int,Int}[], false,
                      EuclideanSheaf{Float64}(fill(D, 1)),
                      Vector{Float64}[], Vector{Float64}[], zeros(0, 0),
                      Vector{Float64}[], Vector{Float64}[],
                      zeros(2), zeros(2), zeros(2),
                      DEFAULT_GAIN, DEFAULT_FEEDFORWARD, 0.0, false, NTuple{3,Float64}[])
    reset!(state)
    length(ranks) == n_agents(state) && set_ranks!(state, ranks)
    return state
end

_regular_offsets(n, radius) =
    [Float64(radius) .* [cos((i - 1) * 2π / n), sin((i - 1) * 2π / n)] for i in 1:n]

# Normalised on the way in. The wrap-around edge is built as (n, 1), and every consumer
# that looks an edge up normalises first, so leaving it unordered made exactly one edge of
# the ring -- and only that one -- impossible to cut.
_cycle_edges(n) = n >= 3 ? [_normalise(i, i % n + 1) for i in 1:n] : [_normalise(i, i + 1) for i in 1:(n - 1)]

_normalise(a, b) = (min(Int(a), Int(b)), max(Int(a), Int(b)))

# Rebuild the sheaf from the current ranks and wiring, and re-cache its null space. The
# null space depends only on the topology, not on where the target is, so it is computed
# here rather than every frame.
function _rebuild!(state::DemoState)
    n = n_agents(state)
    if n == 0
        state.directions = Vector{Float64}[]
        state.null_basis = zeros(0, 0)
        return state
    end
    previous = (state.sheaf, state.null_basis, state.directions)
    try
        tv = target_vertex(state)
        state.sheaf = build_projection_escort_ring(state.offsets, tv;
                                                   ranks=state.ranks, D=D,
                                                   consensus_edges=state.edges)
        state.directions = bisector_directions(state.offsets)
        _, null_basis = harmonic_extension(state.sheaf, Dict(tv => [0.0, 0.0, 1.0]))
        state.null_basis = _orthonormal(null_basis[1:n*D, :])
    catch err
        # A mouse can reach topologies a keyboard could not -- every edge cut and every
        # rank zeroed leaves a sheaf with no edges at all, and the factorisation of an
        # entirely zero Laplacian is a corner the solver need not be expected to enjoy.
        # Keep the session alive on the previous topology rather than dropping the socket.
        @warn "could not rebuild the sheaf; keeping the previous topology" exception = err
        state.sheaf, state.null_basis, state.directions = previous
    end
    return state
end

"""
    set_ranks!(state, ranks) -> DemoState

Rebuild the sheaf for a new per-agent rank vector.

Out-of-range entries are clamped rather than rejected: this is driven by a keyboard and a
mouse, and a stray input should not kill the session.
"""
function set_ranks!(state::DemoState, ranks::AbstractVector{<:Integer})
    wanted = [clamp(Int(r), 0, MAX_RANK) for r in ranks]
    length(wanted) == n_agents(state) || return state
    state.ranks = wanted
    return _rebuild!(state)
end

"""
    connect_agents!(state, a, b) -> DemoState

Wire two agents together with a consensus edge. A no-op if they already share one, or if
either index is out of range.
"""
function connect_agents!(state::DemoState, a::Integer, b::Integer)
    n = n_agents(state)
    (1 <= a <= n && 1 <= b <= n && a != b) || return state
    edge = _normalise(a, b)
    any(e -> _normalise(e...) == edge, state.edges) && return state
    push!(state.edges, edge)
    return _rebuild!(state)
end

"""
    disconnect_agents!(state, a, b) -> DemoState

Remove the consensus edge between two agents, if there is one.

Cutting edges is as consequential as lowering an observation rank, and shows up the same
way: a cycle survives one cut as a path, but a second cut splits the fleet into halves that
are free to drift apart, and `harmonic_extension` reports the extra freedom as null-space
columns.
"""
function disconnect_agents!(state::DemoState, a::Integer, b::Integer)
    edge = _normalise(a, b)
    index = findfirst(e -> _normalise(e...) == edge, state.edges)
    index === nothing && return state
    deleteat!(state.edges, index)
    return _rebuild!(state)
end

"""
    add_node!(state, position) -> DemoState

Place a new agent at `position`, in world coordinates. A no-op once `$MAX_AGENTS` are down.

The agent's *offset* — its share of the formation's shape — is taken relative to the
target, so the agent settles exactly where it was dropped and then travels with the target
from there. It joins the end of the cyclic order, which is what defines its angle bisector,
and starts at rank 0 with no wiring: a new agent sees nothing and talks to no one until you
say otherwise.
"""
function add_node!(state::DemoState, position::AbstractVector{<:Real})
    n_agents(state) >= MAX_AGENTS && return state
    push!(state.offsets, Float64.(position) .- state.target)
    push!(state.ranks, 0)
    push!(state.positions, Float64.(collect(position)))
    push!(state.reference, Float64.(collect(position)))
    return _rebuild!(state)
end

"""
    remove_node!(state, agent) -> DemoState

Delete an agent, along with any edges touching it.

The agents after it shift down to close the gap, so the surviving edges have to be
renumbered to match — and because the cyclic order *is* the index order, deleting a node
also re-cuts its neighbours' angle bisectors.
"""
function remove_node!(state::DemoState, agent::Integer)
    n = n_agents(state)
    (1 <= agent <= n) || return state
    for field in (state.offsets, state.ranks, state.positions, state.reference)
        deleteat!(field, agent)
    end
    kept = filter(e -> agent ∉ e, state.edges)
    state.edges = [(a > agent ? a - 1 : a, b > agent ? b - 1 : b) for (a, b) in kept]
    return _rebuild!(state)
end

"""    clear_nodes!(state) -> DemoState

Remove every agent, leaving an empty board to draw a formation onto.
"""
function clear_nodes!(state::DemoState)
    state.offsets = Vector{Float64}[]
    state.ranks = Int[]
    state.positions = Vector{Float64}[]
    state.reference = Vector{Float64}[]
    state.edges = Tuple{Int,Int}[]
    state.restore_ranks = Int[]
    return _rebuild!(state)
end

"""    reset_edges!(state) -> DemoState

Restore the ring wiring for the current agent count.
"""
function reset_edges!(state::DemoState)
    state.edges = _cycle_edges(n_agents(state))
    return _rebuild!(state)
end

"""    toggle_ghosts!(state) -> Bool

Show or hide the harmonic reference markers. Off by default — they sit on top of the agents
whenever tracking is tight, which is most of the time.
"""
toggle_ghosts!(state::DemoState) = (state.show_ghosts = !state.show_ghosts)

"""
    cycle_rank!(state, agent) -> DemoState

Step one agent's observation rank 0 → 1 → 2 → 0.
"""
function cycle_rank!(state::DemoState, agent::Integer)
    1 <= agent <= n_agents(state) || return state
    ranks = copy(state.ranks)
    ranks[agent] = (ranks[agent] + 1) % (MAX_RANK + 1)
    return set_ranks!(state, ranks)
end

"""
    toggle_all_ranks!(state) -> DemoState

Drop every agent to rank 0, or restore the ranks in force before the last such drop.

With nothing observing, the target vertex is isolated and the formation floats free — the
ring holds its shape and stops tracking, which is exactly what the reference projection in
[`step!`](@ref) arranges.
"""
function toggle_all_ranks!(state::DemoState)
    if all(iszero, state.ranks)
        restored = all(iszero, state.restore_ranks) ?
            [i % 2 == 1 ? 1 : 0 for i in 1:n_agents(state)] : state.restore_ranks
        return set_ranks!(state, restored)
    end
    state.restore_ranks = copy(state.ranks)
    return set_ranks!(state, zeros(Int, n_agents(state)))
end

"""    adjust_gain!(state, factor) -> Float64

Scale the agents' proportional gain, clamped to a sane range. Returns the new gain.
"""
function adjust_gain!(state::DemoState, factor::Real)
    state.gain = clamp(state.gain * Float64(factor), GAIN_LIMITS...)
    return state.gain
end

"""    cycle_feedforward!(state) -> Float64

Step the feedforward scale through 0 → 0.5 → 1 → 0. At 1.0 the reference's own velocity is
fed forward in full and the tracking lag vanishes; at 0 the lag is `speed / gain`.
"""
function cycle_feedforward!(state::DemoState)
    index = findfirst(≈(state.feedforward), FEEDFORWARD_STEPS)
    next = index === nothing ? 1 : index % length(FEEDFORWARD_STEPS) + 1
    state.feedforward = FEEDFORWARD_STEPS[next]
    return state.feedforward
end

"""
    reset!(state) -> DemoState

Return the target to the origin and scatter the agents onto a wider ring, so the formation
is visibly out of place and has to converge.
"""
function reset!(state::DemoState)
    state.target = zeros(2)
    state.velocity = zeros(2)
    state.command = zeros(2)
    state.time = 0.0
    state.offsets = _regular_offsets(MAX_AGENTS, state.radius)
    state.ranks = [i % 2 == 1 ? 1 : 0 for i in 1:MAX_AGENTS]
    state.restore_ranks = zeros(Int, MAX_AGENTS)
    state.edges = _cycle_edges(MAX_AGENTS)
    scatter = 1.6 * state.radius
    state.positions = [scatter .* [cos((i - 0.5) * 2π / MAX_AGENTS), sin((i - 0.5) * 2π / MAX_AGENTS)]
                       for i in 1:MAX_AGENTS]
    state.reference = deepcopy(state.positions)
    _rebuild!(state)
    empty!(state.track)
    state.recording = false
    return state
end

"""
    step!(state, dt) -> DemoState

Advance the demo by `dt` seconds.

The target integrates the current keyboard command; the agents descend onto the harmonic
reference, with the reference's own velocity fed forward in proportion to
`state.feedforward`.
"""
function step!(state::DemoState, dt::Real)
    dt = Float64(dt)
    state.time += dt

    state.velocity .+= dt .* (TARGET_ACCELERATION .* state.command .- TARGET_DAMPING .* state.velocity)
    speed = norm(state.velocity)
    speed > TARGET_MAX_SPEED && (state.velocity .*= TARGET_MAX_SPEED / speed)
    state.target .+= dt .* state.velocity
    _clamp_to_arena!(state)

    reference, rate = _harmonic_reference(state)
    state.reference = reference

    for i in 1:n_agents(state)
        command = state.gain .* (reference[i] .- state.positions[i]) .+ state.feedforward .* rate[i]
        magnitude = norm(command)
        magnitude > AGENT_MAX_SPEED && (command .*= AGENT_MAX_SPEED / magnitude)
        state.positions[i] = state.positions[i] .+ dt .* command
    end

    state.recording && push!(state.track, (state.time, state.target[1], state.target[2]))
    return state
end

# The harmonic reference and its rate. Both come from the same public entry point:
# `harmonic_extension` is linear in the boundary data for a fixed topology, so feeding it
# the target's *velocity* returns the reference's velocity. Note the homogeneous slot is
# 0.0 in the rate problem and 1.0 in the position problem -- the affine gauge is a
# constant, so it has no rate, and putting a 1.0 there would inject a spurious dilation
# into every agent's feedforward.
#
# When the observations do not determine the formation, `harmonic_extension` hands back a
# representative of the solution set plus a basis for the free directions. That
# representative is not the one we want: with every agent at rank 0 it is the formation
# collapsed onto the origin. So we slide along the free directions to the point of the
# solution set *nearest the configuration the agents already hold* -- undetermined means
# "stay where you are", not "jump somewhere arbitrary".
function _harmonic_reference(state::DemoState)
    agents = n_agents(state)
    agents == 0 && return Vector{Float64}[], Vector{Float64}[]
    n = agents * D
    tv = target_vertex(state)
    position, _ = harmonic_extension(state.sheaf,
        Dict(tv => [state.target[1], state.target[2], 1.0]))
    rate, _ = harmonic_extension(state.sheaf,
        Dict(tv => [state.velocity[1], state.velocity[2], 0.0]))
    pv, rv = Vector(position)[1:n], Vector(rate)[1:n]

    N = state.null_basis
    if size(N, 2) > 0
        current = vcat([vcat(p, 1.0) for p in state.positions]...)
        pv .+= N * (N' * (current .- pv))
        rv .-= N * (N' * rv)
    end

    reference = [pv[D*(i-1)+1:D*(i-1)+2] for i in 1:agents]
    velocity = [rv[D*(i-1)+1:D*(i-1)+2] for i in 1:agents]
    return reference, velocity
end

# `harmonic_extension`'s null basis is P⁻¹L⁻ᵀ applied to unit vectors -- a basis, but
# neither orthonormal nor canonically signed. The projection above needs it orthonormal.
function _orthonormal(basis::AbstractMatrix)
    size(basis, 2) == 0 && return zeros(size(basis, 1), 0)
    return Matrix(qr(basis).Q)[:, 1:size(basis, 2)]
end

function _clamp_to_arena!(state::DemoState)
    limit_x = ARENA_X / 2 - 1.2 * state.radius
    limit_y = ARENA_Y / 2 - 1.2 * state.radius
    for (axis, limit) in ((1, limit_x), (2, limit_y))
        if abs(state.target[axis]) > limit
            state.target[axis] = clamp(state.target[axis], -limit, limit)
            state.velocity[axis] = 0.0
        end
    end
    return state
end

"""
    toggle_recording!(state) -> Bool

Start or stop capturing the target's path. Starting clears any previous take.
"""
function toggle_recording!(state::DemoState)
    state.recording = !state.recording
    state.recording && empty!(state.track)
    return state.recording
end

"""
    save_track(state, directory) -> String

Write the recorded target path as `t,x,y` CSV and return the file path.

The format is the one `hexagon_coordination.py --track` reads in the Robotarium demo, so a
path driven here can be replayed on the robots.
"""
function save_track(state::DemoState, directory::AbstractString)
    isempty(state.track) && throw(ArgumentError("no target track has been recorded yet"))
    mkpath(directory)
    path = joinpath(directory, "target_track_$(round(Int, time())).csv")
    open(path, "w") do io
        println(io, "t,x,y")
        for (t, x, y) in state.track
            println(io, join(round.((t, x, y); digits=5), ","))
        end
    end
    return path
end

"""
    snapshot(state) -> NamedTuple

The frame sent to the browser: everything needed to draw, and nothing else.

`null_directions` is empty exactly when the observations determine the formation; each
entry is a uniform translation the formation could undergo at no Dirichlet-energy cost.
"""
function snapshot(state::DemoState)
    observers = findall(>(0), state.ranks)
    projections = [let u = state.directions[i], anchor = state.positions[i]
                       anchor .+ dot(state.target .- anchor, u) .* u
                   end for i in observers]

    return (
        t = round(state.time; digits=3),
        radius = state.radius,
        arena = (ARENA_X, ARENA_Y),
        agents = state.positions,
        reference = state.reference,
        show_ghosts = state.show_ghosts,
        target = state.target,
        edges = [collect(e) for e in state.edges],
        ranks = state.ranks,
        n_agents = n_agents(state),
        max_agents = MAX_AGENTS,
        observers = observers,
        directions = [state.directions[i] for i in observers],
        projections = projections,
        scalar_readings = sum(state.ranks),
        null_directions = _translation_bars(state),
        undetermined = size(state.null_basis, 2),
        gain = round(state.gain; digits=2),
        feedforward = state.feedforward,
        lag = isempty(state.positions) ? 0.0 :
              round(maximum(norm(state.reference[i] .- state.positions[i]) for i in 1:n_agents(state)); digits=3),
        energy = _energy(state),
        recording = state.recording,
        samples = length(state.track),
    )
end

# A bar asserts one specific thing: *this line, and no other, is free*. That is true of an
# agent whose translational freedom is one-dimensional -- the classic case of a single
# rank-1 observer, where everything may slide along the direction its lone reading cannot
# pin. It is false of an agent with no observation at all, which is free in the whole plane
# and has no distinguished direction; the null basis merely happens to pick two, and drawing
# them would dress an arbitrary choice up as structure.
#
# So: intersect the null space with the pure translations, ask each agent how many
# dimensions of that intersection it actually moves in, and draw a bar only for the agents
# whose answer is exactly one. Agents sharing a direction are drawn as a single bar through
# their common centroid.
function _translation_bars(state::DemoState)
    n = n_agents(state)
    bars = NamedTuple{(:at, :dir),Tuple{Vector{Float64},Vector{Float64}}}[]
    (n == 0 || size(state.null_basis, 2) == 0) && return bars

    N = state.null_basis
    translations = zeros(n * D, 2n)
    for i in 1:n, j in 1:2
        translations[D * (i - 1) + j, 2 * (i - 1) + j] = 1.0
    end
    # What is left of each translation after removing its component inside the null space;
    # a translation is free exactly when nothing is left over.
    outside = translations .- N * (N' * translations)
    factors = svd(outside)
    tol = 1e-7 * max(1.0, maximum(factors.S; init = 0.0))
    free = findall(<(tol), factors.S)
    isempty(free) && return bars
    coefficients = factors.V[:, free]

    groups = Tuple{Vector{Float64},Vector{Int}}[]
    for i in 1:n
        block = coefficients[2 * (i - 1) + 1 : 2i, :]
        size(block, 2) == 0 && continue
        local_factors = svd(block)
        scale = maximum(local_factors.S; init = 0.0)
        scale < 1e-9 && continue
        count(>(1e-7 * scale), local_factors.S) == 1 || continue   # 2 means the whole plane
        direction = local_factors.U[:, 1]
        matched = findfirst(g -> abs(dot(g[1], direction)) > 1 - 1e-6, groups)
        matched === nothing ? push!(groups, (direction, [i])) : push!(groups[matched][2], i)
    end

    for (direction, members) in groups
        at = sum(state.positions[i] for i in members) ./ length(members)
        push!(bars, (at = at, dir = direction))
    end
    return bars
end

# Sheaf Dirichlet energy of the *actual* agent configuration -- how far the fleet is from
# satisfying its own coordination constraints right now.
function _energy(state::DemoState)
    isempty(state.edges) && all(iszero, state.ranks) && return 0.0
    x = vcat([vcat(p, 1.0) for p in state.positions]..., vcat(state.target, 1.0))
    return round(sum(abs2, coboundary_map(state.sheaf) * x); sigdigits=4)
end

end # module HexagonDemo
