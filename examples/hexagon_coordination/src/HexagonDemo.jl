"""
    HexagonDemo

Simulation state for the interactive hexagon coordination demo.

Six agents hold a regular hexagon around a target that the user drives with the
keyboard. Agents 1, 3 and 5 each observe only the projection of the target onto
the angle bisector at their own vertex; agents 2, 4 and 6 observe nothing. Every
frame, a harmonic extension over the coordination sheaf fuses those scalar
readings into a full target estimate and hands each agent a position reference.

The sheaf is built by `build_projection_escort_ring`; see its docstring for the
mathematics, including why the observation edges carry a two-dimensional stalk.

This module holds no I/O. `server.jl` drives it and streams `snapshot` out over
a WebSocket.
"""
module HexagonDemo

using CellularSheaves
using LinearAlgebra

export DemoState, step!, snapshot, set_observers!, reset!, toggle_recording!, save_track

const N_AGENTS = 6
const RADIUS = 0.55
const TARGET_VERTEX = N_AGENTS + 1
const D = 3

# The arena the browser draws, in the same units as the ring radius. Sized so
# the ring plus the target's reachable box stays comfortably inside the frame.
const ARENA_X = 4.0
const ARENA_Y = 2.5

const TARGET_ACCELERATION = 6.0     # units/s^2 while a key is held
const TARGET_DAMPING = 3.2          # 1/s, coasts to a stop on release
const TARGET_MAX_SPEED = 2.4
const AGENT_GAIN = 4.0              # first-order descent onto the reference
const AGENT_MAX_SPEED = 3.5

"""
    DemoState(; observers=[1, 3, 5], radius=$RADIUS)

Mutable state of a running demo: the sheaf, the agents, the target, and the
recording buffer.
"""
mutable struct DemoState
    radius::Float64
    observers::Vector{Int}
    sheaf::EuclideanSheaf{Float64}
    offsets::Vector{Vector{Float64}}
    directions::Vector{Vector{Float64}}
    null_basis::Matrix{Float64}
    positions::Vector{Vector{Float64}}
    reference::Vector{Vector{Float64}}
    target::Vector{Float64}
    velocity::Vector{Float64}
    command::Vector{Float64}
    time::Float64
    recording::Bool
    track::Vector{NTuple{3,Float64}}
end

function DemoState(; observers::AbstractVector{<:Integer}=[1, 3, 5], radius::Real=RADIUS)
    state = DemoState(Float64(radius), Int[], EuclideanSheaf{Float64}(fill(D, TARGET_VERTEX)),
                      Vector{Float64}[], Vector{Float64}[], zeros(0, 0),
                      Vector{Float64}[], Vector{Float64}[],
                      zeros(2), zeros(2), zeros(2), 0.0, false, NTuple{3,Float64}[])
    set_observers!(state, observers)
    reset!(state)
    return state
end

"""
    set_observers!(state, observers) -> DemoState

Rebuild the sheaf for a new observer set and recompute its null space.

The null space depends only on the topology, not on where the target is, so it
is cached here rather than recomputed every frame. An empty observer set is
ignored: with nothing pinning the homogeneous coordinate the formation could
collapse to a point at zero energy, which is a degenerate case the demo has
nothing useful to show for.
"""
function set_observers!(state::DemoState, observers::AbstractVector{<:Integer})
    wanted = sort(unique(Int.(observers)))
    filter!(i -> 1 <= i <= N_AGENTS, wanted)
    isempty(wanted) && return state

    state.observers = wanted
    state.sheaf = build_projection_escort_ring(N_AGENTS, TARGET_VERTEX, state.radius;
                                               observers=wanted, D=D)
    angles = [(i - 1) * 2π / N_AGENTS for i in 1:N_AGENTS]
    state.offsets = [state.radius .* [cos(θ), sin(θ)] for θ in angles]
    state.directions = bisector_directions(N_AGENTS)

    _, null_basis = harmonic_extension(state.sheaf, Dict(TARGET_VERTEX => [0.0, 0.0, 1.0]))
    state.null_basis = null_basis
    return state
end

"""
    reset!(state) -> DemoState

Return the target to the origin and scatter the agents onto a wider ring, so the
formation is visibly out of place and has to converge.
"""
function reset!(state::DemoState)
    state.target = zeros(2)
    state.velocity = zeros(2)
    state.command = zeros(2)
    state.time = 0.0
    scatter = 1.6 * state.radius
    state.positions = [scatter .* [cos((i - 0.5) * 2π / N_AGENTS), sin((i - 0.5) * 2π / N_AGENTS)]
                       for i in 1:N_AGENTS]
    state.reference = deepcopy(state.positions)
    empty!(state.track)
    state.recording = false
    return state
end

"""
    step!(state, dt) -> DemoState

Advance the demo by `dt` seconds.

The target integrates the current keyboard command; the agents descend onto the
harmonic reference with a feedforward term for the reference's own velocity.
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

    for i in 1:N_AGENTS
        command = AGENT_GAIN .* (reference[i] .- state.positions[i]) .+ rate[i]
        magnitude = norm(command)
        magnitude > AGENT_MAX_SPEED && (command .*= AGENT_MAX_SPEED / magnitude)
        state.positions[i] = state.positions[i] .+ dt .* command
    end

    state.recording && push!(state.track, (state.time, state.target[1], state.target[2]))
    return state
end

# The harmonic reference and its rate. Both come from the same public entry
# point: `harmonic_extension` is linear in the boundary data for a fixed
# topology, so feeding it the target's *velocity* as boundary data returns the
# reference's velocity. Note the homogeneous slot is 0.0 in the rate problem and
# 1.0 in the position problem -- the affine gauge is a constant, so it has no
# rate, and putting a 1.0 there would inject a spurious dilation into every
# agent's feedforward.
function _harmonic_reference(state::DemoState)
    position, _ = harmonic_extension(state.sheaf,
        Dict(TARGET_VERTEX => [state.target[1], state.target[2], 1.0]))
    rate, _ = harmonic_extension(state.sheaf,
        Dict(TARGET_VERTEX => [state.velocity[1], state.velocity[2], 0.0]))
    pv, rv = Vector(position), Vector(rate)
    reference = [pv[D*(i-1)+1:D*(i-1)+2] for i in 1:N_AGENTS]
    velocity = [rv[D*(i-1)+1:D*(i-1)+2] for i in 1:N_AGENTS]
    return reference, velocity
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

The format is the one `hexagon_coordination.py --track` reads in the Robotarium
demo, so a path driven here can be replayed on the robots.
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

`null_directions` is empty exactly when the observations determine the
formation; each entry is a uniform translation the formation could undergo at no
Dirichlet-energy cost.
"""
function snapshot(state::DemoState)
    projections = [let u = state.directions[i], anchor = state.positions[i]
                       anchor .+ dot(state.target .- anchor, u) .* u
                   end for i in state.observers]

    return (
        t = round(state.time; digits=3),
        radius = state.radius,
        arena = (ARENA_X, ARENA_Y),
        agents = state.positions,
        reference = state.reference,
        target = state.target,
        observers = state.observers,
        directions = [state.directions[i] for i in state.observers],
        projections = projections,
        null_directions = [state.null_basis[1:2, k] for k in 1:size(state.null_basis, 2)],
        energy = _energy(state),
        recording = state.recording,
        samples = length(state.track),
    )
end

# Sheaf Dirichlet energy of the *actual* agent configuration -- how far the
# fleet is from satisfying its own coordination constraints right now.
function _energy(state::DemoState)
    x = vcat([vcat(p, 1.0) for p in state.positions]..., vcat(state.target, 1.0))
    return round(sum(abs2, coboundary_map(state.sheaf) * x); sigdigits=4)
end

end # module HexagonDemo
