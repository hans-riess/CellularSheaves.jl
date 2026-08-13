# Interactive hexagon coordination demo

Drive a target around with the keyboard and watch six agents hold formation
around it — while only three of them can see it, and each of those sees only
*one number*.

```
julia --project=examples/hexagon_coordination -e 'using Pkg; Pkg.instantiate()'
julia --project=examples/hexagon_coordination examples/hexagon_coordination/server.jl
```

Then open <http://localhost:8080>.

## What it demonstrates

Agents 1, 3 and 5 each observe the projection of the target onto the **angle
bisector** at their own vertex — the line through the agent making congruent
angles to its two ring neighbours. Agents 2, 4 and 6 observe nothing at all.

One scalar per observer is not enough for any single agent to locate the target.
Three of them, 120° apart, over-determine the formation's two translational
degrees of freedom, and they do so isotropically: `∑ uᵢuᵢᵀ = (3/2) I`. Every
frame the backend calls
[`harmonic_extension`](../../src/network_sheaves/EuclideanSheaves.jl) to fuse
those readings and propagate the answer to the agents that saw nothing.

Press `3` and `5` to drop to a single observer. A thick bar appears through the
ring: the translation that one scalar reading cannot pin. The formation drifts
along it while the sheaf energy stays at zero — that configuration really is a
valid solution, it just isn't the only one. `harmonic_extension` returns that
direction as a null-space basis rather than silently picking a representative.

Press `1` and `4` together (and nothing else) for the subtler version: two
observers at diametrically opposite vertices have antiparallel bisectors, so the
second reading is redundant and the formation is *still* under-determined.

## Controls

| Key | Action |
|---|---|
| arrows / WASD | drive the target (keys accelerate; it coasts on release) |
| `1` … `6` | toggle whether that agent observes the target |
| `R` | start / stop recording the target's path |
| `enter` | save the recording as CSV under `tracks/` |
| `space` | reset |

## Recording for the Robotarium

`R` then `enter` writes `tracks/target_track_<stamp>.csv` in `t,x,y` format. That
is exactly what the companion Robotarium demo replays:

```bash
python hexagon_coordination.py --track /path/to/target_track_....csv
```

so a path you drive here can be run on the robots.

## How it is put together

| File | Role |
|---|---|
| `src/HexagonDemo.jl` | all the simulation: sheaf, harmonic extension, agent dynamics, recording |
| `server.jl` | static file server (port 8080) and WebSocket loop (port 8081) |
| `www/index.html` | the whole front end — canvas renderer and key handling, no build step |

The simulation runs entirely in Julia. Every frame solves two harmonic
extensions over the coordination sheaf and ships the result to the browser,
which only draws what it is given. Nothing about the mathematics is
reimplemented in JavaScript.

The two harmonic extensions are worth a note. `harmonic_extension` is linear in
the boundary data for a fixed topology, so feeding it the target's *velocity* as
boundary data returns the *reference's* velocity, used as a feedforward term.
The homogeneous slot is `1.0` in the position problem and `0.0` in the rate
problem: the affine gauge is a constant, so it has no rate, and a `1.0` there
would inject a spurious dilation into every agent's feedforward.

`HTTP` and `JSON3` are dependencies of this example only — they are declared in
`examples/hexagon_coordination/Project.toml` and are deliberately not added to
the package's own `Project.toml`.

## The mathematics

See the docstring of `build_projection_escort_ring` in
[`src/network_sheaves/Formations.jl`](../../src/network_sheaves/Formations.jl),
and the worked example at `docs/literate/control/hexagon_bisector_tracking.jl`,
which derives the construction and reproduces these figures statically.
