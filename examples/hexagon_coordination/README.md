# Interactive coordination sheaf demo

Drive a target around with the keyboard and watch a formation hold station around
it — while only some of the agents can see it, and those that can see only
*one number* each. Draw the formation yourself: agents go where you click, and
the wiring between them is yours to cut and re-make.

```
julia --project=examples/hexagon_coordination -e 'using Pkg; Pkg.instantiate()'
julia --project=examples/hexagon_coordination examples/hexagon_coordination/server.jl
```

Then open <http://localhost:8080>.

## What it demonstrates

Each agent carries an **observation rank**, cycled live with the number keys:

| rank | what that agent knows | drawn as |
|---|---|---|
| 0 | nothing at all | bare marker |
| 1 | one number — the target projected onto the **angle bisector** at its vertex | pale bisector line, gold projection foot |
| 2 | the target's position in full (the classic escort pin) | teal sight line |

Every frame the backend calls
[`harmonic_extension`](../../src/network_sheaves/EuclideanSheaves.jl) to fuse
whatever readings exist and propagate the answer to the agents that saw nothing.

The default — agents 1, 3, 5 at rank 1 — reads three scalars 120° apart, which
fuse isotropically (`∑ uᵢuᵢᵀ = (3/2) I`) and over-determine the formation's two
translational degrees of freedom.

**The experiment worth running.** Put a single agent at rank 2 and the rest at
rank 0: two scalar readings, formation determined. Now instead put agents **1 and
4** at rank 1: also two scalar readings, but a thick bar appears through the ring.
Those vertices are diametrically opposite, so their bisectors are antiparallel and
the second reading repeats the first. What determines the formation is not how
many numbers you read but whether the directions you read along *span*.

Press `0` for no tracking at all. The ring keeps its shape and simply stops
following you — it does not collapse, because the demo resolves undetermined
directions by taking the point of the solution set nearest the configuration the
agents already hold. "Undetermined" means stay put, not jump somewhere arbitrary.

## Controls

| Key | Action |
|---|---|
| arrows / WASD | drive the target (keys accelerate; it coasts on release) |
| `1` … `6` | cycle that agent's rank: none → line → full |
| `0` | drop every agent to rank 0 (press again to restore) |
| `[` `]` | agent gain down / up |
| `F` | feedforward 0 → ½ → 1 |
| `G` | show / hide the harmonic reference `q*` (hidden by default) |
| `E` | restore the ring wiring |
| `R` | start / stop recording the target's path |
| `enter` | save the recording as CSV under `tracks/` |
| `C` | clear the board and start from nothing |
| `space` | reset |

| Mouse | Action |
|---|---|
| click empty board | drop a new agent there (up to six) |
| drag agent → agent | wire the two together with a consensus edge |
| click an edge | cut it |
| shift-click an agent | remove it |

## Drawing a formation

The six-agent hexagon is only the starting position. Agents go wherever you click, so the
formation can be a triangle, a square, or something with no symmetry at all.

That matters mathematically rather than cosmetically. The angle bisector at a vertex is the
inward *radial* direction only when the polygon is regular; on any other shape the two part
company, and what an agent observes is the genuine bisector of the angle its two neighbours
subtend. The cyclic order used for that is the order agents were created in — deliberately
**not** the consensus wiring, so a vertex has a well-defined bisector even before it has any
edges. Geometry says what shape is held; topology says who talks to whom; the two stay
independent.

## The board

The background is `static/img/header.jpeg`, served by `server.jl` from a second read-only
root at `/static/`. If that file is ever missing the page falls back to a procedural
chalkboard drawn with an inline SVG filter, so nothing is a hard dependency.

## Wiring is editable too, and it degrades the same way

The consensus edges are not fixed at a ring. Cutting them costs the formation degrees of
freedom exactly as lowering an observation rank does, and the same readout reports it:

| wiring (observers 1, 3, 5 at rank 1) | undetermined directions |
|---|---|
| full cycle | 0 |
| cycle − 1 edge (a path) | 0 — still rigid |
| cycle − 2 edges (split in half) | 1 — the halves drift apart |
| ... with every observer on one side of the cut | 2 |
| no consensus edges at all | 12 |
| no consensus edges, every agent at rank 2 | 0 — wiring is then redundant |

So the cycle turns out to be one edge more than rigidity actually needs, which you can
discover by clicking. *Do the agents see enough?* and *do they talk to enough of each
other?* are different questions, and the null space answers both at once.

## Why the ring lags

By default the ring trails you by roughly 1.4 ring radii, swings wide when you
change direction, and closes in when you slow down. Two knobs control that:

- **gain** (`[` / `]`) — the proportional term. Steady lag is `speed / gain`.
- **feedforward** (`F`) — the reference's own velocity, fed forward. At `1.0` it
  cancels the lag *exactly, at any gain*, which is why the demo shipped feeling
  glued to the target until this was turned down. At `0` you get the full
  `speed / gain` lag.

Agents also cap at 2.2 units/s against a target that tops out at 1.875, so you can
break away briefly on a sharp turn but not outrun the formation.

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
