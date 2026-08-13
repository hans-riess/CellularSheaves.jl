# Interactive hexagon coordination demo: CellularSheaves.jl backend, browser front end.
#
#   julia --project=examples/hexagon_coordination examples/hexagon_coordination/server.jl
#
# then open http://localhost:8080. Drive the target with the arrow keys or WASD.
#
# The simulation loop lives entirely in Julia -- every frame solves a harmonic
# extension over the coordination sheaf. The browser only draws what it is sent
# and forwards keystrokes back, so this really is a CellularSheaves.jl backend
# rather than a reimplementation in JavaScript.
#
# Static files and the WebSocket are served on two adjacent ports (8080 and
# 8081) rather than multiplexed over one, which keeps both sides on HTTP.jl's
# simplest, most stable entry points.

using HTTP
using JSON3
using Printf

include(joinpath(@__DIR__, "src", "HexagonDemo.jl"))
using .HexagonDemo

const HTTP_PORT = parse(Int, get(ENV, "HEXAGON_HTTP_PORT", "8080"))
const WS_PORT = parse(Int, get(ENV, "HEXAGON_WS_PORT", "8081"))
const FRAME_RATE = 60.0
const WWW = joinpath(@__DIR__, "www")
const TRACKS = joinpath(@__DIR__, "tracks")

const CONTENT_TYPES = Dict(".html" => "text/html; charset=utf-8",
                           ".css" => "text/css; charset=utf-8",
                           ".js" => "text/javascript; charset=utf-8")

function serve_static(request::HTTP.Request)
    target = HTTP.URI(request.target).path
    name = (target == "/" || isempty(target)) ? "index.html" : lstrip(target, '/')
    path = normpath(joinpath(WWW, name))
    # Refuse to serve anything that escapes the www directory.
    if !startswith(path, WWW) || !isfile(path)
        return HTTP.Response(404, "not found")
    end
    mime = get(CONTENT_TYPES, lowercase(splitext(path)[2]), "application/octet-stream")
    return HTTP.Response(200, ["Content-Type" => mime], read(path))
end

# Keyboard state arrives as {"command": [x, y]} plus discrete actions. Anything
# unrecognised is ignored rather than fatal: a stray message from a reloading
# browser should not take the simulation down.
function handle_message(state::DemoState, raw)
    message = try
        JSON3.read(raw)
    catch
        return nothing
    end

    if haskey(message, :command)
        command = Float64.(collect(message.command))
        length(command) == 2 && (state.command = command)
    end

    action = get(message, :action, nothing)
    if action == "reset"
        reset!(state)
    elseif action == "observers"
        haskey(message, :observers) && set_observers!(state, Int.(collect(message.observers)))
    elseif action == "record"
        toggle_recording!(state)
    elseif action == "save"
        try
            return (event = "saved", path = save_track(state, TRACKS))
        catch err
            return (event = "error", message = sprint(showerror, err))
        end
    end
    return nothing
end

function run_session(ws)
    state = DemoState()
    dt = 1 / FRAME_RATE
    @info "browser connected"

    # Inbound messages are read on their own task so a quiet keyboard never
    # stalls the simulation, and a busy one never outruns it.
    reader = Threads.@spawn try
        for raw in ws
            reply = handle_message(state, raw)
            reply === nothing || HTTP.WebSockets.send(ws, JSON3.write(reply))
        end
    catch
        nothing
    end

    try
        while !istaskdone(reader)
            step!(state, dt)
            HTTP.WebSockets.send(ws, JSON3.write(snapshot(state)))
            sleep(dt)
        end
    catch err
        err isa HTTP.WebSockets.WebSocketError || @warn "session ended" exception = err
    end
    @info "browser disconnected"
end

function main()
    isdir(WWW) || error("missing front end directory: $WWW")
    mkpath(TRACKS)

    static = HTTP.serve!(serve_static, "127.0.0.1", HTTP_PORT)
    sockets = HTTP.WebSockets.listen!("127.0.0.1", WS_PORT) do ws
        run_session(ws)
    end

    @printf("\n  hexagon coordination demo\n")
    @printf("  open http://localhost:%d  (websocket on :%d)\n", HTTP_PORT, WS_PORT)
    @printf("  arrows/WASD drive the target, 1-6 toggle observers, R records, S saves, space resets\n")
    @printf("  ctrl-c to stop\n\n")

    try
        wait(static)
    catch err
        err isa InterruptException || rethrow()
        @info "shutting down"
    finally
        close(static)
        close(sockets)
    end
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
