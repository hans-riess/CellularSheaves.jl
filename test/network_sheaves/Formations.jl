using Test
using CellularSheaves
using CellularSheaves.Formations
using CellularSheaves: fiber_section_basis
using Graphs
using LinearAlgebra

@testset "Formations" begin
    @testset "SE(3) Matrices" begin
        # Translation
        d = [1.0, 2.0, 3.0]
        Td = se3_translation_matrix(d)
        @test size(Td) == (4, 4)
        @test Td[1:3, 1:3] == I(3)
        @test Td[1:3, 4] == -d
        @test Td[4, 4] == 1.0

        # Rotation
        θz = π/2
        Rz = se3_rotation_matrix(; θz=θz)
        @test Rz[1, 1] ≈ 0.0 atol=1e-10
        @test Rz[1, 2] ≈ -1.0
        @test Rz[2, 1] ≈ 1.0
        @test Rz[2, 2] ≈ 0.0 atol=1e-10
        
        # Rotations by angles
        R_angles = se3_rotation_matrix(; θz=θz)
        @test Rz ≈ R_angles
        
        # Affine
        T_aff = se3_affine_matrix(Rz[1:3, 1:3], d)
        @test T_aff[1:3, 1:3] ≈ Rz[1:3, 1:3]
        @test T_aff[1:3, 4] == -d
    end

    @testset "Escort Ring Builder" begin
        sheaf = build_escort_ring(6, 7, 1.2; observers=[1, 3])

        @test length(sheaf.vertex_stalks) == 7
        @test sheaf.vertex_stalks[1] == 4

        # Edges: 6 ring edges + 2 pinning edges = 8 edges
        @test length(sheaf.edge_stalks) == 8
    end

    @testset "affine_translation_matrix (arbitrary dimension)" begin
        d2 = [1.0, -2.0]
        T2 = affine_translation_matrix(d2)
        @test size(T2) == (3, 3)
        @test T2[1:2, 1:2] == I(2)
        @test T2[1:2, 3] == -d2
        @test T2[3, 3] == 1.0

        # n=3 case matches se3_translation_matrix
        d3 = [1.0, 2.0, 3.0]
        @test affine_translation_matrix(d3) == se3_translation_matrix(d3)
    end

    @testset "build_escort_ring generalized D/affine" begin
        # D=3, affine=true: 2D translation ring (e.g. planar agents)
        sheaf_planar = build_escort_ring(4, 5, 0.5; D=3, affine=true)
        @test length(sheaf_planar.vertex_stalks) == 5
        @test sheaf_planar.vertex_stalks[1] == 3

        # D=6, affine=true: higher-dimensional homogeneous stalks now succeed
        sheaf_hi = build_escort_ring(4, 5, 0.5; D=6, affine=true)
        @test sheaf_hi.vertex_stalks[1] == 6

        # affine=false: non-affine stalks can only represent a zero-radius (consensus) ring
        sheaf_consensus = build_escort_ring(4, 5, 0.0; D=2, affine=false)
        @test sheaf_consensus.vertex_stalks[1] == 2

        @test_throws Exception build_escort_ring(4, 5, 0.5; D=2, affine=false)

        # observers out of range are rejected
        @test_throws Exception build_escort_ring(4, 5, 0.5; observers=[1, 5])
    end

    @testset "build_escort_topology — edge counts per kind" begin
        n = 5
        for (kind, n_consensus) in [(:ring, 5), (:path, 4), (:star, 4), (:clique, 10)]
            s = build_escort_topology(kind, n, n+1, 0.3; observers=[1])
            @test ne(s.underlying_graph) == n_consensus + 1   # +1 observer edge
        end
    end

    @testset "build_escort_topology — every kind is rigid (section space is D-dimensional)" begin
        for kind in (:ring, :path, :star, :clique)
            s = build_escort_topology(kind, 4, 5, 0.3; observers=[1])
            B = fiber_section_basis(s, collect(1:5),
                                    [(src(e), dst(e)) for e in edges(s.underlying_graph)])
            @test size(B, 2) == 4
        end
    end

    @testset "build_escort_topology — ring wrapper reproduces build_escort_ring" begin
        a = build_escort_topology(:ring, 6, 7, 0.3; observers=[1])
        b = build_escort_ring(6, 7, 0.3; observers=[1])
        @test a == b
    end

    @testset "build_escort_topology — clique honours D and affine" begin
        s = build_escort_topology(:clique, 3, 4, 0.0; D=3, affine=false)
        @test all(==(3), s.vertex_stalks)
        @test_throws Exception build_escort_topology(:clique, 3, 4, 0.5; affine=false)
    end

    @testset "build_escort_topology — rejects unknown kind" begin
        @test_throws Exception build_escort_topology(:hexagon, 4, 5, 0.3)
    end

    @testset "bisector_directions" begin
        u = bisector_directions(6)
        @test length(u) == 6
        @test all(v -> length(v) == 2, u)
        @test all(v -> norm(v) ≈ 1.0, u)
        # Bisector at vertex 1 (nominal offset along +x) points back toward the centre.
        @test u[1] ≈ [-1.0, 0.0]
        # Diametrically opposite vertices have antiparallel bisectors.
        @test u[1] ≈ -u[4]
        # Isotropy of the every-other-agent observer set: ∑ uᵢuᵢᵀ = (n/4) I.
        @test sum(v * v' for v in u[1:2:6]) ≈ 1.5 * I(2)
        # Radius-independent, and padded into higher translation dimensions.
        @test bisector_directions(6; trans_dim=3)[3][1:2] ≈ u[3]
        @test bisector_directions(6; trans_dim=3)[3][3] == 0.0
        @test_throws Exception bisector_directions(2)
    end

    @testset "bisector_directions — arbitrary formations" begin
        polygon(n, r) = [r .* [cos(2π * (i - 1) / n), sin(2π * (i - 1) / n)] for i in 1:n]

        # The whole point: on a regular polygon the angle bisector *is* the inward radial
        # direction, so the general method must reproduce the closed-form one exactly.
        for n in (3, 4, 5, 6, 8)
            general = bisector_directions(polygon(n, 0.55))
            radial = bisector_directions(n)
            @test maximum(norm(general[i] - radial[i]) for i in 1:n) < 1e-12
        end
        # ... and independent of scale.
        @test all(bisector_directions(polygon(6, 7.25)) .≈ bisector_directions(polygon(6, 0.3)))

        # On an irregular shape the two genuinely part company, which is what makes free
        # placement worth supporting at all.
        irregular = [[0.8, 0.0], [0.2, 0.6], [-0.7, 0.3], [-0.3, -0.7]]
        bisectors = bisector_directions(irregular)
        @test all(v -> norm(v) ≈ 1.0, bisectors)
        @test maximum(norm(bisectors[i] + normalize(irregular[i])) for i in 1:4) > 1e-2

        # Each bisector really does make congruent angles with its two neighbouring rays.
        for i in 1:4
            back = normalize(irregular[mod1(i - 1, 4)] - irregular[i])
            forth = normalize(irregular[mod1(i + 1, 4)] - irregular[i])
            @test dot(bisectors[i], back) ≈ dot(bisectors[i], forth) atol=1e-12
        end

        # Degenerate vertices fall back to pointing at the centroid.
        straight = [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, -1.0]]
        @test bisector_directions(straight)[2] ≈ [0.0, -1.0] atol=1e-12
        coincident = [[0.5, 0.0], [0.5, 0.0], [-0.5, 0.4], [-0.5, -0.4]]
        @test norm(bisector_directions(coincident)[1]) ≈ 1.0
        # Fewer than three vertices have no interior angle at all.
        @test bisector_directions([[1.0, 0.0], [-1.0, 0.0]]) ≈ [[-1.0, 0.0], [1.0, 0.0]]
    end

    @testset "build_projection_escort_ring — explicit offsets" begin
        R = 0.35
        boundary = Dict(7 => [0.55, -0.30, 1.0])
        polygon(n, r) = [r .* [cos(2π * (i - 1) / n), sin(2π * (i - 1) / n)] for i in 1:n]

        # The regular-polygon method is a wrapper over the explicit-offset one.
        @test build_projection_escort_ring(6, 7, R) ==
              build_projection_escort_ring(polygon(6, R), 7)

        # An irregular formation still recovers its own shape exactly.
        shape = [[0.4, 0.05], [0.1, 0.42], [-0.35, 0.2], [-0.2, -0.38], [0.25, -0.3]]
        sheaf = build_projection_escort_ring(shape, 6; observers=[1, 3, 5])
        target = [0.55, -0.30]
        x, null_basis = harmonic_extension(sheaf, Dict(6 => vcat(target, 1.0)))
        @test size(null_basis, 2) == 0
        xv = Vector(x)
        for i in 1:5
            @test xv[3(i-1)+1:3(i-1)+2] ≈ target .+ shape[i] atol=1e-10
        end

        # Fewer than three agents is allowed; the default wiring is then a path, not a cycle.
        pair = build_projection_escort_ring([[0.3, 0.0], [-0.3, 0.0]], 3; ranks=[2, 0])
        @test ne(pair.underlying_graph) == 2
        @test_throws Exception build_projection_escort_ring([[0.3, 0.0]], 1)
        @test_throws Exception build_projection_escort_ring([[0.3, 0.0, 0.0]], 2; D=3)
    end

    @testset "build_projection_escort_ring — structure" begin
        R = 0.35
        s = build_projection_escort_ring(6, 7, R)
        @test s.vertex_stalks == fill(3, 7)
        # Six ring edges (3-dimensional stalks) + three observation edges (2-dimensional).
        @test ne(s.underlying_graph) == 9
        @test sort(collect(values(s.edge_stalks))) == [2, 2, 2, 3, 3, 3, 3, 3, 3]
        for i in (1, 3, 5)
            @test s.edge_stalks[UnorderedPair(i, 7)] == 2
            # The gauge row: it is what propagates h = 1 from the target into the ring.
            @test s.restriction_maps[i=>7][2, :] == [0.0, 0.0, 1.0]
            @test s.restriction_maps[7=>i][2, :] == [0.0, 0.0, 1.0]
        end
        @test !has_edge(s.underlying_graph, 2, 7)
        @test !has_edge(s.underlying_graph, 4, 7)
        @test !has_edge(s.underlying_graph, 6, 7)

        @test_throws Exception build_projection_escort_ring(2, 3, 0.35)
        @test_throws Exception build_projection_escort_ring(6, 7, 0.0)
        @test_throws Exception build_projection_escort_ring(6, 7, 0.35; D=2)
        @test_throws Exception build_projection_escort_ring(6, 7, 0.35; observers=[7])
        @test_throws Exception build_projection_escort_ring(6, 3, 0.35)
        # Rank 2 is full observation when D = 3; rank 3 has no meaning there.
        @test_throws Exception build_projection_escort_ring(6, 7, 0.35; ranks=[3, 0, 0, 0, 0, 0])
        @test_throws Exception build_projection_escort_ring(6, 7, 0.35; ranks=[1, 0, 1])
        @test_throws Exception build_projection_escort_ring(6, 7, 0.35; observers=[1], ranks=zeros(Int, 6))
    end

    @testset "build_projection_escort_ring — observation ranks" begin
        R = 0.35
        boundary = Dict(7 => [0.55, -0.30, 1.0])
        nullity(s) = size(harmonic_extension(s, boundary)[2], 2)

        # `observers` is shorthand for "rank 1 here, rank 0 elsewhere".
        @test build_projection_escort_ring(6, 7, R; observers=[1, 3, 5]) ==
              build_projection_escort_ring(6, 7, R; ranks=[1, 0, 1, 0, 1, 0])

        # Rank 0 removes the edge; rank 1 gives a 2-dimensional stalk; rank 2 (= D-1) gives 3.
        mixed = build_projection_escort_ring(6, 7, R; ranks=[2, 0, 1, 0, 1, 0])
        @test !has_edge(mixed.underlying_graph, 2, 7)
        @test mixed.edge_stalks[UnorderedPair(1, 7)] == 3
        @test mixed.edge_stalks[UnorderedPair(3, 7)] == 2
        # Every observation edge keeps a gauge row, at every rank.
        @test mixed.restriction_maps[1=>7][end, :] == [0.0, 0.0, 1.0]
        @test mixed.restriction_maps[3=>7][end, :] == [0.0, 0.0, 1.0]

        # Rank D-1 is exactly the identity pin `build_escort_ring` already used.
        full = build_projection_escort_ring(6, 7, R; ranks=fill(2, 6))
        escort = build_escort_ring(6, 7, R; D=3)
        @test full == escort

        # Information is a span, not a count: two scalars determine the formation when read
        # along independent directions and fail to when read along antiparallel ones.
        @test nullity(build_projection_escort_ring(6, 7, R; ranks=[2, 0, 0, 0, 0, 0])) == 0
        @test nullity(build_projection_escort_ring(6, 7, R; ranks=[1, 0, 0, 1, 0, 0])) == 1
        @test nullity(build_projection_escort_ring(6, 7, R; ranks=[1, 0, 0, 0, 0, 0])) == 1

        # Mixed ranks still recover the exact escort formation.
        d = [R .* [cos(2π * (i - 1) / 6), sin(2π * (i - 1) / 6)] for i in 1:6]
        x, nb = harmonic_extension(mixed, boundary)
        @test size(nb, 2) == 0
        for i in 1:6
            @test Vector(x)[3(i-1)+1:3(i-1)+2] ≈ [0.55, -0.30] .+ d[i] atol=1e-10
        end
    end

    @testset "build_projection_escort_ring — consensus edges are configurable" begin
        R = 0.35
        boundary = Dict(7 => [0.55, -0.30, 1.0])
        cycle = [(i, i % 6 + 1) for i in 1:6]
        nullity(edges, ranks) = size(harmonic_extension(
            build_projection_escort_ring(6, 7, R; ranks=ranks, consensus_edges=edges), boundary)[2], 2)

        # The default is the cycle.
        @test build_projection_escort_ring(6, 7, R) ==
              build_projection_escort_ring(6, 7, R; consensus_edges=cycle)
        # Orientation is irrelevant — the edges are unordered.
        @test build_projection_escort_ring(6, 7, R; consensus_edges=reverse.(cycle)) ==
              build_projection_escort_ring(6, 7, R; consensus_edges=cycle)

        observers = [1, 0, 1, 0, 1, 0]
        # A cycle is more wiring than rigidity needs: cutting one edge leaves a path, which
        # still fixes the shape. Cutting a second splits the fleet in two, and the halves
        # are then free to drift apart.
        @test nullity(cycle, observers) == 0
        @test nullity(filter(!=((1, 2)), cycle), observers) == 0
        @test nullity(filter(e -> e ∉ ((1, 2), (4, 5)), cycle), observers) == 1
        # ... and worse when the observers all sit on one side of the cut.
        @test nullity(filter(e -> e ∉ ((1, 2), (4, 5)), cycle), [1, 0, 1, 0, 0, 0]) == 2

        # With no wiring at all each agent stands alone: the three rank-1 agents keep one
        # free direction apiece, and the three that observe nothing are entirely free.
        @test nullity(Tuple{Int,Int}[], observers) == 3 * 1 + 3 * 3
        # Unless every agent sees the target for itself, in which case wiring is redundant.
        @test nullity(Tuple{Int,Int}[], fill(2, 6)) == 0

        @test_throws Exception build_projection_escort_ring(6, 7, R; consensus_edges=[(1, 1)])
        @test_throws Exception build_projection_escort_ring(6, 7, R; consensus_edges=[(1, 2), (2, 1)])
        @test_throws Exception build_projection_escort_ring(6, 7, R; consensus_edges=[(1, 7)])
    end

    @testset "build_projection_escort_ring — every agent at rank 0" begin
        R = 0.35
        # A ring that observes nothing is a legitimate sheaf, not an error: the target
        # vertex is simply isolated and the formation floats free.
        s = build_projection_escort_ring(6, 7, R; ranks=zeros(Int, 6))
        @test ne(s.underlying_graph) == 6
        @test degree(s.underlying_graph, 7) == 0

        x, null_basis = harmonic_extension(s, Dict(7 => [0.55, -0.30, 1.0]))
        # Two translations plus the dilation the target is no longer there to pin.
        @test size(null_basis, 2) == 3
        # The returned representative is the collapsed formation, which is why a caller
        # wanting a usable configuration must pick a different point of the solution set.
        @test norm(Vector(x)[1:18]) < 1e-10

        # The exact escort formation is nonetheless in the solution set.
        d = [R .* [cos(2π * (i - 1) / 6), sin(2π * (i - 1) / 6)] for i in 1:6]
        exact = vcat([vcat([0.55, -0.30] .+ d[i], 1.0) for i in 1:6]...)
        N = Matrix(qr(null_basis[1:18, :]).Q)[:, 1:3]
        @test norm(exact - N * (N' * exact)) < 1e-9
    end

    @testset "build_projection_escort_ring — the escort formation is a zero-energy section" begin
        R = 0.35
        s = build_projection_escort_ring(6, 7, R)
        d = [R .* [cos(2π * (i - 1) / 6), sin(2π * (i - 1) / 6)] for i in 1:6]
        δ = coboundary_map(s)
        for p in ([0.0, 0.0], [0.4, -0.25], [-0.9, 0.6])
            x = vcat([vcat(p .+ d[i], 1.0) for i in 1:6]..., vcat(p, 1.0))
            @test norm(δ * x) < 1e-12
        end
    end

    @testset "build_projection_escort_ring — three observers determine the formation" begin
        R = 0.35
        s = build_projection_escort_ring(6, 7, R)
        d = [R .* [cos(2π * (i - 1) / 6), sin(2π * (i - 1) / 6)] for i in 1:6]
        for p in ([0.0, 0.0], [0.4, -0.25], [-0.9, 0.6])
            x, null_basis = harmonic_extension(s, Dict(7 => vcat(p, 1.0)))
            @test size(null_basis, 2) == 0
            xv = Vector(x)
            for i in 1:6
                # Agents 2, 4 and 6 touch no observation edge; reaching their correct
                # position is the harmonic extension propagating the fused estimate.
                @test xv[3(i-1)+1:3(i-1)+2] ≈ p .+ d[i] atol=1e-10
                @test xv[3i] ≈ 1.0 atol=1e-10
            end
        end
    end

    @testset "build_projection_escort_ring — a single observer leaves one direction undetermined" begin
        R = 0.35
        p = [0.4, -0.25]
        s = build_projection_escort_ring(6, 7, R; observers=[3])
        d = [R .* [cos(2π * (i - 1) / 6), sin(2π * (i - 1) / 6)] for i in 1:6]
        x, null_basis = harmonic_extension(s, Dict(7 => vcat(p, 1.0)))

        @test size(null_basis, 2) == 1
        # `null_basis` spans the full cochain space and is zero on the pinned target
        # block; the agent degrees of freedom are the leading 18 entries.
        @test norm(null_basis[19:21, 1]) < 1e-12
        w = null_basis[1:18, 1]
        # The undetermined direction is a *uniform* translation of the whole ring,
        # orthogonal to agent 3's bisector, with no dilation component.
        u3 = bisector_directions(6)[3]
        head = w[1:2]
        for i in 1:6
            @test w[3(i-1)+1:3(i-1)+2] ≈ head atol=1e-10
            @test abs(w[3i]) < 1e-10
        end
        @test abs(dot(normalize(head), u3)) < 1e-10

        # `harmonic_extension` returns *a* representative of the solution set, not the
        # minimum-norm one, so test membership rather than the specific vector: the exact
        # escort formation differs from `x` by a multiple of the null direction.
        exact = vcat([vcat(p .+ d[i], 1.0) for i in 1:6]...)
        residual = exact - Vector(x)[1:18]
        @test norm(residual - w * (dot(w, residual) / dot(w, w))) < 1e-9
    end

    @testset "build_projection_escort_ring — opposite observers are rank-deficient too" begin
        R = 0.35
        boundary = Dict(7 => [0.4, -0.25, 1.0])
        # Agents 1 and 4 sit diametrically opposite, so their bisectors are antiparallel
        # and two observers buy no more than one.
        _, opposite = harmonic_extension(build_projection_escort_ring(6, 7, R; observers=[1, 4]), boundary)
        @test size(opposite, 2) == 1
        # Any two non-opposite observers do determine the formation.
        for obs in ([1, 3], [3, 5], [1, 2])
            _, nb = harmonic_extension(build_projection_escort_ring(6, 7, R; observers=obs), boundary)
            @test size(nb, 2) == 0
        end
    end
end
