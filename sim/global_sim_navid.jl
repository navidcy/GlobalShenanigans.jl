using Statistics
using JLD2
using Printf
using CUDA
using CairoMakie
using GeoMakie
using Oceananigans

using Oceananigans.Units
using Oceananigans.Fields: interpolate, Field
using Oceananigans.Architectures: arch_array
using Oceananigans.BoundaryConditions
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom, inactive_node, peripheral_node
using Oceananigans.TurbulenceClosures: RiBasedVerticalDiffusivity
using CUDA: @allowscalar, device!
using Oceananigans.Operators
using Oceananigans.Operators: Δzᵃᵃᶜ
using Oceananigans: prognostic_fields
using Oceananigans.Advection: EnergyConservingScheme, VelocityStencil, VorticityStencil
using Oceananigans.TurbulenceClosures: FluxTapering
using SeawaterPolynomials.TEOS10
using Oceananigans.Advection: VelocityStencil
using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: CATKEVerticalDiffusivity,
                                                                  MixingLength


global_filepath = "/storage1/uq-global/GlobalShenanigans.jl/"
input_filepath  = joinpath(@__DIR__, "..", "input/")
output_filepath = joinpath(@__DIR__, "..", "output/")

load_initial_condition = false
ic_filepath = global_filepath * "smooth_ic_7.jld2" # start from 7
qs_filepath = global_filepath * "smooth_ic_10.jld2" # 6 is the last one with ecco forcing

#####
##### Grid
#####

arch = CPU()
reference_density = 1029.0 # kg/m^3

latitude = (-75, 75)

# 1 degree resolution
Nx = 360
Ny = 150
Nz = 48

const Nyears = 40.0

const Nmonths = 12
const thirty_days = 30days

output_prefix = output_filepath * "multithreaded_near_global_lat_lon_$(Nx)_$(Ny)_$(Nz)"

pickup_file = false

# Stretched faces taken from ECCO Version 4 (50 levels in the vertical)
z_faces = jldopen(input_filepath * "zgrid.jld2")["z"][5:end-4]

# An almost-global spherical domain
@show underlying_grid = LatitudeLongitudeGrid(arch;
                                              size = (Nx, Ny, Nz),
                                              longitude = (-180, 180),
                                              latitude,
                                              halo = (5, 5, 5),
                                              z = z_faces,
                                              precompute_metrics = true)

bathymetry = jldopen(input_filepath * "bathymetry-360x150-latitude-75.0.jld2")["bathymetry"]

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bathymetry))

λc, φc, zc = grid.λᶜᵃᵃ[1:grid.Nx], grid.φᵃᶜᵃ[1:grid.Ny], grid.zᵃᵃᶜ[1:grid.Nz]

# plot bathymetry
@info "Plotting input fields..."
bathymetry_for_plot = Array(bathymetry)
bathymetry_for_plot[bathymetry .== 100] .= NaN

fig = Figure()
ga = GeoAxis(fig[1, 1]; dest = "+proj=eqearth", title = "bathymetry", coastlines = true)
hm = heatmap!(ga, λc, φc, bathymetry_for_plot; colorrange = (-9000, 0), nan_color=:black)
Colorbar(fig[1, 2], hm)
save(output_filepath * "bathymetry.png", fig)

# Files contain 1 year (1992) of 12 monthly averages; pick up one month
month = 3 # March
τˣ = jldopen(input_filepath * "boundary_conditions-1degree.jld2")["τˣ"][:, :, month] ./ reference_density
τʸ = jldopen(input_filepath * "boundary_conditions-1degree.jld2")["τˣ"][:, :, month] ./ reference_density
T★ = jldopen(input_filepath * "boundary_conditions-1degree.jld2")["Tˢ"][:, :, month]
S★ = jldopen(input_filepath * "boundary_conditions-1degree.jld2")["Sˢ"][:, :, month]

τˣ_for_plot = Array(τˣ[:, :, 1])
τˣ_for_plot[bathymetry .== 100] .= NaN

fig = Figure()
ga = GeoAxis(fig[1, 1]; dest = "+proj=eqearth", title = "zonal wind stress", coastlines = true)
hm = heatmap!(ga, λc, φc, τˣ_for_plot;
              colormap = :balance,
              colorrange = (-2e-4, 2e-4),
              nan_color=:black)
Colorbar(fig[1, 2], hm)
save(output_filepath * "taux_month$(month)_ecco.png", fig)

T_for_plot = Array(T★[:, :, 1])
T_for_plot[bathymetry .== 100] .= NaN

fig = Figure()
ga = GeoAxis(fig[1, 1]; dest = "+proj=eqearth", title = "SST", coastlines = true)
hm = heatmap!(λc, φc, T_for_plot; nan_color=:black)
Colorbar(fig[1, 2], hm)
save(output_filepath * "temperature_month$(month)_ecco.png", fig)

S_for_plot = Array(S★[:, :, 1])
S_for_plot[bathymetry .== 100] .= NaN

fig = Figure()
ga = GeoAxis(fig[1, 1]; dest = "+proj=eqearth", title = "SSS", coastlines = true)
hm = heatmap!(λc, φc, S_for_plot; nan_color=:black)
Colorbar(fig[1, 2], hm)
save(output_filepath * "salinity_month$(month)_ecco.png", fig)

# Remember the convention!! On the surface a negative flux increases a positive decreases
bathymetry = arch_array(arch, bathymetry)
τˣ = arch_array(arch, -τˣ)
τʸ = arch_array(arch, -τʸ)

τˣ_for_plot = Array(τˣ[:, :, 1])
τˣ_for_plot[bathymetry .== 100] .= NaN

fig = Figure()
ga = GeoAxis(fig[1, 1]; dest = "+proj=eqearth", title = "zonal wind stress", coastlines = true)
hm = heatmap!(ga, λc, φc, τˣ_for_plot;
              colormap = :balance,
              colorrange = (-2e-4, 2e-4),
              nan_color=:black)
Colorbar(fig[1, 2], hm)
save(output_filepath * "taux_month$(month)_used_bc.png", fig)

@info "Plotting completed!"

target_sea_surface_temperature = T★ = arch_array(arch, T★)
target_sea_surface_salinity = S★ = arch_array(arch, S★)


#####
##### Physics and model setup
#####

νz = 1e-5# 1e-5
κz = 1e-5# 1e-5

κ_skew = 900.0     # [m² s⁻¹] skew diffusivity
κ_symmetric = 900.0 # [m² s⁻¹] symmetric diffusivity

using Oceananigans.Operators: Δx, Δy
using Oceananigans.TurbulenceClosures
using Oceananigans.TurbulenceClosures: ExplicitTimeDiscretization

@inline νhb(i, j, k, grid, lx, ly, lz, clock, fields) = (1 / (1 / Δx(i, j, k, grid, lx, ly, lz)^2 + 1 / Δy(i, j, k, grid, lx, ly, lz)^2))^2 / 5days

# vertical_diffusivity = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν=νz, κ=κz)
# convective_adjustment = CATKEVerticalDiffusivity()
# horizontal_diffusivity = HorizontalScalarDiffusivity(ν=1e4)

convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(convective_κz=1e-1, background_κz=1e-5, convective_νz=1e-4, background_νz=1e-5)
biharmonic_viscosity = HorizontalScalarBiharmonicDiffusivity(ν=νhb, discrete_form=true)
gent_mcwilliams_diffusivity = IsopycnalSkewSymmetricDiffusivity(; κ_skew, κ_symmetric, slope_limiter=FluxTapering(1e-2))

closures = (biharmonic_viscosity, convective_adjustment, gent_mcwilliams_diffusivity)

#####
##### Boundary conditions / time-dependent fluxes 
#####

u_wind_stress_bc = FluxBoundaryCondition(τˣ)
v_wind_stress_bc = FluxBoundaryCondition(τʸ)

# Linear bottom drag:
μ = 0.001 # ms⁻¹

@inline u_bottom_drag(i, j, grid, clock, fields, μ) = @inbounds -μ * fields.u[i, j, 1]
@inline v_bottom_drag(i, j, grid, clock, fields, μ) = @inbounds -μ * fields.v[i, j, 1]

@inline is_immersed_drag_u(i, j, k, grid) = Int(peripheral_node(Face(), Center(), Center(), i, j, k - 1, grid) & !inactive_node(Face(), Center(), Center(), i, j, k, grid))
@inline is_immersed_drag_v(i, j, k, grid) = Int(peripheral_node(Center(), Face(), Center(), i, j, k - 1, grid) & !inactive_node(Center(), Face(), Center(), i, j, k, grid))

# Keep a constant linear drag parameter independent on vertical level
@inline u_immersed_bottom_drag(i, j, k, grid, clock, fields, μ) = @inbounds -μ * is_immersed_drag_u(i, j, k, grid) * fields.u[i, j, k] / Δzᵃᵃᶜ(i, j, k, grid)
@inline v_immersed_bottom_drag(i, j, k, grid, clock, fields, μ) = @inbounds -μ * is_immersed_drag_v(i, j, k, grid) * fields.v[i, j, k] / Δzᵃᵃᶜ(i, j, k, grid)

Fu = Forcing(u_immersed_bottom_drag, discrete_form=true, parameters=μ)
Fv = Forcing(v_immersed_bottom_drag, discrete_form=true, parameters=μ)

u_bottom_drag_bc = FluxBoundaryCondition(u_bottom_drag, discrete_form=true, parameters=μ)
v_bottom_drag_bc = FluxBoundaryCondition(v_bottom_drag, discrete_form=true, parameters=μ)

@inline function surface_temperature_relaxation(i, j, grid, clock, fields, p)
    @inbounds T_surface = fields.T[i, j, grid.Nz]
    
    return p.λ * (T_surface - p.T★)
end

@inline function surface_salinity_relaxation(i, j, grid, clock, fields, p)
    @inbounds S_surface = fields.S[i, j, grid.Nz]
    
    return p.λ * (S_surface - p.S★)
end

Δz_top = @allowscalar Δzᵃᵃᶜ(1, 1, grid.Nz, grid.underlying_grid)
Δz_bottom = @allowscalar Δzᵃᵃᶜ(1, 1, 1, grid.underlying_grid)

T_surface_relaxation_bc = FluxBoundaryCondition(surface_temperature_relaxation,
    discrete_form=true,
    parameters=(λ=Δz_top / 28days, T★=target_sea_surface_temperature))

S_surface_relaxation_bc = FluxBoundaryCondition(surface_salinity_relaxation,
    discrete_form=true,
    parameters=(λ=Δz_top / 28days, S★=target_sea_surface_salinity))

u_bcs = FieldBoundaryConditions(top = u_wind_stress_bc, bottom = u_bottom_drag_bc)
v_bcs = FieldBoundaryConditions(top = v_wind_stress_bc, bottom = v_bottom_drag_bc)
T_bcs = FieldBoundaryConditions(top = T_surface_relaxation_bc)
S_bcs = FieldBoundaryConditions(top = S_surface_relaxation_bc)

boundary_conditions = (u = u_bcs, v = v_bcs, T = T_bcs, S = S_bcs)

free_surface = ImplicitFreeSurface()
# free_surface = ImplicitFreeSurface(preconditioner_method = :SparseInverse, preconditioner_settings = (ε = 0.01, nzrel = 10))

# eos = LinearEquationOfState()
eos = TEOS10EquationOfState()
buoyancy = SeawaterBuoyancy(equation_of_state = eos)

forcings = (u = Fu, v = Fv)

model = HydrostaticFreeSurfaceModel(; grid, free_surface,
                                    momentum_advection = WENO(grid = underlying_grid, vector_invariant=VelocityStencil()),
                                    coriolis = HydrostaticSphericalCoriolis(scheme = EnergyConservingScheme()),
                                    buoyancy,
                                    tracers = (:T, :S),
                                    closure = closures,
                                    boundary_conditions,
                                    forcing = forcings,
                                    tracer_advection = WENO(; grid = underlying_grid)
                                    )

#####
##### Initial condition:
#####

u, v, w = model.velocities
η = model.free_surface.η
T = model.tracers.T
S = model.tracers.S

file_initital_conditions = jldopen(input_filepath * "initial_conditions-1degree.jld2")

@info "Reading initial conditions"
T₀ = file_initital_conditions["T"]
S₀ = file_initital_conditions["S"]

set!(model, T=T₀, S=S₀)

@info "Model is initialized with T and S from $(file_initital_conditions)"

#####
##### Simulation setup
#####

Δt = 20minutes

simulation = Simulation(model, Δt = Δt, stop_time = Nyears * years)

start_time = [time_ns()]

function progress(sim)
    wall_time = (time_ns() - start_time[1]) * 1e-9

    u = sim.model.velocities.u

    @info @sprintf("Time: % 12s, iteration: %d, max(|u|): %.2e ms⁻¹, max(|w|): %.2e ms⁻¹, wall time: %s",
        prettytime(sim.model.clock.time), sim.model.clock.iteration,
        maximum(abs, u), maximum(abs, w), prettytime(wall_time))

    start_time[1] = time_ns()

    flush(stdout)

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

using Oceananigans.Models.HydrostaticFreeSurfaceModels: VerticalVorticityField

ζ = VerticalVorticityField(model)
δ = Field(∂x(u) + ∂y(v))

save_interval = 5days

output_fields = (; u, v, T, S, η, ζ)

simulation.output_writers[:surface_fields] = JLD2OutputWriter(model, output_fields,
                                                              schedule = TimeInterval(save_interval),
                                                              dir = output_filepath,
                                                              filename = output_prefix * "_surface",
                                                              indices = (:, :, grid.Nz),
                                                              overwrite_existing = true)

#=
u2 = Field(u * u)
v2 = Field(v * v)
w2 = Field(w * w)
η2 = Field(η * η)
T2 = Field(T * T)

average_outputs = (; u, v, T, S, η, u2, v2, T2, η2)

simulation.output_writers[:averages] = JLD2OutputWriter(model, average_outputs,
                                                        schedule = AveragedTimeInterval(30days, window=30days, stride=10),
                                                        filename = output_prefix * "_averages",
                                                        overwrite_existing = true)


simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                        schedule = TimeInterval(30days),
                                                        prefix = output_prefix * "_checkpoint",
                                                        overwrite_existing = true)
=#

# Let's go!
@info "Running simulation with Δt = $(prettytime(simulation.Δt))"

if load_initial_condition
    @info "load in initial condition from " * ic_filepath
    jlfile = jldopen(ic_filepath)
    if ic_filepath == "initial_conditions-1degree.jld2"
        interior(simulation.model.tracers.T) .= arch_array(arch, jlfile["T"])
        interior(simulation.model.tracers.S) .= arch_array(arch, jlfile["S"])
    elseif ic_filepath == "smooth_initial_condition.jl"
    else
        set!(model, u = arch_array(arch, jlfile["velocities"]["u"]),
                    v = arch_array(arch, jlfile["velocities"]["v"]),
                    w = arch_array(arch, jlfile["velocities"]["w"]),
                    T = arch_array(arch, jlfile["tracers"]["T"]),
                    S = arch_array(arch, jlfile["tracers"]["S"]),
                    η = arch_array(arch, jlfile["free_surface"]["eta"]))
    end
    close(jlfile)
end

run!(simulation, pickup=pickup_file)

@info """

    Simulation took $(prettytime(simulation.run_wall_time))
    Free surface: $(typeof(model.free_surface).name.wrapper)
    Time step: $(prettytime(Δt))
"""

# rm(qs_filepath, force=true)
# jlfile = jldopen(qs_filepath, "a+")
# JLD2.Group(jlfile, "velocities")
# JLD2.Group(jlfile, "tracers")
# JLD2.Group(jlfile, "free_surface") # don't forget free surface

# jlfile["velocities"]["u"] = Array(interior(simulation.model.velocities.u))
# jlfile["velocities"]["v"] = Array(interior(simulation.model.velocities.v))
# jlfile["velocities"]["w"] = Array(interior(simulation.model.velocities.w))
# jlfile["tracers"]["T"] = Array(interior(simulation.model.tracers.T))
# jlfile["tracers"]["S"] = Array(interior(simulation.model.tracers.S))
# jlfile["free_surface"]["eta"] = Array(interior(simulation.model.free_surface.η))
# close(jlfile)
