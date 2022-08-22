using Statistics
using JLD2
using Printf
using CUDA
using CairoMakie
using Oceananigans
using Oceananigans.Units
using Oceananigans.Fields: interpolate, Field
using Oceananigans.Architectures: arch_array
using Oceananigans.Coriolis: HydrostaticSphericalCoriolis
using Oceananigans.BoundaryConditions
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom, inactive_node, peripheral_node
# solid_node is inactive_node and peripheral_node is peripheral_node
using Oceananigans.TurbulenceClosures: RiBasedVerticalDiffusivity
using CUDA: @allowscalar, device!
using Oceananigans.Operators
using Oceananigans.Operators: Δzᵃᵃᶜ
using Oceananigans: prognostic_fields
using Oceananigans.Advection: EnergyConservingScheme, VelocityStencil, VorticityStencil
using Oceananigans.TurbulenceClosures: FluxTapering
using SeawaterPolynomials.TEOS10
using Oceananigans.Advection: VelocityStencil

using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities:
    CATKEVerticalDiffusivity, MixingLength

# Memo to self, change the surface relaxation to be a lot more relaxed
global_filepath = "/storage1/uq-global/GlobalShenanigans.jl/"
output_filepath = "/home/navidcy/NavidGlobalShenanigans.jl/output/"

load_initial_condition = true
# ic_filepath = global_filepath * "smooth_initial_condition.jl"
# ic_filepath = global_filepath * "longer_null_hypothesis_teos10.jld2" # without parameterizations etc.
ic_filepath = global_filepath * "smooth_ic_7.jld2" # start from 7
# qs_filepath = global_filepath * "long_null_hypothesis.jl"
# qs_filepath = global_filepath * "even_longer_null_hypothesis_teos10.jld2"
qs_filepath = global_filepath * "smooth_ic_10.jld2" # 6 is the last one with ecco forcing

#####
##### Grid
#####

arch = GPU()
# device!(3)
reference_density = 1029 # kg/m^3

latitude = (-75, 75)

# 1 degree resolution
Nx = 360
Ny = 150
Nz = 48

const Nyears = 20.0

const Nmonths = 12
const thirty_days = 30days

output_prefix = output_filepath * "near_global_lat_lon_$(Nx)_$(Ny)_$(Nz)_20y_default5"
println("running " * output_prefix)
pickup_file = false

# Stretched faces taken from ECCO Version 4 (50 levels in the vertical)
z_faces = jldopen(global_filepath * "zgrid.jld2")["z"][5:end-4]

# An almost-global spherical domain
@show underlying_grid = LatitudeLongitudeGrid(arch,
                                              size = (Nx, Ny, Nz),
                                              longitude = (-180, 180),
                                              latitude = latitude,
                                              halo = (5, 5, 5),
                                              z = z_faces,
                                              precompute_metrics = true)

bathymetry = jldopen(global_filepath * "bathymetry-360x150-latitude-75.0.jld2")["bathymetry"]

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bathymetry))

λc = grid.λᶜᵃᵃ[1:grid.Nx]
φc = grid.φᵃᶜᵃ[1:grid.Ny]
z = grid.zᵃᵃᶜ[1:grid.Nz]


#####
##### Load forcing files and inital conditions from ECCO version 4
##### https://ecco.jpl.nasa.gov/drive/files
##### Bathymetry is interpolated from ETOPO1 https://www.ngdc.noaa.gov/mgg/global/
#####
#=
using DataDeps

path = "https://github.com/CliMA/OceananigansArtifacts.jl/raw/ss/new_hydrostatic_data_after_cleared_bugs/quarter_degree_near_global_input_data/"

datanames = "z_faces-50-levels"

dh = DataDep("quarter_degree_near_global_lat_lon",
    "Forcing data for global latitude longitude simulation",
    [path * data * ".jld2" for data in datanames]
)

DataDeps.register(dh)

datadep"quarter_degree_near_global_lat_lon"

datadep_path = @datadep_str "quarter_degree_near_global_lat_lon/z_faces-50-levels.jld2"
=#

# plot bathymetry
bathymetry_for_plot = Array(bathymetry)
bathymetry_for_plot[bathymetry .== 100] .= NaN
fig, ax, hm = heatmap(λc, φc, bathymetry_for_plot; colorrange = (-9000, 0), nan_color=:black)
Colorbar(fig[1, 2], hm)
save(output_filepath * "bathymetry.png", fig)

τˣ = zeros(Nx, Ny, Nmonths)
τʸ = zeros(Nx, Ny, Nmonths)
T★ = zeros(Nx, Ny, Nmonths)
S★ = zeros(Nx, Ny, Nmonths)

# Files contain 1 year (1992) of 12 monthly averages
τˣ = jldopen(global_filepath * "boundary_conditions-1degree.jld2")["τˣ"] ./ reference_density
τʸ = jldopen(global_filepath * "boundary_conditions-1degree.jld2")["τˣ"] ./ reference_density
T★ = jldopen(global_filepath * "boundary_conditions-1degree.jld2")["Tˢ"]
S★ = jldopen(global_filepath * "boundary_conditions-1degree.jld2")["Sˢ"]

τˣ_for_plot = Array(τˣ[:, :, 1])
τˣ_for_plot[bathymetry .== 100] .= NaN
fig, ax, hm = heatmap(λc, φc, τˣ_for_plot;
                      colormap = :balance,
                      colorrange = (-2e-4, 2e-4),
                      nan_color=:black)
Colorbar(fig[1, 2], hm)
save(output_filepath * "taux_january_ecco.png", fig)

T_for_plot = Array(T★[:, :, 1])
T_for_plot[bathymetry .== 100] .= NaN
fig, ax, hm = heatmap(λc, φc, T_for_plot; nan_color=:black)
Colorbar(fig[1, 2], hm)
save(output_filepath * "temperature_january_ecco.png", fig)

S_for_plot = Array(S★[:, :, 1])
S_for_plot[bathymetry .== 100] .= NaN
fig, ax, hm = heatmap(λc, φc, S_for_plot; nan_color=:black)
Colorbar(fig[1, 2], hm)
save(output_filepath * "salinity_january_ecco.png", fig)

# Remember the convention!! On the surface a negative flux increases a positive decreases
bathymetry = arch_array(arch, bathymetry)
τˣ = arch_array(arch, -τˣ)
τʸ = arch_array(arch, -τʸ)

τˣ_for_plot = Array(τˣ[:, :, 1])
τˣ_for_plot[bathymetry .== 100] .= NaN
fig, ax, hm = heatmap(λc, φc, τˣ_for_plot;
                      colormap = :balance,
                      colorrange = (-2e-4, 2e-4),
                      nan_color=:black)
Colorbar(fig[1, 2], hm)
save(output_filepath * "taux_january_used_bc.png", fig)

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
gent_mcwilliams_diffusivity = IsopycnalSkewSymmetricDiffusivity(
    κ_skew=κ_skew,
    κ_symmetric=κ_symmetric,
    slope_limiter=FluxTapering(1e-2))

closures = (biharmonic_viscosity, convective_adjustment, gent_mcwilliams_diffusivity)

#####
##### Boundary conditions / time-dependent fluxes 
#####

@inline current_time_index(time, tot_months) = mod(unsafe_trunc(Int32, time / thirty_days), tot_months) + 1
@inline next_time_index(time, tot_months) = mod(unsafe_trunc(Int32, time / thirty_days) + 1, tot_months) + 1
@inline cyclic_interpolate(u₁::Number, u₂, time) = u₁ + mod(time / thirty_days, 1) * (u₂ - u₁)

Δz_top = @allowscalar Δzᵃᵃᶜ(1, 1, grid.Nz, grid.underlying_grid)
Δz_bottom = @allowscalar Δzᵃᵃᶜ(1, 1, 1, grid.underlying_grid)

@inline function surface_wind_stress(i, j, grid, clock, fields, τ)
    time = clock.time
    n₁ = current_time_index(time, Nmonths)
    n₂ = next_time_index(time, Nmonths)

    @inbounds begin
        τ₁ = τ[i, j, n₁]
        τ₂ = τ[i, j, n₂]
    end

    return cyclic_interpolate(τ₁, τ₂, time)
end

u_wind_stress_bc = FluxBoundaryCondition(surface_wind_stress, discrete_form=true, parameters=τˣ)
v_wind_stress_bc = FluxBoundaryCondition(surface_wind_stress, discrete_form=true, parameters=τʸ)

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
    time = clock.time

    n₁ = current_time_index(time, Nmonths)
    n₂ = next_time_index(time, Nmonths)

    @inbounds begin
        T★₁ = p.T★[i, j, n₁]
        T★₂ = p.T★[i, j, n₂]
        T_surface = fields.T[i, j, grid.Nz]
    end

    T★ = cyclic_interpolate(T★₁, T★₂, time)

    return p.λ * (T_surface - T★)
end

@inline function surface_salinity_relaxation(i, j, grid, clock, fields, p)
    time = clock.time

    n₁ = current_time_index(time, Nmonths)
    n₂ = next_time_index(time, Nmonths)

    @inbounds begin
        S★₁ = p.S★[i, j, n₁]
        S★₂ = p.S★[i, j, n₂]
        S_surface = fields.S[i, j, grid.Nz]
    end

    S★ = cyclic_interpolate(S★₁, S★₂, time)

    return p.λ * (S_surface - S★)
end

T_surface_relaxation_bc = FluxBoundaryCondition(surface_temperature_relaxation,
    discrete_form=true,
    parameters=(λ=Δz_top / (4 * 7days), T★=target_sea_surface_temperature))

S_surface_relaxation_bc = FluxBoundaryCondition(surface_salinity_relaxation,
    discrete_form=true,
    parameters=(λ=Δz_top / (4 * 7days), S★=target_sea_surface_salinity))

u_bcs = FieldBoundaryConditions(top = u_wind_stress_bc, bottom = u_bottom_drag_bc)
v_bcs = FieldBoundaryConditions(top = v_wind_stress_bc, bottom = v_bottom_drag_bc)
T_bcs = FieldBoundaryConditions(top = T_surface_relaxation_bc)
S_bcs = FieldBoundaryConditions(top = S_surface_relaxation_bc)

boundary_conditions = (u = u_bcs, v = v_bcs, T = T_bcs, S = S_bcs)

#=
u_bcs = FieldBoundaryConditions(bottom=u_bottom_drag_bc)
v_bcs = FieldBoundaryConditions(bottom=v_bottom_drag_bc)
boundary_conditions = (u=u_bcs, v=v_bcs)
forcings = (u=Fu, v=Fv)
=#

free_surface = ImplicitFreeSurface()
# free_surface = ImplicitFreeSurface(preconditioner_method = :SparseInverse, preconditioner_settings = (ε = 0.01, nzrel = 10))

# eos = LinearEquationOfState()
eos = TEOS10EquationOfState()
buoyancy = SeawaterBuoyancy(equation_of_state = eos)

# CHECK WHY
forcings = (u = Fu, v = Fv)

model = HydrostaticFreeSurfaceModel(; grid, free_surface,
                                    momentum_advection = WENO5(grid = underlying_grid, vector_invariant=VelocityStencil()),
                                    coriolis = HydrostaticSphericalCoriolis(scheme = EnergyConservingScheme()),
                                    buoyancy,
                                    # tracers = (:T, :S, :e),
                                    tracers = (:T, :S),
                                    closure = closures,
                                    boundary_conditions,
                                    forcing = forcings,
                                    tracer_advection = WENO5(; grid = underlying_grid)
                                    )

#####
##### Initial condition:
#####

u, v, w = model.velocities
η = model.free_surface.η
T = model.tracers.T
S = model.tracers.S
# e = model.tracers.e

file_init = jldopen(global_filepath * "initial_conditions-1degree.jld2")

@info "Reading initial conditions"
T_init = file_init["T"]
S_init = file_init["S"]

set!(model, T=T_init, S=S_init)
fill_halo_regions!(T)
fill_halo_regions!(S)

@info "model initialized"

#####
##### Simulation setup
#####

Δt = 20minutes # 20minutes

simulation = Simulation(model, Δt = Δt, stop_time = Nyears * years)

initial_e(x, y, z) = exp((-x^2 -y^2)/30^2 ) * exp(-(z+1000)^2/100^2)

# set!(model, e = initial_e)

start_time = [time_ns()]

using Oceananigans.Models.HydrostaticFreeSurfaceModels: VerticalVorticityField

ζ = VerticalVorticityField(model)
δ = Field(∂x(model.velocities.u) + ∂y(model.velocities.v))

function progress(sim)
    wall_time = (time_ns() - start_time[1]) * 1e-9

    u = sim.model.velocities.u

    @info @sprintf("Time: % 12s, iteration: %d, max(|u|): %.2e ms⁻¹, wall time: %s",
        prettytime(sim.model.clock.time),
        sim.model.clock.iteration, maximum(u),
        prettytime(wall_time))

    start_time[1] = time_ns()

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

output_fields = (; u, v, T, S, η)
save_interval = 5days

u2 = Field(u * u)
v2 = Field(v * v)
w2 = Field(w * w)
η2 = Field(η * η)
T2 = Field(T * T)

outputs = (; u, v, T, S, η)
# average_outputs = (; u, v, T, S, η, e, u2, v2, T2, η2)
average_outputs = (; u, v, T, S, η, u2, v2, T2, η2)


simulation.output_writers[:surface_fields] = JLD2OutputWriter(model, (; u, v, T, S, η),
                                                              schedule = TimeInterval(save_interval),
                                                              filename = output_prefix * "_surface",
                                                              indices = (:, :, grid.Nz),
                                                              overwrite_existing = true)

simulation.output_writers[:averages] = JLD2OutputWriter(model, average_outputs,
                                                        schedule = AveragedTimeInterval(30days, window=30days, stride=10),
                                                        filename = output_prefix * "_averages",
                                                        overwrite_existing = true)

#=
simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                        schedule = TimeInterval(30days),
                                                        prefix = output_prefix * "_checkpoint",
                                                        overwrite_existing = true)
=#
# Let's go!
@info "Running with Δt = $(prettytime(simulation.Δt))"

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
        # interior(simulation.model.velocities.u) .= arch_array(arch, jlfile["velocities"]["u"])
        # interior(simulation.model.velocities.v) .= arch_array(arch, jlfile["velocities"]["v"])
        # interior(simulation.model.velocities.w) .= arch_array(arch, jlfile["velocities"]["w"])
        # interior(simulation.model.tracers.T) .= arch_array(arch, jlfile["tracers"]["T"])
        # interior(simulation.model.tracers.S) .= arch_array(arch, jlfile["tracers"]["S"])
        # interior(simulation.model.free_surface.η) .= arch_array(arch, jlfile["free_surface"]["eta"])
    end
    close(jlfile)
end

run!(simulation, pickup=pickup_file)

@info """

    Simulation took $(prettytime(simulation.run_wall_time))
    Free surface: $(typeof(model.free_surface).name.wrapper)
    Time step: $(prettytime(Δt))
"""

rm(qs_filepath, force=true)
jlfile = jldopen(qs_filepath, "a+")
JLD2.Group(jlfile, "velocities")
JLD2.Group(jlfile, "tracers")
JLD2.Group(jlfile, "free_surface") # don't forget free surface

jlfile["velocities"]["u"] = Array(interior(simulation.model.velocities.u))
jlfile["velocities"]["v"] = Array(interior(simulation.model.velocities.v))
jlfile["velocities"]["w"] = Array(interior(simulation.model.velocities.w))
jlfile["tracers"]["T"] = Array(interior(simulation.model.tracers.T))
jlfile["tracers"]["S"] = Array(interior(simulation.model.tracers.S))
jlfile["free_surface"]["eta"] = Array(interior(simulation.model.free_surface.η))
close(jlfile)
