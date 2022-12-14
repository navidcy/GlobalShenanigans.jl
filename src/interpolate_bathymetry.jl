using ImageMagick
using PyCall
using FFTW
using FastSphericalHarmonics



function interpolate_bathymetry_from_file(filename, passes, degree, latitude)

    file = jldopen(filename)

    bathy = Float64.(file["bathymetry"])

    Nxₒ  = Int(21600)
    Nyₒ  = Int(10800)
    Nxₙ  = Int(360 / degree)
    Nyₙ  = Int(2latitude / degree)

    ΔNx = floor((Nxₒ - Nxₙ) / passes)
    ΔNy = floor((Nyₒ - Nyₙ) / passes)

    Nx = deepcopy(Nxₒ)
    Ny = deepcopy(Nyₒ)

    @assert Nxₒ == Nxₙ + passes * ΔNx
    @assert Nyₒ == Nyₙ + passes * ΔNy

    for pass = 1:passes
        bathy_full = deepcopy(bathy)
        Nxₒ = Nx
        Nyₒ = Ny
        Nx -= Int(ΔNx) 
        Ny -= Int(ΔNy)
        if pass == 1
            oldlat = 89.9999999999999999
        else
            oldlat = latitude
        end
        old_grid = RectilinearGrid(size = (Nxₒ, Nyₒ, 1), y = (-oldlat,   oldlat),   x = (-180, 180), z = (0, 1))
        new_grid = RectilinearGrid(size = (Nx,  Ny,  1), y = (-latitude, latitude), x = (-180, 180), z = (0, 1))
    
        @show Nxₒ, Nyₒ, Nx, Ny, pass
        bathy = interpolate_one_level(bathy_full, old_grid, new_grid, Center)
    end

    # apparently bathymetry is reversed in the longitude direction, therefore we have to swap it

    bathy = reverse(bathy, dims = 2)
    bathy[bathy .> 0] .= ABOVE_SEA_LEVEL

    return bathy
end

function etopo1_to_spherical_harmonics()

    lmax = 10798

    etopo1 = jldopen("data/bathymetry-ice-21600x10800.jld2")["bathymetry"]

    # latitude interpolate and transpose
    etopo1_center = 0.5 *(etopo1[:, 2:end] .+ etopo1[:, 1:end-1])
    etopo1_center = etopo1_center'

    # Drop the 360 point
    etopo1_center = etopo1_center[:, 1:end-1]

    # longitude interpolate
    fft_etopo1_center = rfft(etopo1_center, 2)
    fft_etopo1_center = fft_etopo1_center[:, 1:end-1]
    etopo1_interp     = irfft(fft_etopo1_center, 2lmax +1, 2)

    # Sperical harmonic filtering
    return sph_transform(etopo1_interp)
end

function bathymetry_from_etopo1(Nφ, Nλ, spher_harm_coeff, filter)

    lmax_interp      = Nφ - 1

    spher_harm_coeff_filter = zeros(lmax_interp + 1, 2lmax_interp +1)
    for l = 0:lmax_interp, m = -l:l
        spher_harm_coeff_filter[sph_mode(l,m)] = spher_harm_coeff[sph_mode(l, m)] * filter(l)
    end

    etopo1_filter = sph_evaluate(spher_harm_coeff_filter) # dimensions are Nφ, 2Nφ - 1 

    # longitude interpolate 2
    fft_etopo1_filter = rfft(etopo1_filter, 2) # dimensions are Nφ, Nφ 

    mmax_interp = Nλ ÷ 2 + 1

    fft_etopo1_interp = zeros(Complex{Float64}, Nφ, mmax_interp)

    if mmax_interp <= size(fft_etopo1_filter, 2)
        fft_etopo1_interp .= fft_etopo1_filter[:, 1:mmax_interp]
    else
        fft_etopo1_interp[:, 1:size(fft_etopo1_filter, 2)] .= fft_etopo1_filter
    end

    etopo1_final = irfft(fft_etopo1_interp, Nλ, 2)

    return etopo1_final
end

function cut_to_latitude(bathymetry, latitude)
    

end

function remove_connected_regions(bathymetry)
    batneg = deepcopy(bathymetry)

    batneg[batneg.>0] .= 0
    batneg[batneg.<0] .= 1

    labels = sckikitimage.label(batneg)

    total_elements = zeros(maximum(labels))

    for i in 1:length(total_elements)
        total_elements[i] = sum(labels[labels.==i])
    end

    ocean_idx = findfirst(x -> x == maximum(x), total_elements)
    second_maximum = maximum(filter((x) -> x != total_elements[ocean_idx], total_elements))

    bering_sea_idx = findfirst(x -> x == second_maximum, total_elements)

    labels = Float64.(labels)
    labels[labels.==0] .= NaN

    for i in 1:length(total_elements)
        if (i != ocean_idx) && (i != bering_sea_idx)
            labels[labels.==i] .= NaN
        end
    end

    new_bathymetry = deepcopy(bathymetry)
    new_bathymetry .+= labels
    new_bathymetry[isnan.(bathymetry)] .= ABOVE_SEA_LEVEL

    return new_bathymetry
end

@inline is_coupled_fluid(t, i, j) = land_rectangle(t, i, j, -1) | land_rectangle(t, i, j, -2)

@kernel function _mask_coupled_elements!(bathy) 
    i, j = @index(Global, NTuple)

    # start with index 2
    i′ = i + 2
    j′ = j + 2
    if is_coupled_fluid(bathy, i′, j′) == true
        bathy[i′, j′] = ABOVE_SEA_LEVEL
    end
end

function mask_coupled_elements(bathy, arch)
    dev = device(arch)
    Nx, Ny = size(bathy)

    #copy to GPU
    bathy_GPU = deepcopy(arch_array(arch, bathy))

    loop! = _mask_coupled_elements!(dev, (16, 16), (Nx-4, Ny-4))
    event = loop!(bathy_GPU; dependencies=device_event(arch))
    wait(dev, event)

    #copy back
    return bathy = Array(bathy_GPU)
end

function write_bathymetry_to_file(prefix, bathy, lat)
    Nxₙ, Nyₙ = size(bathy)
    output_prefix = prefix * "-$(Int(Nxₙ))x$(Int(Nyₙ))-latitude-$(lat)"
    jldsave(output_prefix * ".jld2", bathymetry = bathy)
end
