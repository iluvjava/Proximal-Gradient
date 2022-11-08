include("../src/proximal_gradient.jl")
using LaTeXStrings, Plots, SparseArrays, Images, FileIO, Colors

"""

    Give a blur matrix for an n × m sized grid that is flattend by julia. It also support the cases with rgb. 
    * Default Boundary Conditions: Setting it to zero.
    * Not, for rgb, this is  used for tensor of the size (3, m, n). 
"""
function ConstructBlurrMatrix(m::Int, n::Int, k::Int=3; rgb::Bool=true)
    function CoordToLinearIdx(i::Int, j::Int)
        return j*m + i
    end
    function KernelMake(s::Int)
        g(x, y) = (1/sqrt(2pi))*exp(-(x^2 + y^2)/2)
        kernel = [g(x, y) for x in LinRange(-2, 2, 2*s + 1) for y in LinRange(-2, 2, 2*s + 1)]
        return reshape(kernel, 2*s + 1, 2*s + 1)/sum(kernel)
    end
    coefficients = Dict{Tuple{Int, Int}, Float64}()
    # The kernel must be odd size. 
    # kernel = [1/16 1/8 1/16; 1/8 1/4 1/8; 1/16 1/8 1/16]
    kernel = KernelMake(k)
    d = div(size(kernel)[1], 2) # < -- Get the size of the d × d kernel .
    # Use a map to hold the values of the coefficients, index starts with zero. 
    for (I, J) in [(i, j) for i in 1:m for j in 1:n]
        for (III, JJJ) in [(ii + I, jj + J) for ii in -d:d for jj in -d:d]
            coefficients[CoordToLinearIdx(I, J), CoordToLinearIdx(III, JJJ)] = 
                kernel[III - I + d + 1, JJJ - J + d + 1]
        end
    end
    col = Vector{Float64}(); row = Vector{Float64}(); vals = Vector{Float64}()
    for (i, j) in keys(coefficients)
        push!(row, mod(i, n*m) + 1)
        push!(col, mod(j, n*m) + 1)
        push!(vals, coefficients[i, j])
    end
    
    toreturn = sparse(row, col, vals)
    if rgb 
       toreturn = kron(toreturn, Matrix{Float64}(I, 3, 3))
    end
    return toreturn
end

"""
    Reinterpret a 3d tensor that is filled with floats as an image. 
"""
function ImgInterpret(img_arr::Array{Float64, 3})
    flattened = reshape(img_arr, 3, :)
    converted = [RGB(flattened[:, Idx]...) for Idx in 1:size(img_arr, 2)[1]*size(img_arr, 3)[1]]
    reshaped = reshape(converted, size(img_arr)[2:3])
    return reshaped
end

"""
    The main function to run. 
"""
# function Run(lambda::Float64=0.01)
    lambda = 0.001*750000^(-1)
    img_path = "applications/image2.png"
    Img = load(img_path)
    ImgFloat = Float64.(channelview(Img)|>collect)
    ImgFloat = ImgFloat[1:3, :, :]
    @info "Preparing Blurr matrix, parameters and functions. "
    A = ConstructBlurrMatrix(size(ImgFloat, 2), size(ImgFloat, 3), 10)
    b = A*ImgFloat[:]
    g = SquareNormResidual(A, b)
    h = length(b)*lambda*OneNorm()
    Results_Holder = ProxGradResults(100)

    results = ProxGradient(
        g, 
        h, 
        zeros(size(b)), 
        nesterov_momentum=true, 
        itr_max=200, 
        results_holder = Results_Holder
        )
    soln_img = ImgInterpret(reshape(results.soln, size(ImgFloat)))
    blurred_img = ImgInterpret(reshape(b, size(ImgFloat)))
    fig = plot(results.gradient_mapping_norm, yaxis=:log10)
    fig |> display
# end

# Results = Run()
