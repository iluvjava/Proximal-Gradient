include("../src/proximal_gradient.jl")
using LaTeXStrings, Plots, SparseArrays, Images, FileIO, Colors, Distributions

"""
Give a blur matrix for an n × m sized grid that is flattend by julia. It also support the cases with rgb. 
* Default Boundary Conditions: Periodic.
* It's slow but support rgb images as well. But it's only suited to transform rbg images in the tensor dimension of 
[3, m, n]. 
### Arguments 
- `m::Int`: The number of rows for the matrix representing the image. 
- `n::Int` The Number of columns for the matrix representing the image. 
- `rgb::Bool`: If not, the output blur matrix will be in the size of (m*n, m*n), else for RGB, it's support to transform 
images represented by the flattend tensor of size (3, m, n). 
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
    kernel = KernelMake(k)
    d = div(size(kernel)[1], 2) # < -- Get the size of the d × d kernel .
    col = Vector{Float64}(); row = Vector{Float64}(); vals = Vector{Float64}()
    for (I, J) in [(i, j) for i in 1:m for j in 1:n]
        for (III, JJJ) in [(ii + I, jj + J) for ii in -d:d for jj in -d:d]
            push!(row, mod(CoordToLinearIdx(I, J), n*m) + 1)
            push!(col, mod(CoordToLinearIdx(III, JJJ), n*m) + 1)
            push!(vals, kernel[III - I + d + 1, JJJ - J + d + 1])
        end
    end
    toreturn = sparse(row, col, vals)
    if rgb 
       toreturn = kron(toreturn, Matrix{Float64}(I, 3, 3))
    end
    return toreturn
end


"""
Reinterpret a 3d tensor whose element are in the range of [0, 1] to an array of int8 that represents colors. If the 
intputs are not in the right range, then there is gonna be some errors. 

"""
function ImgInterpret(img_arr::Array{Float64, 3})
    flattened = reshape(img_arr, 3, :)
    converted = [RGB(flattened[:, Idx]...) for Idx in 1:size(img_arr, 2)[1]*size(img_arr, 3)[1]]
    reshaped = reshape(converted, size(img_arr)[2:3])
    return reshaped
end



"""
The main function to run the experiment with.
 
### Argument
- `file_name::String`: The name of the file for the prox gradient plot. 
- `mosaic_view_filename::String`: The name of the mosaicview images showing all the intermediate solutions.  
- `alpha::Real=0.01`: the normalized pentalization parameters for the one-norm, λ is 
made by dividing this quantity by the number of elements in `x`, where `x` is the vector that we are 
going to perform the LASSO on. 
"""
function Run(
    file_name::String, 
    alpha::Real=0.01
)
    lambda = alpha*750000^(-1)
    img_path = "applications/image2.png"
    Img = load(img_path)
    Img_Float = Float64.(channelview(Img)|>collect)
    Img_Float = Img_Float[1:3, :, :]
    @info "Preparing Blurr matrix, parameters and functions. "
    "The Blurr Matrix defined under the global scope of the Main"
    A = ConstructBlurrMatrix(size(Img_Float, 2), size(Img_Float, 3), 7)
    b = A*Img_Float[:] + rand(Normal(0, 2e-2), Img_Float|>length)
    g = SquareNormResidual(A, b)
    h = length(b)*lambda*OneNorm()
    Results_Holder = ProxGradResults(80)
    ProxGradient(
        g, 
        h, 
        randn(size(b)), 
        0.01,
        nesterov_momentum=true, 
        line_search=true,
        itr_max=640, 
        results_holder = Results_Holder
        )
    Soln_Img = ImgInterpret(reshape(Results_Holder.soln, size(Img_Float)))
    save("$file_name-soln_img.jpg", Soln_Img)
    Blurred_Img = ImgInterpret(reshape(b, size(Img_Float)))
    save("$(filename)_blurred_img.jpg", Blurred_Img)
    Blurred_Img|>display
    Soln_Img|>display
    fig = plot(
        Results_Holder.gradient_mapping_norm, yaxis=:log10, 
        title="Proximal Mapping Norm with α = $(alpha)",
        ylabel=L"\Vert x^{(k)} - x^{(k - 1)}\Vert", 
        xlabel="Iteration Number: k", 
        dpi=300
    )
    fig|>display
    savefig(fig, file_name*".png")
    soln_imags = GetAllSolns(Results_Holder)
    soln_imags = [reshape(img, size(Img_Float))|>ImgInterpret for img in soln_imags]
    Mosaic_View = mosaic(soln_imags..., nrow=3, rowmajor=true)
    Mosaic_View |> display
    return Results_Holder
end



t1 = Threads.@spawn begin; return Run("inverse_linear_experiment1", 0); end
t2 = Threads.@spawn begin; return Run("inverse_linear_experiment2", 0.01); end
t3 = Threads.@spawn begin; return Run("inverse_linear_experiment3", 0.1); end


