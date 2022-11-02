using Plots, LinearAlgebra 

include("abstract_types.jl")
include("non_smooth_fxns.jl")
include("smooth_fxn.jl")

"""
    This is a struct that models the results obtain from an execution of the 
        prox gradient method. 
"""
mutable struct ProxGradResults
    
    gradient_mappings::Vector{Vector}
    xs::Vector{Vector}
    gradients::Vector{Vector}
    
    function ProxGradResults()
       return new(Vector{Vector{Real}}(), Vector{Vector{Real}}(), Vector{Vector{Real}}()) 
    end
end



"""
    Performs the proximal gradient algorithm, and there are many options to make life easier.
    g:: A smooth function that we can take gradient of. 
    h::A nonsmooth function that we can prox over. 
    x0:: An initial guess for the system. 

"""
function ProxGradient(
        g::SmoothFxn, 
        h::NonsmoothFxn, 
        x0::Vector{T}, 
        step_size::Real;
        itr_max::Int=20, 
        epsilon::AbstractFloat=1e-16, 
        line_search::Bool=false
    ) where {T <: Real}

    @assert itr_max > 0 "The maximum number of iterations is a strictly positive integers. "
    results = ProxGradResults()
    xs = results.xs
    grads = results.gradient_mappings
    push!(xs, x0)
    for k in 1:itr_max
        push!(
            xs, 
            Prox(h, step_size, xs[end] + step_size*Grad(g, xs[end]))
        )
        push!(grads, xs[end] - xs[end - 1])
        if norm(grads[end]) < epsilon
            return results
        end
    end
    return results
end


"""
    Performs the accelerated nesterov gradient descend, and there are many different options to choose from. 
"""





N = 100
A = Diagonal(LinRange(1, 2, N))
b = zeros(N)
h = 1*OneNorm()
g = SquareNormResidual(A, b)

Results = ProxGradient(g, h, ones(N), 1/(2*norm(A, 2)^2), itr_max=5000)

plot((Results.gradient_mappings.|>norm)[1:end - 2], yaxis=:log10) |> display
plot(
        h.(Results.xs[1:end - 2]) + g.(Results.xs[1:end - 2]) .- 
        (h(Results.xs[end]) - g(Results.xs[end])), 
        yaxis=:log10
    )|>display
plot((Results.xs[1:end - 2]).|>norm, yaxis=:log10)


