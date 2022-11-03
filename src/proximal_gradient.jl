using Plots, LinearAlgebra 

include("abstract_types.jl")
include("non_smooth_fxns.jl")
include("smooth_fxn.jl")

"""
    This is a struct that models the results obtain from an execution of the 
        prox gradient method. We see to store the following results: 
        * x - prox(h, L, x) :: The gradient mappings
        * xs, the list of solutions from the prox gradient 
        * gradients: The gradient of the smooth function. 
        * Flags:    
            0. Termninated due to gradient mapping reaches tolerance 
            1. Terminated due to maximal iterations limit has been reached. 
"""
mutable struct ProxGradResults
    
    gradient_mappings::Vector{Vector}
    xs::Vector{Vector}
    flags::Int
    
    function ProxGradResults()
       return new(Vector{Vector{Real}}(), Vector{Vector{Real}}(), 0) 
    end
end



"""
    Performs the proximal gradient algorithm, and there are many options to make life easier. 
    * g, h are smooth and non smooth functions
    * x0 is not optinal
    * step size is optional and it has a default value. 
        * If step size is not explicitly given, line search will be triggered automatically. 
    * Linear Search: 
        Specify whether we should use line search. 
"""
function ProxGradient(
        g::SmoothFxn, 
        h::NonsmoothFxn, 
        x0::Vector{T1}, 
        step_size::T2=1;
        itr_max::Int=1000,
        epsilon::AbstractFloat=1e-10, 
        line_search::Bool=false
    ) where {T1 <: Number, T2 <: Number}

    @assert itr_max > 0 "The maximum number of iterations is a strictly positive integers. "
    
    results = ProxGradResults()
    xs = results.xs
    grads = results.gradient_mappings
    push!(xs, x0)
    
    # Quadratic upper bounding function  
    Q(x, y, L) = g(x) + dot(Grad(g, x), y - x) + (norm(y - x)^2)/(2*L)
    
    for k in 1:itr_max
        x⁺ = Prox(h, step_size, xs[end] - step_size*Grad(g, xs[end]))
        # Line search 
        while line_search && g(x⁺) >= Q(xs[end], x⁺, step_size)
            step_size /= 2
            x⁺ = Prox(h, step_size, xs[end] - step_size*Grad(g, xs[end]))
        end
        push!(
            xs, 
            x⁺
        )
        push!(grads, xs[end] - xs[end - 1])
        if norm(grads[end]) < epsilon
            return results
        end
    end
    results.flags = 1
    return results
end


"""
    Performs the accelerated nesterov gradient descend, and there are many different options to choose from. 
"""


N = 63
A = Diagonal(LinRange(1, 2, N))
b = zeros(N)
for II in eachindex(b)
    if mod(II, 2) == 1
        b[II] = A[II, II]
    else
        b[II] = rand()*1e-4
    end
end
h = 0.1*OneNorm()
g = SquareNormResidual(A, b)

Results = ProxGradient(g, h, ones(N), 1, itr_max=5000, line_search=true)

plot(
    (Results.gradient_mappings.|>norm)[1:end - 2], yaxis=:log10, 
    title="Gradient Mapping Norm"
) |> display
# plot(
#         h.(Results.xs[1:end - 2]) + g.(Results.xs[1:end - 2]) .- 
#         (h(Results.xs[end]) - g(Results.xs[end])), 
#         yaxis=:log10
#     )|>display
# plot((Results.xs[1:end - 2]).|>norm, yaxis=:log10)


Results.xs[end]