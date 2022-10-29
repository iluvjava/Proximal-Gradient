using Plots, LinearAlgebra 

include("abstract_types.jl")
include("non_smooth_fxns.jl")
include("smooth_fxn.jl")

"""
    Performs the proximal gradient algorithm with pre-determined a fixed stepsize. 
"""
function ProxGradient(
        g::SmoothFxn, 
        h::NonsmoothFxn, 
        x0::Vector{T}, 
        step_size::Real, 
        itr_max::Int=20
    ) where {T <: Real}

    @assert itr_max > 0 "The maximum number of iterations is a strictly positive integers. "
    xs = Vector{Vector}()
    push!(xs, x0)
    for k in 1:itr_max
        push!(
            xs, 
            Prox(h, step_size, xs[end] + step_size*Grad(g, xs[end]))
        )
    end
    return xs
end



N = 200
A = rand(N, N)
b = zeros(N)
h = 2*AbsValue()
g = SquareNormResidual(A, b)

Results = ProxGradient(g, h, ones(N), 1/(2*norm(A)^2))
plot(Results.|>norm, yaxis=:log)|>display


