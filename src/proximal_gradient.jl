using LinearAlgebra

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
        * If step size is not explicitly given, line search will be triggered automatically and 
        a default step size will be assigned. 
    * Linear Search: 
        Specify whether we should use line search. This works for both the smooth and the 
        accelerated case. 
"""
function ProxGradient(
        g::SmoothFxn, 
        h::NonsmoothFxn, 
        x0::Vector{T1}, 
        step_size::Union{T2, Nothing}=nothing;
        itr_max::Int=1000,
        epsilon::AbstractFloat=1e-10, 
        line_search::Bool=false, 
        nesterov_momentum::Bool = false
    ) where {T1 <: Number, T2 <: Number}

    @assert itr_max > 0 "The maximum number of iterations is a strictly positive integers. "
    step_size = step_size === nothing ? 1 : step_size

    t = 1 # <-- This is the nesterov momentum term.
    y = x0 # <-- This one is also for the momentum. 
    l = step_size
    results = ProxGradResults()
    xs = results.xs
    grads = results.gradient_mappings
    push!(xs, x0)
    
    # Quadratic upper bounding function  
    Q(x, y, l) = g(x) + dot(Grad(g, x), y - x) + (norm(y - x)^2)/(2*l)
    last_itr = 0

    for k in 1:itr_max
        # Line search 
        x⁺ = Prox(h, l, y - l*Grad(g, y))
        while line_search && g(x⁺) > Q(y, x⁺, l) + eps(T1)
            # "$(l), g(x⁺): $(g(x⁺)), Q(y, x⁺, l): $(Q(y, x⁺, l))" |> println
            l /= 2
            x⁺ = Prox(h, l, y - l*Grad(g, y))
        end
        push!(xs, x⁺)
        push!(grads, xs[end] - xs[end - 1])

        t⁺ = nesterov_momentum ? (1 + sqrt(1 + 4t^2))/2 : 1
        y = x⁺ + ((t - 1)/t⁺)*(xs[end] - xs[end - 1])
        t = t⁺
        
        if norm(grads[end], Inf) < epsilon
            last_itr = k
            break
        end
    end
    if last_itr == itr_max
        results.flags = 1
    end
    popat!(xs, 1)
    return results
end

