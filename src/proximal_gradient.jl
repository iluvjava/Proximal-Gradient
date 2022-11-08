using LinearAlgebra, ProgressMeter

include("abstract_types.jl")
include("non_smooth_fxns.jl")
include("smooth_fxn.jl")


"""
    This is a struct that models the results obtain from an execution of the 
    prox gradient method. We see to store the following results and each result
    is mapped to the iteration number of the algorithm: 
        * x - prox(h, L, x) :: The gradient mappings. 
            * And its norm. 
        * xs, the list of solutions from the prox gradient. 
        * 
        * Flags:    
            0. Termninated due to gradient mapping reaches tolerance 
            1. Terminated due to maximal iterations limit has been reached. 
"""
mutable struct ProxGradResults
    
    gradient_mappings::Dict{Int, Vector}       # collect sparsely 
    gradient_mapping_norm::Vector{Real}     # always collect
    objective_vals::Vector{Real}            # Always collect
    solns::Dict{Int, Vector}                   # collect sparsely 
    soln::Vector                                # the final solution
    step_sizes::Vector{Real}                # always collect. 
    flags::Int                                 # collect at last
    collection_interval::Int                   # specified on initializations. 
    itr_counter::Int                           # updated within this class. 
    
    """
        You have the option to specify how frequently you want the results to be collected, because 
        if we collect all the solutions all the time the program is going to be a memory hog! 
    """
    function ProxGradResults(collection_interval::Int=typemax(Int))
        this = new()
        this.gradient_mappings = Dict{Int, Vector}()
        this.gradient_mapping_norm = Vector{Real}()
        this.objective_vals = Vector{Real}()
        this.solns = Dict{Int, Vector}()
        this.step_sizes = Vector{Real}()
        this.flags = 0
        this.itr_counter = -1
        this.collection_interval = collection_interval
       return this
    end

end

function Initiate!(this::ProxGradResults, x0::Vector, obj_initial::Real, step_size::Real)
    this.itr_counter = 0
    this.solns[this.itr_counter] = x0
    push!(this.objective_vals, obj_initial)
    push!(this.step_sizes, step_size)
    return nothing
end

function Register!(this::ProxGradResults, obj::Real, soln::Vector, pgrad_map::Vector, step_size::Real)
    if this.itr_counter == -1
        error("ProxGrad Results is called without initiation.")
    end
    this.itr_counter += 1
    k = this.itr_counter
    push!(this.objective_vals, obj)
    push!(this.gradient_mapping_norm, norm(pgrad_map))
    push!(this.step_sizes, step_size)

    if mod(k, this.collection_interval) == 0
        this.solns[k] = soln
        this.gradient_mappings[k] = pgrad_map
    end
    return nothing
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
        nesterov_momentum::Bool = false, 
        results_holder::ProxGradResults=ProxGradResults()
    ) where {T1 <: Number, T2 <: Number}

    @assert itr_max > 0 "The maximum number of iterations is a strictly positive integers. "
    line_search = step_size === nothing ? true : line_search
    step_size = step_size === nothing ? 1 : step_size

    t = 1 # <-- This is the nesterov momentum term.
    y = x0 # <-- This one is also for the momentum. 
    l = step_size # <-- initial step size. 
    
    # Quadratic upper bounding function  
    Q(x, y, l) = g(x) + dot(Grad(g, x), y - x) + (norm(y - x)^2)/(2*l)
    last_itr = 0
    last_x = x0
    lastlast_x = x0
    current_x = x0
    Initiate!(results_holder, x0, h(x0) + g(x0), l)

    for k in 1:itr_max
        
        x⁺ = Prox(h, l, y - l*Grad(g, y))
        # Line search 
        while line_search && g(x⁺) > Q(y, x⁺, l) + eps(T1)
            l /= 2
            x⁺ = Prox(h, l, y - l*Grad(g, y))
        end
        # Register the results
        Register!(results_holder, h(x⁺) + g(x⁺), x⁺, y - x⁺, l)
        pgrad_norm = norm(y - x⁺, Inf)
        
        # Update the parameters
        t⁺ = nesterov_momentum ? (1 + sqrt(1 + 4t^2))/2 : 1
        t = t⁺
        y = x⁺ + ((t - 1)/t⁺)*(x⁺ - current_x)
        lastlast_x = last_x
        last_x = current_x
        current_x = x⁺
        # check for termination conditions
        if pgrad_norm < epsilon
            last_itr = k
            break
        end
    end
    # determine exit flag.
    if last_itr == itr_max
        results_holder.flags = 1
    end
    results_holder.soln = current_x
    return results_holder
end

N = 64
A = Diagonal(LinRange(0, 2, N))
b = ones(N)
g = SquareNormResidual(A, b)
h = 0.001*OneNorm()
soln = ProxGradient(g, h, zeros(size(b)), nesterov_momentum=true);
using Plots
plot(soln.gradient_mapping_norm, yaxis=:log10) |> display


