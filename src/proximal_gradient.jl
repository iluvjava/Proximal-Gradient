using LinearAlgebra, ProgressMeter, Distributed

include("abstract_types.jl")
include("non_smooth_fxns.jl")
include("smooth_fxn.jl")


"""
This is a struct that models the results obtain from an execution of the 
prox gradient method. We see to store the following results and each result
is mapped to the iteration number of the algorithm.
### Fields

- `gradient_mappings::Dict{Int, Vector}`: gradient mapping is the difference between the solutions from the current 
iterations and the previous iterations. It's collect sparsely according the policy parameter *collection_interval*. 
- `gradient_mapping_norm::Vector{Real}`: The norm of the gradient mapping vector. 
- `objective_vals::Vector{Real}`: The objective values of the cost function. 
- `solns::Dict{Int, Vector}`: the solutions stored together with the indices. 
- `soln::Vector`: The final solution obtained by the algorithm. 
- `step_sizes::Vector{Real}`: all the stepsizes used by the algorithm. 
- `flags::Int`: exit flag
    * `0` Exited and the algorithm reached desired tolerance. 
    * `1` Maximum iteration reached and then exit is triggered. 
- `collection_interval::Int`
    * Collect one solution per `collection_interval` iteration. 
- `itr_counter::Int`: An inner counter in the struct the keep track of the number of iterations, it's for mapping the 
results and using with the dictionary data structure. 
"""
mutable struct ProxGradResults
    
    gradient_mappings::Dict{Int, Vector}        # collect sparsely 
    gradient_mapping_norm::Vector{Real}         # always collect
    objective_vals::Vector{Real}                # Always collect
    solns::Dict{Int, Vector}                    # collect sparsely 
    soln::Vector                                # the final solution
    step_sizes::Vector{Real}                    # always collect. 
    flags::Int                                  # collect at last
    collection_interval::Int                    # specified on initializations. 
    itr_counter::Int                            # updated within this class. 
    momentums::Vector{Real}                     # Store the momentum sequence
    
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
        this.momentums = Vector{Real}()
        this.flags = 0
        this.itr_counter = -1
        this.collection_interval = collection_interval
       return this
    end

end

"""
Initiate the instance of `ProxGradResults`, it's required, without this this class won't let you do anything. Because 
it at least need the initial conditions for the gradient descend algorithms. 
### Argument
- `ProxGradResults`: The type that this function acts on. 
- `x0::Vecotr`: The initial guess for the algorithm. 
- `objective_initial::Real`: The initial objective value for the optimization problem. 
- `step_size::Real`: The initial step size for running the algorithm. 
"""
function Initiate!(this::ProxGradResults, 
    x0::Vector, 
    obj_initial::Real, 
    step_size::Real, 
    momentum::Real=0
)
    this.itr_counter = 0
    this.solns[this.itr_counter] = x0
    push!(this.objective_vals, obj_initial)
    push!(this.step_sizes, step_size)
    push!(this.momentums, momentum)
    return nothing
end

"""
During each iteration, we have the option to store the parameters when the algorithm is running. 
- `this::ProxGradResults`: This is the type that the function acts on. 
- `soln::Vector`: This is the solution vector at the current iteration of the algorithm. 
"""
function Register!(
    this::ProxGradResults, 
    obj::Real, soln::Vector, 
    pgrad_map::Vector, 
    step_size::Real, 
    momentum::Real
)
    if this.itr_counter == -1
        error("ProxGrad Results is called without initiation.")
    end
    this.itr_counter += 1
    k = this.itr_counter
    push!(this.objective_vals, obj)
    push!(this.gradient_mapping_norm, norm(pgrad_map))
    push!(this.step_sizes, step_size)
    push!(this.momentums, momentum)

    if mod(k, this.collection_interval) == 0
        this.solns[k] = copy(soln)
        this.gradient_mappings[k] = copy(pgrad_map)
    end
    return nothing
end


"""
    Get all the sparsely collected solutions as an array of vectors. 
"""
function GetAllSolns(this::ProxGradResults)
    result = Vector{Vector}()
    for k in sort(keys(this.solns)|>collect)
        push!(result, this.solns[k])
    end
    return result
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

### Positional Arugments
- `g::SmoothFxn`
- `h::NonsmoothFxn`
- `x0::Vector{T1} `
- `step_size::Union{T2, Nothing}=nothing`

### Named Arguments
- `itr_max::Int=1000`
- `epsilon::AbstractFloat=1e-10`
- `line_search::Bool=false`
- `nesterov_momentum::Bool = false`
- `results_holder::ProxGradResults=ProxGradResults()`
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
    @warn "Function ProxGradient is being deprecated, use ProxGrad Custom and its variants instead. "
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
    Initiate!(results_holder, x0, h(x0) + g(x0), l, t)
    @showprogress for k in 1:itr_max
        
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
        t⁺ = nesterov_momentum ? (1 + sqrt(1 + 4*t^2))/2 : 1
        y = x⁺ + ((t - 1)/t⁺)*(x⁺ - last_x)
        t = t⁺
        last_x = x⁺

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
    results_holder.soln = last_x
    return results_holder
end



"""
This function fullfill a template for proximal gradient method. One can susbtitute different updating functions 
for the proximal gradient method. You can only update the momentum term. If more information is needed, make it 
a functor instead of a function. 

### Positional Arugments
- `g::SmoothFxn`
- `h::NonsmoothFxn`
- seq::Union{MomentumTerm, Function},
- `x0::Vector{T1} `
- `step_size::Union{T2, Nothing}=nothing`

### Named Arguments
- `itr_max::Int=1000`
- `epsilon::AbstractFloat=1e-10`
- `line_search::Bool=false`
- `nesterov_momentum::Bool = false`
- `results_holder::ProxGradResults=ProxGradResults()`
"""
function ProxGradMomentum( 
    g::SmoothFxn, 
    h::NonsmoothFxn, 
    seq::Union{MomentumTerm, Function},
    x0::Vector{T1}, 
    step_size::Union{T2, Nothing}=nothing;
    itr_max::Int=1000,
    epsilon::AbstractFloat=1e-10, 
    line_search::Bool=false, 
    results_holder::ProxGradResults=ProxGradResults()
) where {T1 <: Number, T2 <: Number}
    @assert itr_max > 0 "The maximum number of iterations is a strictly positive integers. "
    line_search = step_size === nothing ? true : line_search
    step_size = step_size === nothing ? 1 : step_size
    y = x0                                                                      # <-- This one is also for the momentum. 
    last_x = x0
    x⁺ = x0
    last∇ = Grad(g, y)
    ∇ = last∇
    last_y = y
    l = step_size                                                               # <-- initial step size. 
    Q(x, y, l) = g(x) + dot(Grad(g, x), y - x) + (norm(y - x)^2)/(2*l)          # The descent lemma.  
    last_itr = 0                                                                # Exit iteration
    
    Initiate!(results_holder, x0, h(x0) + g(x0), l)
    @showprogress for k in 1:itr_max
        x⁺ = Prox(h, l, y - l*∇)
        while (line_search && g(x⁺) > Q(y, x⁺, l) + eps(T1))                    # Line search 
            l /= 2
            x⁺ = Prox(h, l, y - l*∇)
        end
        
        pgrad_norm = norm(y - x⁺, Inf)
        θ = seq(
            ;x=y,
            last_x=last_y, 
            grad_current=∇, 
            grad_last=last∇, step_size=l
        )
        Register!(results_holder, h(x⁺) + g(x⁺), x⁺, y - x⁺, l, θ)                 # Register the results
        last_y = y
        y = x⁺ + θ*(x⁺ - last_x)                                            
        last∇ = ∇
        ∇ = Grad(g, y)
        last_x = x⁺
        if pgrad_norm < epsilon                                                 # check for termination conditions
            last_itr = k
            break
        end
    end
    # determine exit flag.
    if last_itr == itr_max
        results_holder.flags = 1
    end
    results_holder.soln = last_x
    return results_holder
end

"""
Perform Proximal Gradient with Nesterov Momentum update. 
------

### Positional Arugments
- `g::SmoothFxn`
- `h::NonsmoothFxn`
- `x0::Vector{T1} `
- `step_size::Union{T2, Nothing}=nothing`

### Named Arguments
- `itr_max::Int=1000`
- `epsilon::AbstractFloat=1e-10`
- `line_search::Bool=false`
- `nesterov_momentum::Bool = false`
- `results_holder::ProxGradResults=ProxGradResults()`
"""
function ProxGradNesterov(
    g::SmoothFxn, 
    h::NonsmoothFxn,
    x0::Vector{T1}, 
    step_size::Union{T2, Nothing}=nothing;
    itr_max::Int=1000,
    epsilon::AbstractFloat=1e-10, 
    line_search::Bool=false, 
    results_holder::ProxGradResults=ProxGradResults()
) where {T1 <: Number, T2 <: Number}
return ProxGradMomentum( 
    g, 
    h, 
    NesterovMomentum(),
    x0, 
    step_size;
    itr_max=itr_max,
    epsilon=epsilon, 
    line_search=line_search, 
    results_holder=results_holder,
) end


function ProxGradISTA(
    g::SmoothFxn, 
    h::NonsmoothFxn,
    x0::Vector{T1}, 
    step_size::Union{T2, Nothing}=nothing;
    itr_max::Int=1000,
    epsilon::AbstractFloat=1e-10, 
    line_search::Bool=false, 
    results_holder::ProxGradResults=ProxGradResults()
) where {T1 <: Number, T2 <: Number}
return ProxGradMomentum(
    g, 
    h, 
    (;kwargs...)->0,
    x0, 
    step_size;
    itr_max=itr_max,
    epsilon=epsilon, 
    line_search=line_search, 
    results_holder=results_holder,
) end


function ProxGradPolyak( 
    g::SmoothFxn, 
    h::NonsmoothFxn, 
    alpha::Number,
    x0::Vector{T1}, 
    step_size::Union{T2, Nothing}=nothing;
    itr_max::Int=1000,
    epsilon::AbstractFloat=1e-10, 
    line_search::Bool=false, 
    results_holder::ProxGradResults=ProxGradResults()
) where {T1 <: Number, T2 <: Number}
    @assert alpha <= 1 "The momentum term for ProxGradPolyak has to be less than one, but we have alpha=$alpha instead."
return ProxGradMomentum( 
    g, 
    h, 
    ()->alpha,
    x0, 
    step_size;
    itr_max=itr_max,
    epsilon=epsilon, 
    line_search=line_search, 
    results_holder=results_holder,
) end



# N = 64
# A = Diagonal(LinRange(0, 2, N))
# b = ones(N)
# g = SquareNormResidual(A, b)
# h = 0.001*OneNorm()
# soln = ProxGradient(g, h, zeros(size(b)), nesterov_momentum=true);
# using Plots
# plot(soln.gradient_mapping_norm, yaxis=:log10) |> display


