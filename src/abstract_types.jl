"""
Can be evaluated at some point and returns a fixed value for each evaluated point. 
"""
abstract type Fxn

end

"""
A type the models non smooth functions, it has a simple prox operator to it. 
* Query the value at some point. 
* Can be proxed.
* Have a gradient, or subgradient. 
"""
abstract type NonsmoothFxn <: Fxn

end



"""
A type the models smooth functions. 
* Ask for the value at some point.
* Has a gradient. 
* Can be proxed (not necessary)
"""
abstract type SmoothFxn <: Fxn

end


"""
A functor for updating the momentum term for the proximal gradient algorithm. 
"""
abstract type MomentumTerm
end



### ====================================================================================================================
mutable struct NesterovMomentum <: MomentumTerm
    k::Int
    t::Vector{Float64}
    theta::Vector{Float64}
    
    function NesterovMomentum()
        this = new()
        this.k = 1
        this.t = Vector{Float64}()
        this.theta = Vector{Float64}()
        push!(this.t, 1)
        return this
    end

end

"""
Functor to compute the Nesterov Momentum Term.

"""
function (this::NesterovMomentum)(; kwargs...)
    t = this.t[this.k]
    t_next = 1/2 + sqrt(1 + 4t^2)/2
    push!(this.t, t_next)
    this.k += 1
    return (t - 1)/t_next
end



### ====================================================================================================================
"""
This is a new one. 
"""
mutable struct CubicMomentum <: MomentumTerm
    k::Int
    t::Vector{Float64}
    s::Vector{Float64}   # an intermediate parameter. 
    theta::Vector{Float64}  # the inertia point. 

    function CubicMomentum()
        this = new()
        this.t = Vector{Float64}()
        push!(this.t, 2.0)
        this.s = Vector{Float64}()
        this.theta = Vector{Float64}()
        return this
    end


end


function (this::CubicMomentum)(; kwargs...)
    t = this.t[end]
    s = t
    t⁺ = cbrt(2t + 1/27 + (1/3)*sqrt(2t/(1 + s))) - 1/3
    theta = (t - 1)/t⁺
    push!(this.t, t⁺)
    push!(this.s, s)
    push!(this.theta, theta)
    return theta
end

### ====================================================================================================================

mutable struct AdaptiveMomentum <: MomentumTerm


end
