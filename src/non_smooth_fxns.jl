### ============================================================================
### The elementwise absolute value with a multiplier
### ============================================================================
"""
A struct that refers to m|x|, where x is a vector and the double bar here 
    denotes the process of applying the absolute value to each element of the 
    vector x. 
### Fields
- `multiplier`: `m` the multiplier on every entries of the one norm, has to be 
non-negative. 
"""
struct OneNorm <: NonsmoothFxn
    multiplier::Real

    function OneNorm(multiplier::Real=1)
        @assert multiplier >= 0 "The multiplier for the elementwise nonsmooth"*
            "function must be non-negative so it's convex for the prox operator. "
        return new(multiplier)
    end
end


"""
The output of the elementwise absolute value on the function. 
"""
function (::OneNorm)(arg::AbstractArray{T}) where {T<:Number}
    return abs.(arg)|>sum
end


"""
returns Argmin_u(this(u) + (1/(2t))‖x - u‖^2)
"""
function (this::OneNorm)(t::T1, x::AbstractArray{T2}) where {T1 <: Number, T2 <: Number}
    @assert t > 0 "The prox constant for the prox operator has to be a strictly positive real, "*
    "however we get t = $(t)"
    λ = this.multiplier
    T(z) = sign(z)*max(abs(z) - t*λ, 0)    
    # result = similar(x)
    # Parellel
    # Threads.@threads for idx in eachindex(x)
    #     result[idx] = T(x[idx])
    # end
    # return result
    return T.(x)
end


"""
Evalue the prox of the type AbsValue with a constaint t at the point x. 

### Arguments
- `this::OneNorm`: The type the function acts on. 
- `t::T1`: The scalar for the proximal operator. 
- `x::AbstractArray{T2}`: The point we are querying the prox of the one norm of. 
### Argument Type Parameters
- `T1<:Number`
- `T2<:Number`
"""
function Prox(this::OneNorm, t::T1, x::AbstractArray{T2}) where {T1 <: Number, T2 <: Number}
    return this(t, x)
end


"""
A multplier function for the abslute value type multiplied with a strictly positive number. 
"""
function Base.:*(m::Real, this::OneNorm)
    return OneNorm(m*this.multiplier)
end


"""
Take the gradient of the one norm function. at some points. 
### Arguments
- `this::OneNorm`: The type the function acts on. 
- `AbstractArray{T<:Number}`: the point we are querying the gradient of. 
"""
function Grad(this::OneNorm, x::AbstractArray{T}) where {T <: Number}
    return this.multiplier*sign.(x)
end



### ============================================================================
### Indicator function for a hyper rectangloid
### ============================================================================
mutable struct HyperRectanguloidIndicator <: NonsmoothFxn
    


end