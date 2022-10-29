

### ============================================================================
### The elementwise absolute value with a multiplier
### ============================================================================
"""
    A struct that refers to m|x|, where x is a vector and the double bar here 
        denotes the process of applying the absolute value to each element of the 
        vector x. 
"""
struct AbsValue <: NonsmoothFxn
    multiplier::Real
    function AbsValue(multiplier::Real=1)
        @assert multiplier > 0 "The multiplier for the elementwise nonsmooth"*
            "function must be strictly positive so it's convex for the prox operator. "
        return new(multiplier)
    end
end


"""
    The output of the elementwise absolute value on the function. 
"""
function (::AbsValue)(arg::AbstractArray{T}) where {T<:Real}
    return abs.(arg)
end


"""
    argmin_u(this(u) + (1/(2λ))‖x - u‖^2)
"""
function (this::AbsValue)(t::T1, at::AbstractArray{T2}) where {T1 <: Real, T2 <: Real}
    @assert t > 0 "The prox constant for the prox operator has to be a strictly positive real. "
    m = this.multiplier
    function Inner(y::Real)
        if y >= t*m
            return y - t*m     
        elseif y <= -t*m
            return y + t*m
        else
            return 0
        end
    end
    return Inner.(at)
end


"""
    Evalue the prox of the type AbsValue with a constaint t at the point x. 
"""
function Prox(this::AbsValue, t::T1, x::AbstractArray{T2}) where {T1 <: Real, T2 <: Real}
    return this(t, x)
end

"""
    A multplier function for the abslute value type multiplied with a strictly positive number. 
"""
function Base.:*(m::Real, this::AbsValue)
    return AbsValue(m*this.multiplier)
end



### ============================================================================
### Indicator for an polyhedral set. 
### ============================================================================
mutable struct PolyedralIndicator
    

end