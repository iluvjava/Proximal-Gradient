"""
‖b - Ax‖^2, where b is specifically, a vector. 
"""
mutable struct SquareNormResidual <: SmoothFxn
    A::AbstractMatrix
    b::AbstractVector

    function SquareNormResidual(A::AbstractMatrix, b::AbstractVector)
        return new(A, b)
    end

end


"""
Returns ‖Ax - b‖^2
"""
function (this::SquareNormResidual)(x::AbstractVector{T}) where {T <: Number}
    return dot(this.A*x - this.b, this.A*x - this.b)/2
end


"""
Returns the gradient of the 2 norm residual function. 
"""
function Grad(this::SquareNormResidual, x::AbstractVector{T}) where {T <: Number}
    A = this.A; 
    b = this.b
    return A'*(A*x - b)
end





"""
 The logistic loss function, binary classifications. 
"""
mutable struct LogisticLoss <: SmoothFxn

end

#### ===========================================================================

"""
Janky Function Number 1. 
- `alpha::Vectors` Strong convexity index, elementwise. 
- `beta::Vectors` Beta smoothness index, elementwise
"""
mutable struct Jancky <: SmoothFxn
    alpha::Vector
    beta::Vector

    function Jancky(alpha::Vector{T}, beta::Vector{T}) where {T <: Number}
        @assert all(i -> i > 0, alpha) "Jancky only accepts α parameter that is strictly larger than zero. "
        @assert all(i -> i >= -2*eps(Float64), beta - alpha) "Janky only accepts β parameter to be larger than α parameter."
        this = new()
        this.alpha = alpha
        this.beta = beta
        return this
    end


end

"""
Evaluate the janky function and return element wise: -sin(x[i]) + (α[i] +1)/2 x^2
"""
function (this::Jancky)(x::AbstractVector{T}) where {T <: Number}
    return sum(@. (x^2)*(this.alpha + this.beta)/4 - (this.beta - this.alpha)*sin(x)/2)
end



"""
Returns the gradient of the 2 norm residual function. 
"""
function Grad(this::Jancky, x::AbstractVector{T}) where {T <: Number}
    return @. (this.alpha + this.beta)*x/2 - (this.beta - this.alpha)*cos(x)/2
end



