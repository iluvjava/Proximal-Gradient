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

#### ===================================================================================================================
#### MR SUPER JANKY FUNCTION 
#### ===================================================================================================================
"""
Janky Function Number 1. 
- `alpha::Vectors` Strong convexity index, elementwise. 
- `beta::Vectors` Beta smoothness index, elementwise

the function is elemenwise of: `(x^2)*(α[i] + β[i])/4 - (β[i] - α[i])*sin(x[i])/2)`, 
The function is lipschitz smooth with beta and strongly convex with alpha. 
"""
mutable struct Jancky <: SmoothFxn
    "vector of strong convexity index. "
    alpha::Vector
    "vector of lipschitz smooth constant. "
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


#### ===================================================================================================================

"""
Function evaluates to `dot(x, A*x)/2 + dot(b, x)+ c`
"""
mutable struct Quadratic <: SmoothFxn
    "Squared matrix. "
    A::AbstractMatrix
    "A vector. "
    b::AbstractVector
    "A constant offset for the quadratic function. "
    c::Number

    function Quadratic(A::AbstractMatrix, b::AbstractVector, c::Number)
        @assert size(A, 1) == size(A, 2) "Type `Quadratic` smooth function requires a squared matrix `A`, but instead we got "*
        "size(A) = $(size(A)). "
        @assert size(A, 1) == size(b, 1) "Type `Quadratic has unmathced dimension between matrix `A` and vector constant `b`. "
        this = new(A, b, c)
        return this 
    end
end


function (this::Quadratic)(x::AbstractVector)
    A, b, c = (this.A, this.b, this.c)
    return 0.5*dot(x, A*x) + dot(b, x) + c
end


function Grad(this::Quadratic, x::AbstractVector)
    A, b, _ = (this.A, this.b, this.c)
    @assert length(x) == length(b) "`x` passed to Grad of `::Quadratic` has the wrong dimension"    
    return 0.5*(A + A')*x + b
end