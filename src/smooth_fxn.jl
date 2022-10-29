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
function (this::SquareNormResidual)(x::AbstractVector{T}) where {T <: Real}
    return norm(this.b - this.A*x)^2
end


"""
    Returns gradient of the 2 norm residual function. 
"""
function Grad(this::SquareNormResidual, x::AbstractVector{T}) where {T <: Real}
    A = this.A
    return 2*A'*(b - A*x)
end