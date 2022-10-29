
"""
‖b - Ax‖^2, where b is specifically, a vector. 
"""
mutable struct SquareNormResidual <: SmoothFxn
A::AbstractMatrix
b::AbstractVector

function SquareNormResidual(A::AbstractMatrix, b::Vector)
    return new(A, b)
end
end


"""
Returns ∇‖Ax - b‖^2
"""
function (this::SquareNormResidual)(x::Vector{Real})
A = this.A
return 2*A'*(b - A*x)
end
