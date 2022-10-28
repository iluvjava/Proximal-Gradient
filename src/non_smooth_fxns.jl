mutable struct AbsValue <: NonsmoothFxn
    input_dims::Int
end


function (::AbsValue)(arg::Vector{Reals})
    return abs.(arg)
end


"""
    argmin_u(this(u) + (1/(2λ))‖x - u‖^2)
"""
function (::AbsValue)(t::Reals, at::Vector{Reals})
    @assert lambda > 0 "The λ constant for the prox operator has to be a strictly positive real. "
    function Inner(x::Reals)
        if y >= t
            return y - t     
        else if y <= -t
            return y + t
        else
            return 0
        end
    end
    return Inner.(at)
end


"""
    ‖b - Ax‖^2, where b is specifically, a vector. 
"""
mutable struct SquareNormResidual <: SmoothFxn
    A::AbstractMatrix
    b::AbstractVector
    
    function SquareNormResidual(A::AbstractMatrix, b::Vector)
        return new(A, B)
    end
end




mutable struct PolyedralIndicator


end