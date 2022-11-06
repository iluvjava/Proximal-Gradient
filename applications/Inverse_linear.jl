include("../src/proximal_gradient.jl")
using LaTeXStrings, Plots, SparseArrays

"""

    Give a blur matrix for an n × m sized grid. 
    * Default Boundary Conditions: Setting it to zero. 
"""
function ConstructBlurrMatrix(m::Int, n::Int)
    function CoordToLinearIdx(i, j)
        return j*m + i
    end
    coefficients = Dict{Tuple{Int, Int}, Float64}()
    kernel = [1/16 1/8 1/16;
              1/8 1/4 1/8; 
              1/16 1/8 1/16]
    # Use a map to hold the values of the coefficients, index starts with zero. 
    for (I, J) in [(i, j) for i in 1:m for j in 1:n]
        for (III, JJJ) in [(ii + I, jj + J) for ii in -1:1 for jj in -1:1]
            coefficients[CoordToLinearIdx(I, J), CoordToLinearIdx(III, JJJ)] = 
                kernel[III - I + 2, JJJ - J + 2]
        end
    end
    
    return coefficients
end
