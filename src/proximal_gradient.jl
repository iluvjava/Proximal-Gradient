using Plots
include("abstract_types.jl")
include("non_smooth_fxns.jl")
include("smooth_fxn.jl")

"""
    Performs the proximal gradient algorithm with pre-determined a fixed stepsize. 
"""
function ProxGradient(f::SmoothFxn, g::NonsmoothFxn, stepsize::Real, itr_max::Int=1000)
    @assert iter_max > 0 "The maximum number of iterations is a strictly positive integers. "
    while k in 1:itr_max
        
    end
end


f = AbsValue()
N = 200
x = LinRange(-2, 2, N)
plot(x, Prox(0.5*f, 1.5, x))


