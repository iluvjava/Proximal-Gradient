
include("../src/proximal_gradient.jl")
using LaTeXStrings, Plots

# Implement the functions here to make the objective function and experiment parameters ================================


# My global variable for the below test instance. 
N = 128
α = 1e-5   # The alpha you want for the Hessian of ||Ax - b||^2
L = 1       # The lipschitz gradient constant. 
κ = sqrt(L/α)

"""
`function GetExperimentFxnInstances()`

Function can take in whatever, but it must return, and in the following order: 
- g::SmoothFxn,
- h::NonsmoothFxn, 
- x0::?????, 
- η::Number
a smooth, nonsmooth function, and an initial guess that is compatible with the functions
returned. 
"""
function TestInstance1()

    A = Diagonal(LinRange(sqrt(α), sqrt(L), N)[end:-1:1])
    b = zeros(N)

    for II in eachindex(b)
        if mod(II, 2) == 1
            b[II] = A[II, II]
        else
            b[II] = rand()*1e-3
        end
    end

    h = 0.0*OneNorm()
    # g = SquareNormResidual(A, b)
    g = Jancky(α*ones(N), diag(A).^2)
    x0 = 1e4*ones(N)
    return g, h, x0, L
end





# Implement the following function to make additional visualization. 


# ======================================================================================================================
# EXPERIMENT PROFILE
# ======================================================================================================================

"The name is for naming the folder for an instance for the experiment. "
EXPERIMENT_NAME = "Experiment1"
"The folder to put the specific test instance plots and data. "
RESULTS_FOLDER = "../experiment_results/"
"""
`TEST_ALGORITHMS` is a list of generic function that must has function header of: 
    `(g::SmoothFxn, h::SmoothFxn, x0::Vector{Number}, step_size::Union{Number, Nothing}; itr_max, epsilon, line_search)`
"""
TEST_ALGORITHMS = [ProxGradNesterov, ProxGradISTA, ProxGradAdaptiveMomentum, 
    # A fancy lambda function due parametric dependence on test isntance parameter. 
    function()(
        g::SmoothFxn, 
        h::NonsmoothFxn, 
        x0::Vector{T1}, 
        step_size::Union{T2, Nothing}=nothing;
        itr_max::Int=1000,
        epsilon::AbstractFloat=1e-10, 
        line_search::Bool=false, 
        results_holder::ProxGradResults=ProxGradResults()
    ) where {T1 <: Number, T2 <: Number}
    return ProxGradGeneric( 
        g, 
        h, 
        (;kwargs...)->(κ - 1)/(κ + 1),
        x0, 
        step_size;
        itr_max=itr_max,
        epsilon=epsilon, 
        line_search=line_search, 
        results_holder=results_holder,
    ) end]

"""
A list of names for the algorithms, they are for displaying on the plots. 
"""
TEST_ALGORITHMS_NAMES = ["FISTA", "ISTA", "Adaptive", "Fixed Momentum"]


# Experiment Settings. 

MAX_ITR = 8000
LINE_SEARCH = true
TOL = 1e-10
INSTANCE = TestInstance1
INSTANCE_PLOTTER = nothing
PARALLEL = false



"""
Use relevant info to construct experiment if possible. 

"""
function PerformExperiment()
    # Sanity check. 
    if (TEST_ALGORITHMS |> length) == 0
        @error "There is no instance to test, var `TEST_ALGORITHMS` empty. "
    end
    @assert length(TEST_ALGORITHMS_NAMES) == length(TEST_ALGORITHMS) "The length of "*
    "`TEST_ALGORITHMS` and `TEST_ALGORITHMS` doesn't match. They should. "


    g, h, x0, η = INSTANCE()
    results_list = Vector{ProxGradResults}()
    for test_algorithm in TEST_ALGORITHMS
        push!(
            results_list, 
            test_algorithm(
                g, h, x0, η, 
                itr_max=MAX_ITR, line_search=LINE_SEARCH, epsilon=TOL
            )
        )
    end
    # Plot the norm of the gradient mapping 
    fig1 = plot(
        (results_list[1].gradient_mapping_norm[1: end - 2]), yaxis=:log10, 
        title="Gradient Mapping Norm", 
        ylabel=L"\left\Vert y_{k} - x_{k + 1} \right\Vert", 
        label=TEST_ALGORITHMS_NAMES[1], 
        xlabel="Iteration Number: k", 
        dpi=300
    )
    for j in 2:length(results_list)
        plot!(
            fig1, results_list[j].gradient_mapping_norm[1:end-2], 
            yaxis=:log10,
            label=TEST_ALGORITHMS_NAMES[j]
        )
    end
    fig1|>display
    return nothing
end

PerformExperiment()


