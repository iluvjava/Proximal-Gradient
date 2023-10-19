
include("../src/proximal_gradient.jl")
using LaTeXStrings, Plots

# Implement the functions here to make the objective function and experiment parameters ================================


"""
Abstract type for packaging the test instance type. It must satisfy list of functions. 
"""
abstract type GenericTestInstance

end

"""
Return a list of function that are proximal gradient algorithms and their variance. These are function in a list, the function 
should have a header of the form 
`(g::SmoothFxn, h::SmoothFxn, x0::Vector{Number}, step_size::Union{Number, Nothing}; itr_max, epsilon, line_search)`, 
the function must return a type `::ProxGradResults`. 

"""
function GetTestAlgorithms(::GenericTestInstance)
    @assert "Abstract type function `GetTestAlgorithms` for `$(typeof(this))` called, you haven't implemented something for this to be triggered. "
end

"""
Get a list of names for displaying the legends on the plot for different convergence rates of the algorithm. 
"""
function GetTestAlgorithmsNames(this::GenericTestInstance)
    @assert "Abstract function `GetTestAlgorithmsNames` for `$(typeof(this))` called, you haven't implemented something for this to be triggered. "
end

"""
Get the test parameters for experiments, Function must return the following variables in the same order specified as: 
    - `g::SmoothFxn`,
    - `h::NonsmoothFxn`, 
    - `x0::?????`, 
    - `η::Number`, the step size
"""
function GetParameters(this::GenericTestInstance)
    @assert "Abstract function `GetParameters` for `$(typeof(this))` called, "
end

# ======================================================================================================================
# EXPERIMENT EXAMPLE TEST CASE | Simple Quadratic Minimizations 
# ======================================================================================================================

mutable struct TestInstanceExample <: GenericTestInstance
    "Dimension for the problem. "
    N
    "Strong convexity index. "
    α
    "Lipschitz constant for the gradient. "
    L
    "Kappa the condition number. "
    κ
    "The matrix used for the quadratic objective function. "
    A
    "The constant vector used for the quadratic objective function. "
    b

    "(Must have) The functions that are algorithm imeplementations that we indended to test. "
    implementations::Vector{Function}
    "(Must Have) The names for the algorithms when plotting the legends. "
    names::Vector{String}
    "(Must Have)The smooth function. "
    g
    "(Must Have)The non-smooth function. "
    h
    "(Must Have)the initial guess for all test algorithms. "
    x0
    

    function TestInstanceExample(N::Int=128, alpha::Number=1e-3, L=1)
        # Establish parameters. 
        b = zeros(N)
        b[1] = -1
        h = 0.0*OneNorm()
        A = Diagonal(LinRange(α, L, N))
        g = Quadratic(A, b, 0)
        x0 = ones(N)
        @assert alpha <= L "The `alpha` should be `≤ L`, but we had L = $(L), and alpha = $(alpha). "
        # Establish test instance. 
        this = new()
        this.N = N
        this.L = L
        this.α = alpha
        this.A = A
        this.h = h
        this.g = g
        this.x0 = x0
        this.test_algorithm = [ProxGradNesterov, ProxGradISTA, ProxGradAdaptiveMomentum, 
        
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
        this.names = ["FISTA", "ISTA", "Adaptive", "Fixed Momentum"]
    
        return this 

    end

end



function GetTestAlgorithms(this::TestInstanceExample)
    return this.implementations
end


function GetTestAlgorithmsNames(this::TestInstanceExample)
    return this.names
end


function GetParameters(this::GenericTestInstance)
    return this.g, this.h, this.x0, 1/this.L
end




# My global variable for the below test instance. 
N = 256
α = 1e-3   # The alpha you want for the Hessian of ||Ax - b||^2
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

    A = Diagonal(LinRange(α, L, N))
    b = zeros(N)
    b[1] = -1

    h = 0.0*OneNorm()
    g = Quadratic(A, b, 0)
    # g = Jancky(α*ones(N), diag(A).^2)
    x0 = ones(N)
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

MAX_ITR = 1000
LINE_SEARCH = true
TOL = 1e-10
INSTANCE = TestInstanceExample()
INSTANCE_PLOTTER = nothing
PARALLEL = false



### ====================================================================================================================
### This is the actual Experiment Code
### ====================================================================================================================


if (INSTANCE|>GetTestAlgorithms |> length) == 0
    @error "There is no instance to test, var `TEST_ALGORITHMS` empty. "
end
@assert length(INSTANCE|>GetTestAlgorithmsNames) == length(INSTANCE|>GetTestAlgorithms) "The length of "*
"`TEST_ALGORITHMS` and `TEST_ALGORITHMS` doesn't match. They should. "

g, h, x0, η = INSTANCE|>GetParameters
RESULTS = Vector{ProxGradResults}()
for test_algorithm in TEST_ALGORITHMS
    push!(
        RESULTS, 
        test_algorithm(
            g, h, x0, η, 
            itr_max=MAX_ITR, line_search=LINE_SEARCH, epsilon=TOL
        )
    )
end

# Plot the norm of the gradient mapping 
FIG1 = plot(
    RESULTS[1].gradient_mapping_norm[1: end - 2], yaxis=:log10, 
    title="Gradient Mapping Norm", 
    ylabel=L"\left\Vert y_{k} - x_{k + 1} \right\Vert", 
    label=TEST_ALGORITHMS_NAMES[1], 
    xlabel="Iteration Number: k", 
    dpi=300
)
for j in 2:length(RESULTS)
    plot!(
        FIG1, RESULTS[j].gradient_mapping_norm[1:end-2], 
        yaxis=:log10,
        label=TEST_ALGORITHMS_NAMES[j]
    )
end
FIG1|>display

# Plot the objective values for different algorithms. 
MIN_OBJALL = vcat([obj.objective_vals for obj in RESULTS]...)|>minimum
MIN_OBJALL -= 2*abs(eps(Float64)*MIN_OBJALL)
FIG2 = plot(
    RESULTS[1].objective_vals .- MIN_OBJALL, 
    title="Objective Values", label=TEST_ALGORITHMS_NAMES[1], 
    ylabel=L"[f + g](x_k) - [f + g](\bar x)", xlabel="Iteration number k", 
    yaxis=:log10; 
    dpi = 300
)
for j in 2:length(RESULTS)
    plot!(
        FIG2, RESULTS[j].objective_vals .- MIN_OBJALL, 
        yaxis=:log10,
        label=TEST_ALGORITHMS_NAMES[j]
    )
end
FIG2|>display





