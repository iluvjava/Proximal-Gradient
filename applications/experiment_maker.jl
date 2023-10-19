
include("../src/proximal_gradient.jl")
using LaTeXStrings, Plots, LuxurySparse, SparseArrays, Arpack

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
    @error "Abstract type function `GetTestAlgorithms` for `$(typeof(this))` called, you haven't implemented something for this to be triggered. "
end

"""
Get a list of names for displaying the legends on the plot for different convergence rates of the algorithm. 
"""
function GetTestAlgorithmsNames(this::GenericTestInstance)
    @error "Abstract function `GetTestAlgorithmsNames` for `$(typeof(this))` called, you haven't implemented something for this to be triggered. "
end

"""
Get the test parameters for experiments, Function must return the following variables in the same order specified as: 
    - `g::SmoothFxn`,
    - `h::NonsmoothFxn`, 
    - `x0::?????`, 
    - `η::Number`, the step size
"""
function GetParameters(this::GenericTestInstance)
    @error "Abstract function `GetParameters` for `$(typeof(this))` called, you haven't implemented the method for your type yet. "
end


"""
The function will be call at the end of the tests to return the results for the test instance. So the test instance 
    can do further specialize testing if it wants. 
"""
function RegisterResultsPostProcessing(this::GenericTestInstance, ::Vector{ProxGradResults})
    @error "Abstract function `RegisterResultsPostProcessing` for `$(typeof(this))` is called, you haven't implemented the method for your type yet. "
end

# ======================================================================================================================
# EXPERIMENT EXAMPLE TEST CASE | Simple Quadratic Minimizations 
# ======================================================================================================================

mutable struct TestInstanceExample <: GenericTestInstance
    "(Must have) The functions that are algorithm imeplementations that we indended to test. "
    implementations::Vector{Function}
    "(Must Have) The names for the algorithms when plotting the legends. "
    names::Vector{String}
    "(Must Have)The smooth function. "
    g
    "(Must Have)The non-smooth function. "
    h
    "(Must Have) the initial guess for all test algorithms. "
    x0
    "The results of the test instances"
    results::Union{Vector{ProxGradResults}, Nothing}

    
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

    function TestInstanceExample(N::Int=2048, alpha::Number=1e-3, L=1)
        # Establish parameters. 
        b = zeros(N)
        b[1] = -1
        κ = L/alpha
        h = 0.1*OneNorm()
        A = Diagonal(LinRange(alpha, L, N))
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
        this.results = nothing
        this.implementations = [ProxGradNesterov, ProxGradISTA, ProxGradAdaptiveMomentum, 
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
                (;kwargs...)->(sqrt(κ) - 1)/(sqrt(κ) + 1),
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


function GetParameters(this::TestInstanceExample)
    return this.g, this.h, this.x0, 1/this.L
end

function RegisterResultsPostProcessing(this::TestInstanceExample, results::Vector{ProxGradResults})
    @info "Experiment done. "
end




"""
Construct an instance of total variation minimization problem for the simple test algorithm. We use Uzawa's algorithm
to seek the minimum of the dual and then solve for the primal solution. 

# Constructor Parameters
- 

"""
mutable struct TVMin1D <: GenericTestInstance
    
    # Test instance parameters. 
    "(Must have) The functions that are algorithm imeplementations that we indended to test. "
    implementations::Vector{Function}
    "(Must Have) The names for the algorithms when plotting the legends. "
    names::Vector{String}
    "(Must Have)The smooth function. "
    g
    "(Must Have)The non-smooth function. "
    h
    "The dual variable solution. "
    x0
    "The results of the test instances"
    results::Union{Vector{ProxGradResults}, Nothing}

    # Problem parameters: 
    "The total signal length including the boundary points. "
    N::Number
    "The noise corrupted signal. "
    u_hat::Vector
    "The time ticks. "
    t::Vector
    "The signal interval diaginal matrix, it's for the approximated Trapzoid rule. "
    D::AbstractMatrix
    "The first order finite difference matrix. "
    C::AbstractMatrix
    "Lipschitz constant. "
    L::Number
    "Strong convexity number. "
    alpha::Number

    

    function TVMin1D(time_ticks::Vector{T1}, signal::Vector{T2}, alpha::Number) where {T1<: Number, T2<:Number}
        @assert length(time_ticks) == length(signal) "The signal length and the time ticks label doesn't equal, we have "*
        "signal length of $(length(signal)), and time tick length of $(length(time_ticks)). "
        interval_lengths = time_ticks[2:end] - time_ticks[1:end - 1]
        @assert all(i -> i > 0, interval_lengths) "The `time_ticks` are not ordered correctly, its element is not strictly monotone increasing. "
        @assert alpha > 0 "Variable alpha should be greater than zero strictly. "
        # Setting up problem parameers -----------------------------------------
        this = new()
        this.results = nothing 
        N = this.N = length(signal)
        û = this.u_hat = signal
        this.t = time_ticks
        # D matrix...
        D = Diagonal(vcat(interval_lengths[1], 2.0*interval_lengths[1:end - 1], interval_lengths[end]))
        D[1, 1] = (1/2)*D[1, 1]
        D[end, end] = (1/2)*D[end, end]
        this.D = convert(Diagonal{Float64}, D)
        # C matrix... 
        row_idx = Vector{Float64}()
        col_idx = Vector{Float64}()
        vals = Vector{Float64}()
        for i = 1:N-1
            push!(row_idx, i)
            push!(col_idx, i)
            push!(vals, -1.0)
            push!(row_idx, i)
            push!(col_idx, i + 1)
            push!(vals, 1.0)
        end
        C = sparse(row_idx, col_idx, vals)
        this.C = C
        # Dual Objective function smooth and non-smooth. 
        A = C*inv(D)*C'
        b = C*û 
        this.g = Quadratic(A, b, 0)
        this.h = HyperRectanguloidIndicator(-fill(alpha, N - 1), fill(alpha, N - 1))
        this.x0 = zeros(N - 1)
        # Setting up experiment parameters -------------------------------------
        # Estimate kappa for fixed step momentum method. 
        L = this.L = eigs(A, nev=1, which=:LM)[1][1]
        α = this.alpha = eigs(A, nev=1, which=:SM, maxiter=3000, tol=1e-10)[1][1]
        ConstructImplementations!(this, L/α)
        
        return this 
    end

    function TVMin1D(N::Int=64,alpha::Number=10) 
        return TVMin1D(1:2N|>collect, vcat(-ones(N), ones(N)) .+ 0.3rand(2N), alpha)
    end

end


function ConstructImplementations!(this::TVMin1D, kappa::Number)
    κ = kappa
    this.implementations = [ProxGradNesterov, ProxGradISTA, ProxGradAdaptiveMomentum, 
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
        (;kwargs...)->(sqrt(κ) - 1)/(sqrt(κ) + 1),
        x0, 
        step_size;
        itr_max=itr_max,
        epsilon=epsilon, 
        line_search=line_search, 
        results_holder=results_holder,
    ) end]
    this.names = ["FISTA", "ISTA", "Adaptive", "Fixed Momentum"]

    return nothing
end

function GetTestAlgorithms(this::TVMin1D)
    return this.implementations
end


function GetTestAlgorithmsNames(this::TVMin1D)
    return this.names
end

function GetParameters(this::TVMin1D)
    return this.g, this.h, this.x0, 1/this.L
end

function RecoverPrimalSolution(this::TVMin1D, soln::AbstractVector)
    û, D, C = (this.u_hat, this.D, this.C)
    return û + inv(D)*C'*soln
end


function RegisterResultsPostProcessing(this::TVMin1D, results::Vector{ProxGradResults})
    getprimal(x) = RecoverPrimalSolution(this, x)
    toplot = [getprimal(r.soln) for r in results]
    println("To plot: ")
    println(typeof(toplot[1]))
    
    fig = plot(this.u_hat, color=RGBA{Float64}(1, 0, 0, 0.2), legend=:bottomright, label="signal with noise")
    for (j, data) in toplot|>enumerate
        plot!(fig, data, label=this.names[j])
    end
    
    display(fig)
    return nothing
end




# ======================================================================================================================
# EXPERIMENT PROFILE, SETTINGS 
# ======================================================================================================================

"The name is for naming the folder for an instance for the experiment. "
EXPERIMENT_NAME = "Experiment1"
"The folder to put the specific test instance plots and data. "
RESULTS_FOLDER = "../experiment_results/"
"""
`TEST_ALGORITHMS` is a list of generic function that must has function header of: 
    `(g::SmoothFxn, h::SmoothFxn, x0::Vector{Number}, step_size::Union{Number, Nothing}; itr_max, epsilon, line_search)`
"""

MAX_ITR = 1500
LINE_SEARCH = true
TOL = 1e-10
INSTANCE = TVMin1D(64, 5)
INSTANCE_PLOTTER = nothing
PARALLEL = false



### ====================================================================================================================
### This is the actual Experiment Code
### ====================================================================================================================


TEST_ALGORITHMS_NAMES = INSTANCE |> GetTestAlgorithmsNames

if (INSTANCE|>GetTestAlgorithms |> length) == 0
    @error "There is no instance to test, var `TEST_ALGORITHMS` empty. "
end
@assert length(INSTANCE|>GetTestAlgorithmsNames) == length(INSTANCE|>GetTestAlgorithms) "The length of "*
"`TEST_ALGORITHMS` and `TEST_ALGORITHMS` doesn't match. They should. "

g, h, x0, η = INSTANCE|>GetParameters
RESULTS = Vector{ProxGradResults}()
for test_algorithm in INSTANCE |> GetTestAlgorithms
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

SEQ_RNG = typemax(Int)
FIG3 = plot(
    RESULTS[1].momentums[1:min(SEQ_RNG, RESULTS[1].momentums|>length)], 
    title="The Momentum Multiplier", 
    xlabel="Iteration number k", 
    label=TEST_ALGORITHMS_NAMES[1], 
    dpi=300
)

for j in 2:length(RESULTS)
    plot!(
        FIG3, 
        RESULTS[j].momentums[1:min(SEQ_RNG, RESULTS[j].momentums|>length)], 
        label=TEST_ALGORITHMS_NAMES[j] 
    )
end

FIG3 |> display

RegisterResultsPostProcessing(INSTANCE, RESULTS)