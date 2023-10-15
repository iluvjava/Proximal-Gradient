include("../src/proximal_gradient.jl")
using LaTeXStrings, Plots

"""
Performs the accelerated nesterov gradient descend, and there are many different options to choose from. 
"""
N = 128
α = 1e-1   # The alpha you want for the Hessian of ||Ax - b||^2
L = 1       # The lipschitz gradient constant. 
κ = sqrt(L/α)

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
scales = rand(N)
g = Jancky(α*ones(N), diag(A).^2)
x0 = 1e4*ones(N)



ResultsA = ProxGradNesterov(g, h, x0, 1/L, itr_max=8000, line_search=true, epsilon=1e-10)
ResultsB = ProxGradISTA(g, h, x0, 1/L, itr_max=8000, line_search=true, epsilon=1e-10)
ResultsC = ProxGradMomentum(g, h, AdaptiveMomentum() , x0, 1/L, itr_max=8000, line_search=true, epsilon=1e-10)
ResultsD = ProxGradPolyak(g, h, (κ - 1)/(κ + 1), x0, 1/L, itr_max=8000, line_search=true, epsilon=1e-10)

# Plotting the gradient mapping error. 
fig = plot(
    (ResultsA.gradient_mapping_norm)[1:end - 2], yaxis=:log10, 
    title="Gradient Mapping Norm", 
    ylabel=L"\left\Vert y_{k} - x_{k + 1} \right\Vert", label="FISTA", 
    xlabel="Iteration Number: k",
    dpi=300
)
plot!(fig, 
    (ResultsB.gradient_mapping_norm)[1:end - 2], 
    yaxis=:log10, label="ISTA"
)
plot!(fig,
    (ResultsC.gradient_mapping_norm)[1:end - 2], 
    yaxis=:log10, label="Adaptive"
)
plot!(fig, 
    ResultsD.gradient_mapping_norm[1:end-2], 
    yaxis=:log10, label="V-FISTA"
)

fig |> display
savefig(fig, "simple_lass_pgrad.png")


MIN_OBJ_IDX = min(MIN_OBJ_IDX1, MIN_OBJ_IDX2, MIN_OBJ3_IDX3, MIN_OBJ_IDX4)
MIN_OF_ALL = vcat(
    ResultsA.objective_vals,
    ResultsB.objective_vals, 
    ResultsC.objective_vals, 
    ResultsD.objective_vals
)|>minimum 
MIN_OF_ALL -= 2*eps(Float64)*MIN_OF_ALL

fig2 = plot(
    ResultsA.objective_vals .- MIN_OF_ALL,
    title="Objective Values", label="FISTA", 
    ylabel=L"[f + g](x_k) - [f + g](\bar x)", xlabel="Iteration Number: k",
    yaxis=:log10;
    dpi=300
)
plot!(
    fig2, 
    ResultsB.objective_vals .- MIN_OF_ALL, 
    label="ISTA"
)
plot!(
    fig2, 
    ResultsC.objective_vals .- MIN_OF_ALL, 
    label="Adaptive"
)

plot!(
    fig2, 
    ResultsD.objective_vals .- MIN_OF_ALL, 
    label="V-FISTA"
)

fig2 |> display
savefig(fig2, "simple_lass_obj.png")

SEQ_SHOW = typemax(Int)
fig3 = plot(
    ResultsA.momentums[1:min(SEQ_SHOW, ResultsA.momentums|>length)], 
    title="The Momentum", 
    label="FISTA θ", 
    dpi=300
)
plot!(
    fig3, ResultsC.momentums[1:min(SEQ_SHOW, ResultsC.momentums|>length)], 
    label="Adaptive"
)
plot!(
    fig3, ResultsD.momentums[1:min(SEQ_SHOW, ResultsD.momentums|>length)], 
    label="Fixed"
)
fig3|>display
