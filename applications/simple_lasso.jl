include("../src/proximal_gradient.jl")
using LaTeXStrings, Plots

"""
Performs the accelerated nesterov gradient descend, and there are many different options to choose from. 
"""
N = 2048
A = Diagonal(LinRange(0.1, 2, N)[end:-1:1])
b = zeros(N)

for II in eachindex(b)
    if mod(II, 2) == 1
        b[II] = A[II, II]
    else
        b[II] = rand()*1e-3
    end
end

h = 0.0*OneNorm()
g = SquareNormResidual(A, b)

ResultsA = ProxGradNesterov(g, h, 3*ones(N), 0.1, itr_max=8000, line_search=true, epsilon=1e-10)
ResultsB = ProxGradISTA(g, h, 3*ones(N), 0.1, itr_max=8000, line_search=true, epsilon=1e-10)
ResultsC = ProxGradMomentum(g, h, AdaptiveMomentum() , 3*ones(N), 0.1, itr_max=8000, line_search=true, epsilon=1e-10)

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
    yaxis=:log10, label="Cubic"
)

fig |> display
savefig(fig, "simple_lass_pgrad.png")

# Plotting the objective value of the function. 
MIN_OBJ_IDX1 = argmin(ResultsA.objective_vals)
MIN_OBJ_IDX2 = argmin(ResultsB.objective_vals)
MIN_OBJ3_IDX3 = argmin(ResultsC.objective_vals)
Min_Obj_Idx = min(MIN_OBJ_IDX1, MIN_OBJ_IDX2, MIN_OBJ3_IDX3)

fig2 = plot(
    ResultsA.objective_vals[1:min(Min_Obj_Idx - 1, length(ResultsA.objective_vals))] .- minimum(ResultsA.objective_vals),
    title="Objective Values", label="FISTA", 
    ylabel=L"[f + g](x_k) - [f + g](\bar x)", xlabel="Iteration Number: k",
    yaxis=:log10;
    dpi=300
)
plot!(
    fig2, 
    ResultsB.objective_vals[1:min(Min_Obj_Idx - 1, length(ResultsB.objective_vals))] .- minimum(ResultsB.objective_vals), 
    label="ISTA"
)
plot!(
    fig2, 
    ResultsC.objective_vals[1:min(Min_Obj_Idx - 1, length(ResultsC.objective_vals))] .- minimum(ResultsC.objective_vals), 
    label="Adaptive"
)
fig2 |> display
savefig(fig2, "simple_lass_obj.png")

SEQ_SHOW = 800
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
fig3|>display
