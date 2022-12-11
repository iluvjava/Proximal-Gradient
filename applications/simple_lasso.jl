include("../src/proximal_gradient.jl")
using LaTeXStrings, Plots

"""
Performs the accelerated nesterov gradient descend, and there are many different options to choose from. 
"""
N = 32
A = Diagonal(LinRange(0, 2, N)[end:-1:1])
b = zeros(N)

for II in eachindex(b)
    if mod(II, 2) == 1
        b[II] = A[II, II]
    else
        b[II] = rand()*1e-3
    end
end

h = 0.01*OneNorm()
g = SquareNormResidual(A, b)

ResultsA = ProxGradNesterov(g, h, 3*ones(N), 0.2, itr_max=8000, line_search=false, epsilon=1e-10)
ResultsB = ProxGradISTA(g, h, 3*ones(N), 0.2, itr_max=8000, line_search=false, epsilon=1e-10)
ResultsC = ProxGradMomentum(g, h, CubicMomentum() , 3*ones(N), 0.2, itr_max=8000, line_search=false, epsilon=1e-10)

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
Fista_Min_Obj = argmin(ResultsA.objective_vals)
Polyak_Min_Obj = argmin(ResultsC.objective_vals)
Min_Obj_Idx = min(Fista_Min_Obj, Polyak_Min_Obj)

fig2 = plot(
    ResultsA.objective_vals[1:min(Min_Obj_Idx - 1, length(ResultsA.objective_vals))] .- minimum(ResultsA.objective_vals),
    title="Objective Values", label="FISTA", 
    ylabel=L"[f + g](x_k) - [f + g](\bar x)", xlabel="Iteration Number: k",
    yaxis=:log10;
    dpi=300
)
plot!(
    ResultsB.objective_vals[1:min(Min_Obj_Idx - 1, length(ResultsB.objective_vals))] .- ResultsB.objective_vals[end], 
    label="ISTA"
)
plot!(
    ResultsC.objective_vals[1:min(Min_Obj_Idx - 1, length(ResultsB.objective_vals))] .- ResultsB.objective_vals[end], 
    label="Cubic"
)
fig2 |> display
savefig(fig2, "simple_lass_obj.png")
