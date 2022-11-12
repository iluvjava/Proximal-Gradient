include("../src/proximal_gradient.jl")
using LaTeXStrings, Plots

"""
Performs the accelerated nesterov gradient descend, and there are many different options to choose from. 
"""
N = 128
A = Diagonal(LinRange(0, 2, N)[end:-1:1])
b = zeros(N)

for II in eachindex(b)
    if mod(II, 2) == 1
        b[II] = A[II, II]
    else
        b[II] = rand()*1e-4
    end
end

h = 0.1*OneNorm()
g = SquareNormResidual(A, b)

ResultsA = ProxGradient(g, h, ones(N), 0.1, itr_max=2000, line_search=true, nesterov_momentum=true)
ResultsB = ProxGradient(g, h, ones(N), 0.1, itr_max=2000, line_search=true, nesterov_momentum=false)
fig = plot(
    (ResultsA.gradient_mapping_norm)[1:end - 2], yaxis=:log10, 
    title="Gradient Mapping Norm", 
    ylabel=L"\left\Vert x_{k + 1} - x_{k} \right\Vert", label="FISTA", 
    xlabel="Iteration Number: k",
    dpi=300
)
plot!(fig, 
    (ResultsB.gradient_mapping_norm)[1:end - 2], 
    yaxis=:log10, label="ISTA"
)
fig |> display
savefig(fig, "simple_lass_pgrad.png")

fig2 = plot(
    ResultsA.objective_vals[1:min(100, length(ResultsA.objective_vals))], title="Objective Values", label="FISTA", 
    ylabel=L"[f + g](x_k)", xlabel="Iteration Number: k",
    dpi=300
)
plot!(
    ResultsB.objective_vals[1:min(100, length(ResultsB.objective_vals))], 
    label="ISTA"
)

fig2 |> display
savefig(fig2, "simple_lass_obj.png")
