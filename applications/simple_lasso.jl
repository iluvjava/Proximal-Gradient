include("../src/proximal_gradient.jl")
using LaTeXStrings, Plots

"""
Performs the accelerated nesterov gradient descend, and there are many different options to choose from. 
"""
N = 64
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

ResultsA = ProxGradient(g, h, ones(N), 1, itr_max=2000, line_search=true, nesterov_momentum=true)
ResultsB = ProxGradient(g, h, ones(N), 1, itr_max=2000, line_search=true, nesterov_momentum=false)
fig = plot(
    (ResultsA.gradient_mapping_norm)[1:end - 2], yaxis=:log10, 
    title="Gradient Mapping Norm", 
    ylabel=L"\left\Vert x_{k + 1} - x_{k} \right\Vert", label="FISTA"
)
plot!(fig, 
    (ResultsB.gradient_mapping_norm)[1:end - 2], 
    yaxis=:log10, label="ISTA"
)
fig |> display

fig2 = plot(
    ResultsA.objective_vals[1:100], title="Objective Values", label="FISTA", 
    ylabel=L"[f + g](x_k)"
)
plot!(ResultsB.objective_vals[1:100], label="ISTA")

fig2 |> display
