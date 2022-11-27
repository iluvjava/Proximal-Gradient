using Plots, LaTeXStrings

g(x) = log(exp(x) + 1)
h(x) = abs(x)
∇g(x) = (exp(x))/(exp(x) + 1)
m(x, y) = h(y) + g(x) + ∇g(x)*(y - x) + (x - y)^2
P(x, b) = sign(x)*max(abs(x) - b, 0)
x_points = LinRange(-3, 3, 100)|>collect
fig = plot(
    x_points, 
    map((x)->m(1, x), x_points), 
    label="Uppder Bounding Functions", 
    dpi=200
)
plot!(fig, x_points, g.(x_points) + h.(x_points), label="log(exp(x) + 1)")
savefig(fig, "the_upperbounding_fxn") 
fig|>display
