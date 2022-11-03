# Setup the Julia Environment. 
using Pkg 
Pkg.activate(".")
Pkg.add(["Plots","Zygote"])
Pkg.instantiate()