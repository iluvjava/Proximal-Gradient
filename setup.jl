# Setup the Julia Environment. 
using Pkg 
Pkg.activate(".")
Pkg.add(["Plots","Zygote", "LaTeXStrings"])
Pkg.instantiate()