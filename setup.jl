# Setup the Julia Environment. 
using Pkg 
Pkg.activate(".")
Pkg.add(["Plots","Zygote", "LaTeXStrings", "Images", "FileIO", "Colors", "ProgressMeter", "QuartzImageIO"])
Pkg.instantiate()