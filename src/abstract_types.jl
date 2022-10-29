"""
    Can be evaluated at some point and returns a fixed value. 
"""
abstract type Fxn

end

"""
    A type the models non smooth functions, it has a simple prox operator to it. 
    * Can be proxed
    * Have a gradient, or subgradient. 
"""
abstract type NonsmoothFxn <: Fxn

end



"""
    A type the models smooth functions. 
    * Ask for the value at some point.
    * Ask for it's value at some points.
    * Has a gradient. 
"""
abstract type SmoothFxn <: Fxn

end
