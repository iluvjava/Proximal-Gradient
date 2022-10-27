"""
    A type the models non smooth functions. 
    * Ask for the proximal operator.
    * Asks for its value at some point. 
    * Asks for any subgradient from a weak oracle. 
"""
abstract type NonsmoothFxn

end

"""
    A type the models smooth functions. 
    * Ask for the value at some point.
    * Ask for it's value at some points.
    * Asks for a subgradient from a weak subgradient oracle. 
    * Ask for a gradient. 
"""
abstract type SmoothFxn <: NonsmoothFxn

end