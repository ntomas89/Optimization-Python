import numpy as np 
import sympy as sy
from AnimateIterations.AnimationFuncs import *
from MyMathFuncs.DescentMethod import GradientDescent_ExactLineSearch

# =========================================================================================================    
# TEST INITIALISATION
# =========================================================================================================
f_counter = 0 # Number of times f was called (for computational analysis)
g_counter = 0

x = sy.IndexedBase('x')
n = 2

# f(x0,x1) = 100*(x0**2 - x1)**2 + (x0 - 1)**2 + x0**3
#   - tiene un minimo local en (0.548, 0.301) >> f = 0.369
#   - tiene un punto de inflexion en (-1.215, 1.477) >> f = 3.113
# Por tanto, dependiendo de donde empecemos la busqueda, el algoritmo puede encontrar uno u otro
fexpr = 100*(x[0]**2 - x[1])**2 + (x[0] - 1)**2 + x[0]**3#3*x[0]**3 + 2*x[0]*x[1] + 2*x[1]**2 + 7
gexpr = [sy.diff(fexpr, x[i]) for i in range(n)]

print("funcion: ", fexpr)
print("gradiente: ", gexpr)

flambdify =   sy.lambdify(x, fexpr,  "numpy") # print(flambdify([1, 0])) >> result is 100
glambdify =  [sy.lambdify(x, g_idx,  "numpy") for g_idx  in gexpr] # print(glambdify[0]([1,0])) >> results in 400

def fun(x):
    # Function handles - to count the number of times we call f(x)
    global f_counter
    f_counter += 1
    return flambdify(x)
def gfun(x):
    # Function handles - to count the number of times we call g(x)
    global g_counter
    g_counter += 1
    return np.array([gf(x) for gf in glambdify])

# =========================================================================================================    
# CALL THE ALGORITHMS
# =========================================================================================================

xVals, yVals, fVals = GradientDescent_ExactLineSearch(fexpr, fun, gfun, [x[0], x[1]], [-50,1])

# =========================================================================================================    
# PLOT THE SEARCH
# =========================================================================================================
x0_min = -100
x0_max = 100
x1_min = -100
x1_max = 100

#Animate2D(fun, x0_min, x0_max, x1_min, x1_max, xVals, yVals)
Animate3D(fun, x0_min, x0_max, x1_min, x1_max, xVals, yVals, fVals)

