import numpy as np 
import matplotlib.pyplot as plt
import sympy as sy
from AnimateIterations.AnimationFuncs import *
from MyMathFuncs.QuasiNewtonMethod import QuasiNewton_DFP, QuasiNewton_BFGS


f_counter = 0 # Number of times f was called (for computational analysis)
g_counter = 0
H_counter = 0

x = sy.IndexedBase('x')
n = 2

# f(x0,x1) = 100*(x0**2 - x1)**2 + (x0 - 1)**2 + x0**3
#   - tiene un minimo local en (0.548, 0.301) >> f = 0.369
#   - tiene un punto de inflexion en (-1.215, 1.477) >> f = 3.113
# Por tanto, dependiendo de donde empecemos la busqueda, el algoritmo puede encontrar uno u otro

fexpr = (1- x[0])**2 + 100*(x[1] - x[0]**2)**2 #100*(x[0]**2 - x[1])**2 + (x[0] - 1)**2 + (x[0]**3)#
gexpr = [sy.diff(fexpr, x[i]) for i in range(n)]
Hexpr = [[sy.diff(g, x[i]) for i in range(n)] for g in gexpr]

flambdify =   sy.lambdify(x, fexpr,  "numpy") # print(flambdify([1, 0])) >> result is 100
glambdify =  [sy.lambdify(x, g_idx,  "numpy") for g_idx  in gexpr] # print(glambdify[0]([1,0])) >> results in 400
Hlambdify = [[sy.lambdify(x, H_idx1, "numpy") for H_idx1 in H_idx2] for H_idx2 in Hexpr] # print(Hlambdify[0][0]([1,0])) >> results in 1202

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
def Hfun(x):
    # Function handles - to count the number of times we call H(x)
    global H_counter
    H_counter += 1
    return np.array([[gf(x)for gf in Hs] for Hs in Hlambdify])

x_init = [75,50]#np.random.normal(0,10,n)*5
print("=================================================================")
print("funcion: ", fexpr)
print("gradiente: ", gexpr)
print("hessiano: ", Hexpr)
print("Iniciamos la busqueda en ", x_init)
#x0_ELS, x1_ELS, f_ELS = SimpleNewton(fexpr, fun, gfun, Hfun, x, x_init, "ExactLineSearch", iterPrint=True)
#x0_BA, x1_BA, f_BA    = SimpleNewton(fexpr, fun, gfun, Hfun, x, x_init, "BacktrackingArmijo", iterPrint=False)
#x0_BAW, x1_BAW, f_BAW = SimpleNewton(fexpr, fun, gfun, Hfun, x, x_init, "BacktrackingArmijoWolfe", iterPrint=False)
x0_ELS, x1_ELS, f_ELS = QuasiNewton_DFP(fexpr, fun, gfun, Hfun, x, x_init, "ExactLineSearch", iterPrint=True)
x0_ELS, x1_ELS, f_ELS = QuasiNewton_BFGS(fexpr, fun, gfun, Hfun, x, x_init, "ExactLineSearch", iterPrint=True)

#==========================================================================
# Animate search
x0_min = -100
x0_max = 100
x1_min = -100
x1_max = 100

#Animate2D(fun, x0_min, x0_max, x1_min, x1_max, x0_ELS, x1_ELS)
Animate3D(fun, x0_min, x0_max, x1_min, x1_max, x0_ELS, x1_ELS, f_ELS)
#==========================================================================