from sympy import lambdify, symbols, solve 
import numpy as np
from MyMathFuncs.TaylorApproximations import Taylor2SympyFunc

# [LIBRO] Introduction to Optimum Design, 4th Ed
#   - Ec. (4.12) Taylor 2 vars
#   - Ec. (4.13) Taylor 2 vars matrix notation

x = symbols('x')
y = symbols('y')
variable_list = [x,y]

# Funcion a aproximar
f_expr = 100*(x**2 - y)**2 + (x - 1)**2 + (x**3)
f = lambdify(variable_list, f_expr,  "numpy")

eval_point = [1, 2]
T2_expr, T2, x_crit, y_crit = Taylor2SympyFunc(f_expr, variable_list, eval_point)

test_point = [1.3, 2.6]
print('===========================================================================')
print('Original function:           ',f_expr)
print('Taylor approximation:        ',T2_expr)
print('Min of Taylor approximation: ','x=',x_crit, ', y=',y_crit)
print('Test point:                  ',test_point)
print('Original function value:     ',f(test_point[0], test_point[1]))
print('Taylor approximation value:  ',T2(test_point[0], test_point[1]))
print('===========================================================================')

# MAS COSITAS
#print(f_expr.series(variable_list, [1,2], 2))
#T2xCoef = T2x.coeff(x, 1)
#RaizX = np.linalg.solve([T2x, T2y],[0, 0])
