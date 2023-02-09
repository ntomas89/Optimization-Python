from sympy import *
import numpy as np
from MyMathFuncs.TaylorApproximations import TaylorNSympyFunc

# TEST

x = symbols('x')
y = symbols('y')
variable_list = [x,y]
evaluation_point = [1,2]
function_expression = 100*(x**2 - y)**2 + (x - 1)**2 + (x**3)
Taylor = TaylorNSympyFunc(function_expression, variable_list, evaluation_point, 2)
flambdify = lambdify(variable_list, function_expression,  "numpy") 
TaylorL = lambdify(variable_list, Taylor,  "numpy")
CheckApproximation = [evaluation_point[0]+np.random.normal(0, 0.3, 1), evaluation_point[1]+np.random.normal(0, 0.3, 1)]

print('===========================================================================')
print('Original function:           ',function_expression)
print('Taylor approximation:        ',Taylor)
print('Evaluation point:            ',CheckApproximation[0], CheckApproximation[1])
print('Original function value:     ',flambdify(CheckApproximation[0], CheckApproximation[1]))
print('Taylor approximation value:  ',TaylorL(CheckApproximation[0], CheckApproximation[1]))
print('===========================================================================')

