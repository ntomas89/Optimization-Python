import sympy as sy
import numpy as np
from sympy.functions import sin,cos
import matplotlib.pyplot as plt
from MyMathFuncs.TaylorApproximations import Taylor1SympyFunc

plt.style.use("ggplot")

# Define the variable and the function to approximate
x = sy.Symbol('x')
f = sin(x) + x**3
flambdify =   sy.lambdify(x, f,  "numpy")

# TEST
n = 4       # Taylor expansion up to grade n
point = 0.5  # Approximation around this point
x_lims = [point-5,point+5]                  # Plot limits
evalRange = np.linspace(x_lims[0], x_lims[1], 500)   # Plot space
y1 = []

print('Function=',f)
for j in range(1,n,1):
    taylor_j = Taylor1SympyFunc(f, point, j)
    print('Taylor expansion at n='+str(j),taylor_j)
    for k in evalRange:
        y1.append(taylor_j.subs(x,k))
    plt.plot(evalRange, y1, linewidth=1, linestyle='dashed', label='order '+str(j))
    y1 = []

# Plot the function to approximate
plt.plot(evalRange, flambdify(evalRange), linewidth=3, label='function to approximate')
plt.xlim(x_lims)
plt.ylim([flambdify(point)-5,flambdify(point)+5])
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.title('Taylor series approximation')
plt.show()

