"""
For an optimization problem with several variables, the direction-finding subproblem must be solved first.
Then, a step size must be determined by searching for the minimum of the cost function along the search direction. 
This is always a 1D minimization problem, also called a line search problem.
There are two type of line search:
ANALYTICAL: we solve the minimisation problem for the one-dimensional function f(alpha)
NUMERICAL: we approximate an alpha_min
"""

def ExactLineSearch(fexpr, f, x0, x, dk):
    # Analytical line search method
    # ------------------------------
    import sympy as sy
    import numpy as np
    alphaSym     = sy.symbols('a')
    x1Sym        = x0 + alphaSym*dk
    fex_alpha    = sy.simplify(fexpr.subs([(x[0],x1Sym[0]), (x[1],x1Sym[1])])) # fex_alpha.subs(alphaSym, 0.0) == fk
    d_fex_alpha  = sy.simplify(sy.diff(fex_alpha, alphaSym))
    # dd_fex_alpha = sy.simplify(sy.diff(fex_alpha, alphaSym, alphaSym))
    alphaRoots   = sy.roots(d_fex_alpha, alphaSym, multiple=True) # only real: [r.n(10) for r in sy.real_roots(d_fex_alpha, alphaSym)]
    # d_fex_alpha.subs(alphaSym, alphaRoots[i])  = 0
    # dd_fex_alpha.subs(alphaSym, alphaRoots[i]) > 0
    alphaAux = []
    f_alphaRoots = []
    alpha = 1.0
    for i in range(len(alphaRoots)):
        if abs(float(sy.im(alphaRoots[i]))) < 1e-10:                
            alphaAux.append(sy.re(alphaRoots[i]))
            f_alphaRoots.append(abs(f(x0+alphaAux[i]*dk)))
    if len(f_alphaRoots) == 0:
        print('no real roots')
        alpha = 1.0
    else:
        alpha_index = np.argmin(f_alphaRoots)
        alpha = sy.re(alphaRoots[alpha_index])
    return alpha

""" GOLDEN SEARCH - NO TERMINADA
def GoldenSearch(fexpr, f, x0, x, dk):
    # The basic idea of the method is: evaluate the function at predetermined points, compare them to
    # bracket the minimum in Phase I, and then converge on the minimum point in Phase II by reducing
    # systematically the interval of uncertainty.
    import sympy as sy
    import numpy as np
    delta = 0.01
    golden = 1.618
    alpha = delta
    alphaList = []
    alphaList.append(alpha)
    fList = []
    fList.append(f(x0))
    fList.append(f(x0 + alpha*dk))
    if fList[i] > fList[i-1]:                
        minRange1 = alphaList[i-1]
        maxRange1 = alphaList[i]
    else:
        for i in range(1,100,1):
            alpha += (golden**i)*delta
            alphaList.append(alpha)
            fList.append(f(x0+alpha*dk))
            if fList[i] > fList[i-1]:                
                minRange1 = alphaList[i-1]
                maxRange1 = alphaList[i]

"""

def BacktrackingArmijo(f, g, p, x0, alpha, sigma, *args):
    import numpy as np 
    condition = f(x0 + alpha*p) < f(x0) + sigma*alpha*np.dot(g(x0), p)
    return condition, 0

def BacktrackingArmijoWolfe(f, g, p, x0, alpha, sigma, gamma):
    import numpy as np 
    Armijo = f(x0 + alpha*p) <= f(x0) + alpha*sigma*np.dot(g(x0), p)
    Wolfe = abs(np.dot(g(x0+alpha*p), p)) <= -gamma*np.dot(g(x0), p)
    condition = Armijo and Wolfe
    return condition, 0