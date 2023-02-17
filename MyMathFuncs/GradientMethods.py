def GradientDescent(fex, f, g, x, xInit, searchMethod):
    import numpy as np 
    import sympy as sy
    from MyMathFuncs.SearchMethods import ExactLineSearch, BacktrackingArmijo, BacktrackingArmijoWolfe

    # =========================================================================================================
    # DESCRIPTION
    # =========================================================================================================
    # Gradient's method starts with a random point and generates a 1st order Taylor expansion of the function
    # to be minimized. Then iterates in the direction of the decreasing gradient:
    #       x_k1 = x_k0 + alpha*p
    # Direction
    # ---------
    # The basic requirement for p is that the cost function be reduced if we take a small step along p. For the
    # gradient method (or steepest-descent), it is:
    #       p = -f'(x_k0)
    # Step:
    # -----
    # Can be computed with an analytical, 1-dimensional (exact line search) method: 
    #       alpha_k = argmin(f(x_k + alpha*p)):
    #       set (d/dalpha)(x_k + alpha*p) = 0 and solve for alpha    #
    # Algorithm drawbacks
    # -------------------
    # 1. Even if convergence of the steepest–descent method is guaranteed, a large number of iterations may be
    #    required to reach the minimum point.
    # 2. Each iteration of the method is started independently of others, which can be inefficient.
    #    Information calculated at the previous iterations is not used.
    # 3. Only first-order information about the function is used at each iteration to determine the search direction. 
    #    This is one reason that convergence of the method is slow. The rate of convergence of the steepest–descent 
    #    method depends on the condition number of the Hessian of the cost function at the optimum point. If the 
    #    condition number is large, the rate of convergence of the method is slow.
    # 4. Practical experience with the steepest–descent method has shown that a substantial decrease in the cost 
    #    function is achieved in the initial few iterations and then this decreases slows considerably in later 
    #    iterations.

    #============================
    # Plot varible initialisation
    xVals = []
    yVals = []
    fVals = []
    xVals.append(xInit[0])
    yVals.append(xInit[1])
    fVals.append(f(xInit))
    #============================
    nIterMax = 500
    epsilon = 1e-10
    k = 0
    x0 = xInit
    
    while k < nIterMax:
        # Compute the “exact” solution, dk, of the well-determined, i.e., full rank, linear
        # matrix equation: Hk * dk = gk
        
        fk = f(x0)
        gk = g(x0)
        gk = gk.astype('float64') # sin esto, gk es datatype = object no se por que. Y da error
        dk = -1.0*gk

        # Check if dk is a direction of descent at point x0: if not, then WHAT
        if np.dot(gk,dk) > 0:
            print('Direction', dk, 'is not a descent direction')

        # Stop condition:
        if k>0 and np.linalg.norm(gk) < epsilon:
            break

        if searchMethod == "ExactLineSearch":             
            alpha = ExactLineSearch(fex, f, x0, x, dk)
        elif searchMethod == "BacktrackingArmijo":
            alpha, idx = BacktrackingArmijo(f, g, dk, x0, 1.0, 0.3)
        elif searchMethod == "BacktrackingArmijoWolfe":
            alpha, idx = BacktrackingArmijoWolfe(f, g, dk, x0, 1.0, 0.1, 0.9)
        else:
            print('please check the input argument for the search method')

        step = alpha*dk
        #step = step.astype('float64')
        # fk1 = f(x0 + step)
        x0 += step        
        # ------------------------------

        # Print at each iteration
        print('Iter ',k, ': ', 'fk:',round(fk,5),', x:',x0, 'alpha: ',round(alpha,6), idx)        
        #print('          ','Exact Line: ',x0[0], x0[1], ' >> ', f(x0))

        #============================        
        # Plot variables updating
        xVals.append(x0[0])
        yVals.append(x0[1])
        fVals.append(f(x0))
        #============================
        k += 1
    
    # Print at final iteration
    print("Iter ", k, ", x = ", x0)
    print("Iter ", k, ", f(x) = ", round(f(x0),3))
    print("Iter ", k, ", error = ", np.linalg.norm(dk))

    # Return values for each iteration: only for plot purposes
    return xVals, yVals, fVals

def ConjugateGradient(fex, f, g, x, xInit, searchMethod):
    import numpy as np 
    import sympy as sy
    from MyMathFuncs.SearchMethods import ExactLineSearch, BacktrackingArmijo, BacktrackingArmijoWolfe

    # =========================================================================================================
    # DESCRIPTION
    # =========================================================================================================
    # The steepest–descent directions at two consecutive steps are orthogonal to each other. This tends to slow 
    # down the steepest–descent method, although it is guaranteed to converge to a local minimum point. The 
    # conjugate gradient directions are not orthogonal to each other. R ather, these directions tend to cut 
    # diagonally through the orthogonal steepest–descent directions. Therefore, they improve the rate of 
    # convergence of the steepest–descent method considerably. 
    #       x_k = x_k-1 + alpha*p
    # Direction
    # ---------
    # In this equation, the current steepest–descent direction is modified by adding a scaled direction that was 
    # used in the previous iteration:
    #       p = -f'(x_k) + b_k*p_k-1
    #       b_k = norm(f'(x_k)) / norm(f'(x_k-1))
    # Step:
    # -----
    # Can be computed with an analytical, 1-dimensional (exact line search) method: 
    #       alpha_k = argmin(f(x_k + alpha*p)):
    #       set (d/dalpha)(x_k + alpha*p) = 0 and solve for alpha    #
    # Algorithm properties
    # -------------------
    # The conjugate gradient algorithm finds the minimum in n iterations for positive definite quadratic functions 
    # having n design variables. or general functions, if the minimum has not been found by then, the iterative 
    # process needs to be restarted every (n + 1)th iteration for computational stability

    #============================
    # Plot varible initialisation
    xVals = []
    yVals = []
    fVals = []
    xVals.append(xInit[0])
    yVals.append(xInit[1])
    fVals.append(f(xInit))
    #============================
    nIterMax = 50
    epsilon = 1e-10
    k = 0
    x0 = xInit
    dk = 0.0
    gk = 0.0
    while k < nIterMax:      
        d_prev = dk
        g_prev = gk
        fk = f(x0)
        gk = g(x0)
        gk = gk.astype('float64') # sin esto, gk es datatype = object no se por que. Y da error

        # Compute the direction:
        if k==0: 
            b = 0.0
        else: 
            b = np.dot(gk,gk)/np.dot(g_prev,g_prev)

        dk = -1.0*gk + b * d_prev

        # Check if dk is a direction of descent at point x0: if not, then WHAT
        if np.dot(gk,dk) > 0:
            print('Direction', dk, 'is not a descent direction')

        # Stop condition:
        if k>0 and np.linalg.norm(gk) < epsilon:
            break

        if searchMethod == "ExactLineSearch":             
            alpha = ExactLineSearch(fex, f, x0, x, dk)
        elif searchMethod == "BacktrackingArmijo":
            alpha = BacktrackingArmijo(f, g, dk, x0, 1.0, 0.3, 2.0)
        elif searchMethod == "BacktrackingArmijoWolfe":
            alpha = BacktrackingArmijoWolfe(f, g, dk, x0, 1.0, 0.3, 2.0)
        else:
            print('please check the input argument for the search method')

        step = alpha*dk
        step = step.astype('float64')
        # fk1 = f(x0 + step)
        x0 += step        
        # ------------------------------

        # Print at each iteration
        print('Iter ',k, ': ', 'fk:',fk,', x:',x0[0], x0[1])        
        #print('          ','Exact Line: ',x0[0], x0[1], ' >> ', f(x0))

        #============================        
        # Plot variables updating
        xVals.append(x0[0])
        yVals.append(x0[1])
        fVals.append(f(x0))
        #============================

        k += 1
    
    # Print at final iteration
    print("Iter ", k, ", x = ", x0)
    print("Iter ", k, ", f(x) = ", round(f(x0),3))
    print("Iter ", k, ", error = ", np.linalg.norm(gk))

    # Return values for each iteration: only for plot purposes
    return xVals, yVals, fVals