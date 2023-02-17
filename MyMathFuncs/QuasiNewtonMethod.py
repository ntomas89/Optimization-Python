def QuasiNewton_DFP(fex, f, g, H, x, x_init, searchMethod, iterPrint = False):
    """
    The DFP method, initially proposed by Davidon (1959), was modified by Fletcher and Powell (1963)
    """
    from MyMathFuncs.SearchMethods import ExactLineSearch, BacktrackingArmijo, BacktrackingArmijoWolfe
    import numpy as np 
    import MyMathFuncs.SearchMethods as sm

    x0 = list(x_init)
    #============================
    # Plot varible initialisation
    x0_vals = []
    x1_vals = []
    f_vals = []
    x0_vals.append(x0[0])
    x1_vals.append(x0[1])
    f_vals.append(f(x0))
    #============================

    # Inicializaciones varias 
    nIterMax = 500
    epsilon = 1e-5
    k = 0
    Ak = np.identity(2)
    Bk = np.zeros(2)
    Ck = np.zeros(2)
    while k < nIterMax:

        gk=g(x0)
        gk = gk.astype('float64')
        Hk = H(x0)
        Hk = Hk.astype('float64')
        dk = -1.0*np.dot(Ak,gk)
        dk = dk.astype('float64')

        # Stop condition:
        if np.linalg.norm(gk) < epsilon:
            break

        if searchMethod == "ExactLineSearch":             
            alpha = ExactLineSearch(fex, f, x0, x, dk)
            idx = 0
        elif searchMethod == "BacktrackingArmijo":
            alpha, idx = BacktrackingArmijo(f, g, dk, x0, 1.0, 0.3)
        elif searchMethod == "BacktrackingArmijoWolfe":
            alpha, idx = BacktrackingArmijoWolfe(f, g, dk, x0, 1.0, 0.1, 0.9)
        else:
            print('please check the input argument for the search method')

        x0 += (alpha)*dk

        # Correction Matrices
        sk = alpha*dk
        yk = g(x0)-gk # gradient with updated x0 value minus gradient calculated at the start of the loop i.e. previous value
        zk = np.dot(Ak, yk)
        Bk = np.outer(sk, sk) / np.dot(sk, yk)
        Ck = np.outer(-zk, zk) / np.dot(yk, zk)
        Ak = Ak + Bk + Ck

        # Print at each iteration
        if iterPrint:
            print('Iter ',k, ': ', 'fk:',round(f(x0),5),', x:',x0, 'alpha: ',round(alpha,6))        

        #============================
        # Plot variables updating
        x0_vals.append(x0[0])
        x1_vals.append(x0[1])
        f_vals.append(f(x0))
        #============================
        k += 1
    
    # Print at final iteration
    print("-----------------------------------------------------------------")
    print("Results for Damped Newton with " + searchMethod + " step size")
    print("-----------------------------------------------------------------")
    print("Iter ", k, ", x = ", x0)
    print("Iter ", k, ", f(x) = ", round(f(x0),3))
    print("Iter ", k, ", error = ", np.linalg.norm(dk))

    # Return values for each iteration: only for plot purposes
    return x0_vals, x1_vals, f_vals

def QuasiNewton_BFGS(fex, f, g, H, x, x_init, searchMethod, iterPrint = False):
    """
    It is possible to update the Hessian rather than its inverse at each and every iteration. There's 
    a popular method that has proven to be most effective in applications. Detailed derivation 
    have been given in the works of Gill et al. (1981) and Nocedal and Wright (2006). It is known as the Broyden–
    Fletcher–Goldfarb-Shanno (BFGS) method
    """
    from MyMathFuncs.SearchMethods import ExactLineSearch, BacktrackingArmijo, BacktrackingArmijoWolfe
    import numpy as np 
    import MyMathFuncs.SearchMethods as sm

    x0 = list(x_init)
    #============================
    # Plot varible initialisation
    x0_vals = []
    x1_vals = []
    f_vals = []
    x0_vals.append(x0[0])
    x1_vals.append(x0[1])
    f_vals.append(f(x0))
    #============================

    # Inicializaciones varias 
    nIterMax = 500
    epsilon = 1e-5
    k = 0
    Hk = np.identity(2)
    Dk = np.zeros(2)
    Ek = np.zeros(2)
    while k < nIterMax:

        gk=g(x0)
        gk = gk.astype('float64')
        #Hk = H(x0)
        #Hk = Hk.astype('float64')
        dk = -1.0*np.linalg.solve(Hk,gk)
        dk = dk.astype('float64')

        # Stop condition:
        if np.linalg.norm(gk) < epsilon:
            break

        if searchMethod == "ExactLineSearch":             
            alpha = ExactLineSearch(fex, f, x0, x, dk)
            idx = 0
        elif searchMethod == "BacktrackingArmijo":
            alpha, idx = BacktrackingArmijo(f, g, dk, x0, 1.0, 0.3)
        elif searchMethod == "BacktrackingArmijoWolfe":
            alpha, idx = BacktrackingArmijoWolfe(f, g, dk, x0, 1.0, 0.1, 0.9)
        else:
            print('please check the input argument for the search method')

        x0 += (alpha)*dk

        # Correction Matrices
        sk = alpha*dk
        yk = g(x0)-gk # gradient with updated x0 value minus gradient calculated at the start of the loop i.e. previous value
        Dk = np.outer(yk, yk) / np.dot(sk, yk)
        Ek = np.outer(gk, gk) / np.dot(gk, dk)
        Hk = Hk + Dk + Ek
        Hk = Hk.astype('float64')
        
        # Print at each iteration
        if iterPrint:
            print('Iter ',k, ': ', 'fk:',round(f(x0),5),', x:',x0, 'alpha: ',round(alpha,6))        

        #============================
        # Plot variables updating
        x0_vals.append(x0[0])
        x1_vals.append(x0[1])
        f_vals.append(f(x0))
        #============================
        k += 1
    
    # Print at final iteration
    print("-----------------------------------------------------------------")
    print("Results for Damped Newton with " + searchMethod + " step size")
    print("-----------------------------------------------------------------")
    print("Iter ", k, ", x = ", x0)
    print("Iter ", k, ", f(x) = ", round(f(x0),3))
    print("Iter ", k, ", error = ", np.linalg.norm(dk))

    # Return values for each iteration: only for plot purposes
    return x0_vals, x1_vals, f_vals
