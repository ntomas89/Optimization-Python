def SimpleNewton(fex, f, g, H, x, x_init, searchMethod, iterPrint = False):
    from MyMathFuncs.SearchMethods import ExactLineSearch, BacktrackingArmijo, BacktrackingArmijoWolfe
    import numpy as np 
    import MyMathFuncs.SearchMethods as sm

    # Metodo de Newton para encontrar el minimo de una funcion. 
    # Argumentos:
    #   f: callable de la funcion a minimizar
    #   g: callable del gradiente de f
    #   H: callable del Hessiano de f
    #   x0: punto de partida para la busqueda del minimo
    #   searchMethd: string con el nombre de una de las funciones situadas en MyMathFuncs.SearchMethods, de modo que
    #                podamos elegir el metodo de busqueda al llamar a DampedNewton
    # Salida:
    #   x0_vals, x1_vals, f_vals por si queremos utilizar animacion en graficos
    # Drawbacks:
    #   1. It requires calculations of second-order derivatives at each and every iteration, which is usually quite 
    #   time-consuming. In some applications it may not even be possible to calculate such derivatives. Moreover, a 
    #   linear system of equations needs to be solved. Therefore, each iteration of the method requires substantially
    #   more calculations compared with the steepest–descent or conjugate gradient method.
    #   2. The Hessian of the cost function may be singular at some iterations. Thus, cannot be used to compute the 
    #   search direction. Moreover, unless the Hessian is positive definite, the search direction cannot be guaranteed 
    #   to be that of descent for the cost function.
    #   3. The method is not convergent unless the Hessian remains positive definite and a step size is calculated along 
    #   the search direction to update the design. However, the method has a quadratic rate of convergence when it works. 


    # Sin esta asignacion, modificar x_init dentro de la funcion afecta a la variable que utilizamos para llamar 
    # a la funcion DampedNewton. Esto es:
    # x_init = np.random.normal(0,10,n)*5
    # x0_BA, x1_BA, f_BA = DampedNewton(fun, gfun, Hfun, x_init, "BacktrackingArmijo", iterPrint=False)
    # Tras finalizar Newton, x_init tambien cambiaria de valor!!
    # La solucion es esto:
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
    SM = getattr(sm, searchMethod) 
    nIterMax = 500
    epsilon = 1e-5
    k = 0
    while k < nIterMax:
        # Compute the “exact” solution, dk, of the well-determined, i.e., full 
        # rank, linear matrix equation: Hk * dk = gk
        fk=f(x0)
        gk=g(x0)
        gk = gk.astype('float64')
        Hk=H(x0)
        Hk = Hk.astype('float64')
        dk = -1.0*np.linalg.solve(Hk,gk)

        # Stop condition:
        #   norm(dk) < epsilor ?                        np.linalg.norm(dk)
        #   norm(gk) < epsilon ?                        np.linalg.norm(dk)
        #   two consecutive values of f(x) very close ? abs(f_vals[k]-f_vals[k-1])
        if k>0 and abs(np.linalg.norm(dk)) < epsilon:
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

        # Print at each iteration
        if iterPrint:
            print('Iter ',k, ': ', 'fk:',round(fk,5),', x:',x0, 'alpha: ',round(alpha,6), idx)        

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

def MarquardtNewton(fex, f, g, H, x, x_init, searchMethod, iterPrint = False):
    from MyMathFuncs.SearchMethods import ExactLineSearch, BacktrackingArmijo, BacktrackingArmijoWolfe

    # Marquardt (1963) suggested a modification to the direction-finding process that has the desirable 
    # features of the steepest–descent and Newton methods. It turns out that far away from the solution point, 
    # the method behaves like the steepest–descent method. Near the solution point, it behaves like the Newton method.

    import numpy as np 
    import MyMathFuncs.SearchMethods as sm

    # Sin esta asignacion, modificar x_init dentro de la funcion afecta a la variable que utilizamos para llamar 
    # a la funcion DampedNewton. Esto es:
    # x_init = np.random.normal(0,10,n)*5
    # x0_BA, x1_BA, f_BA = DampedNewton(fun, gfun, Hfun, x_init, "BacktrackingArmijo", iterPrint=False)
    # Tras finalizar Newton, x_init tambien cambiaria de valor!!
    # La solucion es esto:
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
    nIterMax = 50
    epsilon = 1e-15
    k = 0
    HessianFactor = 10000000
    while k < nIterMax:
        # Compute the “exact” solution, dk, of the well-determined, i.e., full 
        # rank, linear matrix equation: Hk * dk = gk
        fk=f(x0)
        gk=g(x0)
        gk = gk.astype('float64')
        Hk=H(x0)
        Hk = Hk.astype('float64')

        HFI = HessianFactor*np.identity(len(gk))        
        dk  = -1.0*np.linalg.solve(Hk+HFI, gk) # Large HessianFactor >> dk is approx. (-1.0/HessianFactor)*gk
        if f(x0+dk) > f(x0):
            for i in range(20):
                HessianFactor = HessianFactor*(2**i)
                HFI = HessianFactor*np.identity(len(gk))
                dk = -1.0*np.linalg.solve(Hk+HFI, gk)
                if f(x0+dk) > f(x0):
                    break
        HessianFactor = HessianFactor/2

        # Stop condition:
        #   norm(dk) < epsilor ?                        np.linalg.norm(dk)
        #   norm(gk) < epsilon ?                        np.linalg.norm(dk)
        #   two consecutive values of f(x) very close ? abs(f_vals[k]-f_vals[k-1])
        if k>0 and abs(np.linalg.norm(dk)) < epsilon:
            break

        x0 += dk
        # Print at each iteration
        if iterPrint:
            #print('Iter ',k, ': ', 'fk:',round(fk,5),', x:',x0, 'alpha: ',round(alpha,6), idx)        
            print('Iter ',k, ': ', 'fk:',round(fk,5),', x:',x0, 'dk:',dk)

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