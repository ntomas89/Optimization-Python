def DampedNewton(f, g, H, x_init, searchMethod, iterPrint = False):
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
    SM = getattr(sm, searchMethod) 
    nIterMax = 500
    epsilon = 1e-10
    k = 0
    while k < nIterMax:
        # Compute the “exact” solution, dk, of the well-determined, i.e., full 
        # rank, linear matrix equation: Hk * dk = gk
        gk=g(x0)
        Hk=H(x0)
        dk = -1.0*np.linalg.solve(Hk,gk)

        # Stop condition:
        #   norm(dk) < epsilor ?                        np.linalg.norm(dk)
        #   norm(gk) < epsilon ?                        np.linalg.norm(dk)
        #   two consecutive values of f(x) very close ? abs(f_vals[k]-f_vals[k-1])
        if k>0 and abs(np.linalg.norm(dk)) < epsilon:
            break

        # Exact Line Search
        m = 0
        dampingFactor = 0.9
        alpha = 1.0
        sigma = 0.4
        gamma = 0.6
        while m < 20:            
            # If the product of gradient*p is never zero, then the step size will be 1
            # If reducing the step makes gradient*p = zero, then we take that reduced step           
            c, d = SM(f, g, dk, x0, dampingFactor**m, sigma, gamma)          
            if c:
                alpha = dampingFactor**m
                break
            m+=1
        x0 += (alpha)*dk

        # Print at each iteration
        if iterPrint:
            print('Iter ',k, ': ', '(m=', m, ') step size=',alpha, ', LS: ',c, d)        
            print('       ','Exact Line: ',x0[0], x0[1], ' >> ', f(x0))
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