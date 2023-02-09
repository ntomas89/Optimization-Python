def Taylor1SympyFunc(function, aroundPoint, n):
    """ 
    DESCRIPTION: n-th order Taylor approximation of a 1-var function
    """
    import sympy as sy
    x = sy.Symbol('x')
    def factorial(n):
        if n <= 0:
            return 1
        else:
            return n*factorial(n-1)
    i = 0
    p = 0
    while i <= n:
        p = p + (function.diff(x,i).subs(x,aroundPoint))/(factorial(i))*(x-aroundPoint)**i
        i += 1
    return p

def Taylor2SympyFunc(f_expr, vars, aroundPoint):
    """ 
    DESCRIPTION: 2nd order Taylor approximation of a 2-var function
    It also returns the minimum of the quadratic form that approximates f_expr around the specified point
    """
    from sympy import lambdify, symbols, solve 

    x = vars[0] # symbols('x')
    y = vars[1] # symbols('y')
    variable_list = [x,y]

    # Primeras y segundas derivadas de f. Nota, se puede hacer de una: fxy_expr = f_expr.diff(x,y)
    fx_expr  = f_expr.diff(x)
    fy_expr  = f_expr.diff(y)
    fxx_expr = fx_expr.diff(x)
    fxy_expr = fx_expr.diff(y)
    fyx_expr = fy_expr.diff(x)
    fyy_expr = fy_expr.diff(y)

    # Funciones evaluables
    f   = lambdify(variable_list, f_expr,  "numpy")
    fx  = lambdify(variable_list, fx_expr,  "numpy")
    fy  = lambdify(variable_list, fy_expr,  "numpy")
    fxx = lambdify(variable_list, fxx_expr,  "numpy")
    fxy = lambdify(variable_list, fxy_expr,  "numpy")
    fyx = lambdify(variable_list, fyx_expr,  "numpy")
    fyy = lambdify(variable_list, fyy_expr,  "numpy")

    # Aproximamos una funcion cuadratica entorno al punto [x=a, y=b]
    a = aroundPoint[0]
    b = aroundPoint[1]
    Taylor2_expr = f(a,b) + fx(a,b)*(x-a) + fy(a,b)*(y-b) + (1/2)*fxx(a,b)*(x-a)**2 + fxy(a,b)*(x-a)*(y-b) + (1/2)*fyy(a,b)*(y-b)**2
    Taylor2 = lambdify(variable_list, Taylor2_expr,  "numpy")
    
    # Calcular el minimo de la funcion aproximada
    T2x = Taylor2_expr.diff(x)
    T2y = Taylor2_expr.diff(y)
    sols = solve([T2x, T2y], [x, y], dict=True, set=True)

    return Taylor2_expr, Taylor2, sols[0][x], sols[0][y]

def TaylorNSympyFunc(function_expression, variable_list, evaluation_point, degree):
    """
    :param function_expression: Sympy expression of the function
    :param variable_list: list. All variables to be approximated (to be "Taylorized")
    :param evaluation_point: list. Coordinates, where the function will be expressed
    :param degree: int. Total degree of the Taylor polynomial
    :return: Returns a Sympy expression of the Taylor series up to a given degree, of a given multivariate expression, approximated as a multivariate polynomial 
             evaluated at the evaluation_point
    """
    from sympy import factorial, Matrix, prod
    import itertools

    n_var = len(variable_list)

    # list of tuples with variables and their evaluation_point coordinates, to later perform substitution
    point_coordinates = [(i, j) for i, j in (zip(variable_list, evaluation_point))]
    # list with exponentials of the partial derivatives
    deriv_orders = list(itertools.product(range(degree + 1), repeat=n_var))
    # Discarding some higher-order terms
    deriv_orders = [deriv_orders[i] for i in range(len(deriv_orders)) if sum(deriv_orders[i]) <= degree]  
    n_terms = len(deriv_orders)
    # Individual degree of each partial derivative, of each term
    deriv_orders_as_input = [list(sum(list(zip(variable_list, deriv_orders[i])), ())) for i in range(n_terms)]     

    polynomial = 0
    for i in range(n_terms):
        # The * takes an iterable and adds its elements to the parameters of the function call
        # random.randint(*(1,300)) is the same thing as random.randint(1,300)
        # The following is a syntax error, because 300 is not an iterable: random.randint(*300)
        partial_derivatives = function_expression.diff(*deriv_orders_as_input[i])  # e.g. df/(dx*dy**2)=fxyy
        # Substitute all instances of a variable or expression in a mathematical expression with some other variable or expression or value.
        # If the substituted value is numerical then sympy.subs() returns the solution of the resulting expression
        partial_derivatives_at_point = partial_derivatives.subs(point_coordinates) # e.g. fxy(a, b)
        denominator = prod([factorial(j) for j in deriv_orders[i]])  # e.g. (1! * 2!)
        distances_powered = prod([(Matrix(variable_list) - Matrix(evaluation_point))[j] ** deriv_orders[i][j] for j in range(n_var)])  # e.g. (x-x0)*(y-y0)**2
        polynomial += partial_derivatives_at_point / denominator * distances_powered
    return polynomial