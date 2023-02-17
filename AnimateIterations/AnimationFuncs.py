import numpy as np

def Animate3D(fun, x0_min, x0_max, x1_min, x1_max, x0_vals, x1_vals, f_vals):
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import animation

    # Meshgrid sets up the space: points where to draw the function will be evenly
    # distributed between the min an max values of each axis
    X,Y = np.meshgrid(np.linspace(x0_min, x0_max, 100), np.linspace(x1_min, x1_max, 100))

    # Then, zs will evaluate fun in each of these points
    zs = np.array(fun([np.ravel(X), np.ravel(Y)]))
    Z = zs.reshape(X.shape)

    # Create figure
    fig = plt.figure(figsize=(7,7))
    # ax = Axes3D(fig) # Axes3D created problems: unsupported operand type(s) for *: 'NoneType' and 'float' NOT SURE WHY
    ax = fig.add_subplot(projection='3d')

    # Plot Z vs (X,Y)
    ax.plot_surface(X,Y,Z, rstride=5, cstride=5, cmap='jet', alpha=0.5)
    ax.set_xlabel('$x_0$')  # $ for italic characters
    ax.set_ylabel('$y_0$')
    ax.set_zlabel('$f(x_0, y_0)$')
    ax.view_init(45,-45)

    # Create animation
    """
    # Tuple is one of 4 built-in data types in Python used to store collections of data, the other 3 are List, Set, and Dictionary.
    # Comma is what makes something a tuple: 1, >> (1,). The parenthesis are optional in most locations.
    # ax.plot() returns a tuple with one element. By adding the comma to the assignment target list, you ask Python to unpack the 
    # return value and assign it to each variable named to the left in turn. It's the equivalent to do: a,b = func(c) when func
    # returns two values.
    # we could rewrite the below code with parenthesis without changing the meaning:
    #   (line,) = ax.plot([], [], [], color='purple', label='Path', lw=1.5)
    # Or we could use list syntax too:
    #   [line] = ax.plot([], [], [], color='purple', label='Path', lw=1.5)
    # Or we could recast it to lines that do not use tuple unpacking:
    #   line = ax.plot([], [], [], color='purple', label='Path', lw=1.5)[0]
    """
    line, = ax.plot([], [], [], color='purple', label='Path', lw=1.5)
    point, = ax.plot([], [], [], '*', color = 'purple')
    display_value = ax.text(2., 2., 27.5, '', transform = ax.transAxes)

    def init_2():
        # Initialise animation where the line and point properties are empty
        line.set_data([],[])
        line.set_3d_properties([])
        point.set_data([],[])
        point.set_3d_properties([])
        display_value.set_text('here')
        return line, point, display_value
    
    def animate_2(i, ca, line):
        """
        X = np.arange(0, 10, 1) # this is: print(type(X)) == <class 'numpy.ndarray'>
            X[:3] == X[0:3]       is a list   >> [0 1 2]                indexed as: list[1] == 1
            X[X[:3]] == X[X[0:3]] is an array >> [array([0, 1, 2])]     indexed as: array[0][1] == 1
        """
        # Option 1:
        #line.set_data(x0_vals[0:i].T, x1_vals[0:i].T)
        #line.set_3d_properties(f_vals[0:i].T)

        # Option 2:
        line.set_data(ca[0:2, :i])
        line.set_3d_properties(ca[2, :i])

        # animate points
        point.set_data(x0_vals[i:i+1], x1_vals[i:i+1])
        point.set_3d_properties(f_vals[i:i+1])  
        return line, point, display_value

    x0a=np.array(x0_vals)
    x1a=np.array(x1_vals)
    f1a=np.array(f_vals)
    ca=np.concatenate(([x0a], [x1a], [f1a]), axis=0)
    ax.legend(loc=1)

    anim2 = animation.FuncAnimation(fig, animate_2, frames=len(x0_vals), init_func=init_2, fargs=(ca, line), save_count=None, interval=100, repeat_delay=500, blit=True)
    
    plt.show()

    # END OF FUNCTION Animate3D
    #===========================

def Animate2D(fun, x0_min, x0_max, x1_min, x1_max, x0_vals, x1_vals):
    from matplotlib import pyplot as plt
    from matplotlib import animation

    # Meshgrid sets up the space
    X,Y = np.meshgrid(np.linspace(x0_min, x0_max, 100), np.linspace(x1_min, x1_max, 100))

    # Then, zs will evaluate fun in each of these points
    zs=np.array(fun([np.ravel(X), np.ravel(Y)]))
    Z = zs.reshape(X.shape)

    # Create figure
    fig, ax = plt.subplots(figsize=(7,7))
    ax.contour(X,Y,Z,100,cmap='jet')

    # Create animation
    line, = ax.plot([], [], 'purple', label='Path', lw=1.5)
    point, = ax.plot([], [], '*', color = 'purple', markersize=4)
    display_value = ax.text(0.02, 0.02, '', transform = ax.transAxes)

    # Optional - see animate_1 definition
    x0a=np.array(x0_vals)
    x1a=np.array(x1_vals)
    ca=np.concatenate(([x0a], [x1a]), axis=0)

    def init_1():
        # Initialise animation where the line and point properties are empty
        line.set_data([],[])
        point.set_data([],[])
        display_value.set_text('')
        return line, point, display_value
    
    def animate_1(i):
        # Takes iteration number i as input and sets the current data of x0 and x1
        # to plot the line and set the point:

        # Option 1: take directly function input values       
        # line.set_data(x0_vals[:i], x1_vals[:i])

        # Option 2: define ca and access it (did it just to check)
        line.set_data(ca[0:2, :i])

        # animate points
        point.set_data(x0_vals[i], x1_vals[i])
        return line, point, display_value

    ax.legend(loc=1)

    anim1 = animation.FuncAnimation(fig, animate_1, init_func=init_1, frames=len(x0_vals), interval=20, repeat_delay=60, blit=True)
    
    plt.show()

    # END OF FUNCTION Animate2D
    #===========================