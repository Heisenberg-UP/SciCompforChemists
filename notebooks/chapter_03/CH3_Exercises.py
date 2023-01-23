# %% Exercise 1
import matplotlib.pyplot as plt # Imports appropriate plotting library

Volume = list(range(1, 20, 1)) # List of volumes from 1L to 20L


def ideal_gas(V, n=1, R=0.08206, T=298): # Ideal gas function
    '''
    This function will calculate the pressure of an expandable vessel.
    V is Volume and must be a list for the ideal pressure to represent 
    an expandable container.
    Function assumes number of moles is 1, R constant is 0.08206, and
    temperature is 298.
    The set R constant yields pressure atm.
    '''
    P = []
    for i in V:
        Press = (n * R * T) / i
        P.append(Press)
    return P


plt.scatter(Volume, ideal_gas(Volume), c='green') # Sets up data
plt.xlabel('Volume (L)') # X label
plt.ylabel('Pressure (atm)') # Y Label
plt.title('Pressure as a function of Volume') # Title of graph
plt.show()


# %% Exercise 2
import matplotlib.pyplot as plt # Imports plotting library

Xp = [3.98, 3.16, 2.96, 2.66, 2.20] # Electronegative values of halogens
AN = [9, 17, 35, 53, 85] # Atomic number for said halogens
a_radii = [60, 100, 117, 136, 148] # Atomic radii of said halogens

plt.scatter(AN, Xp, c=a_radii) # Scatter plot for AN and Xp with c being colorbar
plt.colorbar() # Implements color bar
plt.xlabel('Atomic Number') # X label
plt.ylabel('Electronegativity') # Y label
plt.title('Electronegativity of halogens')
plt.show()


# %% Exercise 3
import matplotlib.pyplot as plt # Imports plotting library
import numpy as np # Imports math library

x = list(range(-50, 50, 1)) # Lists of x values from -50 to 50


def f(x): # function will depict f(x), g(x), h(x)
    '''
    Function will calculate f(x) = x^2, g(x) = x^2 siin(x),
    h(x) = -x^2
    Enter X as a list for accurate graphing to be applicable
    '''
    f_x = []
    g_x = []
    h_x = []
    for i in x:
        func_f = i**2
        func_g = (i**2) * np.sin(i)
        func_h = -1 * (i**2)
        f_x.append(func_f)
        g_x.append(func_g)
        h_x.append(func_h)
    functions = [f_x, g_x, h_x]
    return functions


plt.plot(x, f(x)[0], color='green', label='f(x)') # Plots f(x) function
plt.plot(x, f(x)[1], color='red', label='g(x)') # Plots g(x) function
plt.plot(x, f(x)[2], color='blue', label='h(x)') # Plots h(x) function
plt.xlabel('X values') # X labels
plt.ylabel('Y values') # Y labels
plt.title('Sandwich Thereom Example') # Title for function
plt.show()


# %% Exercise 4
