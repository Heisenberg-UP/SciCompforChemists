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
import matplotlib.pyplot as plt # Imports plotting library
import numpy as np # Imports math/array library

t = list(range(0, 60, 1))


def first_order_kinetics(time):
    '''
    Function will calculate the change in [A] with respect to time.
    [A] = [A]_0 e^-2kt
    [A](0) = 1
    For this we will set [A]_0 to 1M.
    Enter time as a list for the concentration as a function of time to yield the
    best data.
    '''
    c = []
    for i in time:
        conc = np.exp(-2 * 0.12 * i)
        c.append(conc)
    return c


plt.plot(t, first_order_kinetics(t), color='green') # Plots data provided
plt.xlabel('Time (s)') # X label
plt.ylabel('Concentration [A]') # Y label
plt.title('First Order Kinetics\n2A --> P') # Graph title
plt.show()


# %% Exercise 5
import matplotlib.pyplot as plt # Imports graphing library
import pandas as pd # Imports data manipulation library

Data = pd.read_csv('data/gc_trace.csv') # Pandas reads the .csv file
df = pd.DataFrame(Data, columns=['time', 'abs_intensity']) # Pandas converts .csv into dataframe

plt.plot(df['time'], df['abs_intensity'], color='red') # Sets up graph
plt.xlabel('Time (min)') # X label
plt.ylabel('Intensity') # Y label
plt.title('Gas Chromatography Graph') # Graph title
plt.show()


# %% Exercise 6
import matplotlib.pyplot as plt # Imports plotting library
import pandas as pd # Imports data manipulation library

Data = pd.read_csv('data/ms_bromobenzene.csv') # Pandas reads the .csv file
df = pd.DataFrame(Data, columns=['m/z', 'abs_intensity']) # Pandas converts variable into dataframe

plt.stem(df['m/z'], df['abs_intensity'], markerfmt=' ') # Plots data
plt.xlabel('m/z') # X label
plt.ylabel('Intensity') # Y label
plt.title('Mass Spectrum Bromobenzene') # Graph title
plt.show()


# %% Exercise 7
import matplotlib.pyplot as plt # Imports plotting library

percents = [78, 21, 1] # Percents of gases in atmosphere

plt.pie(percents, labels=['N$_2$', 'O$_2$', 'Other Gases'], explode=(0, 0, 0.2)) # Plots data
plt.title('Atmospheric Composition') # Graph title
plt.axis('equal') # I dont know what this does
plt.show()


# %% Exercise 8
import matplotlib.pyplot as plt # Imports plotting library
import random # Imports library for random value generation

rdn = [random.random() for value in range(1000)] # Creates random list 

plt.hist(rdn, bins=10, edgecolor='k') # Plots data into 10 bins
plt.xlabel('Numerical Value') # X label
plt.ylabel('Number of Numerical Values') # Y label
plt.show()


# %% Exercise 9
import matplotlib.pyplot as plt # Imports plotting library

ppm = [7.52, 4.00, 3.60, 3.44] # ppm shift (x axis)
intensity = [1.52, 3.90, 5.74, 5.78] # Intensity denotes the intensity of fourier transform

plt.stem(ppm, intensity, markerfmt=' ') # Plots data
plt.xlabel('ppm') # x label
plt.ylabel('Intensity') # Y label
plt.title('$^1$H NMR Spectrum of Caffeine in CDCl$_3$') # Graph title
plt.gca().invert_xaxis()
plt.show()


# %% Exercise 10
import matplotlib.pyplot as plt # Imports plotting library

step = [1, 2, 3, 4, 5, 6, 7] # x axis
kcal = [0.0, 11.6, 9.8, 13.4, 5.8, 8.3, 2.7] # y axis

plt.step(step, kcal) # Plots graph
plt.xlabel('Step number') # X label
plt.ylabel('Energy (Kcal)') # Y label
plt.title('Energy per step in the binding and splittle of H$_2$(g)')
plt.show()


# %% Exercise 11
import matplotlib.pyplot as plt # Imports plotting library
