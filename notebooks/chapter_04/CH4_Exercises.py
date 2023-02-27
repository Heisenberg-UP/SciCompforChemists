# %% Exercise 1
# Imports
import numpy as np # Imports numpy library for math and linear arrays

# Variables
AN = np.arange(1, 27, 1) # 1D array of first 26 Atomic numbers

print(AN)


# %% Exercise 2
# Imports
import numpy as np # Imports numpy library for math and linear arrays

# Variables
wl = np.arange((4 * 10**-7), (8.5 * 10**-7), (5 * 10**-8)) # Wavelengths in meters
Energy = ((6.626 * 10**-34) * (2.998 * 10**8)) / wl # Energy calculated from wavelengths

print(wl, '\n', Energy)


# %% Exercise 3
# Imports
import numpy as np # Imports numpy library for math and linear arrays

# Variables
array = 101.325 * np.ones(100) # Creates a 1D array of 100 ones and then multiplied by a scalar

print(array)


# %% Exercise 4
# Imports
import numpy as np # Imports numpy library for math and linear arrays

# Variables
F = np.array([0, 32, 100, 212, 451]) # Creates an array of Fahrenheit temps
C = (F - 32) * (5 / 9)

print(C)


# %% Exercise 5
# Imports
import numpy as np # Imports numpy library for math and linear arrays
import matplotlib.pyplot as plt # Imports plotting library

# Variables and operations
x = np.pi * np.arange(0, 11, 0.1) # xπ values
y1 = np.sin(x) # y1 values
y2 = np.sin(1.1 * x + 0.5) # y2 values
y3 = y1 + y2 # y1 + y2 values

# Graphing data
plt.figure(figsize=(10, 6)) # Sets fig size
plt.plot(x, y1, color='green', label='sin(x)') # plots part a 
plt.plot(x, y2, color='blue', label='sin(1.1x + 0.5)') # plots part a
plt.plot(x, y3, color='red', label='sin(x) + sin(1.1x + 0.5)') # plots part b 
plt.xlabel('xπ') # X label
plt.ylabel('f(xπ)') # Y label

'''
The area of the graph that is smaller is experiencing destructive interference.
'''

plt.show()


# %% Exercise 6
# Imports
import numpy as np # Imports math and linear arrays library
import matplotlib.pyplot as plt # Imports plotting library 

# Variables
k = np.linspace(0.001, 1000, 10000) # Equilibrium constant values
dG = np.array( -1 * 8.314 * 298.15 * np.log(k)) # Calculates ∆G for equilibrium constants

# Plotting
plt.figure(figsize=(10, 6)) # Figsize
plt.scatter(k, dG, color='green') # Scatter plot of equilibrium and gibbs free energy
plt.xlabel('Equilibrium constant (k)') # X label
plt.ylabel('Gibbs free energy (∆G)') # Y label

plt.show()


# %% Exercise 7 
# Imports
import numpy as np # Imports math and linear arrays library
import matplotlib.pyplot as plt # Imports plotting library

# Variable
Ea = np.arange(1000, 21000, 100) # Energies in kJ/mol
k = 1 * np.exp((-1 * Ea) / (8.314 * 298.15)) # Equilibrium constant

# Graphing
plt.figure(figsize=(10, 6)) # Fig size
plt.scatter(Ea, k, color='green') # Scatter plot of data
plt.xlabel('Activation Energy (Ea)') # X label
plt.ylabel('Equilibrium Constant (k)') # Y label

plt.show()


# %% Exercise 8
# Imports
import numpy as np # Imports math and linear arrays library
import matplotlib.pyplot as plt # Imports plotting libraries

# Variables
array = np.arange(0, 15, 1) # Initial Array
array = np.reshape(array, (3, 5)) # Reshape array to 3 by 5

print(array)

array = array.T # Transposes array

print(array)

array = array.flatten() # Flattens array

print(array)


# %% Exercise 9
# Imports
import numpy as np # Imports math and linear array library

# Variables
n = np.linspace(1, 8, 8) # Principal quantum numbers
J = (-2.18 * 10**-18) * (1 / n**2) # Energies of orbitals
matrix = np.dstack((n, J)) # Matrix of energies based off quantum numbers

print(matrix)


# %% Exercise 10
# Imports
import numpy as np # Imports math and linear array library

# Variables
arr = np.random.randint(0, high=10, size=10) # Creates random array 

print(arr[4])


# %% Exercise 11
# Imports 
import numpy as np # Imports math and linear array library

# Variables
arr2 = np.random.randint(0, high=10, size=15).reshape(5, 3) # Creates random array

print(arr2)

print(arr2[1, 2])

print(arr2[2, :])

print(arr2[0, 1:])


# %% Exercise 12
# Imports
import numpy as np # Imports math and linear array library 

# Variables
arr2 = np.array([[1, 1], [2, 2]]) # Array list
arr2 = 1 + arr2 # Adds array with one

print(arr2)


# %% Exercise 13
# Imports
import numpy as np # Imports math and linear array library

# Variables
arr = np.array([[1, 8, 9], [8, 1, 9], [1, 8, 1]]) # Array 3 by 3
arr2 = np.array([[1, 1], [1, 1]]) # Array 2 by 2
matrix = arr + arr2 # Array sum of arrays

'''
These two matrices will never add together because you have a 3 by 3 added to a 
2 by 2.
'''

print(matrix)


# %% Exercise 14
# Imports
import numpy as np # Imports math and linear array libraries

# Variables
arr = np.array([[1, 8], [3, 2]]) # 2 by 2 array
arr2 = np.array([[1, 1], [1, 1]]) # 2 by 2 array
matrix = arr + arr2 # Addition of two arrays

print(matrix)


# %% Exercise 15
# Imports 
import numpy as np # Imports math and linear array library

# Variables
arr = np.random.rand(20) # Random array

print(np.max(arr).argsort()) # STILL NEED TO SOLVE

print(np.mean(arr))

print(np.cumsum(arr))

print(np.sort(arr))


# %% Exercise 16
# Imports
import numpy as np # Imports math and linear array library

# Variables
array = np.random.rand(10) # Generates random array from 0 -> 1 with 10 values

neg_array = array * -1 # Multiplies have the array by -1 to create an array from -1 -> 1

array = np.hstack((neg_array, array)) # Horizontally joins two 1D arrays

median = np.median(array) # Calculates the median value

print(array, median)


# %% Exercise 17
# Imports
import numpy as np # imports math and linear array library

# Variables
array = np.random.randint(0, 36, 36) # Random list of integers from 0 -> 35

array = np.sort(array) # Sorts array based on value

print(array)


# %% Exercise 18
#Imports
import numpy as np # Imports linear array and math library
import matplotlib.pyplot as plt # Imports plotting library for matplotlib

# Variables
array = list(np.random.binomial(1, 0.5, 6)) # Binomial distribution

print(array)
'''
for the binomial distribution, 1 means +1/2 and 0 means -1/2 for nuclei spin
'''

for i in range(len(array)):
    if array[i] == 1:
        array[i] = 0.5
    else:
        array[i] = -0.5

print(array)
plt.stem([1, 2, 3, 4, 5, 6], array)
plt.xlabel('Hydrogen Atoms')
plt.ylabel('Nuclei Spin')
plt.show()
