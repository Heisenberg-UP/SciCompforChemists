# Imports
import sympy # Imports symbolic math library
import numpy as np # Imports linear array library
import matplotlib.pyplot as plt # Imports plotting library
import scipy # Imports science based library


# %% Exercise 1

x, y, z = sympy.symbols('x y z') # Creates variables

func = x**2 + x - 6 # Provided function
func = sympy.factor(func) # Factors symbolic function

print(func)


# %% Exercise 2

z, y, x = sympy.symbols('z y x') # Creates variables

z = 3*x + x**2 + 2*x*y # Provided function
z = sympy.simplify(z) # Simplifies function

print(z)


# %% Exercise 3

z, y, x = sympy.symbols('z, y, x') # Creates variables

func = (x -2) * (x + 5) * (x) # Provided function
func = sympy.expand(func) # Expands function

print(func)


# %% Exercise 4

t = sympy.symbols('t') # Creates variables

equib = (53.2 * 0.128 * (t - 128)) + (238.1 * 4.18 * (t - 25)) # Equation for thermal equilibrium
equib = sympy.solve(equib) # Solves for the final temperature

print(equib)


# %% Exercise 5

k = sympy.symbols('k') # Creates variables

dG = -1220 - (8.314 * 298 * sympy.log(k)) # Gibbs free Equation
dG = sympy.solve(dG) # Solves for equilibrium constant

print(dG)


# %% Exercise 6

# Variables and multiplication

'''
The matrix below is for Carbonate anion. 
C [x, y]
O [x, y]
O [x, y]
O [x, y]
'''

CO3 = np.array([[2.00, 2.00], [2.00, 3.28], [0.27, 1.50], [3.73, 1.50]]) # Carbonate matrix
phi = np.radians(90) # 90 degrees clockwise in radians
M = np.array([[np.cos(phi), -1 * np.sin(phi)], [np.sin(phi), np.cos(phi)]]) # Operating matrix
rot_CO3 = np.dot(CO3, M) # Carries out dot product operation

# Graphing
plt.figure(figsize=(6, 6)) # Fig size
plt.scatter(CO3[:, 0], CO3[:, 1], label='Carbonate') # Carbonate data
plt.scatter(rot_CO3[:, 0], rot_CO3[:, 1], label='Rotated Carbonate') # Rotated Carbonate data
plt.ylim(-5, 5) # Y limits
plt.xlim(-5, 5) # X limits
plt.xlabel('Angstroms') # X label
plt.ylabel('Angstroms') # Y label
plt.title('Rotating Atomic Coordinates on the XY - Plane') # Graph title
plt.legend() # Graph legend
plt.grid(True) # Shows grid

plt.show()

'''
Lambda function compressed the rotation lines above into one function.
'''
Rot_Operator = lambda Coordinates, Angle: np.dot(Coordinates, np.array([[np.cos(np.radians(Angle)),  -1 * np.sin(np.radians(Angle))], [np.sin(np.radians(Angle)), 
                                                 np.cos(np.radians(Angle))]]))


# %% Exercise 7

# Variables
CO3 = np.array([[2.00, 2.00], [2.00, 3.28], [0.27, 1.50], [3.73, 1.50]]) # Carbonate position


# Functions
def center_mass():
    '''
    Calculates the center of mass
    '''
    total_mass = 0.01201 + 3 * 0.01599
    return (0.01201 * CO3[0, :] + 0.01599 * CO3[1, :] + 0.01599 * CO3[2, :] + 0.01599 * CO3[3, :]) / total_mass


CO3_centered = CO3 - center_mass() # Centers the CO3


Rot_Operator = lambda Coordinates, Angle: np.dot(Coordinates, 
                                                 np.array([[np.cos(np.radians(Angle)), 
                                                            -1 * np.sin(np.radians(Angle))], 
                                                            [np.sin(np.radians(Angle)), 
                                                             np.cos(np.radians(Angle))]]))


Rot_CO3_Centered = Rot_Operator(CO3_centered, 90)  # Rotates and shifts back to original position
Rot_CO3 = Rot_CO3_Centered + center_mass() # Moves rotated molecule back to original position

# Plot the results
plt.figure(figsize=(5, 5))
plt.scatter(center_mass()[0], center_mass()[1], label='Center of Mass', linewidths=6)
plt.scatter(CO3[:, 0], CO3[:, 1], label='Carbonate') # Carbonate data
plt.scatter(Rot_CO3[:, 0], Rot_CO3[:, 1], label='Rotated Carbonate') # Rotated Carbonate data
plt.ylim(-5, 5) # Y limits
plt.xlim(-5, 5) # X limits
plt.xlabel('Angstroms') # X label
plt.ylabel('Angstroms') # Y label
plt.title('Rotating Atomic Coordinates about Center of Mass') # Graph title
plt.legend(loc='upper right') # Graph legend
plt.grid(True) # Shows grid

plt.show()


# Exercise 8

# Variables
n, R, T, V, V_f, V_i = sympy.symbols('n R T V V_f V_i') # Creates variables for symbolics

work = sympy.integrate(-n * R * T * (1/V), (V, V_i, V_f)).simplify() # Integrates function
work_numerical = sympy.integrate(-2.44 * 8.314 * 298 * (1/V), (V, 0.552, 1.32)) # Integrates and calculates work 

print(work)
print(round(work_numerical,), ' J')


# Exercise 9

# Variables
A0 = 1 # Initial molar concentration
t = np.arange(0, 61, 1) # Time for reaction to be carried out
k = 0.1 # Rate Constant


# Function
def seconnd_order(A0, k, t):
    '''
    Function will calculate the concentration of the reactants decaying
    '''
    return ((k * t) + (1 / A0)) ** -1
    


def rate_second(A, t):
    '''
    Generic rate for second order reaction: -2k[A] = 2d[A]/dt
    '''
    return -k * (A ** 2)


ODE = scipy.integrate.odeint(rate_second, A0, t) # Takes differential of the generic rate

plt.plot(t, seconnd_order(A0, k, t), label='Second Order Integrated Rate Law')
plt.scatter(t, ODE, color='red', label='Ordinary Differential Equation')
plt.legend(loc='best')
plt.xlabel('time (s)')
plt.ylabel('Concentration [A]')
plt.title('2A -> P \nSecond Order Reaction')

plt.show()


# Exercise 10

S_4 = np.array([[0, -1, 0],[1, 0, 0],[0, 0, -1]]) # S4 operation for group theory
C_2 = np.array([[-1, 0, 0],[0, -1, 0],[0, 0, 1]]) # C2 operation for group theory

S_4S_4 = np.dot(S_4, S_4) # Multiplies two S_4 operations together

print(S_4S_4)
print(C_2)

