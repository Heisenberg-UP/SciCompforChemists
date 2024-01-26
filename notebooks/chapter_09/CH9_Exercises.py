# %% Exercise 1

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

A0 = 1 # [A]_0 is 1 M
t = np.arange(0, 61, 1) # Time in seconds
k = 0.1 # Rate constant


# Functions
def third_rate(A, t):
    '''
    Function represents the rate of a third order reaction.
    '''
    return -k * (A ** 3)


ODE = sp.integrate.odeint(third_rate, A0, t) # Calculates ODE

# Plotting
plt.scatter(t, ODE, label='Ordinary Differential Simulation') # Plots Rate constant ODE
plt.legend(loc='best') # Legend
plt.xlabel('Time (s)') # X Label
plt.ylabel('Concentration [A]') # Y Label
plt.title('3A -> P \n3rd Order Reaction Simulation') # Title

plt.show()


# %% Exercise 2

import numpy as np
import scipy as sp

# Variables
A, P = 1.00, 0.00 # Initial concentrations in molarity
k = 0.28 # Rate constant
length = 60 # Length of simulation
time = range(length + 1) # Time of simulation

# create arrays to hold calculated concentrations
A_conc = np.empty(length + 1)
P_conc = np.empty(length + 1)

# Deterministic Step-wise simulation
for sec in time:
    # record concentration
    A_conc[sec] = A
    P_conc[sec] = P
    # recalculate rate
    rate = k * (A ** 2)
    # recalculate new concentration
    A -= rate
    P += rate

# Plotting
plt.plot(time, A_conc, label='[A]') # Plots reactants
plt.plot(time, P_conc, label='[P]') # Plots products
plt.legend(loc='best') # Legend
plt.xlabel('Time (s)') # X label
plt.ylabel('Concentration (M)') # Y Label
plt.title('Stepwise simulation of second order kinetics') # Graph title

plt.show()


# %% Exercise 3

import matplotlib.pyplot as plt

# Variables
A, B, C, P = 1.50, 0.00, 0.00, 0.00 # Initial concentrations in Molarity
A_k, B_k, C_k = 0.8, 0.4, 0.3 # Rate constants for each step
A_conc, B_conc, C_conc, P_conc = [], [], [], [] # Empty list to change concentrations
length = 30 # Length of simulation

# Multi-step simulation for chemical kinetics
for sec in range(length):
    A_conc.append(A)
    B_conc.append(B)
    C_conc.append(C)
    P_conc.append(P)
    # recalculate rates
    rate_A = A_k * A
    rate_B = B_k * B
    rate_C = C_k * C
    #recalculate concentrations after next time increment
    A = A - rate_A
    B = B + rate_A - rate_B
    C = C - rate_C + rate_B
    P = P + rate_C

# Plotting
plt.plot(range(length), A_conc, label='[A]') # Plot for [A]
plt.plot(range(length), B_conc, label='[B]') # Plot for [B]
plt.plot(range(length), C_conc, label='[C]') # Plot for [C]
plt.plot(range(length), P_conc, label='[P]') # Plot for [P]
plt.legend(loc='best') # Legend
plt.xlabel('Time (s)') # X label
plt.ylabel('Concentration (M)') # Y label
plt.title('Multi-step deterministic simulation \nA -> B -> C -> P') # Title

plt.show()


# %% Exercise 4

import matplotlib.pyplot as plt

# Variables
k1, kr = (1.3 * (10 ** -2)), (6.2 * (10 ** -3)) # Forward and reverse reaction rates
A, B = 2.20, 1.72 # Initial concentrations
A_conc, B_conc = [], [] # Empty lists to append changing concentration to
length = 200 # Duration of simulation

# the simulation
for sec in range(length):
    A_conc.append(A)
    B_conc.append(B)
    # recalculate rates
    rate_A = k1 * (A ** 2)
    rate_B = kr * B
    #recalculate concentrations after next time increment
    A = A - rate_A + rate_B
    B = B + rate_A - rate_B

# Plotting
plt.plot(range(length), A_conc, label='[A]') # Plot for [A]
plt.plot(range(length), B_conc, label='[B]') # Plot for [B]
plt.legend(loc='best') # Legend
plt.xlabel('Time (s)') # X Label
plt.ylabel('Concentration (M)') # Y Label
plt.title('Step-wise function \n2A <-> B') # Title

plt.show()


# %% Exercise 5

import matplotlib.pyplot as plt

# Variables
A_conc, B_conc, I_conc, P_conc = [], [], [], [] # Empty lists for concentrations
A, B, I, P = 1.0, 0.6, 0.0, 0.0  # initial conc, M
k1, k2, kr1, kr2 = 0.091, 0.1, 0.03, 0.01 # rate constant
length = 500

# the simulation
for sec in range(length):
    A_conc.append(A)
    I_conc.append(I)
    B_conc.append(B)
    P_conc.append(P)
    # recalculate rates
    rate_1 = k1 * A
    rate_r1 = kr1 * I
    rate_2 = k2 * B * I
    rate_r2 = kr2 * P
    #recalculate concentrations after next time increment
    A = A - rate_1 + rate_r1
    I = I + rate_1 - rate_2 - rate_r1 + rate_r2
    B = B - rate_2 + rate_r2
    P = P + rate_2 - rate_r2


'''
For the adjusted K_r1 the final product P will be yielded at a faster rate.
'''

# Altered Variables
A_conc_adj, B_conc_adj, I_conc_adj, P_conc_adj = [], [], [], [] # Empty lists for concentrations
A_adj, B_adj, I_adj, P_adj = 1.0, 0.6, 0.0, 0.0  # initial conc, M
k1_adj, k2_adj, kr1_adj, kr2_adj = 0.091, 0.1, 0.001, 0.01 # rate constant
length2 = 500

# Altered simulation
for sec in range(length):
    A_conc_adj.append(A_adj)
    I_conc_adj.append(I_adj)
    B_conc_adj.append(B_adj)
    P_conc_adj.append(P_adj)
    # recalculate rates
    rate_1_adj = k1_adj * A_adj
    rate_r1_adj = kr1_adj * I_adj
    rate_2_adj = k2_adj * B_adj * I_adj
    rate_r2_adj = kr2_adj * P_adj
    #recalculate concentrations after next time increment
    A_adj= A_adj - rate_1_adj + rate_r1_adj
    I_adj = I_adj + rate_1_adj - rate_2_adj - rate_r1_adj + rate_r2_adj
    B_adj = B_adj - rate_2_adj + rate_r2_adj
    P_adj = P_adj + rate_2_adj - rate_r2_adj

# Graphing
plt.plot(range(length), A_conc, label='A', color='blue')
plt.plot(range(length), I_conc, label='I', color='red')
plt.plot(range(length), B_conc, label='B', color='green')
plt.plot(range(length), P_conc, label='P', color='orange')

plt.plot(range(length2), A_conc_adj, label='Adjusted A', color='blue', ls='-.')
plt.plot(range(length2), I_conc_adj, label='Adjusted I', color='red', ls='-.')
plt.plot(range(length2), B_conc_adj, label='Adjusted B', color='green', ls='-.')
plt.plot(range(length2), P_conc_adj, label='Adjusted P', color='orange', ls='-.')

plt.xlim(0.0, 100)
plt.ylim(0.0, 1.0)
plt.xlabel('Time, s')
plt.ylabel('Concentration, M')
plt.legend()


# %% Exercise 6

import matplotlib.pyplot as plt
import numpy as np

# Variables
A_conc, P1_conc, P2_conc = [], [], [] # Empty lists to append to
A, P1, P2 = 2.00, 0, 0 # Concentration M
k1, k2 = 0.02, 0.04 # rate constant M/s
t = np.arange(0, 91, 1) # time in seconds

# Step-wise for loop
for sec in range(len(t)):
    A_conc.append(A)
    P1_conc.append(P1)
    P2_conc.append(P2)
    # Rate calculations
    rate_k1 = k1 * A
    rate_k2 = k2 * A
    # Concentration Calculations
    A = A - rate_k1 - rate_k2
    P1 = P1 + rate_k1
    P2 = P2 + rate_k2


# Graph setup
plt.plot(t, A_conc, label='[A]', ls='-.')
plt.plot(t, P1_conc, label='$[P_1]$', ls='-.')
plt.plot(t, P2_conc, label='$[P_2]$', ls='-.')
plt.legend(loc='best')
plt.xlabel('Time (s)')
plt.ylabel('Concentration (M)')


# %% Exercise 7

import matplotlib.pyplot as plt
import numpy as np

# Variables
Ethylene = 28.06 # g/mol
Styrene = 104.16 # g/mol
monomer_length = 1000 # units or molecules
num_strands = 1000 # Number of polymer stands simulating

# To simulate the addition of these two monomers np.random.binomial() is used
# Ethylene will represent 0 in probability and Styrene will represent 1 in probability

polymer = np.random.binomial(1, p=0.5, size=(num_strands, monomer_length)) 

'''
The size=(1000x2) indicates that there are 1000 trials, and the row indicates this.
The second column indicates the length of the polymer which is also 1000 units.
'''

polymer = polymer * Ethylene + (1 - polymer) * Styrene 

'''
Ethylene is represented by 1 so multiplying polymer by Ethylene we calculate the
amount of mass of the polymer that is ethylene. 1 - polymer inverts the values,
Since Styrene represents 0 in the binomial distrobution we invert the value to 1
so it can be operated on. This also causes the values of ethylene (1) to be negated
to 0. This allows proper operation on the matrix.
'''

polymer = np.sum(polymer, axis=1)

'''
Calculates the sum over the second axis, this allows for a proper distrobution of
the polymers weights based off 1000 strands... I think... I don't know honestly.
'''

# Graph
plt.hist(polymer, bins=175) # creates histogram
plt.xlabel('Weights (g/mol)') # x label
plt.ylabel('Frequency') # y label
plt.title('Frequency of Ethylene and Styrene polymer weights') # title

plt.show()


# %% Exercise 8a

import numpy as np

# Empty polymer list 
polymer = []

# For Loop
for i in range(0,100):
    if len(polymer) >= 100:
        polymer = polymer[:100]
        break
    molecule = np.random.binomial(1, p=0.5, size=1)
    if molecule==0:
        num = np.random.randint(3, 6, size=1)
        for i in range(0,int(num)):
            polymer.append('B')
    if molecule==1:
        num = np.random.randint(3, 6, size=1)
        for i in range(0,int(num)):
            polymer.append('A')

print(len(polymer))
print(polymer)


# %% Exercise 8b

import numpy as np

'''
The previous exercise 8 was completed without the hint.
This exercise for number 8 will be completed using the entirety of the hint.
'''

monomer_length = 100
mono = []  # List to store the monomers

# Simulate the block copolymer
for i in range(monomer_length):
    mono.append(np.random.binomial(1, p=0.5))  # Append a random monomer (0 or 1) to the list

    # Switch between monomer types randomly
    if np.random.binomial(1, p=0.95):  # Adjust the probability as needed
        mono[-1] = 1 - mono[-1]  # Toggle between monomer types

# Convert the list to a string for visualization
block_copolymer = ''.join(['A-' if m == 0 else 'B-' for m in mono])
print(block_copolymer)


# %% Exercise 9 #NOTE: NEED TO FINISH PROBLEM

import numpy as np

rng = np.random.default_rng()
loc = np.cumsum(rng.integers(-1, high=2, size=(3000,2)), axis=0)

'''
I actually do not know how to explain what this code does.
I know numpy wraps data/lists into arrays and will carry-out operations on 
every value of the list without needing a for loop to iterate over every data point.
'''


# %% Exercise 10

import numpy as np

# Amino acid dictionary
amino_acid = {
    'A': 'Alanine',
    'R': 'Arginine',
    'N': 'Asparagine',
    'D': 'Aspartic Acid',
    'C': 'Cysteine',
    'Q': 'Glutamine',
    'H': 'Histidine',
    'I': 'Isoleucine',
    'L': 'Leucine',
    'K': 'Lysine',
    'M': 'Methionine',
    'F': 'Phenylalanine',
    'P': 'Proline',
    'S': 'Serine',
    'T': 'Threonine',
    'W': 'Tryptophan',
    'Y': 'Tyrosine',
    'V': 'Valine'
}

# Place amino acid keys into a list
amino_acid_list = list(amino_acid.keys())

# Generate protein
index = np.random.randint(0, len(amino_acid_list), size=1000)

protein = [amino_acid_list[i] for i in index] # creates protein
print(protein)


# %% Exercise 11

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t 


def calculate_percentage_in_interval(sample_size):
    trials = 100000
    true = 6.2  # true value
    in_interval = 0

    for trial in range(trials):
        # create synthetic data
        error = np.random.random(sample_size)
        data = np.ones(sample_size) * true + (error - 0.5)

        # calculate the 95% CI
        avg = np.mean(data)
        t_value = t.ppf(0.975, df=sample_size - 1)  # Look up t-value
        CI_95 = t_value * np.std(data, ddof=1) / np.sqrt(sample_size)
        lower = avg - CI_95
        upper = avg + CI_95

        # determine if the true value is inside the 95% CI
        if lower <= true <= upper:
            in_interval += 1

    return 100 * in_interval / trials


sample_sizes_dict = {10: 'r', 20: 'g', 30: 'b', 40: 'm', 50: 'c'}  # Replace with your sample sizes and colors

percentages = []
sample_sizes = []
for sample_size, color in sample_sizes_dict.items():
    percentage = calculate_percentage_in_interval(sample_size)
    percentages.append(percentage)
    sample_sizes.append(sample_size)
    plt.scatter(sample_size, percentage, color=color, label=f"Sample Size: {sample_size}")

plt.xlabel('Sample Size')
plt.ylabel('Percentage in Interval')
plt.title('Percentage of True Value in 95% Confidence Interval')
plt.legend()
plt.show()


# %% Exercise 12

#Imports
import numpy as np
import matplotlib.pyplot as plt

# Variables
# Molecules
molecules_pos = np.zeros(1000) # 1000 molecules all start with the same x position being 0

# Weights for diffusion
weights = [0.5, 0, -0.5]

# Simulations
steps = 500 # Steps in simulation

# For loop for simulation
for step in range(steps):
    for i in range(len(molecules_pos)):
        molecules_pos[i] += weights[np.random.choice([0, 1, 2])]

# Build histogram
plt.hist(molecules_pos, bins=100, color="lightblue", edgecolor='black')
plt.xlabel('Distance of Diffusion') 
plt.ylabel('# of Molecules')
plt.title('Molecules Diffusion')

plt.show()
