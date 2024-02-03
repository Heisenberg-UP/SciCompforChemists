# %% Exercise 1

# Imports
import pandas as pd # Imports data manipulation library
import seaborn as sns # Imports fancy manipulation library
import matplotlib.pyplot as plt # Imports data visualization library

# Data import
linear_data = pd.read_csv('data/linear_data.csv')
ld = pd.DataFrame(linear_data, columns=['x', 'y'])

# Plot linear regression graph
sns.regplot(x=ld['x'], y=ld['y'])

plt.show()


# %% Exercise 2

# Imports
import pandas as pd # Imports data manipulation library
import seaborn as sns # Imports fancy manipulation library
import matplotlib.pyplot as plt # Imports data visualization library

# Read desired files
IR_data = pd.read_csv('data/ir_carbonyl.csv') # Reads .csv file
IR = pd.DataFrame(IR_data, columns=['ketones', 'aldehydes', 'acids', 'esters', 'amides']) # Creates dataframe of IR data

# Visualize data using categorical plot
sns.violinplot(data=IR) # Plot data using violins
plt.ylabel('Frequency (cm$^-$$^1$)') # X label
plt.xlabel('Functional Groups') # Y label

plt.show()


# %% Exercise 3

# Imports
import pandas as pd # Imports data manipulation library
import seaborn as sns # Imports fancy manipulation library
import matplotlib.pyplot as plt # Imports data visualization library

# Read desired files
IR_data = pd.read_csv('data/ir_carbonyl.csv') # Reads .csv file
IR = pd.DataFrame(IR_data, columns=['ketones', 'aldehydes', 'acids', 'esters', 'amides']) # Creates dataframe of IR data

# Separate variables
ketones = IR['ketones'] # Creates individual dataset for ketones
aldehydes = IR['aldehydes'] # Creates individual dataset for aldehydes

# Plot data using a kde plot
sns.kdeplot(ketones, label='Ketones') # Kernel Density Estimates for ketones
sns.kdeplot(aldehydes, label='Aldehydes') # KErnel Density Estimates for Aldehydes
plt.legend() # Shows legen
plt.xlabel('Frequency cm$^-$$^1$') # x label

plt.show()


# %% Exercise 4

# Imports
import pandas as pd # Imports data manipulation library
import seaborn as sns # Imports fancy manipulation library
import matplotlib.pyplot as plt # Imports data visualization library

# Read desired files
elements_data = pd.read_csv('data/elements_data.csv') # Read .csv file
ED = pd.DataFrame(elements_data) # Creates datafram for handling data

# Create count block
sns.countplot(x='block', data=ED, hue='block') # Create count block

plt.show()


# %% Exercise 5

# Imports
import pandas as pd # Imports data manipulation library
import seaborn as sns # Imports fancy manipulation library
import matplotlib.pyplot as plt # Imports data visualization library

# Read desired files
blackbody_data = pd.read_csv('data/blackbody.csv') # Read .csv file
BB = pd.DataFrame(blackbody_data) # Creates datafram for handling data

# Create lineplot for visualization
sns.lineplot(x='lambda', y='intensity', data=BB, hue='temp', palette='viridis') # Creates lineplot

plt.show()


# %% Exercise 6

# Imports
import pandas as pd # Imports data manipulation library
import seaborn as sns # Imports fancy manipulation library
import matplotlib.pyplot as plt # Imports data visualization library

# Read desired files
IE_data = pd.read_csv('data/ionization_energies.csv', index_col=0) # Read .csv file
IE = pd.DataFrame(IE_data) # Creates dataframe for data handling

# Create heat map plot
sns.heatmap(IE, annot=True, fmt='d', cmap='viridis', cbar_kws={'label': 'Ionization Energy, kJ/mol'}) # Plot heatmap

plt.show()


# %% Exercise 7

# Imports
import pandas as pd # Imports data manipulation library
import seaborn as sns # Imports fancy manipulation library
import matplotlib.pyplot as plt # Imports data visualization library

# Read desired files
ROH_data = pd.read_csv('data/ROH_data_small.csv') # Read .csv file
ROH = pd.DataFrame(ROH_data) # Create dataframe for handling data

# Create pair plot
sns.pairplot(ROH)

plt.show()


# %% Exercise 8

# Imports
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import sympy
from sympy.physics.hydrogen import R_nl

R = sympy.symbols('R')
r = np.arange(0,60,0.01)

max_radii = []

for n in range(1,5):
    shell_max_radii = []
    for l in range(0, n):
        psi = R_nl(n, l, R)
        f = sympy.lambdify(R, psi, 'numpy')
        max = np.argmax(f(r)**2 * r**2)
        shell_max_radii.append(max/100)
    max_radii.append(shell_max_radii)    
    
columns, index = (0,1,2,3), (4,3,2,1)
max_prob = pd.DataFrame(reversed(max_radii), columns=columns, index=index)

# Create heatmap
sns.heatmap(max_prob, annot=True, cbar_kws={'label': 'Orbital Radius, angstroms'})
plt.ylabel('Quantum Number (n)') # Y label
plt.xlabel('Angular Momentum (l)') # X label

plt.show()