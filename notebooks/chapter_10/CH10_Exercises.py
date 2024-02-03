# %% Exercise 1

# Imports
import pandas as pd # Imports data manipulation library
import seaborn as sns # Imports fancy manipulation library

# Data import
linear_data = pd.read_csv('data/linear_data.csv')
ld = pd.DataFrame(linear_data, columns=['x', 'y'])

# Plot linear regression graph
sns.regplot(x=ld['x'], y=ld['y'])


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


# %% Exercise 3

# Imports
