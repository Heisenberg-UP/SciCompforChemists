# %% Exercise 1
# Imports 
import pandas as pd # Imports pandas for file manipulation and importation

# Variables
mp = [6, -95, -95, -130, -116, -114, -98] # Temps in Celsius
bp = [80, 56, 11, 36, 35, 78, 65] # Temps in Celsius
solvent = ['Benzene', 'Acetone', 'Toluene', 'Pentane', 'Ether', 'Ethanol', 
           'Methanol'] # Names of compounds


# Creates Series
Boiling_Points = pd.Series(bp, solvent) # Creates a series with the solvent name as the index

print(Boiling_Points) # 1a
print('\n', Boiling_Points['Ethanol']) # 1a

# Creates DataFrame
df = pd.DataFrame({'MP': mp, 'BP': bp}, index=solvent) # Creates DataFrame of MP and BP with solvent names as the index

print(df) # 1b
print('\n', df.loc['Benzene', 'MP']) # 1b
print('\n', df.iloc[3, 1]) # 1c


# %% Exercise 2
# Imports
import pandas as pd # Imports pandas and file manipulation library
import matplotlib.pyplot as plt # imports plotting library

# File import
Data = pd.read_csv('data/blue1.csv', index_col=0) # Reads data and sets wavlengths to index
Data.index.name = 'Wavelength' # Names index

print(Data) # 2a

Data.plot(y='abs', use_index=True) #Plots data and uses index as x-axis
plt.xlabel('Wavelength (nm)') # x label
plt.ylabel('Absorbance') # y label
plt.title('Absorption Sprectrum of Blue dye') # Graph title

plt.show() # 2b

print(Data.loc[620]) # 2c


# %% Exercise 3
# Imports
import pandas as pd # Imports file manipulation library
import numpy as np # Imports math and linear array library
import matplotlib.pyplot as plt # Imports plotting library

# File import
Data = pd.read_csv('data/kinetics.csv') # Imports the .csv file
df = pd.DataFrame(Data, columns=['0', '[A](M)', 'time(s)']).drop('0', axis=1) # Creates DataFrame

# Adding columns
df['ln[A]'] = np.log(df['[A](M)']) # Data manipulation
df['[A]^-1'] = (df['[A](M)'])**-1 # Data manipulation
df['[A]^0.5'] = (df['[A](M)']**0.5) # Data manipulation

# Graphing
plt.figure(figsize=(12, 10))

plt.subplot(2,2,1)
plt.scatter(df['[A](M)'], df['time(s)'])
plt.xlabel('Time (s)')
plt.ylabel('[A] (M)')
plt.title('[A] vs Time')

plt.subplot(2, 2, 2)
plt.scatter(df['ln[A]'], df['time(s)'])
plt.xlabel('Time (s)')
plt.ylabel('ln[A] (M)')
plt.title('ln[A] vs Time')

plt.subplot(2, 2, 3)
plt.scatter(df['[A]^-1'], df['time(s)'])
plt.xlabel('Time (s)')
plt.ylabel('[A]^-1 (M)')
plt.title('[A]^-1 vs Time')

plt.subplot(2, 2, 4)
plt.scatter(df['[A]^0.5'], df['time(s)'])
plt.xlabel('Time (s)')
plt.ylabel('[A]^0.5 (M)')
plt.title('[A]^0.5 vs Time')

plt.show()

'''
This data shows the reaction is second order
'''


# %% Exercise 4
# Imports 
import pandas as pd # Imports file manipulation library

# File import
df = pd.DataFrame(pd.read_csv('data/ROH_data.csv'), columns=['bp', 'density', 'MW'])
df.dropna(inplace=True)

print(df)


# %% exercise 5
# Imports 
import pandas as pd # Imports file manipulation library

# .csv read file 
red40 = pd.read_csv('data/red40.csv', names=['Wavelength (nm)', 'Absorbance'])
green3 = pd.read_csv('data/green3.csv', names=['Wavelength (nm)', 'Absorbance'])
blue1 = pd.read_csv('data/blue1.csv', names=['Wavelength (nm)', 'Absorbance'])
yellow6 = pd.read_csv('data/yellow6.csv', names=['Wavelength (nm)', 'Absorbance'])

# Concantation of DataFrames
df = pd.concat([red40, green3['Absorbance'], blue1['Absorbance'], yellow6['Absorbance']], axis=1).dropna()
df.set_index('Wavelength (nm)', inplace=True)
df.columns = ['Red Standard', 'Green Standard', 'Blue Standard', 'Yellow Standard']

print(df)


# %% Exercise 6
# Imports
import pandas as pd # Imports file manipulation library

# .csv read file
alcohol = pd.read_csv('data/alcohols.csv')
alkane = pd.read_csv('data/alkanes.csv')

# Drop columns containing compound names
alcohol.drop(columns=['compound'], inplace=True)
alkane.drop(columns=['compound'], inplace=True)

df = pd.merge(alcohol, alkane, on='carbons')
df.sort_values('carbons', inplace=True)
df.set_index('carbons', inplace=True)

print(df)
