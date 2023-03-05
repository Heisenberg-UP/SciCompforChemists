# %% Exercise 1
# Imports
import matplotlib.pyplot as plt # Imports plotting library
import numpy as np # Imports linear array and math library

# File import
Data = np.genfromtxt('data/CV_K3Fe(CN)6.csv', delimiter=',', 
                     skip_footer=1, skip_header=1) # Imports file from data as a .csv

# Max and Min values
CV_max = np.argmax(Data[:, 1]) # Finds CV Max
CV_min = np.argmin(Data[:, 1]) # Finds CV Min

# Graph set-up
plt.plot(Data[:, 0], Data[:, 1], lw=0.5) # Plots the data from the .csv file
plt.plot(Data[CV_max, 0], Data[CV_max, 1], 'o', color='green') # Plots max value
plt.plot(Data[CV_min, 0], Data[CV_min, 1], 'v' , color='red') # Plots min value
plt.xlabel('Potential (V)') # X label
plt.ylabel('Current (A)') # Y label
plt.title('Cyclic Voltammogram of Potassium Cyanoferrate') # Title
plt.show()


# %% Exercise 2
# Imports
import matplotlib.pyplot as plt # Imports plotting library
import numpy as np # Imports linear array and math library

# File import
Data = np.genfromtxt('data/CV_K3Fe(CN)6.csv', delimiter=',', 
                     skip_footer=1, skip_header=1) # Imports file from data as .csv

# Differentiation
dy_Data = np.diff(Data[:, 1], n=1) # Differentiation of current
Max = np.argmax(dy_Data) # Max value of derivative
Min = np.argmin(dy_Data) # Min value of derivative

# Graph set-up
plt.plot(Data[:, 0], Data[:, 1], lw=0.5, color='blue') # Plots the data from the .csv file
plt.plot(Data[:-1], dy_Data, '-', color='orange') # Plots the differentiation
plt.plot(Data[Max][: -1], dy_Data[Max], 'o', color='green') # Plots positive inflection
plt.plot(Data[Min][:-1], dy_Data[Min], 'o' , color='red') # Plots negative inflection point
plt.xlabel('Potential (V)') # X label
plt.ylabel('Current (A)') # Y label
plt.title('Cyclic Voltammogram of Potassium Cyanoferrate') # Title
plt.show()


# %% Exercise 3
# Imports
import matplotlib.pyplot as plt # Imports plotting library
from scipy.signal import sawtooth, savgol_filter # Imports signal generation and smoothing data
import numpy as np # Impots linear array library

# Variables
t = np.linspace(0, 4, 1000) # Generates time values
sig = sawtooth(2 * np.pi * t) + np.random.rand(1000) # Generates signal with noise

# Smoothing data
smooth_sig = (sig[:-2] + sig[1:-1] + sig[2:]) / 3 # Moving average of smoothed data
savgol_sig = savgol_filter(sig, 21, 5) # Averages using Sacitzky-Golay filter

# Graph
plt.plot(t, sig, color='magenta', label='Signal') # Plots noisy data
plt.plot(t[1:-1], smooth_sig, color='blue', label='Averaged Signal') # Moving Averaged data
plt.plot(t, savgol_sig, color='green', label='Savitzky-Golay Averaging') # Savgol averaging
plt.legend(loc='best') # Places legend in the most appropriate location
plt.title('Generating Noisy Signal') # Graph title
plt.xlabel('Time (s)') # X label
plt.ylabel('Signal (Radians)') # Y label
plt.show()


# %% Exercise 4
# Imports
import matplotlib.pyplot as plt # Imports plotting library
from scipy.fft import fft # Imports fourier transform data
import numpy as np # Imports numpy array

# File import
nmr = np.array(np.loadtxt('data/fid_31P.csv', delimiter=',')) # Loads file and wraps it into an array

# Fourier transform
f_nmr = np.abs(fft(nmr)) # Takes absolute value of fourier transform

# Visualization of Data
plt.figure(figsize=(7,5)) # Sets figures size

plt.subplot(2, 1, 1) # Top plot
plt.title('NMR Data') # Title
plt.xlabel('Time (s)') # X label
plt.ylabel('Amplitude') # Y label
plt.plot(nmr, color='maroon')  # Plots provided data 

plt.subplot(2, 1, 2) # Bottom plot
plt.plot(f_nmr.real, color='maroon') # Plots FT Data
plt.title('Fourier Transformed NMR Data') # Title
plt.xlabel('Frequency (Hz)') # X label
plt.ylabel('Intensity') # Y label

plt.subplots_adjust(hspace=0.5) # Provides spacing between graphs

plt.show() # Shows graph


# %% Exercise 5
# Imports
import numpy as np # Imports linear array and math library
import matplotlib.pyplot as plt # Imports plotting library

# Variables
n_i = np.array([3, 4, 5, 6, 7]) # Initial principle quantum number
wl = np.array([656.1, 485.2, 433.2, 409.1, 396.4]) * (10**-9) # Wavelengths of transition

# Linearization
d_n = (1 / 4) - (1 / (n_i ** 2)) # Calculates 1/nf^2 - 1/ni^2
wl_inv = 1 / wl # Calculates inverse wavelength

# Calculating the Rydberg and comparing
R, b = np.polyfit(d_n, wl_inv, 1) # Calculations intercept and slope
R_theo = 10973731.57 # m^-1

Percent_error = ((R / R_theo) - 1) * 100 # Percent difference

print(R, '\n\n', Percent_error, '% error')


# %% Exercise 6
# Imports
import numpy as np # Imports linear array and math library
import matplotlib.pyplot as plt # Imports plotting library

# Variables
Conc = np.array([0.10, 0.16, 0.20, 0.25, 0.41, 0.55]) # [A] (M)
Rate = [0.0034, 0.0087, 0.014, 0.021, 0.057, 0.10] # Rate (M/s)
X = np.linspace(0, 0.6, 100) # Creating values for fitted data

a, b, c = np.polyfit(Conc, Rate, 2) # Calculates polynomial coefficients and constants

y_fit = a * X ** 2 # represents Rate = K[A]^2

#Graph for visualization
plt.figure(figsize=(8, 6)) # graph size
plt.title('Rate vs [A]') # Title
plt.xlabel('Concentration [A]') # X label
plt.ylabel('Rate ([A]/s)')
plt.scatter(Conc, Rate, label='Experimental data', color='blue') # Plots data
plt.plot(X, y_fit, label='Polynomial fit', linestyle='dashed', color='orange') # plots fitted data
plt.legend(loc='best') # Creates legend

plt.show() # Shows graph


# %% Exercise 7
import numpy as np # Imports linear array and math library

# Variables
ab = np.array([0.125, 0.940, 2.36, 2.63, 3.31, 3.77]) # Absorbance values
Conc = np.array([0.150, 1.13, 2.84, 3.16, 3.98, 4.53]) # [A] of red dye

# Linear fitting
a, b = np.polyfit(Conc, ab, 1) # Fitting data

# Finding the concentration of soft drink
unknown_conc = (0.481 - b) / a

print(unknown_conc, ' M')


# %% Exercise 8
# Imports
import numpy as np # Imports math and linear array library
import matplotlib.pyplot as plt # Imports plotting library
from scipy import interpolate # Imports interpolation for data

# Variables
r = np.array([1, 5, 9, 13, 17]) # Radius in Bohr
psi = np.array([0.21, -0.087, -0.027, -0.0058, -0.00108])  # non-squared probability values
X = np.linspace(1, 17, 100) # For interpolation fitting

# Interpolating data
cubic_int = interpolate.interp1d(r, psi, kind='quadratic') # Make data SMMOOOOOTTTHHH

# Data visualization
plt.figure(figsize=(8,6)) # Graph size
plt.title('Hydrogen Radial Wavefunction') # Graph title
plt.xlabel('r (Bohrs)') # X label
plt.scatter(r, psi, color='blue', label='Sampled Data') # Sampled Data
plt.plot(X, cubic_int(X), linestyle='dashed', color='maroon', label='Cubic fit') # Interpolation data 
plt.legend(loc='best') # Legend

plt.show()


# %% Exercise 9
# Imports
import matplotlib.pyplot as plt # Imports plotting library
import numpy as np # Imports linear array and math library
from scipy.signal import argrelmin # Imports local minima function

# File import
Ferrocene = np.genfromtxt('data/Cp2Fe_Mossbauer.txt', delimiter='   ', 
                          skip_header=10) # Imports the text file and skips the beginning 10 rows

# Variables
F_min = argrelmin(Ferrocene[:, 1], order=35) # Finds 6 local minima

# Data visualization
plt.figure(figsize=(8,5)) # Sets figure size
plt.title('Ferrocene Transmission vs Velocity') # Title
plt.xlabel('Velocity (mm/s)') # X label
plt.ylabel('Relative Transmission') # Y label
plt.plot(Ferrocene[:, 0], Ferrocene[:, 1], color='darkblue', label='Sample Data') # Plots Sample data
plt.plot(Ferrocene[F_min, 0], Ferrocene[F_min, 1], 'C1o') # Plots local minima

plt.show() # Shows graph


# %% Exercise 10
# Imports
import numpy as np # Imports linear Array and math library
import pandas as pd # File manipulation library
from scipy.signal import find_peaks # Finds peaks from data set

# File import
XRF = pd.read_csv("data/XRF_Cu.csv", skiprows=20) # Reads .csv and skips first 20 rows
XRF = pd.DataFrame(XRF, columns=['Channel#', 'Intensity']) # Creates Dataframe

# Wrapping values in an array
XRF['Intensity'] = np.array(XRF['Intensity']) # Wraps Intensity into an array
XRF['Channel#'] = np.array(XRF['Channel#']) * 20.06 # Converts channel to eV

# Finding maxima
XRF_peaks = find_peaks(XRF['Intensity'], height=1000) # Finds peaks with a minimum height barrier
print('Energy of first peak: ', XRF['Channel#'][XRF_peaks[0][0]], ' eV', 
      '\nEnergy of second peak: ', XRF['Channel#'][XRF_peaks[0][1]], ' eV')
