# %% Exercise 1 a & b

# Imports
import nmrglue as ng # Imports nmr data handling library
import matplotlib.pyplot as plt # Import graphical visualization library
import numpy as np # Import array/math library
from scipy.fft import fft # Import fast fourier transform

# Read desired file
dic, data = ng.pipe.read('data/EtOh_1H_NMR_CDCl3.fid') # Read nmr data

# Create necessary variables
udic = ng.pipe.guess_udic(dic, data) # Create metadata library
uc = ng.pipe.make_uc(dic, data) # Make convertable unit data 
ppm = uc.ppm_scale() # Create ppm scale 

# Analyze metadata to see if fourier transform has occured
if bool(udic[0]['time']) == True: # Check if data has been transformed
    fdata = fft(data) # FFT data in time domain
    pdata = ng.process.proc_autophase.autops(fdata, 'acme') # phase data

else: # Data has already been transformed
    pdata = ng.process.proc_autophase.autops(data, 'acme') # phase data

# Check peaks
peak_data = ng.analysis.peakpick.pick(data.real, pthres=0.25) # Creates list of peak data and metadata
peaks = [ppm[int(x[0])] for x in peak_data] # List of peak locations in ppm
print(peaks) # Make sure the CDCl3 peak is at 7.24 ppm

# Integrate CH3- and CH2- regions of nmr
limits = [[1.1, 1.3],[3.6, 3.8]] # Regions of integration
area = ng.analysis.integration.integrate(pdata.real, uc, limits) # Integration
proton_area  = area / np.min(area) # Normalize area for analysis
print(proton_area) # Check to make sure there os a 3:2 ratio of CH3 to CH2 

# Plot spectrum to investigate
plt.plot(ppm, pdata.real) # Plot data
plt.xlim(0, 4) # ppm region of interest
plt.xlabel('Chemical Shiftl, ppm') # X label
plt.ylabel('Abundance') # Y label
plt.gca().invert_xaxis() # Invert axis for OG Chemists


# %% Exercise 2 # TO BE FINISHED

# Imports
import nmrglue as ng # Imports nmr data handling library
import matplotlib.pyplot as plt # Import graphical visualization library
import numpy as np # Import array/math library
from scipy.fft import fft # Import fast fourier transform

# Read desired file
P_dic, P_data = ng.pipe.read('data/2-ethyl-1-hexanol_1H_NMR_CDCl3.fid') # Read 1H nmr data
C_dic, C_data = ng.pipe.read('data/2-ethyl-1-hexanol_13C_NMR_CDCl3.fid') # Read 1H nmr data

# Create necessary variables
P_udic = ng.pipe.guess_udic(P_dic, P_data) # Create metadata library
P_uc = ng.pipe.make_uc(P_dic, P_data) # Make convertable unit data 
P_ppm = P_uc.ppm_scale() # Create ppm scale 

C_udic = ng.pipe.guess_udic(C_dic, C_data) # Create metadata library
C_uc = ng.pipe.make_uc(C_dic, C_data) # Make convertable unit data 
C_ppm = C_uc.ppm_scale() # Create ppm scale 

# Analyze metadata to see if fourier transform has occured
if bool(P_udic[0]['time']) == True: # Check if data has been transformed
    P_fdata = fft(P_data) # FFT data in time domain
    P_pdata = ng.process.proc_autophase.autops(P_fdata, 'acme') # phase data

else: # Data has already been transformed
    P_pdata = ng.process.proc_autophase.autops(P_data, 'acme') # phase data

# Analyze metadata to see if fourier transform has occured
if bool(C_udic[0]['time']) == True: # Check if data has been transformed
    C_fdata = fft(C_data) # FFT data in time domain
    C_pdata = ng.process.proc_autophase.autops(C_fdata, 'acme') # phase data

else: # Data has already been transformed
    C_pdata = ng.process.proc_autophase.autops(C_data, 'acme') # phase data

# Check peaks
P_peak_data = ng.analysis.peakpick.pick(P_data.real, pthres=0.05) # Creates list of peak data and metadata
P_peaks = [P_ppm[int(x[0])] for x in P_peak_data] # List of peak locations in ppm
print(P_peaks) # Make sure the CDCl3 peak is at 7.24 ppm
