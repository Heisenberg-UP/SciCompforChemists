# Imports
import matplotlib.pyplot as plt # Imports plotting library
%matplotlib inline
import skimage # Imports scientific image processing library
from skimage.util import random_noise # Imports random noise module
from skimage import data, io, color, exposure, transform # Imports packages from Skimage library
import numpy as np # Imports linear array and math library

# %% Exercise 1

# Loading image
NaK_THF = io.imread('data/NaK_THF.jpg') # Imports image from data folder

# converting the image
NaK_THF = color.rgb2gray(NaK_THF) # Converts image from RGB to grey-scale

# Saving the image
io.imsave('data/Grey_NaK_THF.png', NaK_THF) # Saves image to data folder

# %% Exercise 2

# Loading image
chelsea = data.chelsea()

# Converting the image to greyscale
chelsea = color.rgb2gray(chelsea)

# Visualizing Image
plt.figure(figsize=(12, 12)) # Creates figure

plt.subplot(2, 1, 1) # Top plot
io.imshow(chelsea) # Uses Skimage to visualize image
plt.subplot(2, 1, 2) # Bottom plot
plt.imshow(chelsea) # Uses matplot to visualize image

'''
The graphs are different because matplotlib does not 
that this is an image with specific color. Instead it
assigns generic heat map.
'''

# %% Exercise 3

chem = data.immunohistochemistry() # Pulls random image
chem = color.rgb2gray(transform.resize(chem, (100, 100))) # Transforms into 100 x 100 pixels
noise = np.random.normal(loc=0.0, scale=0.1, size=chem.shape) # Generates random noise
chem_noisy = chem + noise # Adds noise to image

plt.figure(figsize=(8, 6)) # Creates a figure

plt.subplot(1, 2, 1) # First plot
plt.imshow(chem_noisy) # Shows image

plt.subplot(1, 2, 2) # Second plto
hist = exposure.histogram(chem_noisy) # Creates histogram of exposure
plt.plot(hist[0]) # Plots histogram
plt.xlabel('Values') # X label
plt.ylabel('Counts') # Y label

plt.show()

'''
At this point while working on the chapter I became very annoyed with the basic image processing problems. I am by no means an expert in image processing,
I actually find image processing for computer vision quite interesting. However, I have come to learn that I can enjoy something but dislike the work required.
'''
