# Exercise 1
import itertools

import numpy as np  # Imports NumPy which is a greate mathematical module


    # Exercise 1A
def natural_log():
    list = []  # Creates an empty list to append to
    for numbers in range(2, 24):  # For loop to iterate over range required
        logs = np.log(numbers)  # Performs mathematical operation on list
        list.append(logs)  # Appends mathematical operation into list
    print(list)


    # Exercise 1B
'''List comprehension shows how to iterate over a for loop within the
brackets of an empty list. Extremely efficient!'''
List = [np.log(numbers) for numbers in range(2, 24)]


# Exercise 2
def trans(coord, x=0, y=0, z=0):
    '''
    ([x,y,z], x=0, y=0, z=0) -> [x,y,z]
    You must input coordinates as bracketed list
    You cannot augment a tuple through indexing assignment
    '''
    coord[0] += x  # Recursively adds given x-value for translation
    coord[1] += y  # Recursively adds given y-value for translation
    coord[2] += z  # Recursively adds given z-value for translation
    print(coord)


# Exercise 3
'''Lambda function allows simple functions to be quickly assigned to
variables for easy and simple use. Below is a lambda function assigned to
calculating the square of a input value'''
sqr_func = lambda x: x**2


# Exercise 4
'''Dictionary of all the amino acids with a key of their one letter 
abbreviation and a value of their three letter abbreviation'''
aacid = {'G': 'GLY', 'A': 'ALA', 'V': 'VAL', 'P': 'PRO', 'L': 'LEU',
         'I': 'ILE', 'M': 'MET', 'F': 'PHE', 'W': 'TRP', 'R': 'ARG',
         'E': 'GLU', 'H': 'HIS', 'S': 'SER', 'D': 'ASP', 'N': 'ASN',
         'K': 'LYS', 'Y': 'TYR', 'C': 'CYS', 'T': 'THR', 'Q': 'GLN'}


# Exercise 5
Acid1 = {'HCl', 'HNO3', 'HI', 'H2SO4'}  # Dummy set 1
Acid2 = {'HI', 'HBr', 'HClO4', 'HNO3'}  # Dummy set 2


    # Exercise 5A
Acid_addition = Acid1 | Acid2  # Creates a new set with both sets combine


    # Exercise 5B
Acid_overlap = Acid1 & Acid2  # Yields a set with the overlap from both sets


    # Exercise 5C
Acid1.add('HBrO3')  # Adds a new object to the set


    # Exercise 5D
Acid_subtraction = Acid1 - Acid2  # Generates a set with the differences


# Exercise 6
import os  # Module imports operating system authorizations
os.chdir('/Users/keeganeveritt/Desktop')  # Sets directory to Desktop
for files in os.listdir():  # For loop iterates over all files in Directory
    print(files)
'''Printing the files using the listdir() function will show all hidden 
files. Terminal will hide files that serve a purpose for the OS to use 
without the need of the User.'''


# Exercise 7
import random  # Imports module that supports random processes


    # Exercise 7A
'''Creates a random list with integers from 0 -> 9 and takes the mean 
value of the list'''
random_list = [random.randrange(0, 9) for i in range(11)]
random_mean = np.mean(random_list)


    # Exercise 7B
'''Creates a random list with integers from 0 -> 9 and tajes the mean
values of the list, but for 7B this will use 10,000 values instead of 10'''
random_list = [random.randrange(0, 9) for i in range(10001)]
random_mean = np.mean(random_list)
'''The mean is significantly closer to 4.5 for 7B when the mean for the 
random list in 7A was 2.7. Since the values are between 0 and 9, when 
10,000 random values between said range are averaged you will approach
the mid-point between the range'''


# Exercise 8
# Given code from module
from random import randint  # Imports random integer module from native
atoms = []  # Empty list of atomic coordinates to append to
for i in range(5):  # For loop for iterations of coordinates
    x, y, z = randint(0,20), randint(0,20), randint(0,20)
    atoms.append((x, y, z))  # Appends to empty list "atoms"


def distance(a, b):  # Function defines the magnitude of distance
    x1, y1, z1 = a  # Point a
    x2, y2, z2 = b  # Point b
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)


min_distance = float('inf')
for a, b in itertools.combinations(atoms, 2):
    min_distance = min(min_distance, distance(a, b))
''' I am lost to how Python knows how to calculate this by using itertools
I believe it is doing two combinations of each set of vectors but I dont
know how it knows which to differentiate a, b and assign appropriate values'''


# Exercise 9
    # Exercise 9A
a_symbol = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne']
a_number = list(range(1, 11, 1))


    # Exercise 9B
zipped = zip(a_number, a_symbol)  # Zips two lists into a package
info_lis = []  # Empty list which can be appended to
for pair in zipped:  # For loop for iterates over entire list
    info_lis.append(pair)  # Appends pair (atomic number, symbol) into list


# Exercise 10
sym = ['H', 'He', 'Li', 'Be', 'B', 'C']  # Atomic symbol list
name = ['Hydrogen', 'Helium', 'Lithium', 'Beryllium', 'Boron', 'Carbon']  # Atomic name list
Dict = dict(zip(sym, name))
'''Dict() only allows one argument but the question requires two, by zipping
the two lists together, the Dict() function can be applied to both simultaneously'''


# Exercise 11
numbers = enumerate([random.randrange(0, 21) for i in range(11)])
index_values =[]  # Empty list to append to
for index, values in numbers:  # iterates over list
    if values > 10:  # if statement allows python to use indeces with values of 10 or more
        index_values.append(index)  # Appends to empty list


# Exercise 12
def mult_dim_dist(x=0, y=0, z=0):  # function defines magnitude of unit vector
    return np.sqrt(x**2 + y**2 + z**2)  # Distance formula


# Exercise 13
def alpha_decay(x, p, n):
    '''(alpha decays(x), protons(int), neutrons(int)) -> prints p and n remaining
    Takes in the number of alpha decays(x), protons(p), and number of neutrons(n)
    and all as integers and prints the final number of protons and neutrons.

    # tests
    >> alpha_decay(2, 10, 10)
    6  protons and 6  neutrons remaining.
    >> alpha_decay(1, 6, 6)
    4  protons and 4  neutrons remaining.
    '''
    p -= 2 * x
    n -= 2 * x
    print(str(p), ' protons and', str(n), ' neutrons remaining.')