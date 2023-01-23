# Imports
import math
import random
import numpy as np

# Variables
V = 1.6285  # Volume in L
n = 1.229  # Moles
T = 273.0  # Temperature in K
R = 0.08206  # Constant in L•atm/mol•K
Point = [23, 81]  # Coordinate value of 2D plane
x = 12  # Part of exercise 3
a = 1
b = 2
c = 1
elements = 'NaKBrClNOUP'
CC_single = '345'  # Bond energies in KJ/mol
CC_double = '611'  # Bond energies in KJ/mol
filename = 'Standard.csv'
DNA = 'ATTCGCCGCTTA'  # DNA sequence
Li, C, Na = 3, 6, 11  # Atomic Numbers
Molecules = ['HCl', 'NaOH', 'KCl', 'Ca(OH)2', 'KOH', 'HNO3',
             'Na2SO4', 'KNO3', 'Mg(OH)2', 'HCO2H', 'NaBr']
List = list(range(18, 90, 2))  # Problem 12 List
Tuple = tuple(range(18, 320, 2))  # Problem 13 Tuple
nums = [random.randint(0, 20) for x in range(10)]  # Problem 14 random list
Var = 'I LOVE COOKIES'

# Simple Operations
Hypotenuse = math.hypot(Point[0], Point[1])  # Solution 2a
x = x + 32  # Solution 3
NaK = elements[0:3]  # Solution 5a
UP = elements[-2:0]  # Solution 5b
KBr = elements[2:5]  # Solution 5c
NKrlOP = elements[0] + elements[2] + elements[4] + elements[6] + elements[-3] + elements[-1]
# ^^^ Solution 5 a-d
filename_7a = filename.rstrip('.cvs')  # Solution 7a
'''filename = (list(filename)).pop([8])
print(filename)'''  # I don't care enough Solution 7b
print(List)
List.reverse()  # Problem 12a
print(List)
List.remove(18)  # Problem 12b
print(List)
List.append(16)  # Problem 12c
print(List)
print(Tuple)  # Problem 13a
print('You cannot reverse, remove, or append a tuple')  # Problem 13b

# if statement
if nums.__contains__(7) is True:  # Solution to problem 14
    print(nums, 'The list contains 7')
else:
    print(nums, 'This list does not contain 7')

# Opening a file for problem 20
with open('Test.csv', 'a') as file:
    file.write('time, [A] \n')
    for t in range(20):
        file.write('%s, %s \n' % (t, math.exp(-0.5*t)))
file.close()
file = open('Test.csv')
print('Solution 21: ')
for line in file.readlines()[0:21]:
    print(line)
file.close()

# Appending file using Numpy for solution 21
file = np.genfromtxt('Test.csv', delimiter=',', skip_header=1)
Time = []
Conc = []
for columns in file:
    Time.append(columns[0])
    Conc.append(columns[1])
print('Solution 22: ')
print(str(Time), '\n', str(Conc))


# Functions
def ideal_gas_law(V, n, T, R):
    '''Function only yields pressure value from ideal gas law'''
    P = (n * R * T) / V
    print('Solution 1: ', P, ' atm')


def _2d_distance(Point):
    dx = Point[0] - 0  # dx is redundant since the origin is 0,0
    dy = Point[1] - 0  # dy is redundant since the origin is 0,0
    H = math.sqrt((dx**2) + (dy**2))
    print('Solution 2b: ', H)


def quadratic_formula(a, b, c):
    x_plus = (-b + math.sqrt(b**2 - 4 * a * c)) / (2 * a)
    x_negative = (-b - math.sqrt(b**2 - 4 * a * c)) / (2 * a)
    print('Solution 4: ', x_plus, ', ', x_negative)


def bond_energies(CC_single, CC_double):
    # The bond energy for the sigma bond is just CC_single
    Pi_bond = CC_double - CC_single  # Bond energy in KJ/mol
    print('Solution 6: ', Pi_bond, ' KJ/mol')


def boolean_logic(DNA): # Weird Boolean Logic
    print('Solution 8: ')
    DNA_list = list(DNA)
    print(bool(DNA_list))
    DNA_list.reverse()
    print(bool(DNA_list.reverse))


def atomic_number_logic(Li, C, Na):
    print('Solution 9a-d: ')
    if True:
        if bool(Li > C) is False:
            print('Li is not greater than C')
        if bool(Na <= C) is False:
            print('Na is not less than C')
        if bool(Li > C or Na > C) is True:
            print('Na is greater than C, but Li is not')
        if bool(C > Li and Na > Li) is True:
            print('C and Na are both greater than Li')
    else:
        print('Error')


def acidity(Molecules):
    print('Problem 10: ')
    for i in range(len(Molecules)):
        if Molecules[i].startswith('H') is True:
            print('Acidic Molecule:', Molecules[i])
        elif Molecules[i].__contains__('OH'):
            print('Basic Molecule:', Molecules[i])
        else:
            print('Neurtral Molecule:', Molecules[i])


def ions(Protons, Electrons):
    print('Solution 11: ')
    if int(Protons) == int(Electrons):
        print('Neutral')
    if int(Protons) > int(Electrons):
        print('Cation')
    if int(Protons) < int(Electrons):
        print('Anion')


def string_manipulation(Var):
    print('Solution 15: ')
    print(Var)
    print(Var.lower().split(sep=None))  # Solution 15a
    Var_2 = sorted(Var.lower().split(sep=None))  # Solution 15b
    print(Var_2)


def print_func():
    print('Solution 17: ')
    Ideal_variable = 'PV=nRT ' * 20
    print(Ideal_variable)


def appending():
    print('Solution 16: ')
    A = []
    for i in range(10):
        A.append(i)
        A.append(i)
    print(A)


def mass_division():
    print('Solution 18: ')
    Mass = 1000  # Mass in grams
    for i in range(6):
        print(str(Mass), ' g')
        Mass = Mass / 2


def carbon_half_life():
    print('Solution 19: ')
    C_Mass = 500  # Mass in grams
    C_halflife = 0  # Half-life in years
    while C_Mass >= 10.00:
        C_Mass = C_Mass / 2  # One Half-life
        C_halflife = C_halflife + 1
        Years = 30.2 * C_halflife
    print(str(Years), 'years until only 7.8125 g of Carbon 137 remains')


def docstring_1(V, n, R, T):
    ''' This function solves the Ideal gas law equation for pressure
    using a positional argument. The order is Volume, Moles, R
    constant, and Temperature
    '''
    print('Solution 23a: ')
    P = (n * R * T) / V
    print(P)


def docstring_2(V=1, n=1, R=0.08206, T=0):
    ''' Docstring BABYYYYYYYY!!!!!
    '''
    print('Solution 23b: ')
    P = (n * R * T) / V
    print(P)


def rate(A0, k=1.0, n=1):
    ''' (concentration(M), k = 1.0, n = 1) → rate (M/s)
    Takes in the concentration of A (M), the rate constant (k),
    and the order (n) and returns the rate (M/s)
    '''
    Rate = k * (A0 ** n)
    print('Solution 24: ')
    print(Rate)


def DNA_strands(Strand):

    Opposite_strand = []

    for i in range(len(Strand)):
        if Strand[i].__contains__('A'):
            Opposite_strand.append('T')
        elif Strand[i].__contains__('T'):
            Opposite_strand.append('A')
        elif Strand[i].__contains__('G'):
            Opposite_strand.append('C')
        elif Strand[i].__contains__('C'):
            Opposite_strand.append('G')

    Opposite_strand = ''.join(Opposite_strand)

    print(Opposite_strand)


DNA_strands('ATGGC')