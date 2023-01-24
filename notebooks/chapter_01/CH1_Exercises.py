# %% Exercise 1
P = (1.220 * 273.0 * 0.08206) / 1.6285

print('Pressure = ', str(P), ' atm')


# %% Exercise 2
import math

Hypotenuse = math.hypot(23, 81)


def _2d_distance():
    dx = 23 - 0  # dx is redundant since the origin is 0,0
    dy = 81 - 0  # dy is redundant since the origin is 0,0
    H = math.sqrt((dx**2) + (dy**2))
    return print('Hypotenuse = ', str(H))

print('Hypotenuse = ', str(Hypotenuse))
_2d_distance()


# %% Exercise 3
x = 12
x = x + 32

print(x)


# %% Exercise 4
import math


def quadratic_formula(a=1, b=2, c=1):
    x_plus = (-b + math.sqrt(b**2 - 4 * a * c)) / (2 * a)
    x_negative = (-b - math.sqrt(b**2 - 4 * a * c)) / (2 * a)
    return print('x = ', x_plus, ', ', x_negative)


quadratic_formula()


# %% Exercise 5
elements = 'NaKBrClNOUP'

NaK = elements[0:3]  # Solution 5a
UP = elements[9:]  # Solution 5b
KBr = elements[2:5]  # Solution 5c
NKrlOP = elements[0] + elements[2] + elements[4] + elements[6] + elements[-3] + elements[-1]
# ^^^ Solution 5 a-d


print(str(NaK), '\n', str(UP), '\n', str(KBr), '\n', str(NKrlOP))


# %% Exercise 6
def bond_energies(CC_single=345, CC_double=611):
    # The bond energy for the sigma bond is just CC_single
    Pi_bond = CC_double - CC_single  # Bond energy in KJ/mol
    return print('Pi bond energy = ', str(Pi_bond), ' KJ/mol')


bond_energies()


# %% Exercise 7
filename = 'Standard.csv'

filename_7a = filename.rstrip('.cvs')  # Solution 7a

print(filename_7a, '\nCould not figure out how to solve 7b')


# %% Exercise 8
DNA = 'ATTCGCCGCTTA'  # DNA sequence


def boolean_logic(DNA): # Weird Boolean Logic
    DNA_list = list(DNA)
    print(bool(DNA_list))
    DNA_list.reverse()
    print(bool(DNA_list.reverse))
    

boolean_logic(DNA)


# %% Exercise 9
Li, C, Na = 3, 6, 11  # Atomic Numbers


def atomic_number_logic(Li, C, Na):
    if bool(Li > C) is False:
        print('Li is not greater than C')
    if bool(Na <= C) is False:
        print('Na is not less than C')
    if bool(Li > C or Na > C) is True:
        print('Na is greater than C, but Li is not')
    if bool(C > Li and Na > Li) is True:
        print('C and Na are both greater than Li')


atomic_number_logic(Li, C, Na)


# %% Exercise 10
Molecules = ['HCl', 'NaOH', 'KCl', 'Ca(OH)2', 'KOH', 'HNO3',
             'Na2SO4', 'KNO3', 'Mg(OH)2', 'HCO2H', 'NaBr']


def acidity(Molecules):
    for i in range(len(Molecules)):
        if Molecules[i].startswith('H') is True:
            print('Acidic Molecule:', Molecules[i])
        elif Molecules[i].__contains__('OH'):
            print('Basic Molecule:', Molecules[i])
        else:
            print('Neurtral Molecule:', Molecules[i])


acidity(Molecules)


# %% Exercise 11
def ions(Protons, Electrons):
    if int(Protons) == int(Electrons):
        print('Neutral')
    if int(Protons) > int(Electrons):
        print('Cation')
    if int(Protons) < int(Electrons):
        print('Anion')


ions(10, 10)


# %% Exercise 12
List = list(range(18, 90, 2))  # Problem 12 List

print(List)
List.reverse()  # Problem 12a
print(List)
List.remove(18)  # Problem 12b
print(List)
List.append(16)  # Problem 12c
print(List)


# %% Exercise 13
Tuple = tuple(range(18, 320, 2))  # Problem 13 Tuple

print(Tuple)  # Problem 13a
print('You cannot reverse, remove, or append a tuple')  # Problem 13b


# %% Exercise 14
import random 

nums = [random.randint(0, 20) for x in range(10)]  # Problem 14 random list

if nums.__contains__(7) is True:  # Solution to problem 14
    print(nums, '\nThe list contains 7')
else:
    print(nums, '\nThis list does not contain 7')


# %% Exercise 15
def string_manipulation():
    Var = 'I LOVE COOKIES'
    print(Var)
    print(Var.lower().split(sep=None))  # Solution 15a
    Var_2 = sorted(Var.lower().split(sep=None))  # Solution 15b
    print(Var_2)


string_manipulation()


# %% Exercise 16
def appending():
    double = []
    for i in range(10):
        double.append(i)
        double.append(i)
    return double


appending()


# %% Exercise 17
def print_func():
    Ideal_variable = 'PV=nRT ' * 20
    return print(Ideal_variable)


print_func()


# %% Exercise 18
def mass_division():
    Mass = 1000  # Mass in grams
    for i in range(6):
        print(str(Mass), ' g')
        Mass = Mass / 2


mass_division()


# %% Exercise 19
def carbon_half_life():
    C_Mass = 500  # Mass in grams
    C_halflife = 0  # Half-life in years
    while C_Mass >= 10.00:
        C_Mass = C_Mass / 2  # One Half-life
        C_halflife = C_halflife + 1
        Years = 30.2 * C_halflife
    return print(str(Years), 'years until only 7.8125 g of Carbon 137 remains')


carbon_half_life()


# %% Exercise 20
'''To safe guard against while loops, input a break clause or an instance when the
while loop statement becomes false which will break the loop.'''


# %% Exercise 21
import math

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


# %% Exercise 22
import numpy as np

file = np.genfromtxt('Test.csv', delimiter=',', skip_header=1)
Time = []
Conc = []
for columns in file:
    Time.append(columns[0])
    Conc.append(columns[1])
print(str(Time), '\n', str(Conc))


# %% Exercise 23
def docstring_1(V, n, R, T):
    '''
    This function solves the Ideal gas law equation for pressure
    using a positional argument. The order is Volume, Moles, R
    constant, and Temperature
    '''
    P = (n * R * T) / V
    return P


def docstring_2(V=1, n=1, R=0.08206, T=273.15):
    '''
    Docstring BABYYYYYYYY!!!!!
    '''
    print('Solution 23b: ')
    P = (n * R * T) / V
    return P


print(docstring_1.__doc__, '\n', docstring_1(1, 1, 0.008206, 273.15), '\n\n', 
      docstring_2.__doc__, '\n', docstring_2())


# %% Exercise 24
def rate(A0, k=1.0, n=1):
    '''(concentration(M), k = 1.0, n = 1) → rate (M/s)
    Takes in the concentration of A (M), the rate constant (k),
    and the order (n) and returns the rate (M/s)
    '''
    Rate = k * (A0 ** n)
    return Rate


rate(2)


# %% Exercise 25
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

    return Opposite_strand


DNA_strands('ATGGC')
