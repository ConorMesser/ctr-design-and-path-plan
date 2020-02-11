import numpy as np
from math import cos, sin


def linear_strain(x, dof):
    if dof > 3:
        print(f'Only 3 degrees of freedom are being used out of {dof}.')
    base = np.zeros([6, dof])
    base[1, 0] = 1  # initial y-bending
    if dof > 2:
        base[1, 1] = x  # linear y-bending term
    base[2, dof-1] = x  # linear z-bending term
    return base


def quadratic_strain(x, dof):
    if dof > 3:
        print(f'Only 3 degrees of freedom are being used out of {dof}.')
    base = np.zeros([6, dof])
    base[1, 0] = 1  # initial y-bending
    if dof > 2:
        base[1, 1] = x**2  # quadratic y-bending term
    base[2, dof-1] = x**2  # quadratic z-bending term
    return base


def helix_strain(x, dof):
    if dof > 2:
        print(f'Only 2 degrees of freedom are being used out of {dof}.')
    base = np.zeros([6, dof])
    base[1, 0] = sin(x)  # y-bending
    base[2, 1] = cos(x)  # z-bending
    return base


def get_strains(names):
    strain_functions = []
    for n in names:
        if n == 'helix':
            strain_functions.append(helix_strain)
        elif n == 'quadratic':
            strain_functions.append(quadratic_strain)
        elif n == 'linear':
            strain_functions.append(linear_strain)
        else:
            print(f'{n} is not a defined strain base.')
    return strain_functions
