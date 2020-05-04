import numpy as np
from math import cos, sin


def constant_strain(x, dof):
    if dof > 1:
        print(f'Only 1 degrees of freedom allowed for constant strain.')
    base = np.zeros([6, dof])
    base[1, 0] = 1  # constant y-bending
    return base


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


def pure_helix_strain(x, dof):
    if dof > 2:
        print(f'Only 2 degrees of freedom are being used out of {dof}.')
    base = np.zeros([6, dof])
    base[1, 0] = sin(x/15)  # y-bending
    base[2, 1] = cos(x/15)  # z-bending
    return base


def linear_helix_strain(x, dof):
    if dof > 2:
        print(f'Only 2 degrees of freedom are being used out of {dof}.')
    base = np.zeros([6, dof])
    base[1, 0] = sin(x/15)  # y-bending
    base[2, 1] = x * cos(x/15)  # z-bending
    return base


def get_strains(names):
    strain_functions = []
    for n in names:
        if n == 'helix':
            strain_functions.append(linear_helix_strain)
        elif n == 'pure_helix':
            strain_functions.append(pure_helix_strain)
        elif n == 'quadratic':
            strain_functions.append(quadratic_strain)
        elif n == 'linear':
            strain_functions.append(linear_strain)
        elif n == 'constant':
            strain_functions.append(constant_strain)
        else:
            print(f'{n} is not a defined strain base.')
    return strain_functions
