"""Strain base functions, defining tube shapes."""

import numpy as np
from math import cos, sin


def constant_strain(x, dof):
    """Constant-arc strain base.

    Parameters
    ----------
    x : float
        position along tube
    dof : int
        desired number of degrees of freedom

    Returns
    -------
    np.ndarray
        base at given x position
    """
    base = np.zeros([6, dof])
    base[1, 0] = 1  # constant y-bending
    return base


def linear_strain(x, dof):
    """Linear strain base.

    If two degrees of freedom are used, y-bending is constant
    and z-bending is linearly increasing; q associates with
    [y_constant, z_linear]. If three dof are used, both y and z
    are linear; q: [y_constant, y_linear, z_linear].

    Parameters
    ----------
    x : float
        position along tube
    dof : int
        desired number of degrees of freedom

    Returns
    -------
    np.ndarray
        base at given x position
    """
    base = np.zeros([6, dof])
    base[1, 0] = 1  # initial y-bending
    if dof > 2:
        base[1, 1] = x  # linear y-bending term
    base[2, dof-1] = x  # linear z-bending term
    return base


def quadratic_strain(x, dof):
    """Quadratic strain base.

    If two degrees of freedom are used, y-bending is constant
    and z-bending is quadratically increasing; q associates with
    [y_constant, z_quad]. If three dof are used, both y and z
    are quadratic; q: [y_constant, y_quad, z_quad].

    Parameters
    ----------
    x : float
        position along tube
    dof : int
        desired number of degrees of freedom

    Returns
    -------
    np.ndarray
        base at given x position
    """
    base = np.zeros([6, dof])
    base[1, 0] = 1  # initial y-bending
    if dof > 2:
        base[1, 1] = x**2  # quadratic y-bending term
    base[2, dof-1] = x**2  # quadratic z-bending term
    return base


def pure_helix_strain(x, dof):
    """Sinusoidal helix strain base.

    Gives a helical (non-constant) base based on sinusoidal
    functions with a period equal to 10; this period can
    only be modified in the source code.

    Parameters
    ----------
    x : float
        position along tube
    dof : int
        desired number of degrees of freedom

    Returns
    -------
    np.ndarray
        base at given x position
    """
    base = np.zeros([6, dof])
    base[1, 0] = sin(x/10)  # y-bending
    base[2, 1] = cos(x/10)  # z-bending
    return base


def linear_helix_strain(x, dof):
    """Sinusoidal helix strain with linear growth.

    Gives a helical (non-constant) base based on sinusoidal
    functions with a period equal to 10 (modifiable in source),
    increasing linearly in the z-bending.

    Parameters
    ----------
    x : float
        position along tube
    dof : int
        desired number of degrees of freedom

    Returns
    -------
    np.ndarray
        base at given x position
    """
    base = np.zeros([6, dof])
    base[1, 0] = sin(x/10)  # y-bending
    base[2, 1] = x * cos(x/10)  # z-bending
    return base


def torsion_helix_strain(x, dof):
    """Torsional helix strain base.

    Gives a constant helical base generated by a torsion value.
    Q associates with [torsion, y_constant, z_constant]

    Parameters
    ----------
    x : float
        position along tube
    dof : int
        desired number of degrees of freedom

    Returns
    -------
    np.ndarray
        base at given x position
    """
    base = np.zeros([6, dof])
    base[0, 0] = 1  # torsion
    base[1, 1] = 1  # y-bending
    base[2, 2] = 1  # z-bending
    return base


def get_strains(names, q_dof):
    """Helper function to return strain bases from given names.

    Parameters
    ----------
    names : list[str]
        list of base names
    q_dof : list[int]
        degrees of freedom for each tube

    Returns
    -------
    list[function]
        list of the desired strain base functions

    Raises
    ------
    ValueError
        If the input q_dof doesn't match the base dof options
    """
    strain_functions = []
    for n, this_dof in zip(names, q_dof):
        check_qdof(n, this_dof)
        if n == 'helix':
            strain_functions.append(linear_helix_strain)
        elif n == 'pure_helix':
            strain_functions.append(pure_helix_strain)
        elif n == 'torsion_helix':
            strain_functions.append(torsion_helix_strain)
        elif n == 'quadratic':
            strain_functions.append(quadratic_strain)
        elif n == 'linear':
            strain_functions.append(linear_strain)
        elif n == 'constant':
            strain_functions.append(constant_strain)
        else:
            print(f'{n} is not a defined strain base.')
    return strain_functions


def max_from_base(base, q_max, length, q_dof):
    """Calculates the max allowable q's for given parameters.

    Parameters
    ----------
    base : str
        base type
    q_max : float
        max curvature, based on tube material properties
    length : float or int
        length of tube
    q_dof : int
        degrees of freedom for this base

    Returns
    -------
    list[float]
        maximum q's allowed based on given bases and parameters

    Raises
    ------
    ValueError
        If the input q_dof doesn't match the base dof options
    """
    check_qdof(base, q_dof)
    tube_array = []
    if base == 'linear' or base == 'quadratic':
        if base == 'linear':
            reduced_max = q_max / length
        else:
            reduced_max = q_max / length**2
        if q_dof >= 3:
            tube_array = [q_max, -reduced_max]
        elif q_dof == 2:
            tube_array = [q_max]
        tube_array.append(reduced_max)
    else:
        tube_array = [q_max] * q_dof
    return tube_array


def check_qdof(base, q_dof):
    """Raises an error if the desired q_dof doesn't match the base.

    Parameters
    ----------
    base : str
        base type name
    q_dof : int
        number of desired degrees of freedom

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the input q_dof doesn't match the base dof options
    """
    if base == 'helix' or base == 'pure_helix':
        if q_dof != 2:
            raise ValueError(f'{base} should have 2 degrees of freedom, not {q_dof}.')
    elif base == 'quadratic' or base == 'linear':
        if q_dof < 2 or q_dof > 3:
            raise ValueError(f'{base} should have 2 or 3 degrees of freedom, not {q_dof}.')
    elif base == 'constant':
        if q_dof != 1:
            raise ValueError(f'{base} should have 1 degrees of freedom, not {q_dof}.')
    elif base == 'torsion_helix':
        if q_dof != 3:
            raise ValueError(f'{base} should have 1 degrees of freedom, not {q_dof}.')
    else:
        print(f'{base} is not a defined strain base.')
