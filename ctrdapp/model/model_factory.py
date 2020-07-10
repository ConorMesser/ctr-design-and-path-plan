"""Factory method to create a model."""

import numpy as np

from .model import Model
from .kinematic import Kinematic
from .static import Static, get_basis
from .strain_bases import get_strains


def create_model(configuration, q) -> Model:
    """Creates a model based on the given configuration.

    Parameters
    ----------
    configuration : dict
        configuration dictionary
    q : list[float] or list[list[float]]
        bending parameters for each tube

    Returns
    -------
    Model
        model object with parameters given
        by the configuration dictionary and given q
    """
    name = configuration.get("model_type")
    tube_num = configuration.get('tube_number')
    names = configuration.get('strain_bases')

    config_dof = configuration.get('q_dof')
    if isinstance(config_dof, int):  # q_dof given as int (each tube has same dof)
        output_q_dof = [config_dof] * tube_num
        print(f'Each base has same degrees of freedom: {config_dof}.')
    else:  # q_dof given as list
        output_q_dof = config_dof
    strain_bases = get_strains(names, output_q_dof)

    # q given as single list
    if len(q) == sum(output_q_dof) and not isinstance(q[0], list):
        q_list = []
        n = dof_sum = 0
        while n < tube_num:
            q_list.append(np.asarray(q[dof_sum:dof_sum + output_q_dof[n]]))
            dof_sum += output_q_dof[n]
            n += 1
        output_q = q_list
    elif len(q) == tube_num:  # q given as nested-list, one for each tube
        output_q = [np.asarray(this_q) for this_q in q]
    else:
        raise ValueError(f"Given q of {q} is not the correct size.")

    config_lengths = configuration.get('tube_lengths')
    if isinstance(config_lengths, int):  # given as int
        lengths = [config_lengths] * tube_num
        print(f'Each tube length is the same: {config_lengths}.')
    else:  # given as list
        lengths = config_lengths

    if name == "kinematic":
        return Kinematic(tube_num, output_q, output_q_dof, lengths, configuration.get('delta_x'), strain_bases)
    elif name == "static":
        # radii
        radii = configuration.get('tube_radius')

        # ndof
        if tube_num == 3:

            dof_11 = len(get_basis(1, 1, 1, 1, strain_bases, output_q_dof, tube_num)[0, :])
            dof_21 = len(get_basis(1, 1, 2, 1, strain_bases, output_q_dof, tube_num)[0, :])
            dof_31 = len(get_basis(1, 1, 3, 1, strain_bases, output_q_dof, tube_num)[0, :])
        elif tube_num == 2:
            dof_11 = dof_21 = dof_31 = 0
        else:
            raise ValueError(f'The static model does not support {tube_num} tubes.')
        dof_22 = len(get_basis(1, 1, 2, 2, strain_bases, output_q_dof, tube_num)[0, :])
        dof_32 = len(get_basis(1, 1, 3, 2, strain_bases, output_q_dof, tube_num)[0, :])
        dof_33 = len(get_basis(1, 1, 3, 3, strain_bases, output_q_dof, tube_num)[0, :])
        ndof = {(1, 1): dof_11, (2, 1): dof_21, (3, 1): dof_31, (2, 2): dof_22, (3, 2): dof_32, (3, 3): dof_33}

        return Static(tube_num, output_q, output_q_dof, lengths, configuration.get('delta_x'), strain_bases, ndof, radii)
    else:
        raise UserWarning(f"{name} is not a defined model. "
                          f"Change config file.")
