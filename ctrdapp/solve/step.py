from numpy.linalg import norm
from numpy import array, count_nonzero
from random import random
from math import floor


def step(prev: [float], to: [float], max_bound: float) -> [float]:
    """Returns a point between prev and to, at most max_bound distance from prev

    If the distance between the points, given by the norm is less than the
    max_bound, the to list is returned.

    Parameters
    ----------
    prev : list of float
        the initial/from point
    to : list of float
        the goal/towards point
    max_bound : float
        the maximum distance to travel from prev

    Returns
    -------
    [float] : point with same dimension as prev and to

    Raises
    ------
    ValueError
        If the given points are of different dimensions
    """
    if len(prev) != len(to):
        raise ValueError("Given vectors are not the same dimension")
    vec = []
    for p, t in zip(prev, to):
        vec.append(t - p)

    # if vector magnitude > max, multiply unit vector by max
    vec_mag = norm(vec)
    final = []
    if vec_mag > max_bound:
        for v, p in zip(vec, prev):
            final.append(v*max_bound/vec_mag + p)
        return final
    else:
        return to


def get_single_tube_value(insert_rand, insert_neighbor, neighbor_parent, probability, random_num):
    """Returns an insertion vector with one changed value from insert_neighbor.

    Compares the insert_neighbor vector to its parent's vector to find the tube
    that is currently being inserted. The given probability is for selecting the
    same tube as previous and selecting another tube is
    (1 - probability) / tube_num. Output is the insert_neighbor vector with the
    insert_rand value for the randomly chosen tube (based on the probabilities).

    Parameters
    ----------
    insert_rand : list of float
        The new insertion array
    insert_neighbor : list of float
        The nearest neighbor of the insert_rand array
    neighbor_parent : list of float
        The parent of the insert_neighbor array
    probability : float
        The probability of selecting the same tube as previously changed
    random_num : float
        Random [0, 1) for tube selection (given as parameter for ease of testing)

    Returns
    -------
    list of float : the final insertion array
        combination of insert_neighbor and insert_rand
    int : the tube number with a delta insertion
    """
    tube_num = len(insert_neighbor)
    neighbor_diff = array([n - p for n, p in zip(insert_neighbor, neighbor_parent)])
    if count_nonzero(neighbor_diff) > 1:
        raise ValueError("Multiple tubes inserted in neighbor node.")
    elif count_nonzero(neighbor_diff) == 0:
        last_insertion_tube_num = 0  # default to first tube (esp. for initial insertion)
    else:
        last_insertion_tube_num = neighbor_diff.nonzero()[0][0]
    possible_tubes = list(range(tube_num))
    possible_tubes.pop(last_insertion_tube_num)

    shifted_r = random_num - probability
    if shifted_r < 0:
        this_insertion_tube_num = last_insertion_tube_num
    else:
        probability_other = (1 - probability) / (tube_num - 1)
        this_insertion_tube_num = possible_tubes[floor(shifted_r/probability_other)]

    new_insert = []
    for i in range(tube_num):
        if i == this_insertion_tube_num:
            new_insert.append(insert_rand[i])
        else:
            new_insert.append(insert_neighbor[i])
    return new_insert, this_insertion_tube_num






