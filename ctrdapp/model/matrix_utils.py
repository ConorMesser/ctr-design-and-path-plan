import numpy as np
from numpy.linalg import matrix_power
from math import cos, sin


def big_adjoint(matrix_in):
    """Gets the adjoint matrix from an SE3 matrix

    Usage:
    hat(Adjoint of g * eta) = g * eta * g^-1
    """

    adjoint = np.zeros([6, 6])
    SO3 = np.asarray(matrix_in[0:3, 0:3])
    p_tilde = tilde(matrix_in[0:3, 3]) @ SO3

    adjoint[0:3, 0:3] = SO3
    adjoint[3:6, 0:3] = p_tilde
    adjoint[3:6, 3:6] = SO3
    return adjoint


def little_adjoint(screw_in):
    """Gets the adjoint matrix from a screw

    Usage:
    hat(adjoint of ksi_1 * ksi_2) = ksi_1_hat * ksi_2_hat - ksi_2_hat * ksi_1_hat
    """

    adjoint = np.zeros([6, 6])
    SO3 = tilde(screw_in[0:3])
    p_tilde = tilde(screw_in[3:6])
    adjoint[0:3, 0:3] = SO3
    adjoint[3:6, 0:3] = p_tilde
    adjoint[3:6, 3:6] = SO3
    return adjoint


def tilde(vector):
    """Makes the 3-vector into a 3x3 skew-symmetric matrix"""
    skew = np.zeros([3, 3])
    skew[0, 1] = -vector[2]
    skew[0, 2] = vector[1]
    skew[1, 0] = vector[2]
    skew[1, 2] = -vector[0]
    skew[2, 0] = -vector[1]
    skew[2, 1] = vector[0]
    return skew


def hat(screw):
    """Converts the screw into an se3 matrix"""
    se3 = np.zeros([4, 4])
    se3[0:3, 0:3] = tilde(screw[0:3])
    se3[0:3, 3] = screw[3:6]
    return se3


def exponential_map(theta, gamma_hat):
    """Transforms from se(3) to SE(3) by angle theta

    """
    if theta == 0:
        return np.eye(4) + gamma_hat
    else:
        return np.eye(4) + gamma_hat + \
               ((1-cos(theta)) / theta**2) * matrix_power(gamma_hat, 2) + \
               ((theta-sin(theta)) / theta**3) * matrix_power(gamma_hat, 3)


def t_exponential(h, theta, gamma):
    """Transforms from screw gamma to SE(3) by angle theta and pitch h"""
    adjoint_gamma = little_adjoint(gamma)  # gamma is a screw

    if theta == 0:
        return h * np.eye(6) + (h**2 / 2) * adjoint_gamma
    else:
        sin_exp = h*theta*sin(h*theta)
        cos_exp = h*theta*cos(h*theta)
        return h * np.eye(6) + \
               ((4 - 4*cos(h*theta) - sin_exp) / (2*theta**2)) * adjoint_gamma + \
               ((4*h*theta - 5*sin(h*theta) + cos_exp) / (2*theta**3)) * matrix_power(adjoint_gamma, 2) + \
               ((2 - 2*cos(h*theta) - sin_exp) / (2*theta**4)) * matrix_power(adjoint_gamma, 3) + \
               ((2*h*theta - 3*sin(h*theta) + cos_exp) / (2*theta**5)) * matrix_power(adjoint_gamma, 4)
