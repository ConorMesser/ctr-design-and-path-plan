from ctrdapp.model.kinematic import save_g_positions
from ctrdapp.model.model_factory import create_model
from ctrdapp.solve.visualize_utils import visualize_curve_single
from ctrdapp.scripts.setup_script import setup_script

from math import pi
import numpy as np


def main():
    collision_detector, objects_file, heuristic_factory, this_q, configuration, output_path = setup_script()

    # create model
    this_model = create_model(configuration, this_q)

    try_q = np.array([-1.01338092e+01,  1.16295691e+00, -1.98104425e-02,  9.19414724e-05,
                      6.01577902e+00,  3.81962133e+00, -7.62825629e-02,  3.74874061e-04,
                      -1.05176911e+00,  1.02050308e-01, -1.68539548e-03,  7.70259132e-06,
                      2.69890376e-01,  6.16774319e-02, -1.54744257e-03,  9.12084355e-06,
                      3.06374117e+00, -6.28190146e-02, -3.65086842e-05,  2.66174270e-06,
                      2.65416918e-02, -4.47108927e-04,  1.85629186e-06, -7.49269565e-10,
                      -5.73181894e-02,  2.17707911e-03, -2.02683214e-05,  6.09485549e-08,
                      2.54604827e-02, -1.09619732e-03,  1.33477319e-05, -4.86996371e-08,
                      -2.14492224e-01,  2.81060637e-03, -1.10926421e-05,  1.30145562e-08,
                      3.00000000e-02])

    try_q = np.zeros(13)
    try_q[3] = try_q[6] = try_q[12] = 0.01

    # todo get input from user
    this_g = this_model.solve_g(indices=[0], thetas=[0], full=True)  # , initial_guess=try_q)  # , full=True) initial_guess=try_q)
    # g_filename = output_path / "g_curves.txt"
    # save_g_positions(this_g, g_filename)

    visualize_curve_single(this_g, objects_file, configuration.get("tube_number"),
                           configuration.get("tube_radius").get("outer"),
                           output_path, "curve_visual")


if __name__ == "__main__":
    main()
