import pathlib

from ctrdapp.config.parse_config import parse_config
from ctrdapp.model.model import create_model, save_g_positions
from ctrdapp.solve.visualize_utils import visualize_curve_single


def main():

    path = pathlib.Path().absolute()
    file = path / "configuration" / "config.yaml"
    configuration, dictionaries = parse_config(file)
    objects_file = path / "configuration" / configuration.get("collision_objects_filename")

    # create model
    q = [0.03, 0.0006]  # max q for radius 0.9 tube is 0.0555
    this_model = create_model(configuration, q)

    this_g = this_model.solve_g(indices=[0])
    save_g_positions(this_g, "single_helix_tube_smaller.txt")

    visualize_curve_single(this_g, objects_file, configuration.get("tube_number"), configuration.get("tube_radius"))


if __name__ == "__main__":
    main()
