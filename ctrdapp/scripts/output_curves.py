import pathlib
from time import strftime

from ctrdapp.config.parse_config import parse_config
from ctrdapp.model.kinematic import create_model, save_g_positions
from ctrdapp.solve.visualize_utils import visualize_curve_single


def main():

    path = pathlib.Path().absolute()
    file = path / "configuration" / "config_trial.yaml"
    configuration, dictionaries = parse_config(file)

    run_name = configuration.get("run_identifier")
    output_path = path / "output" / run_name
    try:
        output_path.mkdir(parents=True)
    except FileExistsError:
        current_time = strftime("%H%M")
        current_date = strftime("%m%d%y")
        new_name = f"{run_name}_{current_date}_{current_time}"
        output_path = path / "output" / new_name
        output_path.mkdir(parents=True)

    objects_file = path / "configuration" / configuration.get("collision_objects_filename")

    # create model
    q = [0.03, 0.00002]  # max q for radius 0.9 tube is 0.0555
    this_model = create_model(configuration, q)

    this_g = this_model.solve_g(indices=[0], thetas=[0], full=True)
    g_filename = output_path / "g_curves.txt"
    save_g_positions(this_g, g_filename)

    visualize_curve_single(this_g, objects_file, configuration.get("tube_number"), configuration.get("tube_radius"),
                           output_path, "curve_visual")


if __name__ == "__main__":
    main()
