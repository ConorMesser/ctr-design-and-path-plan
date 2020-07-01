import pathlib
from time import strftime
from math import pi

from ctrdapp.config.parse_config import parse_config
from ctrdapp.model.kinematic import create_model
from ctrdapp.solve.visualize_utils import visualize_curve
from ctrdapp.heuristic.heuristic_factory import create_heuristic_factory


def main():  # todo make a main script to get general config, set things up

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

    heuristic_factory = create_heuristic_factory(configuration,
                                                 dictionaries.get("heuristic"))

    # create model
    q = [0.03, 0.002]  # [pi/100]  # max q for radius 0.9 tube is 0.0555
    this_model = create_model(configuration, q)

    all_curves = [this_model.solve_g(indices=[50], thetas=[0], full=False)]

    for i in range(1, 50):
        g_out, eta_out, new_insert_indices, true_insertions, ftl_out = \
            this_model.solve_integrate([0], [1], [0], [i], all_curves[i-1])

    # g_out, eta_out, new_insert_indices, true_insertions, ftl_out = \
    #     this_model.solve_integrate([-0.1], [1], [0], [50], all_curves[0])

    # g_out, eta_out, new_insert_indices, true_insertions, ftl_out = \
    #     this_model.solve_integrate([0], [1], [0], [50], all_curves[0])
        all_curves.append(g_out)

        heuristic = heuristic_factory.create(follow_the_leader=ftl_out)

    visualize_curve(all_curves, objects_file, configuration.get("tube_number"), configuration.get("tube_radius"),
                    output_path, "FTL_linear_insertion_animated")


if __name__ == "__main__":
    main()
