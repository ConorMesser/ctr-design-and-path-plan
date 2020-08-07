from ctrdapp.model.model_factory import create_model
from ctrdapp.solve.visualize_utils import visualize_curve
from .setup_script import setup_script


def main():
    collision_detector, objects_file, heuristic_factory, this_q, configuration, output_path = setup_script()

    this_model = create_model(configuration, this_q)

    all_curves = [this_model.solve_g(indices=[50], thetas=[0], full=False)]

    for i in range(1, 50):
        g_out, eta_out, new_insert_indices, true_insertions, ftl_out = \
            this_model.solve_integrate([0], [1], [0], [i], all_curves[i - 1])

    # Small insertion w/ rotation
    # g_out, eta_out, new_insert_indices, true_insertions, ftl_out = \
    #     this_model.solve_integrate([-0.1], [1], [0], [50], all_curves[0])

    # Small insertion
    # g_out, eta_out, new_insert_indices, true_insertions, ftl_out = \
    #     this_model.solve_integrate([0], [1], [0], [50], all_curves[0])
        all_curves.append(g_out)

        heuristic = heuristic_factory.create(follow_the_leader=ftl_out)

    visualize_curve(all_curves, objects_file, configuration.get("tube_number"), configuration.get("tube_radius"),
                    output_path, "FTL_linear_insertion_animated")


if __name__ == "__main__":
    main()
