from ctrdapp.scripts.setup_script import setup_script
from ctrdapp.model.model_factory import create_model
from ctrdapp.solve.solver_factory import create_solver


def solver_only():
    collision_detector, objects_file, heuristic_factory, desired_q, configuration, output_path = setup_script()

    this_model = create_model(configuration, desired_q)

    this_solver = create_solver(this_model, heuristic_factory, collision_detector, configuration)

    # call get_best_cost
    cost, best_ind = this_solver.get_best_cost()
    print(cost)

    # this_solver.visualize_best_solution(objects_file)
    this_solver.visualize_full_search(output_path, with_solution=True)
    this_solver.save_tree(output_path)
    this_solver.save_best_solution(output_path)
    this_solver.visualize_best_solution(objects_file, output_path)
    this_solver.visualize_best_solution_path(objects_file, output_path)


if __name__ == "__main__":
    solver_only()
