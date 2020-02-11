from abc import ABC, abstractmethod


class Solver(ABC):
    """Obtains a solution to the given path planning problem.

    The abstract class can be instantiated as any planner (such as RRT and its
    derivatives, PRM, etc.). The public methods get_cost and get_path return the
    minimum cost found and the path (configurations from root to node)
    associated with that minimum cost. The methods vis_full_search and
    vis_best_solution allow 3D plotting of the path for the best solution
    (minimum cost) and the entire search space.
    """

    def __init__(self, model, heuristic_factory, collision_detector, configuration):
        """
        Parameters
        ----------
        heuristic_factory : HeuristicFactory
            Heuristic object containing general parameters and method to create
            new Heuristic object
        collision_detector : CollisionChecker
            Contains obstacles, goal, and methods for collision queries
        configuration : dict
            Dictionary storing configuration variables
        """

        self.model = model
        self.tube_num = configuration.get("tube_number")
        """int : Number of tubes used in this algorithm run"""
        self.tube_rad = configuration.get("tube_radius")
        """List of float : Radii of the tubes"""
        self.heuristic_factory = heuristic_factory
        self.cd = collision_detector

    @abstractmethod
    def get_path(self, index):
        """Retrieves the data from the root to the given index.

        Parameters
        ----------
        index : int
            Index value for desired starting configuration node

        Returns
        -------
        list of list of 4x4 array : g values for each tube
        list of list of float : insertion values for each tube
        list of list of float : rotation values for each tube
        """
        pass

    @abstractmethod
    def get_best_cost(self):
        """Retrieves the minimum cost, with priority given to solutions.

        Returns
        -------
        float : cost of the optimal configuration node
        int : index of the optimal node
        """
        pass

    @abstractmethod
    def visualize_full_search(self):
        """Visualizes the entire search, including obstacles, goal, and tubes

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def visualize_best_solution(self, objects_file):
        """Visualizes path of best solution including obstacles, goal, and tubes

        Parameter
        ---------
        objects_file : Path
            A pathlib PosixPath giving the full path for obstacle/goal json file

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def visualize_from_index(self, index, objects_file):
        """Visualizes path from root to index with obstacles, goal, and tubes

        Parameter
        ---------
        index : int
            Index value for desired ending configuration node
        objects_file : Path
            A pathlib PosixPath giving the full path for obstacle/goal json file

        Returns
        -------
        None
        """
        pass
