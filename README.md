# CTR DaPP
**_Concentric Tube Robot Design and Path Planning_** is an optimization algorithm for the design of concentric tube robots based on a flexible path planner.

The CTR DaPP library was developed to allow for non-constant strain initial curvatures of concentric tube robots. 
The library is made up of multiple packages, each allowing for easy extension:
* **MODEL**: The mechanics of the robot, primarily offering the SE(3) g-curves for a robot given insertion and rotation values
  * Initial models follow Dr. Renda's strain-based static and kinematic models
* **COLLISION**: Provides collision queries between the model output and the environment (obstacles and goal)
  * The collision detection package utilizes the Fast Collision Library (python-fcl)
* **SOLVER**: The path planner, used to explore the environment and find a path to the goal
  * Initial path planners utilize the RRT (randomly-exploring rapid tree) sampling-based framework
* **HEURISTIC**: Defines the cost of a configuration (and path) of a robot
  * Initial heuristics use distance to goal, distance from obstacles, follow the leader path behavior
* **OPTIMIZE**: The algorithm used to optimize the design parameters (q curvature values) for a given environment
  * Initial optimizer uses nelder-mead, requiring only an initial simplex and no derivative information
  
Configuration files allow for specifying the environment and which package types to use (RRT vs. RRT*, static vs. kinematic, etc.).
A few pre-written scripts also allow for easy interaction with the API.


## Usage

```python
from ctrdapp.scripts.full_optimize import full_optimize

full_optimize()  # runs full optimization algorithm
```
```commandline
Enter name of configuration file (with .yaml extension):
> config.yaml
Enter initial guess for the q; should be 5 numbers long, each separated by a comma. (Tubes have these degrees of freedom: [2, 3]).
> 0.02, 0.001, 0.05, 0.0001, 0.002
```
Runs the algorithm with the specified configuration.
Outputs include the optimization process (q's and costs), the optimization summary, and the best design path planner results (full tree, solution path, and path movie).
The outputs are saved in the directory specified in the configuration file.


## Installation

Install the library from source. Python dependencies are defined in the Pipfile, except for the pympnn package which can be installed here.

## License
[MIT](https://choosealicense.com/licenses/mit/)