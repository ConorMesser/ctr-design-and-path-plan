import fcl
from stl import mesh
import json
import numpy as np


def add_goal(init_objects_file):
    """Returns the hard-coded CollisionObject

    In the current formulation, only one object is passed. If multiple goal
    objects are defined in the json file, the whole list should be passed."""

    goals = parse_json('goal', init_objects_file)

    return goals[0]


def add_obstacles(init_objects_file):
    """Returns the hard-coded CollisionObjects in a list"""

    return parse_json('obstacles', init_objects_file)


def parse_mesh(filename):
    """Extracts vertices from given stl file

    STL file will most likely be exported in mm, however no unit information is
    given in the file. Additionally, stl files can be exported wrt the original
    coordinate system or shifted so all coordinates are positive. Regardless,
    care will have to be given aligning other objects to the meshes"""
    this_mesh = mesh.Mesh.from_file(filename)
    vertices = []
    tris = []
    for i in range(len(this_mesh.points)):
        vertices.append(this_mesh.points[i][0:3])
        vertices.append(this_mesh.points[i][3:6])
        vertices.append(this_mesh.points[i][6:9])
        tris.append([i*3, i*3 + 1, i*3 + 2])

    return vertices, tris


def parse_json(object_type, init_objects_file):
    f = init_objects_file.open()
    full_json = json.load(f)
    f.close()

    list_objects = full_json.get(object_type)

    collision_objects = []
    for o in list_objects:
        if o.get('mesh'):
            m = fcl.BVHModel()
            path = init_objects_file.parent
            file = path / o.get('filename')
            vertices, tris = parse_mesh(str(file))
            m.beginModel(len(vertices), len(tris))
            m.addSubModel(vertices, tris)
            m.endModel()
            this_transform = fcl.Transform(o.get('transform'))
            obj = fcl.CollisionObject(m, this_transform)
        else:
            if o.get('shape') == 'Sphere':
                this_obj = fcl.Sphere(o.get('radius'))
                this_transform = fcl.Transform(o.get('transform'))
            elif o.get('shape') == 'Cylinder':
                this_obj = fcl.Cylinder(o.get('radius'), o.get('length'))

                direction = o.get('direction')
                rot = rotate_align(np.asarray(direction))
                this_transform = fcl.Transform(rot, o.get('transform'))
            elif o.get('shape') == 'Box':
                this_obj = fcl.Box(o.get('x_length'), o.get('y_length'), o.get('z_length'))
                this_transform = fcl.Transform(o.get('transform'))
            else:
                raise NotImplementedError(f'Shape type {o.get("shape")} not supported.')

            obj = fcl.CollisionObject(this_obj, this_transform)

        collision_objects.append(obj)

    return collision_objects


def rotate_align(vector_to, vector_from=np.array([0, 0, 1])):
    """Returns a rotation matrix to align the given vectors.

    Adapted from Kevin Moran's article on noacos derivation which can be found:
    https://gist.github.com/kevinmoran/b45980723e53edeb8a5a43c49f134724

    Parameters
    ----------
    vector_to : numpy 3x1 array
        The goal vector
    vector_from : numpy 3x1 array
        The initial vector, with a default of the z-axis unit vector

    Returns
    -------
    3x3 numpy array : the rotation matrix to transform vector_from to vector_to
    """
    axis = np.cross(vector_from, vector_to)
    cos_a = np.dot(vector_from, vector_to)
    k = 1 / (1 + cos_a)

    matrix = np.array([[(axis[0] * axis[0] * k) + cos_a,
                        (axis[1] * axis[0] * k) - axis[2],
                        (axis[2] * axis[0] * k) + axis[1]],
                       [(axis[0] * axis[1] * k) + axis[2],
                        (axis[1] * axis[1] * k) + cos_a,
                        (axis[2] * axis[1] * k) - axis[0]],
                       [(axis[0] * axis[2] * k) - axis[1],
                        (axis[1] * axis[2] * k) + axis[0],
                        (axis[2] * axis[2] * k) + cos_a]])

    return matrix
