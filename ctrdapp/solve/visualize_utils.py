import json
import pyvista as pv
import numpy as np
import time


def parse_json(object_type, init_objects_file):
    f = init_objects_file.open()
    full_json = json.load(f)
    f.close()

    list_objects = full_json.get(object_type)

    collision_objects = []
    for o in list_objects:
        if o.get('mesh'):
            mesh_file = init_objects_file.parent / o.get('filename')
            mesh_type = mesh_file.name.split('.')[-1]
            this_obj = pv.read(mesh_file, file_format=mesh_type.lower())  # todo currently no transform for meshes (for visualization)
        elif o.get('shape') == 'Sphere':
            this_obj = pv.Sphere(radius=o.get('radius'),
                                 center=o.get('transform'))
        elif o.get('shape') == 'Cylinder':
            this_obj = pv.Cylinder(radius=o.get('radius'),
                                   height=o.get('length'),
                                   center=o.get('transform'),
                                   direction=o.get('direction'))
        elif o.get('shape') == 'Box':
            x_half = o.get('x_length') / 2
            y_half = o.get('y_length') / 2
            z_half = o.get('z_length') / 2
            center = o.get('transform')
            this_obj = pv.Box([center[0] - x_half, center[0] + x_half,
                              center[1] - y_half, center[1] + y_half,
                              center[2] - z_half, center[2] + z_half])
        else:
            raise NotImplementedError(f'Shape type {o.get("shape")} not supported.')

        collision_objects.append(this_obj)

    return collision_objects


def visualize_curve(curve, objects_file, tube_num, tube_rad, visualize_from_indices=None):
    plotter = pv.Plotter()
    plotter.open_movie("this_movie.mp4", framerate=3)  # todo

    add_objects(plotter, objects_file)

    plotter.show(auto_close=False, interactive_update=True)
    plotter.write_frame()

    # plot each tube from root to final (only need p values, not R)
    # - plots of tubes must cut off the tube prior to insertion
    #   in movie form or all in one picture?
    for i in range(len(curve)):
        this_time_step = curve[i]
        this_tube_actor = add_single_curve(plotter, this_time_step, tube_num, tube_rad, visualize_from_indices)
        plotter.write_frame()
        plotter.update()
        time.sleep(0.3)
        for a in this_tube_actor:
            plotter.remove_actor(a)

    plotter.close()


def visualize_curve_single(curve, objects_file, tube_num, tube_rad, visualize_from_indices=None):
    plotter = pv.Plotter()

    add_objects(plotter, objects_file)

    _ = add_single_curve(plotter, curve, tube_num, tube_rad, visualize_from_indices)

    plotter.show()


# todo remove visualize_from_indices
def add_single_curve(plotter, curve, tube_num, tube_rad, visualize_from_indices, color='w', invert_insert=False):
    if not visualize_from_indices:
        visualize_from_indices = [0] * tube_num  # plot each whole tube

    this_tube_actor = []
    for n in range(tube_num):
        this_tube = curve[n]
        this_radius = float(tube_rad[n])

        # get the insertion index
        if invert_insert:
            index = len(this_tube) - visualize_from_indices[n]  # check todo
        else:
            index = visualize_from_indices[n]
        # get the P values from the insertion index to the "tip", with given radius
        these_p = [g[0:3, 3] for g in this_tube[index:]]
        if len(these_p) == 1:
            continue
            # todo this_tube_visual = pv.Cylinder(center=these_p[1], direction=)

        else:
            this_tube_spline = pv.Spline(np.array(these_p), len(these_p) * 20)
            this_tube_spline["scalars"] = np.arange(this_tube_spline.n_points)
            this_tube_visual = this_tube_spline.tube(radius=this_radius)
            this_tube_actor.append(plotter.add_mesh(this_tube_visual, color=color))

    return this_tube_actor
    # as movie or picture todo


def add_objects(plotter, objects_file):
    # plot obstacle mesh (with transparency)
    obstacle_meshes = parse_json('obstacles', objects_file)
    for o_m in obstacle_meshes:
        plotter.add_mesh(o_m, opacity=0.5)

    # plot goal mesh (should only be one)
    goal_meshes = parse_json('goal', objects_file)
    for g_m in goal_meshes:
        plotter.add_mesh(g_m, color='b', opacity=0.3)

    # plot insertion plane
    plane = pv.Plane(direction=[1, 0, 0], i_size=30, j_size=30)
    plotter.add_mesh(plane, color='tan', opacity=0.4)


def visualize_tree(from_points, to_points, node_list):
    plotter = pv.Plotter()
    # todo add labels/color based on node indices

    plotter.add_axes_at_origin(xlabel='Tube 1', ylabel='Tube 2', zlabel='Tube 3')
    plotter.add_mesh(pv.Sphere(radius=0.25, center=from_points[0]))

    for ind, points in enumerate(zip(from_points, to_points)):
        p_from = points[0]
        p_to = points[1]
        plotter.add_mesh(pv.Sphere(radius=0.25, center=p_to))

        plotter.add_mesh(pv.Line(p_from, p_to))

    plotter.show()
