import unittest
import fcl
import pathlib
import numpy as np

from ctrdapp.collision.collision_checker import CollisionChecker
from ctrdapp.collision.init_collision import parse_mesh, add_obstacles


class TestCollision(unittest.TestCase):
    def setUp(self) -> None:
        path = pathlib.Path().absolute() / "configuration"
        basic_file = path / "init_objects.json"
        self.basic_checker = CollisionChecker(basic_file)
        box_file = path / "init_box_mesh.json"
        self.box_checker = CollisionChecker(box_file)

        # the double brackets represent a list for each tube (one tube in this case)
        self.curve1 = [[se3_generator([0, 0, 30]), se3_generator([0, 0, -5])]]
        self.curve2 = [[se3_generator([0, 0, 0]), se3_generator([0, 0, 3]),
                       se3_generator([0, 2, 3])]]
        self.curve3 = [[se3_generator([0, 0, 0]), se3_generator([18.99, 0, 0])]]

    def test_build_tube(self):
        self.assertEqual(self.basic_checker._build_tube(
            [[se3_generator([1.0, 1.0, 1.0])]], [5]), [])

        cyl1 = fcl.Cylinder(10, 2)
        trans1 = fcl.Transform([0, 1, 0, 0])
        tube1 = [fcl.CollisionObject(cyl1, trans1)]
        build_tube1 = self.basic_checker._build_tube(self.curve1, [2])

        self.assertEqual(build_tube1[0].getRotation().all(),
                         tube1[0].getRotation().all())
        self.assertEqual(build_tube1[0].getTranslation().all(),
                         tube1[0].getTranslation().all())
        self.assertEqual(len(build_tube1), len(tube1))

        cyl2a = fcl.Cylinder(3, 1)
        trans2a = fcl.Transform([0, 0, 1.5])
        cyl2b = fcl.Cylinder(2, 1)
        trans2b = fcl.Transform([0.52532198881773, -0.850903524534118, 0, 0],
                                [0, 1, 3])
        tube2 = [fcl.CollisionObject(cyl2a, trans2a),
                 fcl.CollisionObject(cyl2b, trans2b)]
        build_tube2 = self.basic_checker._build_tube(self.curve2, [1])

        self.assertEqual(build_tube2[0].getRotation().all(),
                         tube2[0].getRotation().all())
        self.assertEqual(build_tube2[0].getTranslation().all(),
                         tube2[0].getTranslation().all())
        self.assertEqual(build_tube2[1].getRotation().all(),
                         tube2[1].getRotation().all())
        self.assertEqual(build_tube2[1].getTranslation().all(),
                         tube2[1].getTranslation().all())
        self.assertEqual(len(build_tube2), len(tube2))

    def test_check_collision(self):  # todo write better tests
        res = self.box_checker.check_collision(self.curve3, [1])
        self.assertEqual(res[0], -1)
        self.assertAlmostEqual(res[1], 0.01, 6)


def se3_generator(point):
    se3 = np.eye(4)
    se3[0:3, 3] = point
    return se3


class TestInitCollision(unittest.TestCase):

    def test_parse_mesh(self):
        path = pathlib.Path().absolute()
        file = path.parent / "configuration" / "in_simple.STL"
        verts, tris = parse_mesh(str(file))

    def test_cylinder_placement(self):
        path = pathlib.Path().absolute()
        file = path.parent / "test" / "configuration" / "init_cylinder_test.json"
        obstacles = add_obstacles(file)

        for i in range(3):
            for j in [-1, 1]:
                request = fcl.CollisionRequest()
                result = fcl.CollisionResult()
                arr = np.array([0, 0, 0])
                arr[i] = 5 * j
                sph = fcl.CollisionObject(fcl.Sphere(1), fcl.Transform(arr))

                # obstacles are aligned with x (0), y (1), and z (2) axes
                ret = fcl.collide(obstacles[i], sph, request, result)
                self.assertTrue(ret > 0)


if __name__ == '__main__':
    unittest.main()
