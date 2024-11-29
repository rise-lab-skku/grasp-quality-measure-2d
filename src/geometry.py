from typing import Tuple, Union
import copy
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


class ConvexObject():
    def __init__(self, vertices: np.ndarray):
        """Construct a convex object from a set of vertices.

        Args:
            vertices:  A 2D numpy array of shape (n, 2) representing the vertices of the
              convex object.

        Raises:
            ValueError: If the vertices do not form a convex polygon.
        """
        # check vertices are 2D points
        if vertices.shape[1] != 2:
            raise ValueError("Vertices must be 2D points")

        # create convex hull from vertices
        convex_hull = ConvexHull(vertices)
        if len(convex_hull.vertices) != len(vertices):
            raise ValueError("Vertices must form a convex polygon")

        # set properties
        # For 2-D convex hulls, the vertices are in counterclockwise order. 
        self._vertices = convex_hull.points[convex_hull.vertices]
        self._area =  convex_hull.volume

    @property
    def vertices(self) -> np.ndarray:
        """Vertices of the convex object with shape (n, 2)."""
        return  self._vertices

    @property
    def area(self) -> float:
        """Area of the convex object."""
        return self._area

    @property
    def center_mass(self) -> np.ndarray:
        """Center of mass of the convex object."""
        if self._vertices.shape[1] != 2:
            raise ValueError("Center of mass calculation is only implemented for 2D polygons.")
        vertices = self._vertices        
        n = len(vertices)

        # Use the Shoelace formula to compute centroid
        signed_area = 0.0
        centroid_x = 0.0
        centroid_y = 0.0

        for i in range(n):
            x0, y0 = vertices[i]
            x1, y1 = vertices[(i + 1) % n]
            a = x0 * y1 - x1 * y0
            signed_area += a
            centroid_x += (x0 + x1) * a
            centroid_y += (y0 + y1) * a

        signed_area *= 0.5
        if signed_area == 0:
            raise ValueError("Degenerate polygon with zero area cannot\
                              have a well-defined centroid.")

        centroid_x /= (6.0 * signed_area)
        centroid_y /= (6.0 * signed_area)
        return np.array([centroid_x, centroid_y])

    @property
    def matplotlib_patch(self) -> Polygon:
        """Matplotlib patch object for the convex object."""
        return Polygon(self._vertices, closed=True)

    def apply_translation(self, translation: Tuple[float, float]):
        """Apply a translation to the convex object.
        
        Args:
            translation: A tuple (dx, dy) representing the translation to apply.
        """
        self._vertices += np.array(translation)

    def apply_rotation(self, angle: float):
        """Apply a rotation to the convex object.

        Args:
            angle: The angle in [radians] to rotate the convex object by.
        """
        rot_mat = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]])
        self._vertices = np.dot(self._vertices, rot_mat.T)

    def apply_transform(self, transform: np.ndarray):
        """Apply SE(2) transform to the convex object.

        Args:
            transform: A 3x3 numpy array representing the SE(2) transform to apply.
        """
        rot_mat = transform[0:2, 0:2]
        if not np.allclose(np.dot(rot_mat, rot_mat.T), np.eye(2)):
            raise ValueError("Rotation matrix must be orthogonal, but got \n{}".format(rot_mat))
        self._vertices = np.dot(self._vertices, rot_mat.T) + transform[0:2, 2]

    def apply_scaling(self, scaling: Union[float, Tuple[float, float]]):
        """Apply scaling to the convex object.

        Args:
            scaling: A float or tuple (sx, sy) representing the scaling to apply.
        """
        if isinstance(scaling, float):
            scaling = (scaling, scaling)
        self._vertices = (self._vertices - self.center_mass) * np.array(scaling) + self.center_mass

    def copy(self):
        """Return a deep copy of the convex object."""
        return copy.deepcopy(self)

    def smaple_surface(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample points on the surface of the convex object.

        Args:
            num_samples: The number of samples to take.

        Returns:
            A tuple (points, normals) where points is a numpy array of shape (num_samples, 2)
            representing the sampled points on the surface of the convex object, and normals is a
            numpy array of shape (num_samples, 2) representing the normals at each sampled point.
            The normals are outward pointing.
        """
        # get edge vectors and normals
        vertices = np.concatenate([self.vertices, self.vertices[0:1]], axis=0)
        edge_vectors = np.diff(vertices, axis=0)
        edge_normals = np.array([np.array([edge[1], -edge[0]]) for edge in edge_vectors])
        edge_normals /= np.linalg.norm(edge_normals, axis=1)[:, None]

        # sample points on the edges
        edge_length = np.linalg.norm(edge_vectors, axis=1)
        partition = np.cumsum(edge_length)

        # do random uniform sampling
        rand_samples = np.linspace(0, partition[-1], num_samples, endpoint=False)
        rand_samples += rand_samples[1] * np.random.rand(1) * 0.99
        rand_samples.sort()
        indices = np.searchsorted(partition, rand_samples)

        # get rondom points on the edges
        sample_points = np.zeros((num_samples, 2))
        sample_normals = np.zeros((num_samples, 2))
        for i in range(num_samples):
            t = (partition[indices[i]] - rand_samples[i]) / edge_length[indices[i]]
            sample_points[i] = vertices[indices[i]] * (1 - t) + vertices[indices[i] + 1] * t
            sample_normals[i] = edge_normals[indices[i]]
        return sample_points, sample_normals

    @staticmethod
    def create_regular_polygon(num_vertices: int, radius: float) -> 'ConvexObject':
        """Create a regular polygon with a given number of vertices and radius.
        
        Args:
            num_vertices: The number of vertices of the polygon.
            radius: The radius of the polygon.

        Returns:
            A `ConvexObject` representing the regular polygon.

        Raises:
            AssertionError: If the number of vertices is less than 3 or the radius is less than 0.
        """
        assert num_vertices > 2, "Number of vertices must be greater than 2"
        assert radius > 0, "Radius must be greater than 0"

        theta = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
        if num_vertices % 2 == 0:
            theta += np.pi / num_vertices
        else:
            theta += np.pi / 2

        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        vertices = np.stack([x, y], axis=1)
        return ConvexObject(vertices)

    @staticmethod
    def create_random_polygon(num_vertices: int, size: Tuple[float, float]) -> 'ConvexObject':
        """Create a random polygon with a given number of vertices and size.

        Args:
            num_vertices: The number of vertices of the polygon.
              Must be greater than 2 and less than 20.
            size: A tuple (width, height) representing the size of the polygon.
              Must be greater than 0.
        Returns:
            A `ConvexObject` representing the random polygon.

        Raises:
            AssertionError: If the number of vertices is less than 3 or the size is less than 0.
        """
        assert num_vertices > 2, "Number of vertices must be greater than 2"
        assert num_vertices < 20, "Number of vertices must be less than 20"
        assert size[0] > 0 and size[1] > 0, "Size must be greater than 0"

        # sample random points on a circle
        rand_theta = np.random.rand(num_vertices) * 2 * np.pi
        rand_theta.sort()
        x = np.cos(rand_theta) * size[0] / 2
        y = np.sin(rand_theta) * size[1] / 2
        vertices = np.stack([x, y], axis=1)
        return ConvexObject(vertices)
