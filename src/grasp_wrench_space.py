from typing import List
import numpy as np
from scipy.spatial import ConvexHull
import trimesh

# local imports
from src.contact import ContactPoint


def minkowski_sum(
        position_set_a: np.ndarray,
        position_set_b: np.ndarray) -> np.ndarray:
    """Minkowski sum of two sets of position vectors.

    Args:
        position_set_a (np.ndarray): Set of position vectors. (N, dim)
        position_set_b (np.ndarray): Set of position vectors. (M, dim)
        
    Returns:
        The Minkowski sum of the two sets of position vectors. (N*M, dim)

    Raises:
        ValueError: If the dimension of the position vectors is not the same.
    """
    # check dimension
    if position_set_a.shape[1] != position_set_b.shape[1]:
        raise ValueError("The dimension of the position vectors must be the same. {} != {}".format(
            position_set_a.shape[1], position_set_b.shape[1]))

    # compute Minkowski sum
    dim = position_set_a.shape[1]
    wrenches = position_set_a[:, None] + position_set_b[None, :]
    return wrenches.reshape(-1, dim)


class GraspWrenchSpace():
    def __init__(self, contact_points: List[ContactPoint], wrench_hull: bool):
        """Construct the grasp wrench space from contact points.

        Args:
            contact_points: A list of contact points.
            wrench_hull: A boolean indicating whether construct the `grasp wrench hull`.
              If True, the `grasp wrench hull` is constructed from the contact points.
              If False, the `grasp wrench space` is constructed from the contact points.
        """
        if len(contact_points) < 2:
            raise ValueError("The number of contact points must be at least 2.")

        self._contact_points = contact_points
        self._grasp_matrix = None

        if wrench_hull is True:
            self._gws_cvx_hull = self._get_grasp_wrench_hull()
        else:
            self._gws_cvx_hull = self._get_grasp_wrench_space()

    def _get_grasp_wrench_space(self) -> ConvexHull:
        """Compute the grasp wrench space from the contact points.

        1. Compute the Minkowski sum of the basis wrenches.
        2. Compute the convex hull of the Minkowski sum.

        Returns:
            The convex hull of the grasp wrench space.
        """
        # define zero wrench
        zero_wrench = np.zeros((1, 3))

        # minkowski sum of the basis wrenches
        wrench_set = zero_wrench.copy()
        for cp in self._contact_points:
            basis_wrenches = np.concatenate((zero_wrench, cp.basis_wrenches), axis=0)
            np.concatenate((wrench_set, basis_wrenches), axis=0)
            wrench_set = minkowski_sum(wrench_set, basis_wrenches)

        # compute convex hull
        cvxh = ConvexHull(wrench_set)
        return cvxh

    def _get_grasp_wrench_hull(self) -> ConvexHull:
        """Compute the grasp wrench hull from the contact points.

        1. Collect all basis wrenches from the contact points.
        2. Compute the convex hull of the basis wrenches.

        Returns:
            The convex hull of the grasp wrench hull.
        """
        # collect all basis wrenches
        wrench_set = [cp.basis_wrenches for cp in self._contact_points]
        wrench_set = np.concatenate(wrench_set, axis=0)

        # compute convex hull
        cvxh = ConvexHull(wrench_set)
        return cvxh

    @property
    def contact_points(self) -> List[ContactPoint]:
        """A list of contact points."""
        return self._contact_points

    @property
    def convex_hull(self) -> ConvexHull:
        """The convex hull of the grasp wrench space."""
        return self._gws_cvx_hull

    @property
    def mesh(self) -> trimesh.Trimesh:
        """The mesh representation of the grasp wrench space."""
        # check if the convex hull is 3D
        if self.convex_hull.equations.shape[1] != 4:
            raise ValueError("Convex hull must be 3D.")
        mesh = trimesh.Trimesh(
            vertices=self.convex_hull.points,
            faces=self.convex_hull.simplices)
        mesh.fix_normals()
        return mesh