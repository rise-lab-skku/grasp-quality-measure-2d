from typing import List, Tuple
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

# local imports
from src.contact import ContactPoint


class GraspMetrics():
    @staticmethod
    def force_closure_lp(contact_points: List[ContactPoint]) -> bool:
        """Check force closure of the grasp by solving a linear program.

        Args:
            contact_points: A list of contact points.

        Returns:
            A boolean indicating whether the grasp is force closure or not.
        """
        # collect all basis wrenches
        wrenches = [cp.basis_wrenches for cp in contact_points]
        wrenches = np.concatenate(wrenches)

        # if rank of the wrench matrix is less than 3, the grasp is not in force closure
        if np.linalg.matrix_rank(wrenches) < 3:
            return False

        ##############################################################
        # <Linear Program Formulation for Force Closure>                                 
        # F is (3, N) wrench matrix, where N is the number of wrenches. 
        # k is (N, 1) a vector of weights, k >= 0.                      
        # The linear program is:                                        
        #   find: k
        #   min: 1^T * k
        #   s.t.: Fk = 0
        #         k_i >= 1, i = 1, 2, ..., N
        ##############################################################

        # define the linear program
        c = np.ones(wrenches.shape[0])
        A_eq = wrenches.T
        b_eq = np.zeros(3)
        bounds = [(1, None) for _ in range(wrenches.shape[0])]

        # solve the linear program
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

        # if the linear program is successful, the grasp is in force closure
        is_force_closure = res.success
        return is_force_closure

    @staticmethod
    def force_closure_hull(convex_hull: ConvexHull, eps: float=1e-6) -> bool:
        """Check force closure of the grasp using the convex hull of the grasp wrench space.

        Args:
            convex_hull: The convex hull of the grasp wrench space(or hull).
            eps: A small value to prevent numerical errors.

        Returns:
            A boolean indicating whether the grasp is force closure or not.
        """
        ##############################################################
        # <About ConvexHull.equations>
        # A hyperplane is d normal coefficients and an offset.
        # The length of the normal is one. The hyperplane defines a halfspace.
        # If V is a normal, b is an offset, and x is a point inside the convex hull, then Vx+b <0.
        # [ref] Qhull documentation: http://www.qhull.org/html/index.htm
        ##############################################################

        # check the origin is inside the convex hull
        zero_vector = np.array([0, 0, 0, 1], dtype=np.float64)
        ret = np.all(convex_hull.equations @ zero_vector < -eps)
        return ret

    @staticmethod
    def volume(convex_hull: ConvexHull) -> float:
        """Compute the volume of the grasp wrench space.

        Args:
            convex_hull: The convex hull of the grasp wrench space(or hull).

        Returns:
            The volume of the grasp wrench space.
        """
        return convex_hull.volume

    @staticmethod
    def largest_minimum_resisted_wrench(convex_hull: ConvexHull) -> Tuple[float, np.ndarray]:
        """Compute the largest minimum resisted wrench.

        Args:
            convex_hull: The convex hull of the grasp wrench space(or hull).

        Returns:
            The l2 norm of the largest minimum resisted wrench and
            the vector of the largest minimum resisted wrench.
        """
        # if the grasp is not the force closure,
        # largest minimum resisted wrench is 0
        if not GraspMetrics.force_closure_hull(convex_hull):
            return 0.0

        ##############################################################
        # <About ConvexHull.equations>
        # A hyperplane is d normal coefficients and an offset.
        # The length of the normal is one. The hyperplane defines a halfspace.
        # If V is a normal, b is an offset, and x is a point inside the convex hull, then Vx+b <0.
        # [ref] Qhull documentation: http://www.qhull.org/html/index.htm
        ##############################################################

        # find the minimum distance from the origin to the convex hull
        distances = convex_hull.equations @ np.array([0, 0, 0, 1], dtype=np.float64)
        distances = np.abs(distances)
        min_distance_idx = np.argmin(distances)
        min_distance = distances[min_distance_idx]

        # find min vector(foot of perpendicular)
        min_vector = convex_hull.equations[min_distance_idx, :-1] * min_distance
        return min_distance, min_vector

    @staticmethod
    def wrench_resistance_hull(
        convex_hull: ConvexHull,
        external_wrench: np.ndarray,
        eps: float=1e-6) -> bool:
        """Compute the wrench resistance of the grasp from grasp wrench space(or hull).

        Args:
            convex_hull: The convex hull of the grasp wrench space(or hull).
            external_wrench: The external wrench applied to the object.

        Returns:
            A boolean indicating whether the grasp can resist the external wrench.
        """
        ##############################################################
        # <About ConvexHull.equations>
        # A hyperplane is d normal coefficients and an offset.
        # The length of the normal is one. The hyperplane defines a halfspace.
        # If V is a normal, b is an offset, and x is a point inside the convex hull, then Vx+b <0.
        # [ref] Qhull documentation: http://www.qhull.org/html/index.htm
        ##############################################################

        # check the external wrench is inside the convex hull
        external_wrench = np.concatenate((external_wrench, [1]))
        ret = np.all(convex_hull.equations @ external_wrench < -eps)
        return ret

    @staticmethod
    def wrench_resistance_qp(
        contact_points: List[ContactPoint],
        external_wrench: np.ndarray,
        wrench_hull: bool,
        eps: float=1e-5) -> bool:
        """Compute the wrench resistance of the grasp by solving a quadratic program.

        Args:
            contact_points: A list of contact points.
            external_wrench: The external wrench applied to the object.
            wrench_hull: A boolean indicating whether to use the grasp wrench hull.
            eps: A small value to prevent numerical errors.

        Returns:
            A boolean indicating whether the grasp can resist the external wrench.
        """
        # collect all basis wrenches
        wrenches = [cp.basis_wrenches for cp in contact_points]
        wrenches = np.concatenate(wrenches)  # (N, 3)

        #################################
        # <cvxopt quadratic programming>
        # minimize 1/2 x^T Q x + p^T x
        # subject to Gx <= h
        #            Ax = b
        #################################
        Q = wrenches @ wrenches.T  # (N, N)
        p = wrenches @ external_wrench  # (N, 1)
        G1 = -np.eye(wrenches.shape[0])  # wrenches >= 0
        h1 = np.zeros(wrenches.shape[0])
        if wrench_hull is True:
            # if Grasp Wrench Hull, sum of weights <= 1
            G2 = np.ones(wrenches.shape[0]) 
            h2 = 1
        else:
            # if Grasp Wrench Space, sum of weights for each contact point <= 1
            num_contact_points = len(contact_points)
            G2 = np.zeros((num_contact_points, wrenches.shape[0]))
            for i in range(num_contact_points):
                G2[i, 2*i:2*(i+1)] = 1
            h2 = np.ones(num_contact_points)
        G = np.concatenate((G1, G2))
        h = np.concatenate((h1, h2))

        # convert to cvxopt matrix
        Q = matrix(Q)
        p = matrix(p)
        G = matrix(G)
        h = matrix(h)

        # solve the quadratic program
        sol = solvers.qp(Q, p, G, h)

        # check the result
        weights = np.array(sol['x'])
        error = np.linalg.norm(wrenches.T @ weights + external_wrench)
        can_resist = error < eps
        return can_resist
