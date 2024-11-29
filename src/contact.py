import numpy as np


class ContactPoint():
    def __init__(self, point: np.ndarray, normal: np.ndarray, friction: float, contact_force: float):
        """Initialize a contact point.

        Args:
            point: The contact point in the object's coordinate frame.
            normal: The surface normal(pointing outward) in the object's coordinate frame.
            friction: The coefficient of friction at the contact point.
            contact_force: The magnitude of the contact force at the contact point.
        """
        self._point = point
        self._normal = normal
        self._friction = friction
        self._contact_force = contact_force

        # internal variables
        self._pose = None

        self.test = 1

    def __str__(self):
        return f"ContactPoint(point={self.point}, normal={self.normal},\
                 friction={self.friction}, contact_force={self.contact_force})"

    @property
    def point(self):
        """The contact point in the object's coordinate frame."""
        return self._point

    @property
    def normal(self):
        """The contact normal in the object's coordinate frame."""
        return self._normal

    @property
    def friction(self):
        """The coefficient of friction at the contact point."""
        return self._friction

    @property
    def contact_force(self):
        """The magnitude of the contact force at the contact point."""
        return self._contact_force

    @property
    def pose(self):
        """The pose of the contact point in the object's coordinate frame.

        x-axis is tangent to the object and y-axis is normal to the object (pointing inwards).

        Returns:
            A 3x3 SE(2) transformation matrix representing the pose of the contact point.
        """
        if self._pose is None:
            y_axis = -self.normal
            x_axis = np.array([y_axis[1], -y_axis[0]])
            rot_matrix = np.array([x_axis, y_axis]).T
            tf = np.eye(3)
            tf[0:2, 0:2] = rot_matrix
            tf[0:2, 2] = self.point
            self._pose = tf
        return self._pose

    @property
    def adjoint(self):
        """The adjoint matrix [Ad_T_oc] of the pose of the contact point.

        Given SE(2) transformation matrix T_ab:
            [[R11 R12 px]
             [R21 R22 py]
             [  0   0  1]]

        The adjoint matrix [Ad_T_ab] is defined as:
           [[1   0   0]
            [ py R11 R12]
            [-px R21 R22]]

        Relation between adjoint and wrench:
            wrench W = [t_z, f_x, f_y]
            W_b  = [Ad_T_ab]^T * W_a

        Returns:
            A 3x3 adjoint matrix of the pose of the contact point.
        """
        adjoint = np.eye(3)
        adjoint[1, 0] = self.point[1]
        adjoint[2, 0] = -self.point[0]
        adjoint[1:3, 1:3] = self.pose[0:2, 0:2]
        return adjoint

    @property
    def basis_wrenches(self) -> np.ndarray:
        """Compute the basis wrenches of the friction cone in object's coordinate frame.

        The wrench is defined as [t_z, f_x, f_y] where t_z is the torque and f_x, f_y are the forces.

        Returns:
            A 2x3 numpy array representing the basis wrenches at the contact point in the object's coordinate frame.
        """
        # define the basis wrenches in contact coordinate frame
        theta = np.arctan2(self.friction, 1)
        wrenches_c = np.array([
            [0, np.sin(theta), np.cos(theta)],
            [0, -np.sin(theta), np.cos(theta)]
        ])

        # scale the basis wrenches by the contact force
        wrenches_c *= self.contact_force

        # transform the basis wrenches to object coordinate frame
        # w_o = [Ad_T_co]^T * w_c
        adjoint_co = np.linalg.inv(self.adjoint)
        wrenches_o = np.dot(adjoint_co.T, wrenches_c.T).T
        return wrenches_o
