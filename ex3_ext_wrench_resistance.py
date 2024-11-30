import numpy as np
import matplotlib.pyplot as plt
import trimesh

from src.geometry import ConvexObject
from src.contact import ContactPoint
from src.grasp_metrics import GraspMetrics
from src.grasp_wrench_space import GraspWrenchSpace
from src.visualizer import plot_scene, plot_grasp_wrench_space

def main():
    # set parameters
    friction_coef = 0.5
    contact_force = 1  # N
    small_external_wrench = np.array([0, 0, -1])  # [tz, fx, fy] Nm, N, N
    large_external_wrench = np.array([0, 0, -2])  # [tz, fx, fy] Nm, N, N

    # create a random convex object
    cvx_obj = ConvexObject.create_regular_polygon(
        num_vertices=4, radius=np.sqrt(2))

    # define contact points for grasp examples
    grasp1_points = np.array([
        [-0.8, 1.0],
        [-0.8, -1.1],
        [ 0.8, -1.0],
        ])

    # sample contact points on the object
    grasp1_contact_points = []
    for i in range(len(grasp1_points)):
        point, normal = cvx_obj.sample_closest_point(grasp1_points[i])
        cp = ContactPoint(point, normal, friction_coef, contact_force)
        grasp1_contact_points.append(cp)

    # create grasp wrench space for grasp examples
    grasp1_gws = GraspWrenchSpace(grasp1_contact_points, wrench_hull=False)

    # compute all metrics for grasp examples
    wrench_resistance_s = GraspMetrics.wrench_resistance_qp(
        contact_points=grasp1_contact_points,
        external_wrench=small_external_wrench,
        wrench_hull=False)
    wrench_resistance_l = GraspMetrics.wrench_resistance_qp(
        contact_points=grasp1_contact_points,
        external_wrench=large_external_wrench,
        wrench_hull=False)

    # plot grasp examples
    fig = plt.figure(figsize=(16, 5))

    # set boundary of 3d plot
    axis_range = [-2, 2, -2, 2, -2, 2]

    # plot grasp1 example
    ax1 = fig.add_subplot(131)
    plot_scene(
        ax1,
        convex_object=cvx_obj,
        contact_points=grasp1_contact_points,
        external_wrench=small_external_wrench)

    # 2) plot grasp wrench space
    ax2 = fig.add_subplot(132, projection='3d')
    plot_grasp_wrench_space(
        ax=ax2,
        grasp_wrench_space=grasp1_gws,
        axis_range=axis_range,
        external_wrench=small_external_wrench)
    title = f"External Wrench(tz, fx, fy): {small_external_wrench}\n"
    title += f"Wrench Resistance (small): {wrench_resistance_s}\n"
    ax2.set_title(title, loc='left')

    # 3) plot grasp wrench hull
    ax3 = fig.add_subplot(133, projection='3d')
    plot_grasp_wrench_space(
        ax=ax3,
        grasp_wrench_space=grasp1_gws,
        axis_range=axis_range,
        external_wrench=large_external_wrench)
    title = f"External Wrench(tz, fx, fy): {large_external_wrench}\n"
    title += f"Wrench Resistance (large): {wrench_resistance_l}\n"
    ax3.set_title(title, loc='left')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
