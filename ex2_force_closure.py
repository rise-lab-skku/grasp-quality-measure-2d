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

    # create a random convex object
    cvx_obj = ConvexObject.create_regular_polygon(
        num_vertices=4, radius=np.sqrt(2))

    # define contact points for grasp examples
    grasp1_points = np.array([
        [-0.8,  1.0],
        [ 0.8, -1.0],
        ])
    grasp2_points = np.array([
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
    grasp2_contact_points = []
    for i in range(len(grasp2_points)):
        point, normal = cvx_obj.sample_closest_point(grasp2_points[i])
        cp = ContactPoint(point, normal, friction_coef, contact_force)
        grasp2_contact_points.append(cp)

    # create grasp wrench space for grasp examples
    grasp1_gws = GraspWrenchSpace(grasp1_contact_points, wrench_hull=False)
    grasp2_gws = GraspWrenchSpace(grasp2_contact_points, wrench_hull=False)

    # compute all metrics for grasp examples
    grasp1_fc = GraspMetrics.force_closure_lp(grasp1_contact_points)
    grasp1_volume = GraspMetrics.volume(grasp1_gws.convex_hull)
    grasp1_lrw, _ = GraspMetrics.largest_minimum_resisted_wrench(grasp1_gws.convex_hull)

    grasp2_fc = GraspMetrics.force_closure_lp(grasp2_contact_points)
    grasp2_volume = GraspMetrics.volume(grasp2_gws.convex_hull)
    grasp2_lrw, _ = GraspMetrics.largest_minimum_resisted_wrench(grasp2_gws.convex_hull)

    # plot grasp examples
    fig = plt.figure(figsize=(10, 10))

    # set boundary of 3d plot
    axis_range = [-2, 2, -2, 2, -2, 2]

    # plot grasp1 example
    ax1 = fig.add_subplot(221)
    plot_scene(ax1, cvx_obj, grasp1_contact_points)

    # 2) plot grasp wrench space
    ax2 = fig.add_subplot(222, projection='3d')
    plot_grasp_wrench_space(
        ax=ax2,
        grasp_wrench_space=grasp1_gws,
        axis_range=axis_range,
        draw_lrw=True)
    title = 'Grwa Wrench Space\n'
    title += f"Force Closure: {grasp1_fc}\n"
    title += f"Volume: {grasp1_volume:.3g}\n"
    title += f"Largest Min. Resisted Wrench: {grasp1_lrw:.3g}\n"
    ax2.set_title(title, loc='right')

    # plot grasp2 example
    ax3 = fig.add_subplot(223)
    plot_scene(ax3, cvx_obj, grasp2_contact_points)

    # 2) plot grasp wrench space
    ax4 = fig.add_subplot(224, projection='3d')
    plot_grasp_wrench_space(
        ax=ax4,
        grasp_wrench_space=grasp2_gws,
        axis_range=axis_range,
        draw_lrw=True)
    title = 'Grwa Wrench Space\n'
    title += f"Force Closure: {grasp2_fc}\n"
    title += f"Volume: {grasp2_volume:.3g}\n"
    title += f"Largest Min. Resisted Wrench: {grasp2_lrw:.3g}\n"
    ax4.set_title(title, loc='right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
