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
    grasp1_gwh = GraspWrenchSpace(grasp1_contact_points, wrench_hull=True)

    # compute all metrics for grasp examples
    fc = GraspMetrics.force_closure_lp(grasp1_gws.contact_points)
    gws_volume = GraspMetrics.volume(grasp1_gws.convex_hull)
    gws_lrw, _ = GraspMetrics.largest_minimum_resisted_wrench(grasp1_gws.convex_hull)

    gwh_volume = GraspMetrics.volume(grasp1_gwh.convex_hull)
    gwh_lrw, _ = GraspMetrics.largest_minimum_resisted_wrench(grasp1_gwh.convex_hull)


    # plot grasp examples
    fig = plt.figure(figsize=(16, 5))

    # set boundary of 3d plot
    axis_range = [-2, 2, -2, 2, -2, 2]

    # plot grasp1 example
    ax1 = fig.add_subplot(131)
    plot_scene(ax1, cvx_obj, grasp1_contact_points)

    # 2) plot grasp wrench space
    ax2 = fig.add_subplot(132, projection='3d')
    plot_grasp_wrench_space(
        ax=ax2,
        grasp_wrench_space=grasp1_gws,
        axis_range=axis_range,
        draw_lrw=True)
    title = 'Grasp Wrench Space\n'
    title += f"Force Closure: {fc}\n"
    title += f"Volume: {gws_volume:.3g}\n"
    title += f"Largest Min. Resisted Wrench: {gws_lrw:.3g}\n"
    ax2.set_title(title, loc='right')

    # 3) plot grasp wrench hull
    ax3 = fig.add_subplot(133, projection='3d')
    plot_grasp_wrench_space(
        ax=ax3,
        grasp_wrench_space=grasp1_gwh,
        axis_range=axis_range,
        draw_lrw=True)
    title = 'Grasp Wrench Hull\n'
    title += f"Force Closure: {fc}\n"
    title += f"Volume: {gwh_volume:.3g}\n"
    title += f"Largest Min. Resisted Wrench: {gwh_lrw:.3g}\n"
    ax3.set_title(title, loc='right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
