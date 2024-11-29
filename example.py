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
    contact_force = 10  # N
    external_wrench = np.array([0, 0, -20])  # (Nm, N, N)

    # create a convex object
    # cvx_obj = ConvexObject.create_regular_polygon(12, 1)
    cvx_obj = ConvexObject.create_regular_polygon(4, 0.1)

    # sample random uniform points on the surface
    points, normals = cvx_obj.smaple_surface(3)

    # create contact points
    contact_points = []
    for i in range(len(points)):
        cp = ContactPoint(points[i], normals[i], friction_coef, contact_force)
        contact_points.append(cp)

    # create grasp wrench space and grasp wnench hull
    gws = GraspWrenchSpace(contact_points, wrench_hull=False)
    gwh = GraspWrenchSpace(contact_points, wrench_hull=True)

    # compute all metrics
    gws_fc = GraspMetrics.force_closure_hull(gws.convex_hull)
    gws_volume = GraspMetrics.volume(gws.convex_hull)
    gws_lrw, _ = GraspMetrics.largest_minimum_resisted_wrench(gws.convex_hull)
    gws_wr = GraspMetrics.wrench_resistance_hull(gws.convex_hull, external_wrench)

    gwh_fc = GraspMetrics.force_closure_hull(gwh.convex_hull)
    gwh_volume = GraspMetrics.volume(gwh.convex_hull)
    gwh_lrw, _ = GraspMetrics.largest_minimum_resisted_wrench(gwh.convex_hull)
    gwh_wr = GraspMetrics.wrench_resistance_hull(gwh.convex_hull, external_wrench)

    # plot grasp examples
    fig = plt.figure(figsize=plt.figaspect(0.3))

    # get boundary of 3d plot
    x_bound = np.max(np.abs(gws.convex_hull.points[:, 1])) * 1.1
    y_bound = np.max(np.abs(gws.convex_hull.points[:, 2])) * 1.1
    z_bound = np.max(np.abs(gws.convex_hull.points[:, 0])) * 1.1
    axis_range = [-x_bound, x_bound, -y_bound, y_bound, -z_bound, z_bound]

    # 1) plot grasp scene
    ax1 = fig.add_subplot(131)
    plot_scene(ax1, cvx_obj, contact_points, external_wrench)

    # 2) plot grasp wrench space
    ax2 = fig.add_subplot(132, projection='3d')
    plot_grasp_wrench_space(
        ax=ax2,
        grasp_wrench_space=gws,
        axis_range=axis_range,
        external_wrench=external_wrench,
        draw_lrw=True)
    title = 'Grwa Wrench Space\n'
    title += f"Force Closure: {gws_fc}\n"
    title += f"Volume: {gws_volume:.3g}\n"
    title += f"Largest Min. Resisted Wrench: {gws_lrw:.3g}\n"
    title += f"Wrench Resistance: {gws_wr}"
    ax2.set_title(title)

    # 3) plot grasp wrench hull
    ax3 = fig.add_subplot(133, projection='3d')
    plot_grasp_wrench_space(
        ax=ax3,
        grasp_wrench_space=gwh,
        axis_range=axis_range,
        external_wrench=external_wrench,
        draw_lrw=True)
    title = 'Grwa Wrench Hull\n'
    title += f"Force Closure: {gwh_fc}\n"
    title += f"Volume: {gwh_volume:.3g}\n"
    title += f"Largest Min. Resisted Wrench: {gwh_lrw:.3g}\n"
    title += f"Wrench Resistance: {gwh_wr}"
    ax3.set_title(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
