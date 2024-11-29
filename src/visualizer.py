from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Polygon

# local imports
from src.contact import ContactPoint
from src.geometry import ConvexObject
from src.grasp_wrench_space import GraspWrenchSpace
from src.grasp_metrics import GraspMetrics


def get_com_patches(center: Tuple[float, float], radius: float) -> List[Wedge]:
    center = (center[0], center[1])
    wedge1 = Wedge(center, radius, 0, 90, facecolor='black', edgecolor='black')
    wedge2 = Wedge(center, radius, 90, 180, facecolor='white', edgecolor='black')
    wedge3 = Wedge(center, radius, 180, 270, facecolor='black', edgecolor='black')
    wedge4 = Wedge(center, radius, 270, 360, facecolor='white', edgecolor='black')
    return [wedge1, wedge2, wedge3, wedge4]

def plot_scene(
        ax: plt.Axes,
        convex_object: ConvexObject,
        contact_points: List[ContactPoint]=None,
        external_wrench: np.ndarray=None):
    # get boundary of plot
    x_bound = [
        np.min(convex_object.vertices),
        np.max(convex_object.vertices)]
    y_bound = [
        np.min(convex_object.vertices),
        np.max(convex_object.vertices)]

    # get scale of plot
    x_size = x_bound[1] - x_bound[0]
    y_size = y_bound[1] - y_bound[0]
    obj_size = np.max([x_size, y_size])

    # set axis limits
    ax.set_xlim([x_bound[0] - 0.3*obj_size, x_bound[1] + 0.3*obj_size])
    ax.set_ylim([y_bound[0] - 0.3*obj_size, y_bound[1] + 0.3*obj_size])
    ax.set_aspect("equal")
    ax.set_axisbelow(True)
    ax.grid(True)

    # set labels
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    # plot obj polygon path
    polygon_patch = convex_object.matplotlib_patch
    polygon_patch.set_facecolor("pink")
    polygon_patch.set_edgecolor((0.3, 0.3, 0.3))
    ax.add_patch(polygon_patch)

    # plot contact points
    if contact_points is not None:
        # collect contact points and normals
        points = np.array([cp.point for cp in contact_points])
        normals = np.array([cp.normal for cp in contact_points]) * obj_size * 0.15

        # plot contact points
        ax.scatter(points[:, 0], points[:, 1], c="red")

        # plot contact normals
        for i in range(len(points)):
            ax.arrow(
                points[i, 0]+normals[i, 0], points[i, 1]+normals[i, 1],
                -normals[i, 0], -normals[i, 1],
                width=obj_size*0.005,
                head_width=obj_size*0.02,
                head_length=obj_size*0.05,
                length_includes_head=True,
                fc='black', ec='black')

        # plot friction cone
        for cp in contact_points:
            # get [fx, fy] basis wrenches
            basis_wrenches = cp.basis_wrenches[:, 1:3]
            basis_wrenches = basis_wrenches / np.linalg.norm(basis_wrenches, axis=1)[:, None]
            basis_wrenches *= obj_size * 0.2
            basis_wrenches += cp.point
            basis_wrenches = np.vstack([cp.point, basis_wrenches])
            fc_patch = Polygon(
                basis_wrenches, closed=True, facecolor=(0.6, 0, 0),
                # alpha=0.9,
                )
            ax.add_patch(fc_patch)

    # plot external wrench
    if external_wrench is not None:
        vec = external_wrench[1:3]
        vec = vec / np.linalg.norm(vec) * obj_size * 0.7
        ax.arrow(
            0, 0,
            vec[0], vec[1],
            width=obj_size*0.005,
            head_width=obj_size*0.02,
            head_length=obj_size*0.05,
            color=(0.9, 0, 0))
        
    # plot com patches
    com_patches = get_com_patches([0, 0], obj_size * 0.05)
    for patch in com_patches:
        ax.add_patch(patch)

def plot_grasp_wrench_space(
        ax: plt.Axes,
        grasp_wrench_space: GraspWrenchSpace,
        axis_range: Tuple[float, float, float, float, float, float]=None,
        external_wrench: np.ndarray=None,
        draw_lrw: bool=False,  # draw largest minimum resisted wrench
        ):
    # plot origin
    ax.scatter(0, 0, 0, c='black')

    # plot convex hull
    color = '#0E4C92'
    vertices = grasp_wrench_space.mesh.vertices
    faces = grasp_wrench_space.mesh.faces

    # [tz, fx, fy] to [fx, fy, tz]
    vertices = vertices[:, [1, 2, 0]]
    external_wrench = external_wrench[[1, 2, 0]] if external_wrench is not None else None

    # plot 3d mesh
    ax.plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        triangles=faces,
        color=color,
        alpha=0.2)

    # plot wireframe
    for tri in faces:
        tri_vertices = vertices[tri]
        for i in range(3):
            start, end = tri_vertices[i], tri_vertices[(i + 1) % 3]
            ax.plot(
                [start[0], end[0]], 
                [start[1], end[1]], 
                [start[2], end[2]],
                color=color,
                alpha=0.5)

    # set axis limits
    if axis_range is not None:
        ax.set_xlim(axis_range[0:2])
        ax.set_ylim(axis_range[2:4])
        ax.set_zlim(axis_range[4:6])

    # set label
    ax.set_xlabel("fx [N]")
    ax.set_ylabel("fy [N]")
    ax.set_zlabel("tz [Nm]")

    # plot external wrench
    if external_wrench is not None:
        # ax.quiver(
        #     0, 0, 0,
        #     external_wrench[0], external_wrench[1], external_wrench[2],
        #     color='red')
        ax.plot(
            [0, external_wrench[0]],
            [0, external_wrench[1]],
            [0, external_wrench[2]],
            color='red')

    # plot largest minimum resisted wrench
    is_force_closure = GraspMetrics.force_closure_hull(grasp_wrench_space.convex_hull)
    if draw_lrw and is_force_closure:
        min_dist, _ = GraspMetrics.largest_minimum_resisted_wrench(grasp_wrench_space.convex_hull)
        # draw sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = min_dist * np.outer(np.cos(u), np.sin(v))
        y = min_dist * np.outer(np.sin(u), np.sin(v))
        z = min_dist * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='gray', alpha=0.3)
