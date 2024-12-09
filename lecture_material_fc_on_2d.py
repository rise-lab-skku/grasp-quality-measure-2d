import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

from src.geometry import ConvexObject
from src.contact import ContactPoint
from src.grasp_metrics import GraspMetrics
from src.grasp_wrench_space import GraspWrenchSpace, minkowski_sum
from src.visualizer import plot_scene, plot_grasp_wrench_space


def run_2d_example(cvx_obj: ConvexObject,
                   grasp_points: np.ndarray,
                   friction_coef: float,
                   contact_force: float,
                   save_dir: str,
                   ) -> None:
    ##########################
    # compute grasp wrenches #
    ##########################
    # sample contact points on the object
    grasp_contact_points = []
    for i in range(len(grasp_points)):
        point, normal = cvx_obj.sample_closest_point(grasp_points[i])
        cp = ContactPoint(point, normal, friction_coef, contact_force)
        grasp_contact_points.append(cp)

    # collect [fy, tz] wrenches
    fytz_wrenches = []
    for cp in grasp_contact_points:
        basis_wrenches = cp.basis_wrenches
        for bw in basis_wrenches:
            fytz_wrenches.append(bw)
    fytz_wrenches = np.array(fytz_wrenches)
    print('[[ft, tx]] wrenches: \n', fytz_wrenches)
    fytz_wrenches = fytz_wrenches[:, [2, 0]]

    # get minkovski sum of wrenches as polygon patch
    fytz_wrenches_minkovski = np.zeros((1, 2))
    for cp in grasp_contact_points:
        basis_wrenches = cp.basis_wrenches[:, [2, 0]]
        basis_wrenches = np.concatenate([basis_wrenches, np.zeros((1, 2))], axis=0)
        fytz_wrenches_minkovski = minkowski_sum(fytz_wrenches_minkovski, basis_wrenches)
    fytz_wrenches_minkovski = ConvexHull(fytz_wrenches_minkovski)
    fytz_wrenches_minkovski = fytz_wrenches_minkovski.points[fytz_wrenches_minkovski.vertices]
    minkowski_polygon = Polygon(fytz_wrenches_minkovski, edgecolor='grey', facecolor='grey', alpha=0.8)

    # create save directory
    os.makedirs(save_dir, exist_ok=True)

    # save grasp wrench space
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axisbelow(True)
    ax.grid(True)
    ax.add_patch(minkowski_polygon)
    dark_red = (0.7, 0, 0)
    ax.scatter(fytz_wrenches[:, 0], fytz_wrenches[:, 1], c=dark_red)
    plt.xlim(-40, 40)
    plt.ylim(-40, 40)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('fy (N)', fontsize=12)
    plt.ylabel('tz (Nm)', fontsize=12)
    plt.xticks(np.arange(-40, 41, 10))
    plt.yticks(np.arange(-40, 41, 10))
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gws.png'), dpi=300)
    plt.close()

    # save grasp example
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_scene(ax, cvx_obj, grasp_contact_points)
    plt.xlabel('x [m]', fontsize=12)
    plt.ylabel('y [m]', fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'scene.png'), dpi=300)
    plt.close()

def main():
    ##################
    # set parameters #
    ##################
    # friction_coef = 1/np.sqrt(3)
    friction_coef = 5/12
    contact_force = 13  # N

    # create a random convex object
    cvx_obj = ConvexObject.create_regular_polygon(
        num_vertices=4, radius=np.sqrt(2)*2)
    cvx_obj.apply_scaling([1, 0.5])

    # define contact points for grasp examples
    non_fc_grasp = np.array([
        [-1.0,  1.0],
        [ 1.0, -1.0],
        ])
    fc_grasp = np.array([
        [-1.0,  1.0],
        [ 1.0, -1.0],
        [-1.0, -1.0],
        ])

    # run examples
    run_2d_example(cvx_obj, non_fc_grasp, friction_coef, contact_force, 'figs_lecture/non_fc_grasp')
    run_2d_example(cvx_obj, fc_grasp, friction_coef, contact_force, 'figs_lecture/fc_grasp', )


if __name__ == "__main__":
    main()
