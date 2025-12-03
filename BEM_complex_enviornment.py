import numpy as np
import scipy.io
import time
import multiprocessing
import sys
import os
from scipy.spatial import Delaunay
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import os
import scipy.io as sio
from _if_hit_obstacles import *


def legendre_polynomial(n, a, b):
    """
    Gauss–Legendre nodes/weights mapped from [-1,1] → [a,b].
    Returns (nodes, weights), both 1D arrays of length n.
    """
    n = max(1, int(np.round(n)))  # ensure at least 1
    xi, wi = np.polynomial.legendre.leggauss(n)  # on [-1,1]
    # affine map
    t = 0.5 * (b - a) * xi + 0.5 * (b + a)
    # w = 0.5 * (b - a) * wi
    w = abs(b - a) / 2 * wi
    return t, w


def _rot_mat(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def legendre_polynomial(n, a, b):
    n = max(1, int(np.round(n)))
    xi, wi = np.polynomial.legendre.leggauss(n)  # on [-1,1]
    t = 0.5 * (b - a) * xi + 0.5 * (b + a)
    w = 0.5 * abs(b - a) * wi  # <-- always positive weights for BEM
    return t, w


def _rot_mat(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _wrap(th):
    th = (th + np.pi) % (2 * np.pi) - np.pi
    if th <= -np.pi:
        th = np.pi
    return th


def rounded_box_quadrature(
    corner, R=0.5, dens=100, densW=100, rotation=0.0, rotate_normals=True
):
    """
    Returns
    -------
    x, y, w, nx, ny, kappa,
    arc_centers,        # shape (4,2): centers of 4 fillet arcs AFTER rotation
    arc_theta_ranges    # shape (4,2): (theta_start, theta_end) AFTER rotation (global)
    """
    # starts from Bottom straight edge, goes CCW around box
    corner = np.asarray(corner, dtype=float)
    if corner.shape[0] not in (4, 5) or corner.shape[1] != 2:
        raise ValueError(
            "corner must be (4,2) or (5,2): 4 corners (+ optional center)."
        )

    # center from 5th row if provided; else average of corners
    if corner.shape[0] == 5:
        center = corner[4]
        corner = corner[:4]
    else:
        center = corner.mean(axis=0)

    # Unrotated helper nodes (axis-aligned in given corner coords)
    node1 = np.array([corner[0, 0] + R, corner[0, 1]])
    node2 = np.array([corner[1, 0] - R, corner[1, 1]])
    node3 = np.array([corner[1, 0], corner[1, 1] + R])
    node4 = np.array([corner[2, 0], corner[2, 1] - R])
    node5 = np.array([corner[2, 0] - R, corner[2, 1]])
    node6 = np.array([corner[3, 0] + R, corner[3, 1]])
    node7 = np.array([corner[3, 0], corner[3, 1] - R])
    node8 = np.array([corner[0, 0], corner[0, 1] + R])
    line_segments = np.array(
        [
            [node1, node2],  # bottom
            [node3, node4],  # right
            [node5, node6],  # top
            [node7, node8],  # left
        ],
        dtype=float,
    )

    # 1→2 (bottom)
    len1_2 = abs(node1[0] - node2[0])
    n1 = max(1, int(np.round(dens * len1_2)))
    x1_2, w1_2 = legendre_polynomial(n1, node1[0], node2[0])
    y1_2 = np.full_like(x1_2, node1[1])
    n1_2 = np.column_stack((np.zeros(n1), -np.ones(n1)))
    R1_2 = np.zeros(n1)

    # 2→3 (BR arc; angles 3π/2→2π); center = (node2.x, node2.y + R)
    n2 = max(1, int(np.round(densW * (0.5 * np.pi * R))))
    t2_3, w2_3 = legendre_polynomial(n2, 1.5 * np.pi, 2 * np.pi)
    c23 = np.array([node2[0], node2[1] + R])
    x2_3 = R * np.cos(t2_3) + c23[0]
    y2_3 = R * np.sin(t2_3) + c23[1]
    w2_3 = R * w2_3
    n2_3 = np.column_stack((np.cos(t2_3), np.sin(t2_3)))
    R2_3 = np.full(n2, 1.0 / R)

    # 3→4 (right)
    len3_4 = abs(node3[1] - node4[1])
    n3 = max(1, int(np.round(dens * len3_4)))
    y3_4, w3_4 = legendre_polynomial(n3, node3[1], node4[1])
    x3_4 = np.full_like(y3_4, node3[0])
    n3_4 = np.column_stack((np.ones(n3), np.zeros(n3)))
    R3_4 = np.zeros(n3)

    # 4→5 (TR arc; angles 0→π/2); center = (node5.x, node4.y)
    n4 = max(1, int(np.round(densW * (0.5 * np.pi * R))))
    t4_5, w4_5 = legendre_polynomial(n4, 0.0, 0.5 * np.pi)
    c45 = np.array([node5[0], node4[1]])
    x4_5 = R * np.cos(t4_5) + c45[0]
    y4_5 = R * np.sin(t4_5) + c45[1]
    w4_5 = R * w4_5
    n4_5 = np.column_stack((np.cos(t4_5), np.sin(t4_5)))
    R4_5 = np.full(n4, 1.0 / R)

    # 5→6 (top)
    len5_6 = abs(node5[0] - node6[0])
    n5 = max(1, int(np.round(dens * len5_6)))
    x5_6, w5_6 = legendre_polynomial(n5, node5[0], node6[0])
    y5_6 = np.full_like(x5_6, node5[1])
    n5_6 = np.column_stack((np.zeros(n5), np.ones(n5)))
    R5_6 = np.zeros(n5)

    # 6→7 (TL arc; angles π/2→π); center = (node6.x, node6.y - R)
    n6 = max(1, int(np.round(densW * (0.5 * np.pi * R))))
    t6_7, w6_7 = legendre_polynomial(n6, 0.5 * np.pi, np.pi)
    c67 = np.array([node6[0], node6[1] - R])
    x6_7 = R * np.cos(t6_7) + c67[0]
    y6_7 = R * np.sin(t6_7) + c67[1]
    w6_7 = R * w6_7
    n6_7 = np.column_stack((np.cos(t6_7), np.sin(t6_7)))
    R6_7 = np.full(n6, 1.0 / R)

    # 7→8 (left)
    len7_8 = abs(node7[1] - node8[1])
    n7 = max(1, int(np.round(dens * len7_8)))
    y7_8, w7_8 = legendre_polynomial(n7, node7[1], node8[1])
    x7_8 = np.full_like(y7_8, node7[0])
    n7_8 = np.column_stack((-np.ones(n7), np.zeros(n7)))
    R7_8 = np.zeros(n7)

    # 8→1 (BL arc; angles π→3π/2); center = (node8.x + R, node8.y)
    n8 = max(1, int(np.round(densW * (0.5 * np.pi * R))))
    t8_1, w8_1 = legendre_polynomial(n8, np.pi, 1.5 * np.pi)
    c81 = np.array([node8[0] + R, node8[1]])
    x8_1 = R * np.cos(t8_1) + c81[0]
    y8_1 = R * np.sin(t8_1) + c81[1]
    w8_1 = R * w8_1
    n8_1 = np.column_stack((np.cos(t8_1), np.sin(t8_1)))
    R8_1 = np.full(n8, 1.0 / R)

    # concatenate (pre-rotation)
    x = np.concatenate([x1_2, x2_3, x3_4, x4_5, x5_6, x6_7, x7_8, x8_1])
    y = np.concatenate([y1_2, y2_3, y3_4, y4_5, y5_6, y6_7, y7_8, y8_1])
    w = np.concatenate([w1_2, w2_3, w3_4, w4_5, w5_6, w6_7, w7_8, w8_1])
    n = np.vstack([n1_2, n2_3, n3_4, n4_5, n5_6, n6_7, n7_8, n8_1])
    kappa = np.concatenate([R1_2, R2_3, R3_4, R4_5, R5_6, R6_7, R7_8, R8_1])

    # arc centers (pre-rotation) in the same order as the arcs above
    arc_centers = np.vstack([c23, c45, c67, c81])

    # arc theta ranges BEFORE rotation (local frame)
    theta_ranges_local = np.array(
        [
            [1.5 * np.pi, 2.0 * np.pi],  # 2→3 (BR)
            [0.0, 0.5 * np.pi],  # 4→5 (TR)
            [0.5 * np.pi, np.pi],  # 6→7 (TL)
            [np.pi, 1.5 * np.pi],  # 8→1 (BL)
        ]
    )
    # [ bottom-right, top-right, top-left, bottom-left ]

    # rotate points (and optionally normals) about the center
    if rotation != 0.0:
        Rmat = _rot_mat(rotation)

        pts = np.column_stack((x, y)) - center
        pts = pts @ Rmat.T + center
        x, y = pts[:, 0], pts[:, 1]

        # rotate arc centers too
        ac_shift = arc_centers - center
        arc_centers = (ac_shift @ Rmat.T) + center
        ls = line_segments.reshape(-1, 2)  # (8,2)
        ls = (ls - center) @ Rmat.T + center  # rotate about center
        line_segments = ls.reshape(4, 2, 2)  # back to (4,2,2)
        if rotate_normals:
            n = n @ Rmat.T  # rotate normals too

        # rotate theta ranges to GLOBAL by adding rotation
        theta_ranges = theta_ranges_local + rotation
    else:
        theta_ranges = theta_ranges_local.copy()

    # wrap theta ranges to (-pi, pi]
    theta_ranges = np.vectorize(_wrap)(theta_ranges)

    nx, ny = n[:, 0], n[:, 1]
    return x, y, w, nx, ny, kappa, arc_centers, theta_ranges, line_segments


def _rotate_xy_n(x, y, nx, ny, angle, center):
    if angle == 0.0:
        return x, y, nx, ny
    cx, cy = center
    c, s = np.cos(angle), np.sin(angle)
    # rotate points about center
    X = x - cx
    Y = y - cy
    xr = c * X - s * Y + cx
    yr = s * X + c * Y + cy
    # rotate normals (no translation)
    nxr = c * nx - s * ny
    nyr = s * nx + c * ny
    return xr, yr, nxr, nyr


def Get_Circ_Setup(R, density, center=(0.0, 0.0)):
    theta = np.linspace(0, 2 * np.pi, int(density * 2 * np.pi * R), endpoint=False)
    x = center[0] + R * np.cos(theta)
    y = center[1] + R * np.sin(theta)
    w = np.ones_like(x) * 2 * np.pi * R / len(x)
    nx = np.cos(theta)
    ny = np.sin(theta)
    kappa = np.ones_like(x) / R
    return x, y, w, nx, ny, kappa


safedis = -0.01
high_order = 6

density = 90
local_density = 407
R_wedge = 0.1
center1 = np.array([1, 0])
center2 = np.array([3, -1])
center3 = np.array([0, 0])
center4 = np.array([-1.8, -1.8])
center5 = np.array([1.2, 3.0])
center8 = np.array([3, 1.7])
center9 = np.array([-3.5, -1])
center10 = np.array([-3.5, 1])
center11 = np.array([-2, 3])
center12 = np.array([-0.3, -4])
R1 = 0.6
R2 = 0.5
R3 = 5
R4 = 0.7
R5 = 1
R8 = 0.7
R9 = 0.5
R10 = 0.3
R11 = 0.4
R12 = 0.3
x1, y1, w1, nx1, ny1, kappa1 = Get_Circ_Setup(R1, density, center=center1)
x2, y2, w2, nx2, ny2, kappa2 = Get_Circ_Setup(R2, density, center=center2)
x3, y3, w3, nx3, ny3, kappa3 = Get_Circ_Setup(R3, density, center=center3)
x4, y4, w4, nx4, ny4, kappa4 = Get_Circ_Setup(R4, density, center=center4)
x5, y5, w5, nx5, ny5, kappa5 = Get_Circ_Setup(R5, density, center=center5)
x8, y8, w8, nx8, ny8, kappa8 = Get_Circ_Setup(R8, density, center=center8)
x9, y9, w9, nx9, ny9, kappa9 = Get_Circ_Setup(R9, density, center=center9)
x10, y10, w10, nx10, ny10, kappa10 = Get_Circ_Setup(R10, density, center=center10)
x11, y11, w11, nx11, ny11, kappa11 = Get_Circ_Setup(R11, density, center=center11)
x12, y12, w12, nx12, ny12, kappa12 = Get_Circ_Setup(R12, density, center=center12)
shift_from_origin1 = np.array([-1, 1.5])
L = 0.7
corner = np.array([[-L, -L], [L, -L], [L, L], [-L, L]]) + shift_from_origin1
x7, y7, w7, nx7, ny7, kappa7, arc_centers1, theta_ranges1, segment1 = (
    rounded_box_quadrature(corner, R_wedge, density, local_density, -3 * np.pi / 10)
)

shift_from_origin2 = np.array([1.5, -2.6])
L = 0.6
corner = (
    np.array([[-L, -1.5 * L], [L, -1.5 * L], [L, 1.5 * L], [-L, 1.5 * L]])
    + shift_from_origin2
)
x6, y6, w6, nx6, ny6, kappa6, arc_centers2, theta_ranges2, segment2 = (
    rounded_box_quadrature(corner, R_wedge, density, local_density, 5 * np.pi / 6)
)
print(theta_ranges1, theta_ranges2)
# input("pause")
# x3, y3, n_out, n_in, kappa3, w3 = smoothstar_geometry(
#     a=0.3, w=5, center=(2.0, 3.0), R0=0.5, density=density
# )
# --- Combine into one global set ---
x_coord = np.concatenate([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12])
y_coord = np.concatenate([y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12])
weights = np.concatenate([w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12])
normal_x = np.concatenate(
    [nx1, nx2, nx3, nx4, nx5, nx6, nx7, nx8, nx9, nx10, nx11, nx12]
)
normal_y = np.concatenate(
    [ny1, ny2, ny3, ny4, ny5, ny6, ny7, ny8, ny9, ny10, ny11, ny12]
)
kappas = np.concatenate(
    [
        kappa1,
        kappa2,
        kappa3,
        kappa4,
        kappa5,
        kappa6,
        kappa7,
        kappa8,
        kappa9,
        kappa10,
        kappa11,
        kappa12,
    ]
)
N1 = len(x1)
N2 = len(x2)
N3 = len(x3)
N4 = len(x4)
N5 = len(x5)
N6 = len(x6)
N7 = len(x7)
N8 = len(x8)
N9 = len(x9)
N10 = len(x10)
N11 = len(x11)
N12 = len(x12)
I1 = np.arange(0, N1)  # indices of inner hole 1
I2 = np.arange(N1, N1 + N2)  # indices of inner hole 2
I3 = np.arange(N1 + N2, N1 + N2 + N3)  # indices of outer big circle
I4 = np.arange(N1 + N2 + N3, N1 + N2 + N3 + N4)  # indices of inner hole 3
I5 = np.arange(N1 + N2 + N3 + N4, N1 + N2 + N3 + N4 + N5)  # indices of rounded box
I6 = np.arange(
    N1 + N2 + N3 + N4 + N5, N1 + N2 + N3 + N4 + N5 + N6
)  # indices of rounded box 2
I7 = np.arange(N1 + N2 + N3 + N4 + N5 + N6, N1 + N2 + N3 + N4 + N5 + N6 + N7)
I8 = np.arange(N1 + N2 + N3 + N4 + N5 + N6 + N7, N1 + N2 + N3 + N4 + N5 + N6 + N7 + N8)
I9 = np.arange(
    N1 + N2 + N3 + N4 + N5 + N6 + N7 + N8,
    N1 + N2 + N3 + N4 + N5 + N6 + N7 + N8 + N9,
)
I10 = np.arange(
    N1 + N2 + N3 + N4 + N5 + N6 + N7 + N8 + N9,
    N1 + N2 + N3 + N4 + N5 + N6 + N7 + N8 + N9 + N10,
)
I11 = np.arange(
    N1 + N2 + N3 + N4 + N5 + N6 + N7 + N8 + N9 + N10,
    N1 + N2 + N3 + N4 + N5 + N6 + N7 + N8 + N9 + N10 + N11,
)
I12 = np.arange(N1 + N2 + N3 + N4 + N5 + N6 + N7 + N8 + N9 + N10 + N11, len(x_coord))
sign_half = np.zeros_like(x_coord)
# small inner circles → exterior Neumann limit (+1/2)
sign_half[I1] = -0.5
sign_half[I2] = -0.5
sign_half[I4] = -0.5
sign_half[I5] = -0.5
sign_half[I6] = -0.5
sign_half[I7] = -0.5
sign_half[I8] = -0.5
sign_half[I9] = -0.5
sign_half[I10] = -0.5
sign_half[I11] = -0.5
sign_half[I12] = -0.5
# outer boundary → interior Neumann limit (−1/2)
sign_half[I3] = +0.5

print("Total number of singularities:", len(x_coord))


obstacles = []

# 1) A rounded box you built earlier
obstacles.append(
    {
        "type": "rounded_box",
        "line_segments": segment1,  # (4,2,2)
        "arc_centers": arc_centers1,  # (4,2)
        "theta_ranges": theta_ranges1,  # (4,2) AFTER rotation
        "R": R_wedge,
    }
)
obstacles.append(
    {
        "type": "rounded_box",
        "line_segments": segment2,  # (4,2,2)
        "arc_centers": arc_centers2,  # (4,2)
        "theta_ranges": theta_ranges2,  # (4,2) AFTER rotation
        "R": R_wedge,
    }
)

# 2) A circle (repeat for as many as you have)
obstacles.append({"type": "circle", "center": center1, "radius": R1})
obstacles.append({"type": "circle", "center": center2, "radius": R2})
obstacles.append({"type": "circle", "center": center3, "radius": R3})
obstacles.append({"type": "circle", "center": center4, "radius": R4})
obstacles.append({"type": "circle", "center": center5, "radius": R5})
obstacles.append({"type": "circle", "center": center8, "radius": R8})
obstacles.append({"type": "circle", "center": center9, "radius": R9})
obstacles.append({"type": "circle", "center": center10, "radius": R10})
obstacles.append({"type": "circle", "center": center11, "radius": R11})
obstacles.append({"type": "circle", "center": center12, "radius": R12})


def kernel(index_source, index_target):
    # kernel function
    if index_source != index_target:

        x1, y1 = x_coord[index_target], y_coord[index_target]
        x2, y2 = x_coord[index_source], y_coord[index_source]

        nx, ny = normal_x[index_source], normal_y[index_source]  # outward normal
        dx = x1 - x2  # target-source
        dy = y1 - y2
        rsquare = np.square(dx) + np.square(dy)
        return 1 / 2 / np.pi / rsquare * (nx * (dx) + ny * (dy))
    else:
        # self evaluation
        return -kappas[index_source] / (4 * np.pi)


N_singular = len(x_coord)
# calculate A and f matrix
current_path = os.getcwd()
load_dir = current_path + "/LU_py"
print(load_dir)
lufilename = load_dir + "/lu_R_{}.npy".format(int(R3))
pivfilename = load_dir + "/piv_R_{}.npy".format(int(R3))
if os.path.isfile(lufilename) and os.path.isfile(pivfilename):
    print("Load LU")
    lu = np.load(lufilename)
    piv = np.load(pivfilename)

else:
    print("Compute LU")
    A = np.zeros((N_singular, N_singular), dtype=np.float64)
    for i in range(len(x_coord)):
        for j in range(len(x_coord)):
            # source and target are inversed for interiour neuman problem
            A[i, j] = (
                (kernel(i, j) + 1) * weights[j]
                if sign_half[i] == 0.5
                else (kernel(i, j)) * weights[j]
            )
        A[i, i] += sign_half[i]
    lu, piv = linalg.lu_factor(A)
    np.save(lufilename, lu)
    np.save(pivfilename, piv)
    print("LU computed")


geo_file_name = (
    current_path + "/geo_data/BEM_geometry_complex_enviornment_R{}.mat".format(R3)
)
if not os.path.isfile(geo_file_name):
    normal_vec = np.stack((normal_x, normal_y), axis=-1)
    weights_mat = weights.reshape(-1, 1)
    kappas_mat = kappas.reshape(-1, 1)
    x_coord_mat = x_coord.reshape(-1, 1)
    y_coord_mat = y_coord.reshape(-1, 1)
    scipy.io.savemat(
        geo_file_name,
        mdict={
            "normal_vec": normal_vec,
            "Sx": x_coord_mat,
            "Sy": y_coord_mat,
            "Nper": np.array([N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11, N12]),
            "weights": weights_mat,
            "kappas": kappas_mat,
            "R": R3,
        },
    )


def cal_signed_distance_function(loc):

    dis_list = []
    have_list = []

    def _distance_to_circle(
        loc, center, radius, th_min, th_max, inner=False, debug=False, across=False
    ):
        have = True
        x0, y0 = loc - center
        th = np.arctan2(y0, x0)
        if debug:
            print(th, th_min, th_max)
            # input()
        if across:
            if th > th_min and th < th_max:
                have = False
                return 1000, have
            if inner:
                dis = radius - np.sqrt(x0**2 + y0**2)
            else:
                dis = np.sqrt(x0**2 + y0**2) - radius
        else:
            if th < th_min or th > th_max:
                have = False
                return 1000, have
            if inner:
                dis = radius - np.sqrt(x0**2 + y0**2)
            else:
                dis = np.sqrt(x0**2 + y0**2) - radius

        return dis, have

    def _edge_outward_normal(a, b, center):
        """Unit outward normal for edge [a,b] using the shape center."""
        v = b - a
        L = np.hypot(v[0], v[1])
        if L == 0.0:
            n = 0.5 * (a + b) - center
            n /= np.hypot(n[0], n[1]) + 1e-15
            return n
        t = v / L
        n = np.array([-t[1], t[0]])  # CCW normal
        mid = 0.5 * (a + b)
        if np.dot(n, mid - center) < 0:  # flip to be outward-from-interior
            n = -n
        return n

    def _line_distance_if_projected(p, a, b, n_out):
        """
        Distance to the infinite line (| (p-a)·n_out |) but ONLY if the projection
        of p onto the line lies on the segment interior; otherwise return +inf.
        """
        v = b - a
        vv = np.dot(v, v)
        if vv == 0.0:
            return np.inf  # let the arc (corner) handle the endpoint
        u = np.dot(p - a, v) / vv
        if 0.0 <= u <= 1.0:
            return abs(np.dot(p - a, n_out))
        return np.inf

    def signed_distance_rounded_box(
        loc, line_segments, arc_centers, R, center, inner=True
    ):
        """
        Signed distance to a rotated rounded rectangle (fillet radius R),
        using outputs from your rounded_box_quadrature.

        - line_segments: (4,2,2)  -> straight edges: [bottom, right, top, left]
        - arc_centers:   (4,2)    -> quarter-arc centers (order doesn't matter here)
        - R: fillet radius
        - center: same center used to construct/rotate the shape
        - inner: True => +inside / -outside   (matches your circle SDF)

        Returns: dis, have   (have=True)
        """
        p = np.asarray(loc, float)
        segs = np.asarray(line_segments, float)
        acs = np.asarray(arc_centers, float)
        c = np.asarray(center, float)

        # ---- 1) candidate unsigned distances: edges (only if projected inside) ----
        d_candidates = []
        for k in range(4):
            a, b = segs[k, 0], segs[k, 1]
            n_out = _edge_outward_normal(a, b, c)
            d_line = _line_distance_if_projected(p, a, b, n_out)
            d_candidates.append(d_line)
        d_min = np.min(d_candidates)

        # ---- 3) inside test: all edge half-planes OR inside any corner disk ----
        inside_edges = True
        for k in range(4):
            a, b = segs[k, 0], segs[k, 1]
            n_out = _edge_outward_normal(a, b, c)
            if np.dot(p - a, n_out) > 1e-12:  # outside that edge’s half-plane
                inside_edges = False
                break
        inside_corners = np.any(np.hypot(*(p - acs).T) <= R + 1e-12)
        is_inside = inside_edges or inside_corners

        # ---- 4) signed distance with your convention ----
        sgn = +1.0 if inner else -1.0
        dis = d_min * (sgn if is_inside else -sgn)

        return float(dis), True

    debug = False
    dis1, have_first = _distance_to_circle(
        loc, center1, R1, -np.pi, np.pi, True, debug, False
    )
    dis_list.append(dis1)
    have_list.append(have_first)

    debug = False
    dis2, have_second = _distance_to_circle(
        loc, center2, R2, -np.pi, np.pi, True, debug, False
    )
    dis_list.append(dis2)
    have_list.append(have_second)

    debug = False
    dis3, have_third = _distance_to_circle(
        loc, center3, R3, -np.pi, np.pi, False, debug, False
    )
    dis_list.append(dis3)
    have_list.append(have_third)
    debug = False
    dis4, have_fourth = _distance_to_circle(
        loc, center4, R4, -np.pi, np.pi, True, debug, False
    )
    dis_list.append(dis4)
    have_list.append(have_fourth)
    debug = False
    dis5, have_fifth = _distance_to_circle(
        loc, center5, R5, -np.pi, np.pi, True, debug, False
    )
    dis_list.append(dis5)
    have_list.append(have_fifth)
    debug = False
    dis8, have_eighth = _distance_to_circle(
        loc, center8, R8, -np.pi, np.pi, True, debug, False
    )
    dis_list.append(dis8)
    have_list.append(have_eighth)
    debug = False
    dis9, have_ninth = _distance_to_circle(
        loc, center9, R9, -np.pi, np.pi, True, debug, False
    )
    dis_list.append(dis9)
    have_list.append(have_ninth)
    debug = False
    dis10, have_tenth = _distance_to_circle(
        loc, center10, R10, -np.pi, np.pi, True, debug, False
    )
    dis_list.append(dis10)
    have_list.append(have_tenth)
    debug = False
    dis11, have_eleventh = _distance_to_circle(
        loc, center11, R11, -np.pi, np.pi, True, debug, False
    )
    dis_list.append(dis11)
    have_list.append(have_eleventh)
    debug = False
    dis12, have_twelfth = _distance_to_circle(
        loc, center12, R12, -np.pi, np.pi, True, debug, False
    )
    dis_list.append(dis12)
    have_list.append(have_twelfth)
    # debug = False
    ########################### Rounded box 1 ###########################
    dis13, have_thirteen = _distance_to_circle(
        loc,
        arc_centers1[0],
        R_wedge,
        theta_ranges1[0, 0],
        theta_ranges1[0, 1],
        True,
        False,
        False,
    )
    dis_list.append(dis13)
    have_list.append(have_thirteen)
    dis14, have_fourteen = _distance_to_circle(
        loc,
        arc_centers1[1],
        R_wedge,
        theta_ranges1[1, 0],
        theta_ranges1[1, 1],
        True,
        False,
        False,
    )
    dis_list.append(dis14)
    have_list.append(have_fourteen)
    dis15, have_fifteen = _distance_to_circle(
        loc,
        arc_centers1[2],
        R_wedge,
        theta_ranges1[2, 0],
        theta_ranges1[2, 1],
        True,
        False,
        False,
    )
    dis_list.append(dis15)
    have_list.append(have_fifteen)
    dis16, have_sixteen = _distance_to_circle(
        loc,
        arc_centers1[3],
        R_wedge,
        theta_ranges1[3, 0],
        theta_ranges1[3, 1],
        True,
        False,
        True,
    )
    dis_list.append(dis16)
    have_list.append(have_sixteen)
    dis17, have_seventeenth = signed_distance_rounded_box(
        loc,
        segment1,
        arc_centers1,
        R_wedge,
        shift_from_origin1,
        inner=True,
    )
    dis_list.append(dis17)
    have_list.append(have_seventeenth)

    ########################################################################

    ########################### Rounded box 2 ###########################
    dis18, have_eighteen = _distance_to_circle(
        loc,
        arc_centers2[0],
        R_wedge,
        theta_ranges2[0, 0],
        theta_ranges2[0, 1],
        True,
        False,
        False,
    )
    dis_list.append(dis18)
    have_list.append(have_eighteen)
    dis19, have_nineteen = _distance_to_circle(
        loc,
        arc_centers2[1],
        R_wedge,
        theta_ranges2[1, 0],
        theta_ranges2[1, 1],
        True,
        False,
        True,
    )
    dis_list.append(dis19)
    have_list.append(have_nineteen)
    dis20, have_twenty = _distance_to_circle(
        loc,
        arc_centers2[2],
        R_wedge,
        theta_ranges2[2, 0],
        theta_ranges2[2, 1],
        True,
        False,
        False,
    )
    dis_list.append(dis20)
    have_list.append(have_twenty)
    dis21, have_twentyone = _distance_to_circle(
        loc,
        arc_centers2[3],
        R_wedge,
        theta_ranges2[3, 0],
        theta_ranges2[3, 1],
        True,
        False,
        False,
    )
    dis_list.append(dis21)
    have_list.append(have_twentyone)
    dis22, have_twentytwo = signed_distance_rounded_box(
        loc,
        segment2,
        arc_centers2,
        R_wedge,
        shift_from_origin2,
        inner=True,
    )
    dis_list.append(dis22)
    have_list.append(have_twentytwo)
    ########################################################################
    dis_list = np.array(dis_list)
    have_list = np.array(have_list)

    dis_list = dis_list[have_list]

    if dis_list.size == 0:
        print("No distance")
        # print(loc)
        # _distance_to_circle(loc, circle_center6, r, -np.pi, 0, True, True)
        return 1000
    else:
        return dis_list[np.argmin(np.abs(dis_list))]


# Function for Result Analysis
def Reshape(
    A, type, style
):  # input style here is the shape of each slice, for example, 10*3*501 is the shape of original form of the matrix
    # type 1 is to reshape a 3d matrix into 2d matrix by stacking each slice
    if type == 1:
        B = A.transpose(2, 0, 1).reshape(-1, A.shape[1])
    # type 2 is to reshape a 10*n rows and k columns matrix into a 10 rows and k*n columns by stacking by columns
    if type == 2:
        B = A.reshape(np.roll((style[0], style[1], style[2]), 1)).transpose(1, 2, 0)
        B = B.transpose(0, 2, 1).reshape(10, int(style[1] * style[2]))
    # inverse of type 2
    if type == 3:
        temp = np.split(A, style[2], axis=1)
        B = np.vstack(temp)
    if type == 4:
        temp = np.split(A, style[2], axis=0)
        B = np.vstack(temp)
    return B


def GetInterc(p):

    loc = p[0:2]
    orientation = p[2]

    dis_list = []
    th_list = []
    have_list = []
    # calculate distance to large circle (1-2 and 7-8)
    x0, y0 = loc
    th0 = orientation

    # solves for t (len_to_intersect) in the parametric equations of the ray and the circle's equation
    # to find the point(s) of intersection.
    # find t (len_to_intersect) using the discriminant of the quadratic equation
    # the input th_min and th_max is in the range of [-pi,pi]
    # across true means that the interval is across pi, otherwise, it is not across pi
    def _distance_to_segment(
        loc, orientation, pA, pB, body_center=None, debug=False, swimmer_outside=True
    ):
        """
        Ray (loc, orientation) vs finite line segment [pA, pB] (all in GLOBAL coords).

        Returns:
        dis, th, have_
            dis:   if have_: forward ray distance t to intersection (>=0)
                else:    Euclidean distance from loc to closest point on the segment
            th:    outward wall-normal angle at the hit/closest point, in (-pi, pi]
            have_: True if ray intersects the segment with t>=0 and segment-param u∈[0,1]
        Notes:
        - 'body_center' is optional; when given, the outward normal is chosen to
            point *away* from the body center (useful for walls of a box).
        - For vertical walls you'll naturally get th ∈ {0, π}; for horizontal, th ∈ {±π/2}.
        """
        loc = np.asarray(loc, float)
        pA = np.asarray(pA, float)
        pB = np.asarray(pB, float)

        d = np.array(
            [np.cos(orientation), np.sin(orientation)], float
        )  # ray dir (unit)
        v = pB - pA  # segment vector
        vv = np.dot(v, v)
        eps = 1e-14

        # Solve pA + u v = loc + t d  ⇒  [d, -v] [t; u] = pA - loc
        rhs = pA - loc
        det = d[0] * (-v[1]) - d[1] * (-v[0])  # = -(d × v) (scalar 2D cross)

        have_ = False
        if abs(det) > eps:
            # Solve 2x2 by Cramer's rule (fast, no conditionals)
            t = (rhs[0] * (-v[1]) - rhs[1] * (-v[0])) / det
            u = (d[0] * rhs[1] - d[1] * rhs[0]) / det
            if debug:
                print(f"t={t:.6g}, u={u:.6g}")

            if t >= 0.0 and -eps <= u <= 1.0 + eps:
                # valid intersection (allow tiny tolerance on u)
                have_ = True
                hit = pA + np.clip(u, 0.0, 1.0) * v
                dis = float(t)
            else:
                have_ = False

        if not have_:
            # No intersection: closest point on segment to ray origin
            if vv < eps:
                # degenerate segment: treat as a point
                closest = pA
            else:
                u0 = np.dot(loc - pA, v) / vv
                u0 = np.clip(u0, 0.0, 1.0)
                closest = pA + u0 * v
            dis = float(np.hypot(*(closest - loc)))
            hit = closest

        # Outward normal at the hit/closest point:
        # Choose the CCW left normal of the segment then flip if needed.
        if vv < eps:
            # degenerate: fall back to a sensible normal (point away from center if provided)
            if body_center is not None:
                n = hit - body_center
                n_norm = np.hypot(n[0], n[1])
                n = n / (n_norm + eps)
            else:
                # arbitrary
                n = np.array([0.0, 1.0])
        else:
            tseg = v / np.sqrt(vv)  # unit tangent
            n = np.array([-tseg[1], tseg[0]])  # CCW normal
            if body_center is not None:
                # ensure normal points outward (away from body center)
                mid = 0.5 * (pA + pB)
                out_dir = mid - body_center
                if np.dot(n, out_dir) < 0:
                    n = -n

        # Ensure normal faces the swimmer (ray origin) if desired
        if swimmer_outside:
            # to_swimmer = loc - hit
            # if np.dot(n, to_swimmer) > 0:    swimmer_outside=False,       # NEW: True -> return inward normal; False -> outward
            n = -n  # change to inward normal
        # ------------------------------------------------------------

        th = _wrap(np.arctan2(n[1], n[0]))
        return dis, th, have_

    def _distance_to_circle(
        loc, center, radius, th_min, th_max, debug=False, across=False, inner=False
    ):
        def check_root(t, ori, th_min, th_max, across):
            have = True
            x = x0 + t * np.cos(ori)
            y = y0 + t * np.sin(ori)
            th = np.arctan2(y, x)
            if t < 0:
                have = False
                th = 0
                t = 1000
                return t, th, have
            if across:
                if th > th_min and th < th_max:
                    have = False
                    th = 0
                    t = 1000
                return t, th, have
            else:
                if th < th_min or th > th_max:
                    have = False
                    th = 0
                    t = 1000
                return t, th, have

        if debug:
            print("begin")
        have = True

        x0, y0 = loc - center
        th0 = orientation
        delta = (x0 * np.cos(th0) + y0 * np.sin(th0)) ** 2 - (x0**2 + y0**2 - radius**2)
        if debug:
            print(x0, y0, th0, delta)
        if delta < 0:
            have = False
            return 1000, 0, have  # dis,th,have
        else:
            len_to_intersect_1 = -(x0 * np.cos(th0) + y0 * np.sin(th0)) - np.sqrt(delta)
            len_to_intersect_2 = -(x0 * np.cos(th0) + y0 * np.sin(th0)) + np.sqrt(delta)
            if debug:
                print(len_to_intersect_1, len_to_intersect_2)
            t_1, theta_1, have_1 = check_root(
                len_to_intersect_1, th0, th_min, th_max, across
            )
            t_2, theta_2, have_2 = check_root(
                len_to_intersect_2, th0, th_min, th_max, across
            )
            if have_1 == True and have_2 == False:
                if inner:
                    theta_1 += np.pi
                return t_1, theta_1, have_1
            elif have_1 == False and have_2 == True:
                if inner:
                    theta_2 += np.pi
                return t_2, theta_2, have_2
            elif have_1 == True and have_2 == True:
                if t_1 < t_2:
                    if inner:
                        theta_1 += np.pi
                    return t_1, theta_1, have_1
                else:
                    if inner:
                        theta_2 += np.pi
                    return t_2, theta_2, have_2
            else:
                return 1000, 0, have

    # calculate distance to the wedge 1 (node1-node2)
    debug = False
    dis1, th_1, have_first = _distance_to_circle(
        loc, center1, R1, -np.pi, np.pi, debug, False, True
    )
    dis_list.append(dis1)
    th_list.append(th_1)
    have_list.append(have_first)

    # calculate distance to the left circle (node2-node3)
    debug = False
    dis2, th_2, have_second = _distance_to_circle(
        loc, center2, R2, -np.pi, np.pi, debug, False, True
    )
    # print(have_fifth)
    dis_list.append(dis2)
    th_list.append(th_2)
    have_list.append(have_second)

    # calculate distance to the wedge 3 (node3-node4)
    debug = False
    dis3, th_3, have_third = _distance_to_circle(
        loc, center3, R3, -np.pi, np.pi, debug, False, False
    )
    dis_list.append(dis3)
    th_list.append(th_3)
    have_list.append(have_third)

    debug = False
    dis4, th_4, have_forth = _distance_to_circle(
        loc, center4, R4, -np.pi, np.pi, debug, False, True
    )
    dis_list.append(dis4)
    th_list.append(th_4)
    have_list.append(have_forth)

    debug = False
    dis5, th_5, have_fifth = _distance_to_circle(
        loc, center5, R5, -np.pi, np.pi, debug, False, True
    )
    dis_list.append(dis5)
    th_list.append(th_5)
    have_list.append(have_fifth)

    ######## square 1 ###########
    debug = False
    dis6, th_6, have_sixth = _distance_to_circle(
        loc,
        arc_centers1[0],
        R_wedge,
        theta_ranges1[0, 0],
        theta_ranges1[0, 1],
        debug,
        False,
        True,
    )
    dis_list.append(dis6)
    th_list.append(th_6)
    have_list.append(have_sixth)

    debug = False
    dis7, th_7, have_seventh = _distance_to_circle(
        loc,
        arc_centers1[1],
        R_wedge,
        theta_ranges1[1, 0],
        theta_ranges1[1, 1],
        debug,
        False,
        True,
    )
    dis_list.append(dis7)
    th_list.append(th_7)
    have_list.append(have_seventh)

    debug = False
    dis8, th_8, have_eighth = _distance_to_circle(
        loc,
        arc_centers1[2],
        R_wedge,
        theta_ranges1[2, 0],
        theta_ranges1[2, 1],
        debug,
        False,
        True,
    )
    dis_list.append(dis8)
    th_list.append(th_8)
    have_list.append(have_eighth)
    debug = False
    dis9, th_9, have_ninth = _distance_to_circle(
        loc,
        arc_centers1[3],
        R_wedge,
        theta_ranges1[3, 0],
        theta_ranges1[3, 1],
        debug,
        True,
        True,
    )
    dis_list.append(dis9)
    th_list.append(th_9)
    have_list.append(have_ninth)

    (seg_bottom_A, seg_bottom_B) = segment1[0]
    (seg_right_A, seg_right_B) = segment1[1]
    (seg_top_A, seg_top_B) = segment1[2]
    (seg_left_A, seg_left_B) = segment1[3]
    # bottom wall (replaces your "horizontal bottom (node4-node5)" block)
    dis14, th_14, have_fortheen = _distance_to_segment(
        loc=np.array([x0, y0]),
        orientation=th0,
        pA=seg_bottom_A,
        pB=seg_bottom_B,
        body_center=shift_from_origin1,  # ensures outward normal (downward for bottom)
        debug=False,
        swimmer_outside=True,
    )
    dis_list.append(dis14)
    th_list.append(th_14)
    have_list.append(have_fortheen)

    # right wall (vertical)
    dis15, th_15, have_fiftheen = _distance_to_segment(
        loc=np.array([x0, y0]),
        orientation=th0,
        pA=seg_right_A,
        pB=seg_right_B,
        body_center=shift_from_origin1,
        swimmer_outside=True,
    )
    dis_list.append(dis15)
    th_list.append(th_15)
    have_list.append(have_fiftheen)

    # top wall
    dis16, th_16, have_sixtheen = _distance_to_segment(
        loc=np.array([x0, y0]),
        orientation=th0,
        pA=seg_top_A,
        pB=seg_top_B,
        body_center=shift_from_origin1,
        swimmer_outside=True,
    )
    dis_list.append(dis16)
    th_list.append(th_16)
    have_list.append(have_sixtheen)
    # left wall
    dis17, th_17, have_seventeenth = _distance_to_segment(
        loc=np.array([x0, y0]),
        orientation=th0,
        pA=seg_left_A,
        pB=seg_left_B,
        body_center=shift_from_origin1,
        swimmer_outside=True,
    )
    dis_list.append(dis17)
    th_list.append(th_17)
    have_list.append(have_seventeenth)

    # (similarly for top/left if you need them)

    ######## square 2 ###########
    debug = False
    dis10, th_10, have_tenth = _distance_to_circle(
        loc,
        arc_centers2[0],
        R_wedge,
        theta_ranges2[0, 0],
        theta_ranges2[0, 1],
        debug,
        False,
        True,
    )
    dis_list.append(dis10)
    th_list.append(th_10)
    have_list.append(have_tenth)
    debug = False
    dis11, th_11, have_eleventh = _distance_to_circle(
        loc,
        arc_centers2[1],
        R_wedge,
        theta_ranges2[1, 0],
        theta_ranges2[1, 1],
        debug,
        True,
        True,
    )
    dis_list.append(dis11)
    th_list.append(th_11)
    have_list.append(have_eleventh)
    debug = False
    dis12, th_12, have_twelfth = _distance_to_circle(
        loc,
        arc_centers2[2],
        R_wedge,
        theta_ranges2[2, 0],
        theta_ranges2[2, 1],
        debug,
        False,
        True,
    )
    dis_list.append(dis12)
    th_list.append(th_12)

    have_list.append(have_twelfth)
    debug = False
    dis13, th_13, have_thirteenth = _distance_to_circle(
        loc,
        arc_centers2[3],
        R_wedge,
        theta_ranges2[3, 0],
        theta_ranges2[3, 1],
        debug,
        False,
        True,
    )
    dis_list.append(dis13)
    th_list.append(th_13)
    have_list.append(have_thirteenth)

    (seg2_bottom_A, seg2_bottom_B) = segment2[0]
    (seg2_right_A, seg2_right_B) = segment2[1]
    (seg2_top_A, seg2_top_B) = segment2[2]
    (seg2_left_A, seg2_left_B) = segment2[3]
    # bottom wall (replaces your "horizontal bottom (node4-node5)" block)
    dis18, th_18, have_eighteen = _distance_to_segment(
        loc=np.array([x0, y0]),
        orientation=th0,
        pA=seg2_bottom_A,
        pB=seg2_bottom_B,
        body_center=shift_from_origin2,  # ensures outward normal (downward for bottom)
        debug=False,
        swimmer_outside=True,
    )
    dis_list.append(dis18)
    th_list.append(th_18)
    have_list.append(have_eighteen)
    # right wall (vertical)
    dis19, th_19, have_nineteen = _distance_to_segment(
        loc=np.array([x0, y0]),
        orientation=th0,
        pA=seg2_right_A,
        pB=seg2_right_B,
        body_center=shift_from_origin2,
        swimmer_outside=True,
    )
    dis_list.append(dis19)
    th_list.append(th_19)
    have_list.append(have_nineteen)
    # top wall
    dis20, th_20, have_twenty = _distance_to_segment(
        loc=np.array([x0, y0]),
        orientation=th0,
        pA=seg2_top_A,
        pB=seg2_top_B,
        body_center=shift_from_origin2,
        swimmer_outside=True,
    )
    dis_list.append(dis20)
    th_list.append(th_20)
    have_list.append(have_twenty)
    # left wall
    dis21, th_21, have_twentyone = _distance_to_segment(
        loc=np.array([x0, y0]),
        orientation=th0,
        pA=seg2_left_A,
        pB=seg2_left_B,
        body_center=shift_from_origin2,
        swimmer_outside=True,
    )
    dis_list.append(dis21)
    th_list.append(th_21)
    have_list.append(have_twentyone)
    ############################ other circles ###########################
    debug = False
    dis22, th_22, have_twotwo = _distance_to_circle(
        loc, center8, R8, -np.pi, np.pi, debug, False, True
    )
    dis_list.append(dis22)
    th_list.append(th_22)
    have_list.append(have_twotwo)
    debug = False
    dis23, th_23, have_twothree = _distance_to_circle(
        loc, center9, R9, -np.pi, np.pi, debug, False, True
    )
    dis_list.append(dis23)
    th_list.append(th_23)
    have_list.append(have_twothree)
    debug = False
    dis24, th_24, have_twofour = _distance_to_circle(
        loc, center10, R10, -np.pi, np.pi, debug, False, True
    )
    dis_list.append(dis24)
    th_list.append(th_24)
    have_list.append(have_twofour)
    debug = False
    dis25, th_25, have_twofive = _distance_to_circle(
        loc, center11, R11, -np.pi, np.pi, debug, False, True
    )
    dis_list.append(dis25)
    th_list.append(th_25)
    have_list.append(have_twofive)
    debug = False
    dis26, th_26, have_twosix = _distance_to_circle(
        loc, center12, R12, -np.pi, np.pi, debug, False, True
    )
    dis_list.append(dis26)
    th_list.append(th_26)
    have_list.append(have_twosix)

    #######################################################
    # return the minimum distance and the corresponding angle
    dis_list = np.array(dis_list)
    th_list = np.array(th_list)
    have_list = np.array(have_list)
    # print(dis_list, th_list, have_list)
    # print(dis_list, th_list, have_list)
    dis_list = dis_list[have_list]
    th_list = th_list[have_list]
    # print(dis_list, th_list, have_list)

    if dis_list.size == 0:

        print("error: no wall", loc, orientation)
        return 0, 1000
    else:
        return th_list[np.argmin(dis_list)], np.min(dis_list)


# def GetVoronoi(location, para):
#     tri = Delaunay(np.transpose(location))
#     v1 = np.ndarray.flatten(tri.simplices)
#     v2 = np.ndarray.flatten(tri.simplices[:, [1, 2, 0]])
#     vn = np.zeros((para["num"], para["num"]))
#     vn[v1, v2] = 1
#     vn = np.logical_or(vn, vn.T)
#     return vn


# def ccw(A, B, C):
#     return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


# def intersect(A, B, C, D):
#     return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


# def check_inersect_circle(loc1, loc2, center, radius):
#     d = loc2 - loc1
#     f = loc1 - center
#     a = np.dot(d, d)
#     b = 2 * np.dot(f, d)
#     c = np.dot(f, f) - radius**2
#     discriminant = b**2 - 4 * a * c
#     if discriminant < 0:
#         return False
#     discriminant = np.sqrt(discriminant)
#     t1 = (-b - discriminant) / (2 * a)
#     t2 = (-b + discriminant) / (2 * a)
#     if t1 >= 0 and t1 <= 1:
#         return True
#     if t2 >= 0 and t2 <= 1:
#         return True
#     return False


# def check_intersections(segments, target_segment):
#     A, B = target_segment
#     intersections = [intersect(A, B, np.array(C), np.array(D)) for C, D in segments]
#     return any(intersections)


# def CheckVoronoi(vn, location, para):

#     for i in range(para["num"]):
#         for j in range(para["num"]):
#             if vn[i, j] == True:
#                 target_segment = [location[:, i], location[:, j]]
#                 if check_intersections(segments, target_segment):
#                     vn[i, j] = False
#                     continue
#                 if check_inersect_circle(
#                     location[:, i], location[:, j], circle_center6, r
#                 ):
#                     vn[i, j] = False
#                     continue
#                 if check_inersect_circle(
#                     location[:, i], location[:, j], circle_center5, r
#                 ):
#                     vn[i, j] = False
#                     continue
#                 # print(location[0][i], location[0][j], location[1][i], location[1][j])
#                 # print(len(xxx))
#     return vn


def getangle(phi, rhox, rhoy, rhom=None):
    if rhom is None:
        rhom = np.sqrt(rhox**2 + rhoy**2)
    rhox = rhox / rhom
    rhoy = rhoy / rhom
    ex = np.cos(phi)
    ey = np.sin(phi)
    sgn = np.array(np.sign(ex * rhoy - ey * rhox))
    sgn[sgn == 0] = 1
    return sgn * np.arccos(np.clip(ex * rhox + ey * rhoy, -1, 1))


# def GetVision(vn, state, para):
#     listI = np.tile(np.arange(0, para["num"]), (para["num"]))
#     ns = listI[np.ndarray.flatten(vn)]
#     nn = np.sum(vn, axis=0)
#     nnmax = np.amax(nn)

#     neighborI = np.zeros((para["num"], nnmax + 1))
#     neighborI[np.arange(0, para["num"]), nn] = para["num"]
#     neighborI = np.cumsum(neighborI[:, :-1], axis=1).astype(int)
#     neighborI[neighborI == 0] = ns
#     x = state[0][:]
#     y = state[1][:]
#     a = state[2][:]
#     xN = np.append(x, np.nan)
#     yN = np.append(y, np.nan)
#     aN = np.append(a, np.nan)
#     phi = aN[neighborI] - a[:, None]
#     rho1 = xN[neighborI] - x[:, None]
#     rho2 = yN[neighborI] - y[:, None]
#     rhon = np.sqrt(rho1**2 + rho2**2)
#     theta = getangle(a[:, None], rho1, rho2, rhon)

#     visual = 1 + np.cos(theta)
#     sum_visual = np.nansum(visual, axis=1)
#     # w_vision = (
#     #     np.nansum((para["Ip"] * np.sin(phi) + rhon * np.sin(theta)) * visual, axis=1)
#     #     / sum_visual
#     # )
#     w_vision = (
#         np.nansum(
#             (para["Ia"] * rhon * np.sin(theta)) * visual,
#             axis=1,
#         )
#         / sum_visual
#     )
#     if (sum_visual == 0).any():
#         print("no vision")
#         w_vision[sum_visual == 0] = 0

#     return w_vision


def GetVision(vn, state, para):
    num = para["num"] + 1
    listI = np.tile(np.arange(0, num), (num))
    ns = listI[np.ndarray.flatten(vn)]
    nn = np.sum(vn, axis=0)
    nnmax = np.amax(nn)

    neighborI = np.zeros((num, nnmax + 1))
    neighborI[np.arange(0, num), nn] = num
    neighborI = np.cumsum(neighborI[:, :-1], axis=1).astype(int)
    neighborI[neighborI == 0] = ns
    x = state[0][:]
    y = state[1][:]
    a = state[2][:]
    xN = np.append(x, np.nan)
    yN = np.append(y, np.nan)
    aN = np.append(a, np.nan)
    rho1 = xN[neighborI] - x[:, None]
    rho2 = yN[neighborI] - y[:, None]
    rhon = np.sqrt(rho1**2 + rho2**2)
    theta = getangle(a[:, None], rho1, rho2, rhon)

    visual = 1 + np.cos(theta)
    sum_visual = np.nansum(visual, axis=1)
    # w_vision = (
    #     np.nansum((para["Ip"] * np.sin(phi) + rhon * np.sin(theta)) * visual, axis=1)
    #     / sum_visual
    # )
    w_vision = (
        np.nansum(
            (para["Ia"] * rhon * np.sin(theta)) * visual,
            axis=1,
        )
        / sum_visual
    )
    if (sum_visual == 0).any():
        print("no vision")
        w_vision[sum_visual == 0] = 0

    return w_vision


def GetHydro(state, para):
    # generate index mat
    otherI = np.tile(np.arange(0, para["num"]), (para["num"], 1))
    otherI = np.delete(otherI, np.arange(0, otherI.size, para["num"] + 1)).reshape(
        (para["num"], para["num"] - 1)
    )
    # extract fish info
    # x = state[0][:]
    # y = state[1][:]
    ori = state[2][:]
    # computing hydro interactions
    # dZr = x[:, None] - x[otherI]
    # dZi = y[:, None] - y[otherI]
    dZr = state[0][:][:, None] - state[0][:][otherI]
    dZi = state[1][:][:, None] - state[1][:][otherI]
    dZ = dZr + 1j * dZi
    o_di = ori[otherI]
    Uc = (
        np.sum((np.exp(1j * o_di) / (dZ**2 + para["delta"] ** 2)), axis=1)
        * para["If"]
        / np.pi
    )
    wc = (
        np.sum(
            np.imag(
                np.exp(1j * (2 * ori[:, None] + o_di))
                * dZ
                / (dZ**2 + para["delta"] ** 2) ** 2
            ),
            axis=1,
        )
        * 2
        * para["If"]
        / np.pi
    )
    Ux = np.real(Uc)
    Uy = -np.imag(Uc)
    w = wc
    U = np.vstack([Ux, Uy])
    return U, w


def GetHydroBEM(state, para):
    # np.savetxt("state.txt", state)
    # define boundary

    dipole_z = state[0][:] + 1j * state[1][:]
    dipole_angle = state[2][:]
    dipole_strength = para["If"]
    xi, yi = x_coord[:, np.newaxis], y_coord[:, np.newaxis]
    nx, ny = normal_x[:, np.newaxis], normal_y[:, np.newaxis]
    dz = xi + 1j * yi - dipole_z
    w = (
        (dipole_strength)
        * np.exp(1j * dipole_angle)
        / (dz**2 + para["delta"] ** 2)
        / np.pi
    )
    f = np.sum((nx * np.real(w) - ny * np.imag(w)), axis=1)

    # print(np.sum(f * np.sqrt(arclength)))

    # sigma = linalg.solve(A, f)
    sigma = linalg.lu_solve((lu, piv), -f)

    # print("integral of f = ", np.sum(f * weights))
    # print("integral of sigma = ", np.sum(sigma * weights))

    # calculate the velocity and angular velocity of dipole
    U_BEM = np.zeros((2, para["num"]), dtype=np.float64)

    #  sigularities location - evaluate location
    dx = x_coord - state[0][:, np.newaxis]
    dy = y_coord - state[1][:, np.newaxis]
    dz = dx + 1j * dy

    w = sigma * weights / 2 / np.pi / dz
    U_BEM[0, :] = np.sum(np.real(w), axis=1)
    U_BEM[1, :] = -np.sum(np.imag(w), axis=1)

    dudx = sigma * weights / 2 / np.pi * ((dx**2 - dy**2) / (dx**2 + dy**2) ** 2)
    dudy = sigma * weights / 2 / np.pi * ((2 * dx * dy) / (dx**2 + dy**2) ** 2)
    dvdx = sigma * weights / 2 / np.pi * ((2 * dx * dy) / (dx**2 + dy**2) ** 2)
    dvdy = sigma * weights / 2 / np.pi * ((dy**2 - dx**2) / (dx**2 + dy**2) ** 2)

    dudx = np.sum(dudx, axis=1)
    dudy = np.sum(dudy, axis=1)
    dvdx = np.sum(dvdx, axis=1)
    dvdy = np.sum(dvdy, axis=1)
    w_BEM = -(
        np.sin(dipole_angle) * np.cos(dipole_angle) * (dudx - dvdy)
        + np.sin(dipole_angle) ** 2 * dvdx
        - np.cos(dipole_angle) ** 2 * dudy
    )

    #################### validation ####################
    # xmin = -5
    # xmax = 5
    # ymin = -5
    # ymax =
    # Nx = 300
    # Ny = 300
    # eval_angle = np.pi / 4
    # x = np.linspace(xmin, xmax, Nx, dtype=np.float64)
    # y = np.linspace(ymin, ymax, Ny, dtype=np.float64)
    # xmesh, ymesh = np.meshgrid(x, y)
    # umesh = np.zeros_like(xmesh, dtype=np.float64)
    # vmesh = np.zeros_like(xmesh, dtype=np.float64)
    # omega_mesh = np.zeros_like(xmesh, dtype=np.float64)
    # for dipole_index in range(para["num"]):
    #     dipole_z = state[0][dipole_index] + 1j * state[1][dipole_index]
    #     dipole_angle = state[2][dipole_index]
    #     dipole_strength = para["If"]
    #     dz = xmesh + 1j * ymesh - dipole_z
    #     dz = dz[:, :, np.newaxis]
    #     # wc = (
    #     #     (1j * np.exp(2 * 1j * eval_angle) * np.exp(1j * dipole_angle) / dz**3)
    #     #     * 2
    #     #     * dipole_strength
    #     #     / np.pi
    #     # )
    #     wc = np.imag(
    #         np.exp(1j * (2 * eval_angle + dipole_angle))
    #         / (dz**3)
    #         * 2
    #         * para["If"]
    #         / np.pi
    #     )
    #     w = 1 / dz / dz * (dipole_strength) * np.exp(1j * dipole_angle) / np.pi
    #     # print(np.shape(w))
    #     umesh += np.sum(np.real(w), axis=-1)
    #     vmesh += -np.sum(np.imag(w), axis=-1)
    #     omega_mesh += np.sum(wc, axis=-1)
    # xmesh = xmesh[:, :, np.newaxis]
    # ymesh = ymesh[:, :, np.newaxis]

    # # sigularities location - evaluate location
    # dx = x_coord - xmesh
    # dy = y_coord - ymesh
    # dz = dx + 1j * dy
    # w = sigma * weights / 2 / np.pi / dz
    # uBEMmesh = np.sum(np.real(w), axis=-1)
    # vBEMmesh = -np.sum(np.imag(w), axis=-1)
    # xmesh = np.squeeze(xmesh)
    # ymesh = np.squeeze(ymesh)

    # index = np.zeros_like(xmesh)
    # for i in range(np.size(xmesh, 0)):
    #     for j in range(np.size(xmesh, 1)):
    #         index[i, j] = cal_signed_distance_function([xmesh[i, j], ymesh[i, j]])

    # distance = (xmesh - state[0, :, np.newaxis]) ** 2 + (ymesh - state[1][:]) ** 2

    # error = np.sqrt((uBEMmesh - umesh) ** 2 + (vBEMmesh - vmesh) ** 2) / np.sqrt(
    #     umesh**2 + vmesh**2
    # )
    # log_error = np.log10(error)
    # log_error[np.isinf(log_error)] = -100
    # log_error[index > 0.1] = np.nan
    # plt.figure(figsize=(10, 9 * (ymax - ymin) / (xmax - xmin)))
    # matplotlib.rcParams["xtick.direction"] = "in"
    # matplotlib.rcParams["ytick.direction"] = "in"
    # # plt.rcParams["text.usetex"] = True
    # bwith = 0.25
    # ax = plt.gca()
    # ax.spines["bottom"].set_linewidth(bwith)
    # ax.spines["left"].set_linewidth(bwith)
    # ax.spines["top"].set_linewidth(bwith)
    # ax.spines["right"].set_linewidth(bwith)
    # ax.tick_params("both", which="major", width=bwith)
    # plt.imshow(
    #     log_error, extent=[xmin, xmax, ymin, ymax], origin="lower", vmin=-16, vmax=0
    # )
    # plt.plot(x_coord, y_coord, "black")
    # plt.axis("equal")
    # plt.title("log$_{10}$ error in flow field")
    # plt.colorbar()
    # plt.savefig("error_velocity.pdf", format="pdf")
    # plt.show()

    # dudx = sigma * weights / 2 / np.pi * ((dx**2 - dy**2) / (dx**2 + dy**2) ** 2)
    # dudy = sigma * weights / 2 / np.pi * ((2 * dx * dy) / (dx**2 + dy**2) ** 2)
    # dvdx = sigma * weights / 2 / np.pi * ((2 * dx * dy) / (dx**2 + dy**2) ** 2)
    # dvdy = sigma * weights / 2 / np.pi * ((dy**2 - dx**2) / (dx**2 + dy**2) ** 2)

    # dudx = np.sum(dudx, axis=2)
    # dudy = np.sum(dudy, axis=2)
    # dvdx = np.sum(dvdx, axis=2)
    # dvdy = np.sum(dvdy, axis=2)

    # # print(dudx.shape, dudy.shape, dvdx.shape, dvdy.shape)
    # w_BEM = -(
    #     np.sin(eval_angle) * np.cos(eval_angle) * (dudx - dvdy)
    #     + np.sin(eval_angle) ** 2 * dvdx
    #     - np.cos(eval_angle) ** 2 * dudy
    # )

    # error_omega = np.sqrt((w_BEM - omega_mesh) ** 2) / np.sqrt(omega_mesh**2)
    # log_error = np.log10(error_omega)
    # log_error[np.isinf(log_error)] = -100
    # log_error[index > 0.1] = np.nan

    # plt.figure(figsize=(10, 9 * (ymax - ymin) / (xmax - xmin)))
    # matplotlib.rcParams["xtick.direction"] = "in"
    # matplotlib.rcParams["ytick.direction"] = "in"
    # bwith = 0.25
    # ax = plt.gca()
    # ax.spines["bottom"].set_linewidth(bwith)
    # ax.spines["left"].set_linewidth(bwith)
    # ax.spines["top"].set_linewidth(bwith)
    # ax.spines["right"].set_linewidth(bwith)
    # ax.tick_params("both", which="major", width=bwith)
    # plt.imshow(
    #     log_error,
    #     extent=[xmin, xmax, ymin, ymax],
    #     origin="lower",
    #     vmin=-16,
    #     vmax=0,
    # )
    # plt.plot(x_coord, y_coord, "black")
    # plt.axis("equal")
    # plt.title("log$_{10}$ error of $\omega$")
    # plt.colorbar()
    # plt.savefig("error_omega.pdf", format="pdf")
    # plt.show()

    # plt.figure(figsize=(10, 9 * (ymax - ymin) / (xmax - xmin)))
    # matplotlib.rcParams["xtick.direction"] = "in"
    # matplotlib.rcParams["ytick.direction"] = "in"
    # bwith = 0.25
    # ax = plt.gca()
    # ax.spines["bottom"].set_linewidth(bwith)
    # ax.spines["left"].set_linewidth(bwith)
    # ax.spines["top"].set_linewidth(bwith)
    # ax.spines["right"].set_linewidth(bwith)
    # ax.tick_params("both", which="major", width=bwith)
    # plt.imshow(
    #     w_BEM,
    #     extent=[xmin, xmax, ymin, ymax],
    #     origin="lower",
    #     vmin=-0.005,
    #     vmax=0.005,
    # )
    # plt.plot(x_coord, y_coord, "black")
    # plt.axis("equal")
    # plt.title("$\omega$ generated by BEM")
    # plt.colorbar()
    # plt.savefig("omega_BEM.pdf", format="pdf")
    # plt.show()

    return U_BEM, w_BEM


def GetNoise(para):
    wNoise = np.zeros([3, para["num"]])
    wNoise[2][:] = (
        para["In"] * np.sqrt(para["dt"]) * np.random.normal(0, 1, para["num"])
    )
    return wNoise


def GetWallAvoid(state, para):
    w = np.zeros([1, para["num"]])
    d = np.zeros(para["num"])
    phi = np.zeros(para["num"])
    interc_angle = np.zeros(para["num"])
    for i in range(para["num"]):
        # print(i)
        [interc_angle[i], d[i]] = GetInterc(state[:, i])

        # print(interc_angle[i], d[i])
        # print(state[2, i] - interc_angle[i])
        phi[i] = state[2, i] - interc_angle[i]
        phi[i] = np.arctan2(np.sin(phi[i]), np.cos(phi[i]))
        w[:, i] = para["Iw"] * np.sign(phi[i]) / d[i]
        if np.abs(d[i]) <= np.abs(safedis):
            w[:, i] = w[:, i] * (1 + 1 / d[i] ** (high_order - 1))
    dt = 1e-3
    w = np.clip(w, -np.pi / dt, np.pi / dt)

    return w


def Check_jumping(Cur_state, Cur_rdot, para):
    Cur_loc = Cur_state[0:2, :]
    Nstep_loc = Cur_loc + Cur_rdot * para["dt"]
    if_jump = np.zeros(para["num"], dtype=bool)
    for i in range(para["num"]):
        target = [Cur_loc[:, i], Nstep_loc[:, i]]
        check_intersection, _ = segment_crosses_any(
            Cur_loc[:, i], Nstep_loc[:, i], obstacles
        )
        if check_intersection:
            if_jump[i] = True
    return if_jump


def Initialization(para):
    step = int(para["total_time"] / para["dt"] + 1)
    state = np.zeros((3, para["num"], step + 1))
    rDotList = np.zeros((2, para["num"], step + 1))
    targetsearch = np.zeros((1, para["num"], step + 1))
    # wNoiseList = np.zeros((3, 10, step + 1))

    # np.random.seed(int(time.time()))
    # alpha = np.random.uniform(-np.pi, np.pi, para["num"])
    # r = randcircular2(para["R"] / 1.5, para["num"], right=False, bothside=False)

    ### for validation
    # dipole_x = 11  # np.float128(2.2)
    # dipole_y = 3  # np.float128(2.15)

    r = [[0], [-4.5]]
    # r = [[-4], [-2]]
    # r = [[0], [-4.5]]
    # r = [[3], [-3.8]]
    # r = [[4.5], [-1]]
    # alpha = 1 * np.pi / 4
    # alpha = np.pi/4
    alpha = 7 * np.pi / 6

    count = 0
    initial_state = np.vstack([r, alpha])
    state[:, :, count] = np.vstack([r, alpha])
    target_pos = np.array([-1.2, 4.7])
    target_state = np.vstack([target_pos[:, np.newaxis], 0])

    # data = sio.loadmat(input_file)
    # state1 = data["state"]
    # state[:, :, count] = state1[:, :, -1]
    rDotList[:, :, count] = np.zeros((2, para["num"]))
    # wNoiseList[:, :, count] = np.zeros((3, para['num']))
    return (
        state,
        rDotList,
        targetsearch,
        initial_state,
        target_pos,
        target_state,
        count,
        step,
    )


def step_Computation(para):
    [
        state,
        rDotList,
        targetsearch,
        initial_state,
        target_pos,
        target_state,
        count,
        step,
    ] = Initialization(para)
    locationCheck_last = None
    dis_last = None
    for t in tqdm(range(step)):
        # for t in tqdm(range(step)):

        # print t when t is a multiple of 10
        if t % 1000 == 0 and t > 0:
            file_name = para[
                "root"
            ] + "If{}_Ia{:02d}_Iw{}_In{:02d}_R{:}_num{:}_Initr_{:.1f}_{:.1f}_ori_{:.2f}_n{:}_".format(
                "00" if para["If"] == 0 else "{:.0e}".format(para["If"]),
                int(para["Ia"] * 10),
                "{:.01e}".format(para["Iw"]),
                int(para["In"] * 10),
                int(para["R"]),
                int(para["num"]),
                (initial_state[0, 0]),
                (initial_state[1, 0]),
                (initial_state[2, 0]),
                int(para["n"]),
            )

            savestate = state[:, :, :]
            saverDotList = rDotList[:, :, :]
            scipy.io.savemat(
                file_name + "BEM.mat",
                mdict={
                    "state": savestate,
                    "rdot": saverDotList,
                    "para": para,
                    "targetsearch": targetsearch,
                    "initial_state": initial_state,
                    "target_pos": target_pos,
                    "target_state": target_state,
                },
            )
        CurState = state[:, :, count]
        CurLocation = state[0:2, :, count]
        # CurState_test = np.array([[2.48669], [4.01577], [9 / 10 * np.pi]])
        # cc,_ = segment_crosses_any(CurState_test[:2, 0], target_pos, obstacles)
        CrossObstacle, _ = segment_crosses_any(
            CurLocation[:, 0], target_pos, obstacles
        )  # True for connnecting path hits obstacle
        CurHeading = state[2, :, count]

        ex = np.cos(CurHeading)
        ey = np.sin(CurHeading)
        e = np.vstack([ex, ey])

        # [U, wHydro] = GetHydro(CurState, para)
        U, wHydro = 0, 0

        [U_BEM, w_BEM] = GetHydroBEM(CurState, para) if para["If"] != 0 else (0, 0)
        # U_BEM, w_BEM = 0, 0
        # voro = GetVoronoi(CurLocation, para)
        voro = np.array([[0, 1], [1, 0]], dtype="bool")  # for two fish case
        # GetWallAvoid(CurState_test, para)
        # GetVision(voro, np.hstack((CurState_test, target_state)), para)[0]

        wVision = (
            GetVision(voro, np.hstack((CurState, target_state)), para)[0]
            if not CrossObstacle
            else 0.0
        )
        wNoise = GetNoise(para)
        wWall = GetWallAvoid(CurState, para)
        # wAttr = AddAttractionPoint(CurState, para)
        thetaDot = wHydro + w_BEM + para["light"] * (wWall + wVision)  # + wAttr
        thetaDot = np.clip(thetaDot, -np.pi / para["dt"], np.pi / para["dt"])
        rDot = e + U + U_BEM

        thetaDot_nohydro = para["light"] * (wWall + wVision)
        thetaDot_nohydro = np.clip(
            thetaDot_nohydro, -np.pi / para["dt"], np.pi / para["dt"]
        )
        if_jump = Check_jumping(CurState, rDot, para)
        if if_jump.any():
            rDot[:, if_jump] = e[:, if_jump]
            thetaDot[:, if_jump] = thetaDot_nohydro[:, if_jump]

        if np.any(abs(rDot) > 500):
            where_exceed = np.unique(np.where(abs(rDot) > 500)[1])
            rDot[:, where_exceed] = e[:, where_exceed]
            thetaDot[:, where_exceed] = thetaDot_nohydro[:, where_exceed]

        count += 1
        state[:, :, count] = (
            CurState + np.vstack([rDot, thetaDot]) * para["dt"] + wNoise
        )

        dis = np.zeros(para["num"])

        for i in range(para["num"]):
            dis[i] = cal_signed_distance_function(state[0:2, i, count])

        if any((dis - safedis) >= 0):
            locationCheck = np.sign(dis - safedis)
            # print(np.sum(locationCheck))
            # if not (type(locationCheck_last) is np.ndarray):
            #     warningFish = np.where(locationCheck >= 0)[0]
            # else:
            #     warningFish = np.where(
            #         np.logical_and(
            #             locationCheck >= 0, locationCheck >= locationCheck_last
            #         )
            #     )[0]
            # locationCheck_last = np.copy(locationCheck)
            if dis_last is None:
                warningFish = np.where(locationCheck >= 0)[0]
            else:
                warningFish = np.where(
                    np.logical_or(
                        np.logical_and(locationCheck >= 0, dis_last < dis), dis > -0.01
                    )
                )[0]
            for k in range(len(warningFish)):
                # test_state = CurState[0:2, warningFish[k]] + rDot[:, warningFish[k]]*para['dt']
                # test_distance = np.sqrt(test_state[0]**2+test_state[1]**2)
                # if test_distance - checkDist[warningFish[k]] >= -1e-12:
                rDot[:, warningFish[k]] = np.zeros(2)  # 0*rDot[:, warningFish[k]]

                # thetaDot[:, warningFish[k]] = (
                #     para["light"] * wWall[:, warningFish[k]]
                #     + wHydro[warningFish[k]]
                #     + w_BEM[warningFish[k]]
                # )
                # wNoise[:, warningFish[k]] = np.zeros(3)
            state[:, :, count] = (
                CurState + np.vstack([rDot, thetaDot]) * para["dt"] + wNoise
            )
        dis_last = np.copy(dis)
        rDotList[:, :, count] = rDot
        targetsearch[0, 0, count] = 1 if not CrossObstacle else 0
        if (
            np.sqrt(
                (CurState[0][0] - target_pos[0]) ** 2
                + (CurState[1][0] - target_pos[1]) ** 2
            )
            < 0.1
        ):
            count_stop = count
            return (
                state[:, :, : count_stop + 1],
                rDotList[:, :, : count_stop + 1],
                targetsearch[:, :, : count_stop + 1],
                initial_state,
                target_pos,
                target_state,
            )

    return state, rDotList, targetsearch, initial_state, target_pos, target_state


def Computation(n):
    # np.random.seed(n + int(time.time()))
    np.random.seed(n)
    # root = "/storage/hao/Complex_enviornment/Diff_IC"
    root = "/Volumes/reynolds/Complex_enviornment/Circle_enclosure/"
    root = root + "/BEM_data/"
    if not os.path.exists(root):
        os.makedirs(root)

    para = dict(
        [
            ("In", 0.5),
            ("Ia", 1),
            ("light", 1),
            ("dt", 1e-3),
            ("Iw", 0.94),
            ("If", 1e-2),
            ("num", 1),
            ("delta", 1e-2),
            ("total_time", 300),
            ("root", root),
            ("n", n),
            ("R", R3),
        ]
    )

    sys.stdout.write(
        "If = %e; Ia = %f; In = %f; Radius = %d; Light = %f; Number = %d;\n"
        % (
            para["If"],
            para["Ia"],
            para["In"],
            para["R"],
            para["light"],
            para["num"],
        )
    )
    [state, rDotList, targetsearch, initial_state, target_pos, target_state] = (
        step_Computation(para)
    )
    file_name = (
        root
        + "If{}_Ia{:02d}_Iw{}_In{:02d}_R{:}_num{:}_Initr_{:.1f}_{:.1f}_ori_{:.2f}_n{:}_".format(
            "00" if para["If"] == 0 else "{:.0e}".format(para["If"]),
            int(para["Ia"] * 10),
            "{:.01e}".format(para["Iw"]),
            int(para["In"] * 10),
            int(para["R"]),
            int(para["num"]),
            (initial_state[0, 0]),
            (initial_state[1, 0]),
            (initial_state[2, 0]),
            int(para["n"]),
        )
    )

    # final state saving
    savestate = state[:, :, :]
    saverDotList = rDotList[:, :, :]
    scipy.io.savemat(
        file_name + "BEM.mat",
        mdict={
            "state": savestate,
            "rdot": saverDotList,
            "para": para,
            "targetsearch": targetsearch,
            "initial_state": initial_state,
            "target_pos": target_pos,
            "target_state": target_state,
        },
    )

    return None

    # for i in range(N_instances):


# Computation(1)


# ########################################################
N_instances = 5
task_queue = np.arange(2, N_instances)
task_queue = task_queue.tolist()
total_tasks = len(task_queue)
if __name__ == "__main__":
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:

        completed_tasks = 0
        task_sent = 0
        # Distribute tasks to workers
        while completed_tasks < total_tasks:
            status = MPI.Status()
            # Receive a request from any worker
            msg = comm.recv(source=MPI.ANY_SOURCE, tag=0, status=status)
            worker_rank = status.Get_source()

            if task_queue:
                # Send the next task to the requesting worker
                task = task_queue.pop(0)
                comm.send(task, dest=worker_rank, tag=1)
                task_sent += 1
            else:
                # No more tasks, signal the worker to stop
                comm.send(None, dest=worker_rank, tag=1)

            completed_tasks += 1
        for worker_rank in range(1, size):
            # Check if the worker has already been sent a termination signal
            # This is necessary only if workers might not have received it yet
            # Depending on the above loop logic, this might be redundant
            if task_sent < total_tasks:
                continue  # Tasks are still being sent
            comm.send(None, dest=worker_rank, tag=1)
    else:
        # Worker processes
        while True:
            # Request a task
            comm.send(None, dest=0, tag=0)
            # Receive a task
            task = comm.recv(source=0, tag=1)
            if task is None:
                # No more tasks, exit
                break
            # Perform the computation
            seed_task = task
            # print(seed_task, In_task, Iw_task)
            Computation(seed_task)
