import numpy as np
import scipy as sp
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


# dipole_x = np.array([1.5, 0, 2, 2, -2]) + 1.5
# dipole_y = np.array([0, 0, 3, -2, 2]) + 1
# alpha = np.array([np.pi / 6, np.pi / 3, -np.pi / 4, np.pi / 9, -np.pi / 5])
# dipole_x = np.array([0])
# dipole_y = np.array([-1.5])
# alpha = np.array([np.pi / 6])
dipole_x = np.array([1.3370])
dipole_y = np.array([-1.6183])
alpha = np.array([0.8453])
state = np.vstack([dipole_x, dipole_y, alpha])
para = dict([("num", len(dipole_x)), ("If", 1e-2), ("delta", 0)])

N_singular = len(x_coord)
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


# solve the density of the singularities
dipole_z = state[0][:] + 1j * state[1][:]
dipole_angle = state[2][:]
dipole_strength = para["If"]
xi, yi = x_coord[:, np.newaxis], y_coord[:, np.newaxis]
nx, ny = normal_x[:, np.newaxis], normal_y[:, np.newaxis]
dz = xi + 1j * yi - dipole_z
w = (dipole_strength) * np.exp(1j * dipole_angle) / (dz**2 + para["delta"] ** 2) / np.pi
f = np.sum((nx * np.real(w) - ny * np.imag(w)), axis=1)
print("RHS computed")
# print(np.sum(f * np.sqrt(arclength)))

# sigma = linalg.solve((lu, piv), f)
sigma = linalg.lu_solve((lu, piv), -f)
print("Density solved")

xmin = min(x_coord) - 0.1
xmax = max(x_coord) + 0.1
ymin = min(y_coord) - 0.1
ymax = max(y_coord) + 0.1


################### validation ####################
eval_angle = np.pi / 4  # for evaluating angular velocity
ds = 0.02
Nx = int((xmax - xmin) / ds)
Ny = int((ymax - ymin) / ds)
x = np.linspace(xmin, xmax, Nx, dtype=np.float64)
y = np.linspace(ymin, ymax, Ny, dtype=np.float64)
xmesh, ymesh = np.meshgrid(x, y)
xmesh, ymesh = np.meshgrid(x, y)
umesh = np.zeros_like(xmesh, dtype=np.float64)
vmesh = np.zeros_like(xmesh, dtype=np.float64)
omega_mesh = np.zeros_like(xmesh, dtype=np.float64)

for dipole_index in range(para["num"]):
    dipole_z = state[0][dipole_index] + 1j * state[1][dipole_index]
    dipole_angle = state[2][dipole_index]
    dipole_strength = para["If"]
    dz = xmesh + 1j * ymesh - dipole_z
    dz = dz[:, :, np.newaxis]
    wc = np.imag(
        np.exp(1j * (2 * eval_angle + dipole_angle))
        * dz
        / ((dz**2 + para["delta"] ** 2) ** 2)
        * 2
        * para["If"]
        / np.pi
    )
    w = (
        1
        / (dz**2 + para["delta"] ** 2)
        * (dipole_strength)
        * np.exp(1j * dipole_angle)
        / np.pi
    )
    # print(np.shape(w))
    umesh += np.sum(np.real(w), axis=-1)
    vmesh += -np.sum(np.imag(w), axis=-1)
    omega_mesh += np.sum(wc, axis=-1)
xmesh = xmesh[:, :, np.newaxis]
ymesh = ymesh[:, :, np.newaxis]
print("xmesh shape:", xmesh.shape)
# sigularities location - evaluate location
# for k in range(np.size(x)
dx = x_coord - xmesh
dy = y_coord - ymesh
dz = dx + 1j * dy
w = sigma * weights / 2 / np.pi / dz

uBEMmesh = np.sum(np.real(w), axis=-1)
vBEMmesh = -np.sum(np.imag(w), axis=-1)
xmesh = np.squeeze(xmesh)
ymesh = np.squeeze(ymesh)

############test#############
dipole_x = np.array([1.3370])
dipole_y = np.array([-1.6183])
test_pos = np.vstack([dipole_x, dipole_y])
for dipole_index in range(para["num"]):
    dipole_z = state[0][dipole_index] + 1j * state[1][dipole_index]
    dipole_angle = state[2][dipole_index]
    dipole_strength = para["If"]
    dz = dipole_x + 1j * dipole_y - dipole_z
    # dz = dz[:, :, np.newaxis]
    wc = np.imag(
        np.exp(1j * (2 * eval_angle + dipole_angle))
        * dz
        / ((dz**2 + para["delta"] ** 2) ** 2)
        * 2
        * para["If"]
        / np.pi
    )
    w = (
        1
        / (dz**2 + para["delta"] ** 2)
        * (dipole_strength)
        * np.exp(1j * dipole_angle)
        / np.pi
    )
    # print(np.shape(w))
    u_test += np.sum(np.real(w), axis=-1)
    v_test += -np.sum(np.imag(w), axis=-1)
    omega_mesh += np.sum(wc, axis=-1)
print("u_test:", u_test, "v_test:", v_test)
##########################
from matplotlib.path import Path


def _make_path(x, y):
    """Create a closed polygon Path from a boundary polyline."""
    # ensure it's explicitly closed
    XY = np.column_stack([x, y])
    if not (x[0] == x[-1] and y[0] == y[-1]):
        XY = np.vstack([XY, XY[0]])
    return Path(XY, closed=True)


# 1) Build a Path for each obstacle you have active
obstacle_paths = [
    _make_path(x1, y1),
    _make_path(x2, y2),
    # _make_path(x3, y3),
    _make_path(x4, y4),
    _make_path(x5, y5),
    _make_path(x6, y6),  # include if/when you use the wedge
    _make_path(x7, y7),
    _make_path(x8, y8),
    _make_path(x9, y9),
    _make_path(x10, y10),
    _make_path(x11, y11),
    _make_path(x12, y12),
]

# 2) Compute a boolean mask of grid points that lie inside ANY obstacle
XY = np.c_[xmesh.ravel(), ymesh.ravel()]  # (Nx*Ny, 2)
inside_any = np.zeros(XY.shape[0], dtype=bool)
for P in obstacle_paths:
    inside_any |= P.contains_points(XY)
inside_any = inside_any.reshape(xmesh.shape)  # (Ny, Nx)

# 3) Mask velocity (and any other fields) inside obstacles
U_total = umesh + uBEMmesh
V_total = vmesh + vBEMmesh

U_total[inside_any] = np.nan
V_total[inside_any] = np.nan
# if you visualize vorticity or anything else, mask that too:
# omega_mesh[inside_any] = np.nan
##########################


# log_error[index < -0.1] = np.nan
plt.figure(figsize=(10, 9 * (ymax - ymin) / (xmax - xmin)))
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"
# plt.rcParams["text.usetex"] = True
bwith = 0.25
ax = plt.gca()
ax.spines["bottom"].set_linewidth(bwith)
ax.spines["left"].set_linewidth(bwith)
ax.spines["top"].set_linewidth(bwith)
ax.spines["right"].set_linewidth(bwith)
ax.tick_params("both", which="major", width=bwith)
# plt.imshow(log_error, extent=[xmin, xmax, ymin, ymax], origin="lower", vmin=-16, vmax=0)
plt.streamplot(
    xmesh,
    ymesh,
    # (umesh + uBEMmesh),
    U_total,
    # (vmesh + vBEMmesh),
    V_total,
    broken_streamlines=False,
    linewidth=0.2,
    arrowstyle="-",
    density=1.5,
    color="grey",
)
# plt.plot(x_coord, y_coord, "black")
plt.plot(x1, y1, "black")
plt.plot(x2, y2, "black")
plt.plot(x3, y3, "black")
plt.plot(x4, y4, "black")
plt.plot(x5, y5, "black")
plt.plot(x6, y6, "black")
plt.plot(x7, y7, "black")
plt.plot(x8, y8, "black")
plt.plot(x9, y9, "black")
plt.plot(x10, y10, "black")
plt.plot(x11, y11, "black")
plt.plot(x12, y12, "black")
plt.scatter(dipole_x, dipole_y, color="red", s=20, marker="o")
plt.axis("equal")
# plt.title("log$_{10}$ error in velocity")
plt.colorbar()
plt.savefig("./streamlines.png", format="png")
plt.show()

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
# log_error[index < -0.1] = np.nan

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
# # plt.savefig("./figures_geo_fixed/error_omega.png", format="png")
# plt.show()
