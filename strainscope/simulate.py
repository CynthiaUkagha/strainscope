"""Synthetic HRTEM-style lattice images with a known displacement field.

Because the displacement is known exactly, every measurement made on these
images can be checked against ground truth.
"""
import numpy as np


def reciprocal_vectors(a1, a2):
    """Reciprocal vectors g1, g2 (cycles/pixel) with a_i . g_j = delta_ij."""
    A = np.column_stack([a1, a2]).astype(float)
    B = np.linalg.inv(A).T
    return B[:, 0], B[:, 1]


def pixel_grid(shape):
    ny, nx = shape
    y, x = np.mgrid[0:ny, 0:nx].astype(float)
    return x, y


def rotate(v, deg):
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def strain_step(shape, eps_xx=0.02, x0=None, width=6.0):
    """Unstrained on the left, uniform strain eps_xx on the right, smooth interface.

    Returns ((ux, uy), exx_true).
    """
    x, _ = pixel_grid(shape)
    if x0 is None:
        x0 = shape[1] / 2
    t = (x - x0) / width
    ux = eps_xx * width * np.logaddexp(0.0, t)   # d(ux)/dx = eps_xx * sigmoid(t)
    uy = np.zeros_like(ux)
    exx = eps_xx / (1.0 + np.exp(-t))
    return (ux, uy), exx


def edge_dislocation(shape, center, burgers, nu=0.3):
    """Isotropic-elasticity displacement field of an edge dislocation.

    `burgers` is a 2-vector in pixels; it should be a lattice vector so the
    lattice closes around the core.
    """
    x, y = pixel_grid(shape)
    b = float(np.hypot(*burgers))
    ang = np.degrees(np.arctan2(burgers[1], burgers[0]))
    X, Y = x - center[0], y - center[1]
    xr, yr = rotate((X, Y), -ang)
    r2 = xr ** 2 + yr ** 2 + 1e-9
    ux_r = b / (2 * np.pi) * (np.arctan2(yr, xr) + xr * yr / (2 * (1 - nu) * r2))
    uy_r = -b / (2 * np.pi) * ((1 - 2 * nu) / (4 * (1 - nu)) * np.log(r2)
                               + (xr ** 2 - yr ** 2) / (4 * (1 - nu) * r2))
    ux, uy = rotate((ux_r, uy_r), ang)
    return ux, uy


def lattice_image(shape=(256, 256), a=8.0, rotation_deg=12.0, displacement=None,
                  noise=0.3, seed=0):
    """Image of a square lattice (spacing `a` px) under a displacement field.

    Returns (image, a1, a2).
    """
    a1 = rotate((a, 0.0), rotation_deg)
    a2 = rotate((0.0, a), rotation_deg)
    g1, g2 = reciprocal_vectors(a1, a2)
    x, y = pixel_grid(shape)
    if displacement is not None:
        x = x - displacement[0]
        y = y - displacement[1]
    img = np.zeros(shape)
    for g, w in ((g1, 1.0), (g2, 1.0), (g1 + g2, 0.5)):
        img += w * np.cos(2 * np.pi * (g[0] * x + g[1] * y))
    rng = np.random.default_rng(seed)
    img += noise * img.std() * rng.standard_normal(shape)
    return img, a1, a2
