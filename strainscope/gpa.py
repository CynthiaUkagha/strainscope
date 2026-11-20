"""Geometric phase analysis (GPA) of lattice images.

Hytch, Snoeck & Kilaas, Ultramicroscopy 74 (1998) 131-146.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _freqs(shape):
    ny, nx = shape
    return np.meshgrid(np.fft.fftfreq(nx), np.fft.fftfreq(ny))


def find_peaks(img, n=12, min_radius=0.02):
    """Strongest Bragg peaks in the FFT as (gx, gy, amplitude), plus the FFT background level."""
    ny, nx = img.shape
    win = np.outer(np.hanning(ny), np.hanning(nx))
    A = np.abs(np.fft.fft2((img - img.mean()) * win))
    KX, KY = _freqs(img.shape)
    A[np.hypot(KX, KY) < min_radius] = 0
    is_max = (A == ndimage.maximum_filter(A, size=5, mode="wrap")) & (A > 0)
    idx = np.argwhere(is_max)
    order = np.argsort(A[is_max])[::-1][:n]
    peaks = [(KX[tuple(idx[i])], KY[tuple(idx[i])], A[tuple(idx[i])]) for i in order]
    return peaks, float(np.median(A))


def choose_g(peaks, min_angle=20.0):
    """Pick the strongest peak and the strongest one at least `min_angle` away from it."""
    g1 = np.array(peaks[0][:2])
    for p in peaks[1:]:
        v = np.array(p[:2])
        cos = abs(v @ g1) / (np.linalg.norm(v) * np.linalg.norm(g1))
        if np.degrees(np.arccos(min(1.0, cos))) >= min_angle:
            return g1, v
    raise ValueError("No two non-collinear lattice peaks found.")


def _demodulate(F, KX, KY, X, Y, g, sigma):
    mask = np.exp(-((KX - g[0]) ** 2 + (KY - g[1]) ** 2) / (2 * sigma ** 2))
    return np.fft.ifft2(F * mask) * np.exp(-2j * np.pi * (g[0] * X + g[1] * Y))


def _phase_gradient(H):
    dx = np.angle(np.roll(H, -1, axis=1) * np.conj(np.roll(H, 1, axis=1))) / 2
    dy = np.angle(np.roll(H, -1, axis=0) * np.conj(np.roll(H, 1, axis=0))) / 2
    return dx, dy


@dataclass
class GPAResult:
    exx: np.ndarray
    eyy: np.ndarray
    exy: np.ndarray
    rotation: np.ndarray
    g1: np.ndarray
    g2: np.ndarray
    sigma: float
    H1: np.ndarray
    H2: np.ndarray
    H3: np.ndarray
    s3: int
    reference: tuple
    interior: tuple
    peaks: list = field(default_factory=list)
    fft_background: float = 0.0


def interior_slices(shape, sigma, factor=3.0):
    """Region far enough from the image edge to be free of FFT edge artefacts."""
    m = int(np.ceil(factor / (2 * np.pi * sigma))) + 4
    return (slice(m, shape[0] - m), slice(m, shape[1] - m))


def gpa(img, g=None, sigma=None, reference=None):
    """Strain maps (relative to `reference`) from a lattice image.

    g:         optional (g1, g2) in cycles/pixel; found automatically if omitted.
    sigma:     Gaussian mask width in cycles/pixel; sets the spatial resolution.
    reference: (row_slice, col_slice) of a region taken as unstrained.
    """
    img = np.asarray(img, dtype=float)
    peaks, bg = find_peaks(img)
    g1, g2 = (np.asarray(v, float) for v in (g if g is not None else choose_g(peaks)))
    if sigma is None:
        sigma = 0.2 * min(np.linalg.norm(g1), np.linalg.norm(g2))
    interior = interior_slices(img.shape, sigma)
    if reference is None:
        reference = interior

    X, Y = np.meshgrid(np.arange(img.shape[1], dtype=float), np.arange(img.shape[0], dtype=float))
    KX, KY = _freqs(img.shape)
    F = np.fft.fft2(img - img.mean())

    # Refine each g so the reference region has zero mean phase gradient.
    refined = []
    for gv in (g1, g2):
        dx, dy = _phase_gradient(_demodulate(F, KX, KY, X, Y, gv, sigma))
        refined.append(gv + np.array([dx[reference].mean(), dy[reference].mean()]) / (2 * np.pi))
    g1, g2 = refined

    H1 = _demodulate(F, KX, KY, X, Y, g1, sigma)
    H2 = _demodulate(F, KX, KY, X, Y, g2, sigma)
    # Third reflection for the closure check: whichever of g1+g2 / g1-g2 is present.
    def _amp_near(gv):
        return max((p[2] for p in peaks if np.hypot(p[0] - gv[0], p[1] - gv[1]) < 2 * sigma), default=0.0)
    s3 = 1 if _amp_near(g1 + g2) >= _amp_near(g1 - g2) else -1
    H3 = _demodulate(F, KX, KY, X, Y, g1 + s3 * g2, sigma)

    d1 = _phase_gradient(H1)
    d2 = _phase_gradient(H2)
    P = np.array([[d1[0], d1[1]], [d2[0], d2[1]]])           # dP_i / dx_j
    G = np.array([g1, g2])
    D = -np.einsum("ik,kj...->ij...", np.linalg.inv(G), P) / (2 * np.pi)   # du_i / dx_j

    return GPAResult(
        exx=D[0, 0], eyy=D[1, 1], exy=0.5 * (D[0, 1] + D[1, 0]),
        rotation=0.5 * (D[1, 0] - D[0, 1]),
        g1=g1, g2=g2, sigma=float(sigma), H1=H1, H2=H2, H3=H3, s3=s3,
        reference=reference, interior=interior, peaks=peaks, fft_background=bg,
    )


def phase_closure_residual(res):
    """Phase of H(g1 +/- g2) minus the phases of H(g1) and +/- H(g2).

    For a real lattice these must agree up to a constant; this is an
    independent consistency check that the strain fit alone cannot pass by luck.
    """
    H2 = np.conj(res.H2) if res.s3 == 1 else res.H2
    r = np.angle(res.H3 * np.conj(res.H1) * H2)
    return _wrap(r - np.angle(np.mean(np.exp(1j * r[res.interior]))))


def find_dislocations(H, interior, block=16, amp_frac=0.3):
    """Locate phase vortices (dislocation cores) by phase winding around blocks.

    Returns a list of (x, y) block centres where the winding number is non-zero.
    """
    P = np.angle(H)
    A = np.abs(H)
    thr = amp_frac * np.median(A[interior])
    ys = range(interior[0].start, interior[0].stop - block, block // 2)
    xs = range(interior[1].start, interior[1].stop - block, block // 2)
    hits = np.zeros((len(ys), len(xs)), bool)
    for i, y0 in enumerate(ys):
        for j, x0 in enumerate(xs):
            y1, x1 = y0 + block, x0 + block
            path_y = np.r_[np.full(block, y0), np.arange(y0, y1), np.full(block, y1), np.arange(y1, y0, -1)]
            path_x = np.r_[np.arange(x0, x1), np.full(block, x1), np.arange(x1, x0, -1), np.full(block, x0)]
            if A[path_y, path_x].min() < thr:
                continue
            ph = P[path_y, path_x]
            w = _wrap(np.diff(np.r_[ph, ph[0]])).sum() / (2 * np.pi)
            hits[i, j] = abs(round(w)) >= 1
    labels, n = ndimage.label(hits)
    centres = []
    for k in range(1, n + 1):
        ii, jj = np.nonzero(labels == k)
        centres.append((xs[0] + jj.mean() * block / 2 + block / 2, ys[0] + ii.mean() * block / 2 + block / 2))
    return centres
