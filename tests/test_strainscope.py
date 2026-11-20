import numpy as np
from scipy.ndimage import gaussian_filter1d

from strainscope import gpa, run_checks, simulate

SHAPE = (256, 256)


def _step_case(eps=0.02, noise=0.3, seed=0):
    disp, exx_true = simulate.strain_step(SHAPE, eps_xx=eps)
    img, a1, a2 = simulate.lattice_image(SHAPE, displacement=disp, noise=noise, seed=seed)
    ref = (slice(40, 216), slice(30, 90))
    return img, exx_true, ref


def test_reciprocal_vectors():
    a1, a2 = np.array([8.0, 1.0]), np.array([-2.0, 7.0])
    g1, g2 = simulate.reciprocal_vectors(a1, a2)
    assert np.allclose([a1 @ g1, a1 @ g2, a2 @ g1, a2 @ g2], [1, 0, 0, 1])


def test_recovers_known_uniform_strain():
    img, exx_true, ref = _step_case(eps=0.02)
    res = gpa(img, reference=ref)
    strained = (slice(40, 216), slice(180, 220))
    assert abs(res.exx[strained].mean() - 0.02) < 0.002
    assert abs(res.eyy[strained].mean()) < 0.002
    assert abs(res.exx[ref].mean()) < 0.001


def _profile_error(noise):
    img, exx_true, ref = _step_case(eps=0.015, noise=noise)
    res = gpa(img, reference=ref)
    rows, cols = slice(60, 196), slice(40, 200)  # inside the edge-artefact-free interior
    meas = res.exx[rows, cols].mean(axis=0)
    # GPA blurs strain with a Gaussian of width 1/(2*pi*sigma); compare like with like.
    true = gaussian_filter1d(exx_true[rows].mean(axis=0), 1 / (2 * np.pi * res.sigma))[cols]
    return meas - true


def test_strain_profile_exact_without_noise():
    assert np.max(np.abs(_profile_error(noise=0.0))) < 1e-4


def test_strain_profile_precision_with_noise():
    assert np.sqrt(np.mean(_profile_error(noise=0.3) ** 2)) < 0.003


def test_clean_lattice_passes():
    img, _, ref = _step_case(eps=0.01, noise=0.2)
    grade, _ = run_checks(gpa(img, reference=ref))
    assert grade == "PASS"


def test_closure_check_independent_of_g_sign():
    img, _, ref = _step_case(eps=0.01, noise=0.2)
    res = gpa(img, reference=ref)
    flipped = gpa(img, g=(res.g1, -res.g2), reference=ref)
    for r in (res, flipped):
        grade, checks = run_checks(r)
        closure = next(c for c in checks if c.name.startswith("Phase closure"))
        assert closure.status == "PASS"


def test_pure_noise_fails():
    rng = np.random.default_rng(1)
    img = rng.standard_normal(SHAPE)
    grade, checks = run_checks(gpa(img))
    assert grade == "FAIL"


def test_dislocation_is_found_and_flagged():
    ux, uy = simulate.edge_dislocation(SHAPE, center=(128, 128), burgers=simulate.rotate((8.0, 0.0), 12.0))
    img, a1, a2 = simulate.lattice_image(SHAPE, displacement=(ux, uy), noise=0.2)
    grade, checks = run_checks(gpa(img))
    cores = next(c for c in checks if c.name == "Dislocation cores")
    assert int(cores.value) >= 1
    assert grade in ("REVIEW", "FAIL")
