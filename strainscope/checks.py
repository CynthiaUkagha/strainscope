"""Self-checks: decide whether a strain map should be trusted.

A GPA strain map always "succeeds" numerically, even on pure noise. These
checks grade each result PASS, REVIEW or FAIL and say why.
"""
from dataclasses import dataclass

import numpy as np

from .gpa import find_dislocations, phase_closure_residual

PASS, REVIEW, FAIL = "PASS", "REVIEW", "FAIL"


@dataclass
class Check:
    name: str
    value: str
    status: str
    why: str


def _peak_snr(res, g):
    best = min(res.peaks, key=lambda p: np.hypot(p[0] - g[0], p[1] - g[1]))
    return best[2] / res.fft_background


def _reference_spread(res):
    rs, cs = res.reference
    ym = (rs.start + rs.stop) // 2
    xm = (cs.start + cs.stop) // 2
    quarters = [(slice(rs.start, ym), slice(cs.start, xm)), (slice(rs.start, ym), slice(xm, cs.stop)),
                (slice(ym, rs.stop), slice(cs.start, xm)), (slice(ym, rs.stop), slice(xm, cs.stop))]
    spread = 0.0
    for comp in (res.exx, res.eyy):
        means = [comp[q].mean() for q in quarters]
        spread = max(spread, max(means) - min(means))
    return float(spread)


def run_checks(res, closure_review=0.35, closure_fail=0.8):
    checks = []

    snr = min(_peak_snr(res, res.g1), _peak_snr(res, res.g2))
    checks.append(Check(
        "Lattice peak strength", f"{snr:.0f}x background",
        PASS if snr >= 50 else REVIEW if snr >= 15 else FAIL,
        "Weak Bragg peaks mean the phase is mostly noise."))

    cos = abs(res.g1 @ res.g2) / (np.linalg.norm(res.g1) * np.linalg.norm(res.g2))
    angle = np.degrees(np.arccos(min(1.0, cos)))
    checks.append(Check(
        "g-vector angle", f"{angle:.0f} deg",
        PASS if angle >= 30 else REVIEW if angle >= 15 else FAIL,
        "Nearly parallel g-vectors make the 2D strain tensor ill-conditioned."))

    gmin = min(np.linalg.norm(res.g1), np.linalg.norm(res.g2), np.linalg.norm(res.g1 - res.g2))
    ratio = res.sigma / gmin
    checks.append(Check(
        "Mask size", f"{ratio:.2f} of nearest peak spacing",
        PASS if ratio <= 0.3 else REVIEW if ratio <= 0.45 else FAIL,
        "A mask that overlaps neighbouring peaks mixes their phases."))

    # Compare the mean strain in the four quarters of the reference region.
    # Random noise averages out of the quarter means; a real gradient does not.
    spread = _reference_spread(res)
    checks.append(Check(
        "Reference flatness", f"{spread * 100:.2f}% spread between quarters",
        PASS if spread <= 0.003 else REVIEW if spread <= 0.01 else FAIL,
        "The 'unstrained' reference region should actually be flat."))

    r = phase_closure_residual(res)[res.interior]
    closure = float(np.sqrt(np.mean(r ** 2)))
    checks.append(Check(
        "Phase closure (3rd reflection)", f"{closure:.2f} rad rms",
        PASS if closure <= closure_review else REVIEW if closure <= closure_fail else FAIL,
        "An independent third reflection must agree with the other two."))

    cores = find_dislocations(res.H1, res.interior) + find_dislocations(res.H2, res.interior)
    n = len({(round(x / 16), round(y / 16)) for x, y in cores})
    checks.append(Check(
        "Dislocation cores", str(n),
        PASS if n == 0 else REVIEW,
        "Near a core the map is not elastic strain; interpret it separately."))

    statuses = [c.status for c in checks]
    grade = FAIL if FAIL in statuses else REVIEW if REVIEW in statuses else PASS
    return grade, checks


def report(grade, checks):
    lines = [f"Grade: {grade}", ""]
    w = max(len(c.name) for c in checks)
    for c in checks:
        lines.append(f"  [{c.status:6}] {c.name:<{w}}  {c.value}")
    return "\n".join(lines)
