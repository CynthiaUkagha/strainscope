"""Make the README figures and the grade table.

    python examples/demo.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from strainscope import gpa, report, run_checks, simulate

OUT = Path(__file__).resolve().parent.parent / "docs"
OUT.mkdir(exist_ok=True)
SHAPE = (256, 256)
REF = (slice(40, 216), slice(30, 90))

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold"})


def strained_with_dislocation():
    (ux, uy), exx_true = simulate.strain_step(SHAPE, eps_xx=0.02)
    dx, dy = simulate.edge_dislocation(SHAPE, center=(170, 128), burgers=simulate.rotate((8.0, 0.0), 12.0))
    img, _, _ = simulate.lattice_image(SHAPE, displacement=(ux + dx, uy + dy), noise=0.3)
    return img, exx_true


def overview():
    img, _ = strained_with_dislocation()
    res = gpa(img, reference=REF)
    (ux, uy), exx_true = simulate.strain_step(SHAPE, eps_xx=0.02)
    clean, _, _ = simulate.lattice_image(SHAPE, displacement=(ux, uy), noise=0.3)
    res_clean = gpa(clean, reference=REF)
    it = res.interior

    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2), constrained_layout=True)
    ax[0].imshow(img[64:192, 106:234], cmap="gray")
    ax[0].set_title("1. Lattice image (simulated)")

    F = np.fft.fftshift(np.abs(np.fft.fft2(img - img.mean())))
    n = SHAPE[0] // 2
    ax[1].imshow(np.log1p(F)[n - 48:n + 48, n - 48:n + 48], cmap="magma",
                 extent=[-48 / SHAPE[1], 48 / SHAPE[1], 48 / SHAPE[0], -48 / SHAPE[0]])
    for g, c in ((res.g1, "#4cc9f0"), (res.g2, "#f8961e")):
        ax[1].add_patch(plt.Circle(g, 2 * res.sigma, fill=False, color=c, lw=2))
    ax[1].set_title("2. FFT with chosen g-vectors")

    exx = np.full(SHAPE, np.nan)
    exx[it] = res.exx[it] * 100
    m = ax[2].imshow(exx, cmap="RdBu_r", vmin=-4, vmax=4)
    ax[2].add_patch(plt.Rectangle((REF[1].start, REF[0].start), REF[1].stop - REF[1].start,
                                  REF[0].stop - REF[0].start, fill=False, ls="--", color="k"))
    ax[2].annotate("dislocation", (170, 128), (190, 60), arrowprops=dict(arrowstyle="->"))
    ax[2].set_title("3. Measured strain εxx (%)")
    fig.colorbar(m, ax=ax[2], shrink=0.8)

    rows, cols = slice(60, 196), slice(it[1].start, it[1].stop)
    x = np.arange(SHAPE[1])[cols]
    truth = gaussian_filter1d(exx_true[rows].mean(0), 1 / (2 * np.pi * res_clean.sigma))[cols]
    ax[3].plot(x, exx_true[rows].mean(0)[cols] * 100, color="0.6", lw=1, label="true (sharp)")
    ax[3].plot(x, truth * 100, color="k", lw=2, label="true at GPA resolution")
    ax[3].plot(x, res_clean.exx[rows, cols].mean(0) * 100, color="#d1495b", lw=1.5, label="measured")
    ax[3].set_xlabel("x (pixels)")
    ax[3].set_ylabel("εxx (%)")
    ax[3].set_title("4. Measured vs known strain")
    ax[3].legend(frameon=False, loc="upper left")
    for a in ax[:3]:
        a.set_xticks([]); a.set_yticks([])
    fig.savefig(OUT / "overview.png", dpi=130)


def grade_table():
    rng = np.random.default_rng(1)
    (ux, uy), _ = simulate.strain_step(SHAPE, eps_xx=0.01)
    good, _, _ = simulate.lattice_image(SHAPE, displacement=(ux, uy), noise=0.2)
    disl, _ = strained_with_dislocation()
    faint, _, _ = simulate.lattice_image(SHAPE, displacement=(ux, uy), noise=6.0)
    cases = [
        ("Strained lattice, clean", good, REF),
        ("Strained lattice + dislocation", disl, REF),
        ("Very noisy lattice", faint, REF),
        ("Pure noise (no lattice)", rng.standard_normal(SHAPE), None),
    ]
    rows = []
    for name, img, ref in cases:
        grade, checks = run_checks(gpa(img, reference=ref))
        print(f"\n== {name}\n{report(grade, checks)}")
        reasons = ", ".join(c.name.lower() for c in checks if c.status != "PASS") or "all checks pass"
        rows.append(f"| {name} | **{grade}** | {reasons} |")
    table = "| Input | Grade | Flagged by |\n|---|---|---|\n" + "\n".join(rows)
    (OUT / "grades.md").write_text(table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    overview()
    grade_table()
