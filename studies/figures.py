"""Generate all paper figures from cached study data (paper section 5.3).

Fig 1  power-balance control-volume schematic (drawn, not computed)
Fig 2  IBL verification: Blasius + turbulent flat plate with error panels
Fig 3  D8 dissipation decomposition vs Hall 2017 measurement
Fig 4  headline heatmap: subsystem PSC and net saving vs (f_Phi, FPR)
Fig 5  diminishing returns: normalized saving vs f_Phi + design curves
Fig 6  Sobol total-order indices per output (pilot or production)
Fig 7  UQ histograms with 5-95% bands and published point estimates
Fig 8  block-fuel delta vs mission range at three ingestion fractions

Run:  uv run python studies/figures.py [--uq-tag ""]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plotstyle  # noqa: E402
from plotstyle import C_BLUE, C_GREEN, C_ORANGE, C_VERM  # noqa: E402

plotstyle.apply()
FIG = plotstyle.FIGDIR
DATA = plotstyle.DATADIR


def uq_file(stem: str) -> Path:
    """Resolve a UQ artifact: data/ first, then data/production/ (where the
    production run writes so its large samples stay out of git)."""
    p = DATA / stem
    if not p.exists() and (DATA / "production" / stem).exists():
        return DATA / "production" / stem
    return p

INPUT_LABELS = {
    "f_phi": r"$f_\Phi$",
    "fpr": "FPR",
    "x_tr": r"$x_{tr}/L$",
    "n_powerlaw": r"$n$ (profile)",
    "eta_pol": r"$\eta_{pol}$",
    "k_dist": r"$k_{dist}$",
    "eta_elec": r"$\eta_{elec}$",
}


# ---------------------------------------------------------------- Fig 1
def fig1_schematic() -> None:
    from blipb import BLIComparator

    comp = BLIComparator()
    bl = comp.bl
    xL = bl.x / comp.fuselage.length
    rL = bl.r / comp.fuselage.length

    fig, ax = plt.subplots(figsize=(6.5, 2.9))
    ax.fill_between(xL, -rL, rL, color="0.85", edgecolor="0.4", linewidth=0.8)
    # boundary-layer edge (exaggerated x3 for legibility)
    delta_vis = 3.0 * bl.theta_star / comp.fuselage.length * 8
    ax.plot(xL, rL + delta_vis, color=C_BLUE, linewidth=1.2)
    ax.annotate(
        r"boundary layer: $\theta^*(x)$ = cumulative $\Phi_{surf}$",
        xy=(0.62, float(rL[int(0.62 * len(xL))] + delta_vis[int(0.62 * len(xL))])),
        xytext=(0.35, 0.115),
        arrowprops=dict(arrowstyle="->", lw=0.8),
        fontsize=8.5,
    )
    # fan disk at tail
    r_te = rL[-1]
    y_cap = (bl.r_te + comp.profile.delta) / comp.fuselage.length
    ax.plot([1.0, 1.0], [r_te, y_cap], color=C_VERM, linewidth=3.5, solid_capstyle="butt")
    ax.annotate(
        "BLI fan\n(ingested annulus)",
        xy=(1.0, 0.5 * (r_te + y_cap)),
        xytext=(0.80, 0.14),
        arrowprops=dict(arrowstyle="->", lw=0.8),
        fontsize=8.5,
        color=C_VERM,
    )
    # jet
    xj = np.linspace(1.0, 1.12, 20)
    ax.fill_between(xj, -0.1 * (xj - 1) / 0.12 * 0 + r_te * 0.0, y_cap * np.exp(-(xj - 1) * 8) + 0.01,
                    color=C_VERM, alpha=0.25, linewidth=0)
    ax.annotate(r"jet: $\Phi_{jet}=\frac{1}{2}\dot m (V_j-V_\infty)^2$",
                xy=(1.06, 0.035), fontsize=8.5, color=C_VERM)
    # wake (residual, non-ingested)
    ax.annotate(r"residual wake: $\Phi_{wake}=\dot E_{a,res}$",
                xy=(1.02, -0.05), fontsize=8.5, color=C_BLUE)
    # Trefftz plane
    ax.axvline(1.15, color="0.3", linestyle="--", linewidth=0.9)
    ax.annotate("Trefftz plane\n($p=p_\\infty$)", xy=(1.152, 0.10), fontsize=8)
    # control volume
    ax.add_patch(
        plt.Rectangle((-0.05, -0.16), 1.20, 0.34, fill=False, edgecolor="0.5",
                      linestyle=":", linewidth=1.0)
    )
    ax.annotate("control volume (shared by podded and BLI cases)",
                xy=(-0.04, 0.165), fontsize=8, color="0.35")
    ax.annotate(r"$V_\infty$", xy=(-0.045, 0.01), fontsize=10)
    ax.annotate("", xy=(0.02, 0.0), xytext=(-0.02, 0.0),
                arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.set_xlim(-0.07, 1.30)
    ax.set_ylim(-0.17, 0.20)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(FIG / "fig1_schematic.pdf")
    fig.savefig(FIG / "fig1_schematic.png")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 2
def fig2_verification() -> None:
    from blipb.ibl.head import solve_head
    from blipb.ibl.thwaites import solve_thwaites

    NU, U = 1.5e-5, 50.0
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.6))

    # Blasius
    x = np.linspace(1e-4, 1.0, 800)
    lam = solve_thwaites(x, np.full_like(x, U), NU)
    re_x = U * x / NU
    ax = axs[0]
    ax.loglog(re_x, lam.cf, color=C_BLUE, label="Thwaites (this work)")
    ax.loglog(re_x, 0.664 / np.sqrt(re_x), "--", color="0.25", label="Blasius exact")
    ax.set_xlabel(r"$Re_x$")
    ax.set_ylabel(r"$C_f$")
    ax.set_title("(a) laminar flat plate", fontsize=9)
    ax.legend(fontsize=8)

    # Turbulent
    xt = np.geomspace(0.09, 30.0, 700)
    turb = solve_head(xt, np.full_like(xt, U), NU,
                      theta0=0.036 * 0.09 * (U * 0.09 / NU) ** -0.2, H0=1.40)
    re_xt = U * xt / NU
    ax = axs[1]
    ax.loglog(re_xt, turb.cf, color=C_VERM, label="Head + L-T (this work)")
    ax.loglog(re_xt, 0.0592 * re_xt**-0.2, "--", color="0.25", label=r"$0.0592\,Re_x^{-1/5}$")
    ax.loglog(re_xt, 0.370 * np.log10(re_xt) ** -2.584, ":", color=C_GREEN,
              label="Schultz-Grunow")
    ax.set_xlabel(r"$Re_x$")
    ax.set_title("(b) turbulent flat plate", fontsize=9)
    ax.legend(fontsize=8)
    fig.savefig(FIG / "fig2_verification.pdf")
    fig.savefig(FIG / "fig2_verification.png")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 3
def fig3_d8_decomposition() -> None:
    d8 = pd.read_csv(DATA / "validation_d8.csv")

    def get(q):
        return float(d8.loc[d8.quantity == q, "value"].iloc[0])

    cats = ["jet", "surface", "wake", "total"]
    this = [get("jet contribution (this work)"), 0.0,
            get("wake contribution (this work)"), get("PSC this work")]
    hall = [0.052, 0.024, 0.006, 0.082]

    x = np.arange(len(cats))
    w = 0.36
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    b1 = ax.bar(x - w / 2, np.array(hall) * 100, w, color=C_BLUE,
                label="D8 measured (Hall 2017)")
    b2 = ax.bar(x + w / 2, np.array(this) * 100, w, color=C_VERM,
                label="this work (low-order)", hatch="//", edgecolor="white", linewidth=0.5)
    ax.errorbar([len(cats) - 1 - w / 2], [8.2], yerr=[0.8], fmt="none",
                ecolor="0.2", capsize=3, linewidth=1)
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.1f", fontsize=7.5, padding=1.5)
    ax.set_xticks(x, ["jet", "surface", "wake", "total"])
    ax.set_ylabel("PSC contribution [%]")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(0, 11.5)
    fig.savefig(FIG / "fig3_d8_decomposition.pdf")
    fig.savefig(FIG / "fig3_d8_decomposition.png")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 4
def fig4_heatmap() -> None:
    df = pd.read_parquet(DATA / "atlas_fphi_fpr.parquet")
    f = np.sort(df.f_phi.unique())
    p = np.sort(df.fpr.unique())
    psc = df.pivot(index="fpr", columns="f_phi", values="psc").values * 100
    net = df.pivot(index="fpr", columns="f_phi", values="net_saving").values * 100

    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True)
    for ax, z, title in (
        (axs[0], psc, "(a) subsystem PSC [%]"),
        (axs[1], net, "(b) aircraft net power saving [%]"),
    ):
        im = ax.pcolormesh(f, p, z, cmap="viridis", shading="nearest")
        cs = ax.contour(f, p, z, colors="white", linewidths=0.7,
                        levels=6)
        ax.clabel(cs, fontsize=7, fmt="%.1f")
        ax.set_xlabel(r"ingested dissipation fraction $f_\Phi$")
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.9)
    axs[0].set_ylabel("FPR")
    # STARC-ABL operating point marker
    axs[1].plot([0.78], [1.25], marker="*", markersize=11, color=C_VERM,
                markeredgecolor="white", markeredgewidth=0.6, linestyle="none")
    axs[1].annotate("STARC-ABL\n(1/3 thrust)", xy=(0.78, 1.25), xytext=(0.45, 1.30),
                    fontsize=7.5, color=C_VERM,
                    arrowprops=dict(arrowstyle="->", lw=0.7, color=C_VERM))
    fig.savefig(FIG / "fig4_heatmap.pdf")
    fig.savefig(FIG / "fig4_heatmap.png")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 5
def fig5_saturation() -> None:
    sat = pd.read_parquet(DATA / "atlas_saturation.parquet")
    des = pd.read_parquet(DATA / "atlas_design_curves.parquet")

    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7))
    ax = axs[0]
    styles = ["-", "--", ":"]
    for (fpr, g), ls in zip(sat.groupby("fpr"), styles):
        ax.plot(g.f_phi, g.saving_norm, ls, label=f"FPR = {fpr:.2f}")
    ax.plot([0, 1], [0, 1], color="0.7", linewidth=0.8, label="linear return")
    ax.set_xlabel(r"$f_\Phi$")
    ax.set_ylabel(r"saving / saving($f_\Phi{=}0.95$)")
    ax.set_title("(a) diminishing returns", fontsize=9)
    ax.legend(fontsize=7.5)

    ax = axs[1]
    for (share, g), ls in zip(des.groupby("thrust_share"), styles):
        ax.plot(g.f_phi, g.psc * 100, ls, label=f"thrust share {share:.2f}")
    ax.set_xlabel(r"achieved $f_\Phi$")
    ax.set_ylabel("subsystem PSC [%]")
    ax.set_title("(b) force-driven design curves (FPR swept)", fontsize=9)
    ax.legend(fontsize=7.5)
    fig.savefig(FIG / "fig5_saturation.pdf")
    fig.savefig(FIG / "fig5_saturation.png")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 6
def fig6_sobol(tag: str = "") -> None:
    outputs = ["psc_aero", "psc_net", "delta_fuel"]
    titles = {
        "psc_aero": r"PSC$_{aero}$ (subsystem)",
        "psc_net": "net power saving",
        "delta_fuel": r"$\Delta$ block fuel",
    }
    fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.6), sharey=True)
    order = None
    for ax, name in zip(axs, outputs):
        st = pd.read_parquet(uq_file(f"uq{tag}_sobol_{name}.parquet"))
        pce = pd.read_parquet(uq_file(f"uq{tag}_pce_sobol_{name}.parquet"))
        if order is None:
            order = list(st["ST"].sort_values().index)
        y = np.arange(len(order))
        ax.barh(y + 0.18, st.loc[order, "ST"], height=0.34, color=C_BLUE,
                xerr=st.loc[order, "ST_conf"], error_kw=dict(lw=0.7), label="Saltelli-Sobol")
        ax.barh(y - 0.18, pce.loc[order, "ST"], height=0.34, color=C_ORANGE,
                hatch="//", edgecolor="white", linewidth=0.4, label="PCE")
        ax.set_yticks(y, [INPUT_LABELS[k] for k in order])
        ax.set_title(titles[name], fontsize=9)
        ax.set_xlabel(r"total-order $S_T$")
    axs[0].legend(fontsize=7.5, loc="lower right")
    fig.savefig(FIG / "fig6_sobol.pdf")
    fig.savefig(FIG / "fig6_sobol.png")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 7
def fig7_uq_histograms(tag: str = "") -> None:
    mc = pd.read_parquet(uq_file(f"uq{tag}_mc_samples.parquet"))
    pct = pd.read_parquet(uq_file(f"uq{tag}_mc_percentiles.parquet"))

    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7))

    ax = axs[0]
    vals = mc["psc_aero"] * 100
    ax.hist(vals, bins=60, color=C_BLUE, alpha=0.75, density=True)
    for p in ("p5", "p50", "p95"):
        ax.axvline(pct.loc["psc_aero", p] * 100, color="0.2", linestyle="--", linewidth=0.9)
    ax.axvline(8.2, color=C_VERM, linewidth=1.4)
    ax.annotate("Uranga D8\n8.2%", xy=(8.2, ax.get_ylim()[1] * 0.72), fontsize=7.5,
                color=C_VERM, ha="right")
    ax.set_xlabel(r"subsystem PSC$_{aero}$ [%]")
    ax.set_ylabel("density")
    ax.set_title("(a) aerodynamic PSC across the input space", fontsize=9)

    ax = axs[1]
    vals = mc["delta_fuel"] * 100
    ax.hist(vals, bins=60, color=C_GREEN, alpha=0.75, density=True)
    for p in ("p5", "p50", "p95"):
        ax.axvline(pct.loc["delta_fuel", p] * 100, color="0.2", linestyle="--", linewidth=0.9)
    for val, lbl, col in ((-3.4, "NASA TM\n-3.4%", C_VERM), (1.7, "Giannakakis\n+1.7%", C_ORANGE)):
        ax.axvline(val, color=col, linewidth=1.4)
        ax.annotate(lbl, xy=(val, ax.get_ylim()[1] * 0.7), fontsize=7.5, color=col,
                    ha="left" if val > 0 else "right")
    ax.set_xlabel(r"$\Delta$ block fuel [%] (3000 nmi)")
    ax.set_title("(b) block-fuel delta with literature estimates", fontsize=9)
    fig.savefig(FIG / "fig7_uq_histograms.pdf")
    fig.savefig(FIG / "fig7_uq_histograms.png")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 8
def fig8_fuel_vs_range() -> None:
    from blipb import BLIComparator
    from blipb.atmosphere import G0
    from blipb.mission.breguet import NMI, delta_block_fuel
    from blipb.uq.model import DEFAULT_CONFIG as cfg

    comp = BLIComparator(fuselage=cfg.fuselage(), flight=cfg.flight())
    d_total = cfg.w_initial * G0 / cfg.lift_drag
    p_ref = d_total * cfg.flight().V / cfg.eta_prop_ref

    ranges = np.linspace(500, 3500, 25)
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    styles = ["-", "--", ":"]
    for f_phi, ls in zip((0.3, 0.5, 0.8), styles):
        r = comp.run_design(f_phi=f_phi, fpr=1.25)
        psc_net = (r.p_shaft_pod - r.p_shaft_bli / 0.92) / p_ref
        dfuel = [
            delta_block_fuel(psc_net, range_m=rr * NMI, v=cfg.flight().V,
                             lift_drag=cfg.lift_drag, tsfc=cfg.tsfc,
                             w_initial=cfg.w_initial, snowball=1.35) * 100
            for rr in ranges
        ]
        ax.plot(ranges, dfuel, ls, label=rf"$f_\Phi$ = {f_phi}")
    ax.set_xlabel("mission range [nmi]")
    ax.set_ylabel(r"$\Delta$ block fuel [%]")
    ax.legend(fontsize=8)
    ax.set_title("turboelectric chain, FPR 1.25, snowball 1.35", fontsize=9)
    fig.savefig(FIG / "fig8_fuel_vs_range.pdf")
    fig.savefig(FIG / "fig8_fuel_vs_range.png")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--uq-tag", type=str, default="",
                    help='"" for pilot data, "_prod" for production run')
    args = ap.parse_args()
    for fn in (fig1_schematic, fig2_verification, fig3_d8_decomposition,
               fig4_heatmap, fig5_saturation):
        fn()
        print(f"{fn.__name__} done")
    for fn in (fig6_sobol, fig7_uq_histograms):
        fn(args.uq_tag)
        print(f"{fn.__name__} done")
    fig8_fuel_vs_range()
    print("fig8_fuel_vs_range done")
    print(f"\nAll figures written to {FIG}")

