from __future__ import annotations

"""Animacion 3D didactica de la deriva por parches para el anexo.

Este es el unico codigo separado de las corridas principales. No introduce un
modelo nuevo: importa los codigos de 4 K y 0,1 K, toma la realizacion
representativa de cada ensamble y anima la difusion del centro guia:

    dX, dY ~ N(0, sqrt(2 D_gc dt))

La vista 3D muestra la deriva secular en el plano z=0. La oscilacion axial es
mucho mas rapida que la deriva de horas, por eso se representa como escala
vertical de referencia y no como una helice artificial.

Salidas:
  animA_01_deriva_centro_guia_4K_01K.gif
  figA_03_frame_animacion_deriva.pdf
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import penning_clasico_parches as clasico
import penning_01K_contexto_cuantico as cuantico


HOUR = 3600.0


def build_classic_case():
    constants = clasico.get_constants()
    config = clasico.get_default_config()
    config = clasico.replace(config, save_figures=False)
    trap = clasico.resolve_trap_parameters(config, constants)
    freqs = clasico.compute_frequencies(trap, constants)
    result = clasico.run_ensemble(config.temperature_K, config, trap, freqs, constants, config.seed_base)
    envelope = (
        result.representative.amplitudes.A_plus
        + result.representative.amplitudes.A_minus
    )
    axial_amplitude = result.representative.amplitudes.A_z
    return {
        "label": "4 K, clasico",
        "title": "Deriva 3D por parches a T = 4 K",
        "trace": result.representative.trace,
        "trap": trap,
        "result": result,
        "envelope": envelope,
        "axial_amplitude": axial_amplitude,
        "color": "tab:blue",
    }


def build_quantum_case():
    constants = cuantico.get_constants()
    config = cuantico.get_default_config()
    config = cuantico.replace(config, save_figures=False)
    trap = cuantico.resolve_trap_parameters(config, constants)
    freqs = cuantico.compute_frequencies(trap, constants)
    thermal_state = cuantico.compute_quantum_thermal_state(config, freqs, constants)
    result = cuantico.run_ensemble(config, trap, freqs, thermal_state, config.seed_base)
    envelope = (
        result.representative.thermal_state.A_eff_plus
        + result.representative.thermal_state.A_eff_minus
    )
    axial_amplitude = result.representative.thermal_state.A_eff_z
    return {
        "label": "0,1 K, cuantico",
        "title": "Deriva 3D por parches a T = 0,1 K",
        "trace": result.representative.trace,
        "trap": trap,
        "result": result,
        "envelope": envelope,
        "axial_amplitude": axial_amplitude,
        "color": "tab:orange",
    }


def resample_trace(trace: dict[str, np.ndarray], n_frames: int) -> dict[str, np.ndarray]:
    t = trace["t_gc"]
    idx = np.unique(np.linspace(0, t.size - 1, n_frames).astype(int))
    return {
        "t_gc": trace["t_gc"][idx],
        "X_gc": trace["X_gc"][idx],
        "Y_gc": trace["Y_gc"][idx],
        "R_gc": trace["R_gc"][idx],
    }


def configure_3d_axis(ax, case: dict) -> None:
    trap = case["trap"]
    rw_mm = trap.r_wall * 1e3
    theta = np.linspace(0.0, 2.0 * np.pi, 480)
    z_visual_mm = max(case["axial_amplitude"] * 1e3 * 1.25, 0.02)
    ax.plot3D(
        rw_mm * np.cos(theta),
        rw_mm * np.sin(theta),
        np.zeros_like(theta),
        color="tab:red",
        ls="--",
        lw=1.3,
        label=r"Pared radial $r_{wall}$",
    )
    safe_radius_mm = (trap.r_wall - case["envelope"]) * 1e3
    ax.plot3D(
        safe_radius_mm * np.cos(theta),
        safe_radius_mm * np.sin(theta),
        np.zeros_like(theta),
        color="0.35",
        ls=":",
        lw=1.1,
        label=r"$r_{wall} - A_{rad}$",
    )
    for z_ref, ls, alpha in [(-z_visual_mm, ":", 0.42), (z_visual_mm, ":", 0.42)]:
        ax.plot3D(
            0.20 * rw_mm * np.cos(theta),
            0.20 * rw_mm * np.sin(theta),
            np.full_like(theta, z_ref),
            color="0.55",
            ls=ls,
            lw=0.8,
            alpha=alpha,
        )
    ax.plot3D([0.0, 0.0], [0.0, 0.0], [-z_visual_mm, z_visual_mm],
              color="0.25", lw=1.2, label=r"Escala axial $A_z$")
    ax.scatter([0.0], [0.0], [0.0], s=32, color="tab:green", label="Inicio", depthshade=False)
    ax.set_xlim(-rw_mm * 1.08, rw_mm * 1.08)
    ax.set_ylim(-rw_mm * 1.08, rw_mm * 1.08)
    ax.set_zlim(-z_visual_mm, z_visual_mm)
    ax.set_box_aspect((1.0, 1.0, 0.42))
    ax.view_init(elev=48, azim=-58)
    ax.set_xlabel(r"$X_{gc}$ [mm]")
    ax.set_ylabel(r"$Y_{gc}$ [mm]")
    ax.set_zlabel("z [mm]")
    ax.set_title(case["title"])
    ax.grid(True, alpha=0.22, linewidth=0.6)


def configure_projection_axis(ax: plt.Axes, case: dict) -> None:
    trap = case["trap"]
    rw_mm = trap.r_wall * 1e3
    theta = np.linspace(0.0, 2.0 * np.pi, 480)
    safe_radius_mm = (trap.r_wall - case["envelope"]) * 1e3
    ax.plot(rw_mm * np.cos(theta), rw_mm * np.sin(theta),
            color="tab:red", ls="--", lw=1.2, label=r"$r_{wall}$")
    ax.plot(safe_radius_mm * np.cos(theta), safe_radius_mm * np.sin(theta),
            color="0.35", ls=":", lw=1.0, label=r"$r_{wall}-A_{rad}$")
    ax.scatter([0.0], [0.0], s=24, color="tab:green", label="Inicio", zorder=5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-rw_mm * 1.08, rw_mm * 1.08)
    ax.set_ylim(-rw_mm * 1.08, rw_mm * 1.08)
    ax.set_xlabel(r"$X_{gc}$ [mm]")
    ax.set_ylabel(r"$Y_{gc}$ [mm]")
    ax.set_title("Proyección radial del centro guía")
    ax.grid(True, alpha=0.22, linewidth=0.6)


def make_animation(out_dir: Path, frames: int, fps: int, dpi: int) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = [build_classic_case(), build_quantum_case()]
    for case in cases:
        if case["trace"] is None:
            raise RuntimeError(f"La realizacion representativa de {case['label']} no tiene traza.")
        case["anim_trace"] = resample_trace(case["trace"], frames)

    fig = plt.figure(figsize=(12.6, 9.4), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=(4.6, 2.55, 1.18))
    axes_3d = [fig.add_subplot(gs[0, 0], projection="3d"), fig.add_subplot(gs[0, 1], projection="3d")]
    axes_2d = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    info_ax = fig.add_subplot(gs[2, :])
    info_ax.axis("off")

    artists = []
    for ax3d, ax2d, case in zip(axes_3d, axes_2d, cases):
        configure_3d_axis(ax3d, case)
        configure_projection_axis(ax2d, case)
        (line3d,) = ax3d.plot3D([], [], [], color=case["color"], lw=1.8, label="Centro guia")
        point3d = ax3d.scatter([], [], [], s=76, color=case["color"], edgecolor="black", depthshade=False)
        (line2d,) = ax2d.plot([], [], color=case["color"], lw=1.65, label="Centro guia")
        point2d = ax2d.scatter([], [], s=58, color=case["color"], edgecolor="black", zorder=7)
        progress3d = ax3d.text2D(
            0.03,
            0.96,
            "",
            transform=ax3d.transAxes,
            ha="left",
            va="top",
            bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "0.75", "alpha": 0.9},
            fontsize=9,
        )
        progress2d = ax2d.text(
            0.03,
            0.95,
            "",
            transform=ax2d.transAxes,
            ha="left",
            va="top",
            bbox={"boxstyle": "round,pad=0.26", "fc": "white", "ec": "0.75", "alpha": 0.9},
            fontsize=8.5,
        )
        ax3d.legend(loc="upper left", fontsize=7.6)
        ax2d.legend(loc="lower right", fontsize=7.8)
        artists.append((line3d, point3d, line2d, point2d, progress3d, progress2d))

    model_text = info_ax.text(
        0.02,
        0.72,
        "",
        transform=info_ax.transAxes,
        ha="left",
        va="center",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "0.78", "alpha": 0.95},
    )
    comparison_text = info_ax.text(
        0.98,
        0.72,
        "",
        transform=info_ax.transAxes,
        ha="right",
        va="center",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "0.78", "alpha": 0.95},
    )

    ratio = cases[0]["result"].median_t_conf / cases[1]["result"].median_t_conf

    def update(frame_idx: int):
        updated = []
        for case, (line3d, point3d, line2d, point2d, progress3d, progress2d) in zip(cases, artists):
            trace = case["anim_trace"]
            x_mm = trace["X_gc"][: frame_idx + 1] * 1e3
            y_mm = trace["Y_gc"][: frame_idx + 1] * 1e3
            z_mm = np.zeros_like(x_mm)
            line3d.set_data_3d(x_mm, y_mm, z_mm)
            point3d._offsets3d = ([x_mm[-1]], [y_mm[-1]], [0.0])
            line2d.set_data(x_mm, y_mm)
            point2d.set_offsets([[x_mm[-1], y_mm[-1]]])

            r_eff_mm = (trace["R_gc"][frame_idx] + case["envelope"]) * 1e3
            rw_mm = case["trap"].r_wall * 1e3
            progress3d.set_text(
                f"t = {trace['t_gc'][frame_idx] / HOUR:.2f} h\n"
                f"$R_{{eff}}/r_{{wall}}$ = {r_eff_mm / rw_mm:.2f}\n"
                f"$D_{{gc}}$ = {case['trap'].D_gc:.2e} m²/s"
            )
            progress2d.set_text(
                f"$A_{{rad}}$ = {case['envelope']*1e6:.3f} µm\n"
                f"$A_z$ = {case['axial_amplitude']*1e6:.3f} µm\n"
                f"$t_{{50}}$ = {case['result'].median_t_conf/HOUR:.2f} h"
            )
            updated.extend([line3d, point3d, line2d, point2d, progress3d, progress2d])

        model_text.set_text(
            "Animación 3D didáctica: la deriva lenta se dibuja en el plano secular z=0; "
            "la escala vertical representa la amplitud axial.\n"
            r"$dX,dY \sim \mathcal{N}(0,\sqrt{2D_{gc}\,dt})$; "
            r"perdida cuando $R_{gc}+A_{rad}\geq r_{wall}$"
        )
        comparison_text.set_text(
            f"Medianas: 4 K = {cases[0]['result'].median_t_conf/HOUR:.2f} h, "
            f"0,1 K = {cases[1]['result'].median_t_conf/HOUR:.2f} h\n"
            f"Cociente = {ratio:.2f}  (esperado $\\sqrt{{40}}$ = {np.sqrt(40):.2f})"
        )
        updated.extend([model_text, comparison_text])
        return updated

    update(frames - 1)
    frame_path = out_dir / "figA_03_frame_animacion_deriva.pdf"
    fig.savefig(frame_path, dpi=dpi, bbox_inches="tight")

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    gif_path = out_dir / "animA_01_deriva_centro_guia_4K_01K.gif"
    ani.save(gif_path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return gif_path, frame_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera la animacion didactica de deriva por parches.")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--fps", type=int, default=18)
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()
    gif_path, frame_path = make_animation(args.out_dir, args.frames, args.fps, args.dpi)
    print(f"Animacion: {gif_path}")
    print(f"Frame estatico: {frame_path}")


if __name__ == "__main__":
    main()
