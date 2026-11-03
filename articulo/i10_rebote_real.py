"""
i10_rebote_real.py — A1, ronda 3, tarea A1-2 (R3-2b).

Prueba de tránsito con el REBOTE de un pozo de fondo plano (velocidad constante y reflexión en los
extremos, z(t) en onda triangular) en lugar de la oscilación armónica de transito.py.
Unidades escaladas: w_- = 1, B = U0 = 1. Amplitud del rebote = 1 (media longitud del tubo).
Cociente de la microtrampa: w_b/w_- = 161 (L = 10 cm, 5 eV; i10_microtrampa_r3.py).
El patrón de parches estático es el de transito.patrones_axiales: correlación gaussiana de
longitud ELL, con ELL = 0,1 y 0,02 (el positrón cruza ~20 y ~100 parches por recorrido).

Uso: MPLBACKEND=Agg python3 i10_rebote_real.py [--codigo RUTA] [--salida CARPETA]   (determinista, ~1-2 min)
"""
import argparse
import csv
import os
import sys

sys.dont_write_bytecode = True
ap = argparse.ArgumentParser()
ap.add_argument("--codigo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap.add_argument("--salida", default=os.path.dirname(os.path.abspath(__file__)))
args = ap.parse_args()
sys.path.insert(0, args.codigo)

import numpy as np  # noqa: E402

import transito as tr  # noqa: E402

RAZON = 161.0


def z_rebote(fase):
    """Onda triangular de amplitud 1: velocidad constante y reflexión en +-1. fase en [0, 2 pi)."""
    u = (fase / (2 * np.pi)) % 1.0
    return np.where(u < 0.5, -1 + 4 * u, 3 - 4 * u)


def z_armonico(fase):
    return np.cos(fase)


def prueba(perfil, ell, n_tray=300, n_periodos=6, pasos_axial=256, semilla=2026):
    tr.ELL = ell
    tr.W_Z = RAZON
    rng = np.random.default_rng(semilla)
    E_de_z = tr.patrones_axiales(n_tray, rng)
    fase0 = rng.uniform(0, 2 * np.pi, n_tray)
    Tz = 2 * np.pi / RAZON
    dt = Tz / pasos_axial
    E_per = np.array([E_de_z(perfil(RAZON * j * dt + fase0)) for j in range(pasos_axial)])
    G0 = E_per.mean(axis=0)
    n_pasos = int(round(n_periodos * 2 * np.pi / dt))
    Z = np.zeros(n_tray, complex)
    t, R2 = [0.0], [0.0]
    for n in range(1, n_pasos + 1):
        Z = tr.paso(Z, E_per[(n - 1) % pasos_axial], E_per[n % pasos_axial], dt)
        if n % pasos_axial == 0:
            t.append(n * dt); R2.append(np.mean(np.abs(Z) ** 2))
    t, R2 = np.array(t), np.array(R2)
    teo = 2 * np.mean(np.abs(G0) ** 2) * (1 - np.cos(t))
    kP = [np.argmin(abs(t - 2 * np.pi * k)) for k in range(1, n_periodos + 1)]
    return t, R2, teo, R2.max() / teo.max(), R2[kP].max() / R2.max(), np.mean(np.abs(G0) ** 2)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    filas, curvas = [], {}
    for ell in (0.1, 0.02):
        for nom, perfil in (("bounce (flat well)", z_rebote), ("harmonic", z_armonico)):
            t, R2, teo, r_max, r_kP, g0 = prueba(perfil, ell)
            veredicto = "acotado" if r_kP < 0.02 and abs(r_max - 1) < 0.05 else "revisar"
            print(f"ELL = {ell:4}: {nom:18s}: max<|Z|^2>/teoría = {r_max:.4f}; max <|Z|^2>(t = k P_-)/max = {r_kP:.1e};"
                  f" <|G0|^2> = {g0:.4f} -> {veredicto}")
            filas.append((ell, nom, r_max, r_kP, g0, veredicto))
            curvas[(ell, nom)] = (t, R2, teo)
    tr.W_Z, tr.ELL = 50.0, 0.1
    with open(os.path.join(args.salida, "i10_rebote_real.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ELL", "perfil", "max_R2_sobre_teoria", "R2_en_kP_sobre_max", "G0_cuadrado_medio", "veredicto"])
        for f in filas:
            w.writerow([f[0], f[1], f"{f[2]:.5f}", f"{f[3]:.3e}", f"{f[4]:.5f}", f[5]])
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    for (ell, nom), (t, R2, teo) in curvas.items():
        ls = "-" if "bounce" in nom else "--"
        c = "tab:blue" if ell == 0.1 else "tab:red"
        ax.plot(t / (2 * np.pi), R2, ls=ls, color=c, label=f"{nom}, ℓ = {ell}")
        ax.plot(t / (2 * np.pi), teo, ls=":", color="k", lw=0.8)
    ax.set_xlabel(r"$t/P_-$")
    ax.set_ylabel(r"$\langle|Z|^2\rangle$ (scaled units)")
    ax.set_title(r"Static patch sampled by the axial motion, $\omega_b/\omega_-=161$: bounded, no diffusion", fontsize=8)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.salida, "i10_rebote_real.pdf"))
    fig.savefig(os.path.join(args.salida, "i10_rebote_real.png"), dpi=120)
