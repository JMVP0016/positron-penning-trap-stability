"""
fig_regimes_extended.py — A1, ronda 3, tarea A1-3 (R3-2d, I3d): mapa de regímenes ampliado.

Es la figura fig4_08 (ruido S_E(f_-) en abscisas, campo estático |E| en ordenadas) con estos añadidos:
  - canal axial T2 (A1 r2, i3a): su densidad equivalente isótropa es S_eq = S_G/2, porque el canal
    es de una sola componente (D = S_G/(8 B0^2) = S_eq/(4 B0^2)). Como S_G ∝ E0^2, en este mapa es
    una franja de pendiente 2 entre (k = 1/a, tau_z = 1 s) y (k = pi/z0, tau_z = 0,01 s), a 4 K; a 0,1 K
    se dibuja la línea del peor caso;
  - ruido de superficie SOLO como cota: banda mín-máx de los escenarios A y B a 4 K (M8), sin curva central;
  - gas residual: solo una anotación con la cota de presión de Sellner 2017 (vía A3 r2). Sin cifra de
    aniquilación, porque Z_eff está pendiente de verificar (A3).
Idioma: FIG_IDIOMA=en (por defecto, punto decimal) o FIG_IDIOMA=es, como el mecanismo del parche de A4.

Uso: FIG_IDIOMA=en MPLBACKEND=Agg python3 fig_regimes_extended.py [--codigo RUTA] [--salida CARPETA]
"""
import argparse
import os
import sys
from math import factorial

sys.dont_write_bytecode = True
ap = argparse.ArgumentParser()
ap.add_argument("--codigo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap.add_argument("--salida", default=os.path.dirname(os.path.abspath(__file__)))
args = ap.parse_args()
sys.path.insert(0, args.codigo)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from centro_guia import E_critico  # noqa: E402
from params import B0, F_M, K_B, M_E, R_WALL, UN_ANIO, W_Z, Z0  # noqa: E402
from ruido import SE_desde_D, escenarios  # noqa: E402
from parches_fluctuantes import banda_superficie  # noqa: E402

IDIOMA = os.environ.get("FIG_IDIOMA", "en")
TXT = {
    "en": dict(xl=r"field noise at $f_-$, $S_E(f_-)$ (V$^2$ m$^{-2}$ Hz$^{-1}$)", yl=r"static patch field $|E|$ (V/m)",
               stable="stable: bounded orbit\nand $\\bar t$ > 1 yr", inst="immediate loss by the static patch",
               thr="50–100 V/m: depends on how the field appears", noise="loss by noise\nin less than 1 yr",
               t2a="axial channel T2, 4 K\n(band: k = 1/a, τ_z = 1 s → k = π/z₀, τ_z = 0.01 s)",
               t2b="T2 worst case, 0.1 K", surf="surface noise, scenarios A–B at 4 K\n(bound only: min–max)",
               filt="circuits: real filter / strong filter + EMI (4 K)",
               gas="residual gas: p < 1.2×10⁻¹⁸ mbar (Sellner 2017);\nannihilation time pending Z_eff (not plotted)"),
    "es": dict(xl=r"ruido del campo a $f_-$, $S_E(f_-)$ (V$^2$ m$^{-2}$ Hz$^{-1}$)", yl=r"campo estático del parche $|E|$ (V/m)",
               stable="estable: órbita acotada\ny $\\bar t$ > 1 año", inst="pérdida inmediata por el parche estático",
               thr="50–100 V/m: depende de cómo aparece el campo", noise="pérdida por ruido\nen menos de un año",
               t2a="canal axial T2, 4 K\n(franja: k = 1/a, τ_z = 1 s → k = π/z₀, τ_z = 0,01 s)",
               t2b="T2, peor caso, 0,1 K", surf="ruido de superficie, escenarios A–B a 4 K\n(solo cota: mín–máx)",
               filt="circuitos: filtro real / filtro fuerte + EMI (4 K)",
               gas="gas residual: p < 1,2×10⁻¹⁸ mbar (Sellner 2017);\ntiempo de aniquilación pendiente de Z_eff (no se dibuja)"),
}[IDIOMA]


def S_G_por_E0cuad(T, k, tau, nmax=30):
    """S_G(f_-)/E0^2 del canal T2 (forma cerrada de Laguerre, A1 r2 i3a_canal_axial.py)."""
    beta = k * np.sqrt(2 * K_B * T / (M_E * W_Z**2))
    b = beta**2 / 4
    a = np.array([b**n * np.exp(-b) / factorial(n) for n in range(nmax + 1)])
    n = np.arange(1, nmax + 1)
    return np.sum(a[1:] ** 2 * 4 * (tau / n) / (1 + (2 * np.pi * F_M * tau / n) ** 2))


x0, x1, y0, y1 = 1e-21, 1e-8, 1e-5, 1e4
S_anio = SE_desde_D(R_WALL**2 / (4 * UN_ANIO))
fig, ax = plt.subplots(figsize=(15 / 2.54, 9 / 2.54))
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
ax.fill_between([x0, S_anio], y0, E_critico("golpe"), color="#1baf7a", alpha=0.10, lw=0)
ax.fill_between([x0, x1], E_critico("golpe"), E_critico("lento"), color="0.5", alpha=0.2, lw=0)
ax.axhline(E_critico("golpe"), color="k", lw=0.9); ax.axhline(E_critico("lento"), color="k", lw=0.9, ls="--")
ax.axvline(S_anio, color="k", lw=0.9, ls=":")
ax.text(2e-21, 1e-4, TXT["stable"], fontsize=7)
ax.text(2e-21, 400, TXT["inst"], fontsize=7)
ax.text(2e-21, 70, TXT["thr"], fontsize=6, va="center")
ax.text(S_anio * 30, 3, TXT["noise"], fontsize=7)
# superficie: solo cota (mín-máx de A y B a 4 K)
lo = min(banda_superficie(np.array([4.0]), e)[0][0] for e in ("A", "B"))
hi = max(banda_superficie(np.array([4.0]), e)[1][0] for e in ("A", "B"))
ax.axvspan(lo, hi, ymax=0.62, facecolor="none", edgecolor="tab:purple", hatch="////", lw=0.6, label=TXT["surf"])
# circuitos a 4 K (puntos sobre E = 1 V/m, como fig4_08)
esc = escenarios(4.0)
ax.plot([esc["filtro real (corte 1,2 kHz)"][0], esc["filtro fuerte (corte 12 Hz) + EMI"][0]], [1.0, 1.0], "o",
        mfc="none", color="tab:blue", ms=5, label=TXT["filt"])
# canal T2: S_eq(E0) = S_G/2
E = np.logspace(-5, np.log10(E_critico("lento")), 200)
s_lo = S_G_por_E0cuad(4.0, 1 / R_WALL, 1.0) / 2
s_hi = S_G_por_E0cuad(4.0, np.pi / Z0, 0.01) / 2
ax.fill_betweenx(E, s_lo * E**2, s_hi * E**2, color="tab:red", alpha=0.25, lw=0, label=TXT["t2a"])
ax.plot(S_G_por_E0cuad(0.1, np.pi / Z0, 0.01) / 2 * E**2, E, color="tab:red", ls=":", lw=1, label=TXT["t2b"])
ax.text(1.5e-21, 3e3, TXT["gas"], fontsize=6, color="0.25")
ax.set_xlabel(TXT["xl"]); ax.set_ylabel(TXT["yl"])
ax.legend(loc="lower right", fontsize=5.8, framealpha=0.9)
fig.tight_layout()
suf = "" if IDIOMA == "en" else "_es"
fig.savefig(os.path.join(args.salida, f"fig_regimes_extended{suf}.pdf"))
fig.savefig(os.path.join(args.salida, f"fig_regimes_extended{suf}.png"), dpi=130)
E_cruce = np.sqrt(S_anio / s_hi)
print(f"S_E de 1 año = {S_anio:.3e}; T2 a 4 K: S_eq/E0^2 entre {s_lo:.3e} y {s_hi:.3e} (Hz^-1);"
      f" peor caso: t < 1 año si E0 > {E_cruce:.3g} V/m;  superficie a 4 K: S_E entre {lo:.2e} y {hi:.2e}")
