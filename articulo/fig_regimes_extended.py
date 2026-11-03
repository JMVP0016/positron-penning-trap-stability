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
Estilo (3 nov.): el de la tesis (estilo.mplstyle y la paleta de figuras.py), con letra legible, los rótulos de
las regiones dentro del mapa y la leyenda de las fuentes de ruido debajo. La cota del gas residual va en la
leyenda escrita de la figura (Ilustr. 4-11), no dentro del mapa.

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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import figuras  # noqa: E402  (estilo y paleta comunes de la tesis)
from figuras import COLOR, tamano  # noqa: E402

IDIOMA = os.environ.get("FIG_IDIOMA", "en")
TXT = {
    "en": dict(xl=r"field noise at $f_-$, $S_E(f_-)$ (V$^2$ m$^{-2}$ Hz$^{-1}$)", yl=r"static patch field $|E|$ (V/m)",
               stable="Stable:\nbounded orbit\nand $\\bar t$ > 1 yr", inst="Immediate loss by the static patch",
               noise="Loss by noise\nin less than 1 yr", anio=r"$\bar t$ = 1 yr",
               thr="static-patch threshold: 50 V/m if sudden, 100 V/m if slow",
               t2a=r"axial channel T2, 4 K (band: range of $k$ and $\tau_z$)",
               t2b="axial channel T2, worst case at 0.1 K", surf="surface noise at 4 K, scenarios A and B (bound)",
               filt3=r"three-stage filter + EMI, 4 K (at $|E|$ = 1 V/m)", filt12=r"12 Hz filter + EMI, 4 K (at $|E|$ = 1 V/m)"),
    "es": dict(xl=r"ruido del campo a $f_-$, $S_E(f_-)$ (V$^2$ m$^{-2}$ Hz$^{-1}$)", yl=r"campo estático del parche $|E|$ (V/m)",
               stable="Estable:\nórbita acotada\ny $\\bar t$ > 1 año", inst="Pérdida inmediata por el parche estático",
               noise="Pérdida por ruido\nen menos de un año", anio=r"$\bar t$ = 1 año",
               thr="umbral del parche: 50 V/m si es súbito, 100 V/m si es lento",
               t2a=r"canal axial, 4 K (franja: según $k$ y $\tau_z$)",
               t2b="canal axial, peor caso a 0,1 K", surf="ruido de superficie a 4 K, escenarios A y B (cota)",
               filt3=r"filtro de tres etapas + EMI, 4 K (con $|E|$ = 1 V/m)", filt12=r"filtro de 12 Hz + EMI, 4 K (con $|E|$ = 1 V/m)"),
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
E_golpe, E_lento = E_critico("golpe"), E_critico("lento")
# superficie: solo cota (mín-máx de A y B a 4 K); circuitos a 4 K; canal T2: S_eq(E0) = S_G/2
lo = min(banda_superficie(np.array([4.0]), e)[0][0] for e in ("A", "B"))
hi = max(banda_superficie(np.array([4.0]), e)[1][0] for e in ("A", "B"))
esc = escenarios(4.0)
E = np.logspace(-5, np.log10(E_lento), 200)
s_lo = S_G_por_E0cuad(4.0, 1 / R_WALL, 1.0) / 2
s_hi = S_G_por_E0cuad(4.0, np.pi / Z0, 0.01) / 2
with plt.style.context(os.path.join(figuras._DIR, "estilo.mplstyle")):
    fig, ax = plt.subplots(figsize=tamano(2, 11.0))
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    # regiones: estable (verde muy claro), franja entre los dos umbrales (gris) y sus límites
    ax.fill_between([x0, S_anio], y0, E_golpe, color=COLOR["serie3"], alpha=0.08, lw=0)
    ax.fill_between([x0, x1], E_golpe, E_lento, color=COLOR["secundario"], alpha=0.3, lw=0)
    ax.axhline(E_golpe, color=COLOR["teoria"], lw=0.9); ax.axhline(E_lento, color=COLOR["teoria"], lw=0.9, ls="--")
    ax.axvline(S_anio, color=COLOR["teoria"], lw=0.9, ls=":")
    # rótulos de las regiones, dentro del mapa
    ax.text(1.25e-21, 1.0, TXT["stable"], fontsize=8.5, va="center")
    ax.text(2e-21, 1e3, TXT["inst"], fontsize=8.5, va="center")
    ax.text(S_anio * 20, 2e-5, TXT["noise"], fontsize=8.5, va="bottom")
    ax.text(S_anio * 1.4, 1e-3, TXT["anio"], fontsize=7.5, rotation=90, va="bottom")
    for Ec in (E_golpe, E_lento):
        ax.text(1.01, Ec, f"{Ec:.0f} V/m", transform=ax.get_yaxis_transform(), fontsize=7.5, va="center")
    # fuentes de ruido
    ax.fill_betweenx([y0, E_golpe], lo, hi, facecolor="none", edgecolor=COLOR["secundario"], hatch="////", lw=0.6)
    ax.plot(esc["filtro real (corte 1,2 kHz)"][0], 1.0, "o", color=COLOR["serie3"], ms=6)
    ax.plot(esc["filtro fuerte (corte 12 Hz) + EMI"][0], 1.0, "s", color=COLOR["secundario"], ms=5.5)
    ax.fill_betweenx(E, s_lo * E**2, s_hi * E**2, color=COLOR["positron"], alpha=0.22, lw=0)
    ax.plot(S_G_por_E0cuad(0.1, np.pi / Z0, 0.01) / 2 * E**2, E, color=COLOR["positron"], ls=":", lw=1.3)
    ax.set_xlabel(TXT["xl"]); ax.set_ylabel(TXT["yl"])
    # una sola leyenda debajo (en columnas: superficie y circuitos; canal axial y umbral)
    manejadores = [Patch(facecolor="none", edgecolor=COLOR["secundario"], hatch="////", lw=0.6),
                   Line2D([], [], color=COLOR["serie3"], marker="o", ls="none", ms=6),
                   Line2D([], [], color=COLOR["secundario"], marker="s", ls="none", ms=5.5),
                   Patch(facecolor=COLOR["positron"], alpha=0.22, lw=0),
                   Line2D([], [], color=COLOR["positron"], ls=":", lw=1.3),
                   Patch(facecolor=COLOR["secundario"], alpha=0.3, lw=0)]
    etiquetas = [TXT["surf"], TXT["filt3"], TXT["filt12"], TXT["t2a"], TXT["t2b"], TXT["thr"]]
    fig.legend(manejadores, etiquetas, loc="outside lower center", ncol=2, fontsize=7.5, frameon=False,
               columnspacing=1.6, handlelength=2.4)
    suf = "" if IDIOMA == "en" else "_es"
    fig.savefig(os.path.join(args.salida, f"fig_regimes_extended{suf}.pdf"))
    fig.savefig(os.path.join(args.salida, f"fig_regimes_extended{suf}.png"))
E_cruce = np.sqrt(S_anio / s_hi)
print(f"S_E de 1 año = {S_anio:.3e}; T2 a 4 K: S_eq/E0^2 entre {s_lo:.3e} y {s_hi:.3e} (Hz^-1);"
      f" peor caso: t < 1 año si E0 > {E_cruce:.3g} V/m;  superficie a 4 K: S_E entre {lo:.2e} y {hi:.2e}")
