"""
fig_regimes_extended.py — Mapa de regímenes ampliado.

Es la figura fig4_08 (ruido S_E(f_-) en abscisas, campo estático |E| en ordenadas) con estos añadidos:
  - canal axial T2 (i3a_canal_axial.py): su densidad equivalente isótropa es S_eq = S_G/2, porque el canal
    es de una sola componente (D = S_G/(8 B0^2) = S_eq/(4 B0^2)). Como S_G ∝ E0^2, en este mapa es
    una franja de pendiente 2 entre (k = 1/a, tau_z = 1 s) y (k = pi/z0, tau_z = 0,01 s), a 4 K; a 0,1 K
    se dibuja la línea del peor caso;
  - ruido de superficie SOLO como cota: banda mín-máx de los escenarios A y B a 4 K (M8), sin curva central;
  - gas residual: no se dibuja; la cota de presión de Sellner 2017 no da una cifra de aniquilación,
    porque Z_eff no está verificado;
  - circuitos con disipación (v2.0.2): el filtro de tres etapas lleva un marcador vacío en
    S_eq = B0^2 r_wall^2 / t_con, el ruido que en el modelo de solo ruido daría el mismo tiempo medio que el
    circuito con disipación (disipacion.py). En el filtro de 12 Hz el desplazamiento (factor 1,3) es menor que
    el símbolo, así que solo se escriben sus dos tiempos. Los rótulos usan el criterio temporal t* = 1 año y
    llaman «pérdida determinista» a la del campo estático.
Idioma: FIG_IDIOMA=en (por defecto, punto decimal) o FIG_IDIOMA=es, como el mecanismo del parche de A4.
Estilo: el de la tesis (estilo.mplstyle y la paleta de figuras.py). Desde la v2.0.2 la leyenda de abajo explica
cada símbolo y da los tiempos medios de los circuitos, y los umbrales se rotulan en el borde derecho. La letra es de 8 a 9 pt para imprimir la figura a todo el ancho del texto (Ilustración 4-9 de la
tesis).

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
from disipacion import escenarios_con_disipacion  # noqa: E402
from parches_fluctuantes import banda_superficie  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import figuras  # noqa: E402  (estilo y paleta comunes de la tesis)
from figuras import COLOR, tamano  # noqa: E402

IDIOMA = os.environ.get("FIG_IDIOMA", "en")
TXT = {
    "en": dict(xl=r"field noise at $f_-$, $S_E(f_-)$ (V$^2$ m$^{-2}$ Hz$^{-1}$)", yl=r"static patch field $|E|$ (V/m)",
               stable="Stable:\nbounded orbit\nand $\\bar t > t^*$", inst="Deterministic loss by the static field",
               noise="Loss by noise:\n$\\bar t < t^* = 1$ yr", golpe="sudden", lento="slow",
               l3="three-stage filter, noise only: {a} yr", l3d=r"three-stage filter, with dissipation ($S_{{\rm eq}}$): {b} yr",
               l12="12 Hz filter: {a} yr ({b} with dissipation)", lanio=r"$\bar t = t^* = 1$ yr",
               surf="surface, 4 K, scenarios A and B (bound)", t2a=r"axial channel, 4 K (range of $k$ and $\tau_z$)",
               t2b="axial channel, 0.1 K (worst case)", franja="threshold: sudden (50 V/m) or slow (100 V/m)"),
    "es": dict(xl=r"ruido del campo a $f_-$, $S_E(f_-)$ (V$^2$ m$^{-2}$ Hz$^{-1}$)", yl=r"campo estático del parche $|E|$ (V/m)",
               stable="Estable:\nórbita acotada\ny $\\bar t > t^*$", inst="Pérdida determinista por el campo estático",
               noise="Pérdida por ruido:\n$\\bar t < t^* = 1$ año", golpe="súbito", lento="lento",
               l3="filtro de tres etapas, solo ruido: {a} años", l3d=r"filtro de tres etapas, con disipación ($S_{{\mathrm{{eq}}}}$): {b} años",
               l12="filtro de 12 Hz: {a} años ({b} con disipación)", lanio=r"$\bar t = t^* = 1$ año",
               surf="superficie, 4 K, escenarios A y B (cota)", t2a=r"canal axial, 4 K (según $k$ y $\tau_z$)",
               t2b="canal axial, 0,1 K (peor caso)", franja="umbral: súbito (50 V/m) o lento (100 V/m)"),
}[IDIOMA]


def cifra(v, dec):
    """Número con dec decimales; coma decimal en la versión española."""
    s = f"{v:.{dec}f}"
    return s.replace(".", ",") if IDIOMA == "es" else s


def S_G_por_E0cuad(T, k, tau, nmax=30):
    """S_G(f_-)/E0^2 del canal T2 (forma cerrada de Laguerre, i3a_canal_axial.py)."""
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
S3, S12 = esc["filtro real (corte 1,2 kHz)"][0], esc["filtro fuerte (corte 12 Hz) + EMI"][0]
# con disipación: ruido equivalente que, sin disipación, daría el mismo tiempo medio (t = r_wall^2 / (4 D))
ed = escenarios_con_disipacion(4.0)
S3d = SE_desde_D(R_WALL**2 / (4 * ed["filtro de tres etapas + EMI"][1]))
S12d = SE_desde_D(R_WALL**2 / (4 * ed["filtro fuerte + EMI"][1]))
E = np.logspace(-5, np.log10(E_lento), 200)
s_lo = S_G_por_E0cuad(4.0, 1 / R_WALL, 1.0) / 2
s_hi = S_G_por_E0cuad(4.0, np.pi / Z0, 0.01) / 2
t3, t3d = (v / UN_ANIO for v in ed["filtro de tres etapas + EMI"][:2])
t12, t12d = (v / UN_ANIO for v in ed["filtro fuerte + EMI"][:2])
S_G01 = S_G_por_E0cuad(0.1, np.pi / Z0, 0.01) / 2
with plt.style.context(os.path.join(figuras._DIR, "estilo.mplstyle")):
    fig, ax = plt.subplots(figsize=tamano(2, 11.0))
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    # regiones: estable (verde claro), franja entre los dos umbrales (gris) y sus límites. La línea de un año
    # llega solo hasta el umbral lento, porque por encima la pérdida es determinista
    ax.fill_between([x0, S_anio], y0, E_golpe, color=COLOR["serie3"], alpha=0.14, lw=0)
    ax.fill_between([x0, x1], E_golpe, E_lento, color=COLOR["secundario"], alpha=0.3, lw=0)
    ax.axhline(E_golpe, color=COLOR["teoria"], lw=0.9); ax.axhline(E_lento, color=COLOR["teoria"], lw=0.9, ls="--")
    ax.plot([S_anio, S_anio], [y0, E_lento], color=COLOR["teoria"], lw=0.9, ls=":")
    # rótulos de las regiones dentro del mapa y de los umbrales en el borde derecho
    ax.text(hi * 1.8, 2e-5, TXT["stable"], fontsize=9, va="bottom")
    ax.text(2e-21, 1e3, TXT["inst"], fontsize=9, va="center")
    ax.text(S_anio * 20, 2e-5, TXT["noise"], fontsize=9, va="bottom")
    for Ec, modo in ((E_golpe, TXT["golpe"]), (E_lento, TXT["lento"])):
        ax.text(1.01, Ec, f"{Ec:.0f} V/m ({modo})", transform=ax.get_yaxis_transform(), fontsize=8, va="center")
    # superficie (solo cota) y canal axial, con el modelo de solo ruido
    trama = to_rgba(COLOR["secundario"], 0.65)
    ax.fill_betweenx([y0, E_golpe], lo, hi, facecolor="none", edgecolor=trama, hatch="///", lw=0.5)
    ax.fill_betweenx(E, s_lo * E**2, s_hi * E**2, color=COLOR["positron"], alpha=0.22, lw=0)
    for s in (s_lo, s_hi):
        ax.plot(s * E**2, E, color=COLOR["positron"], lw=0.5, alpha=0.7)
    ax.plot(S_G01 * E**2, E, color=COLOR["positron"], ls=":", lw=1.3)
    # circuitos en |E| = 1 V/m: lleno = solo ruido; vacío = con disipación, en S_eq
    ax.annotate("", xy=(S3d, 1.0), xytext=(S3, 1.0),
                arrowprops=dict(arrowstyle="->", color=COLOR["serie3"], lw=1.0, shrinkA=4.5, shrinkB=4.5))
    ax.plot(S3, 1.0, "o", color=COLOR["serie3"], ms=7, mec="white", mew=0.7, zorder=5)
    ax.plot(S3d, 1.0, "o", mfc="white", mec=COLOR["serie3"], mew=1.4, ms=6.5, zorder=5)
    ax.plot(S12, 1.0, "s", color=COLOR["secundario"], ms=6.5, mec="white", mew=0.7, zorder=5)
    ax.set_xlabel(TXT["xl"]); ax.set_ylabel(TXT["yl"])
    # leyenda con todos los símbolos: columna 1 = circuitos y línea de un año; columna 2 = fuentes y umbral
    manejadores = [Line2D([], [], color=COLOR["serie3"], marker="o", ls="none", ms=7, mec="white", mew=0.7),
                   Line2D([], [], marker="o", ls="none", ms=6.5, mfc="white", mec=COLOR["serie3"], mew=1.4),
                   Line2D([], [], color=COLOR["secundario"], marker="s", ls="none", ms=6.5, mec="white", mew=0.7),
                   Line2D([], [], color=COLOR["teoria"], ls=":", lw=0.9),
                   Patch(facecolor="none", edgecolor=trama, hatch="///", lw=0.5),
                   Patch(facecolor=COLOR["positron"], alpha=0.22, edgecolor=COLOR["positron"], lw=0.5),
                   Line2D([], [], color=COLOR["positron"], ls=":", lw=1.3),
                   Patch(facecolor=COLOR["secundario"], alpha=0.3, lw=0)]
    etiquetas = [TXT["l3"].format(a=cifra(t3, 2)), TXT["l3d"].format(b=cifra(t3d, 3)),
                 TXT["l12"].format(a=cifra(t12, 0), b=cifra(t12d, 0)), TXT["lanio"],
                 TXT["surf"], TXT["t2a"], TXT["t2b"], TXT["franja"]]
    fig.legend(manejadores, etiquetas, loc="outside lower center", ncol=2, fontsize=8, frameon=False,
               columnspacing=1.5, handlelength=2.0, handletextpad=0.5)
    suf = "" if IDIOMA == "en" else "_es"
    fig.savefig(os.path.join(args.salida, f"fig_regimes_extended{suf}.pdf"))
    fig.savefig(os.path.join(args.salida, f"fig_regimes_extended{suf}.png"))
E_cruce = np.sqrt(S_anio / s_hi)
print(f"S_E de 1 año = {S_anio:.3e}; T2 a 4 K: S_eq/E0^2 entre {s_lo:.3e} y {s_hi:.3e} (Hz^-1);"
      f" peor caso: t < 1 año si E0 > {E_cruce:.3g} V/m;  superficie a 4 K: S_E entre {lo:.2e} y {hi:.2e}")
print(f"circuitos a 4 K: tres etapas S_E = {S3:.3e} -> S_eq con disipación = {S3d:.3e}"
      f" (t = {ed['filtro de tres etapas + EMI'][1] / UN_ANIO:.3f} años); 12 Hz S_E = {S12:.3e} -> S_eq = {S12d:.3e}"
      f" (t = {ed['filtro fuerte + EMI'][1] / UN_ANIO:.0f} años)")
