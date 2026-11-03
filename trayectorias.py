"""
trayectorias.py — Ilustración 4-7 (fig:res-trayectorias): el movimiento del positrón en las cuatro
temperaturas del cuerpo de la tesis, 4; 1; 0,3 y 0,1 K (tarea 1.10 del WP1). Base:
codigo/v11_figs/regenerar_fig4_08_09.py del proyecto LaTeX, que solo dibujaba 0,1 K.

    python3 trayectorias.py                                      # figures/fig4_06b_trayectorias.pdf y .png
    FIG_IDIOMA=en FIG_CARPETA=figures/en python3 trayectorias.py  # etiquetas en inglés

Una columna por temperatura y una fila por modo. Dentro de cada fila los ejes son comunes, para que
se vea cuánto se encoge cada movimiento al enfriar:
  (a) axial: z(t) = A_z sen(w_z t) durante cuatro periodos axiales (300 ns);
  (b) magnetrón: órbita del centro guía, de radio A_-, con n_- = n_z (magnetrón reenfriado por
      bandas laterales, la hipótesis de cuantico.py);
  (c) ciclotrón: giro de radio A_+ alrededor del centro guía. La línea discontinua es el punto cero
      sqrt(hbar / (m w_1)) = 36,3 nm: a 0,1 K el ciclotrón ya está en él.
Las amplitudes incluyen el punto cero, A = sqrt(2 hbar (n + 1/2) / (m w_ef)), y salen de
cuantico.tabla_modos: son las de la Ilustr. 4-6 y las de datos/temperaturas_cuerpo.csv.
Colores de las columnas como en marcar_temperaturas: 4 K azul, 0,1 K vino, 1 K y 0,3 K gris.
El nombre fig4_06b deja el archivo entre fig4_06_modos (Ilustr. 4-6) y fig4_07_canales_temperatura
(Ilustr. 4-8). Los datos se guardan en datos/trayectorias_amplitudes.csv, sin el prefijo fig4_0 para que no
entre en la comprobación md5 de los 10 CSV de la V12 (receta del WP1). No modifica ningún otro archivo.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from cuantico import radio_punto_cero, tabla_modos
from figuras import COLOR, guardar, tamano, usar_estilo
from figuras_tesis import CARPETA, L, guardar_datos, num
from params import W_1, W_Z
from temperaturas import T_CUERPO

PERIODOS = 4        # periodos axiales dibujados en (a)
HOLGURA = 1.6       # límite de los ejes / amplitud mayor de la fila: deja sitio al rótulo de cada panel


def color_T(T):
    """Color de la columna: el de la línea vertical de esa temperatura en marcar_temperaturas."""
    return {4.0: COLOR["4K"], 0.1: COLOR["01K"]}.get(T, COLOR["secundario"])


def cifra(x, unidad):
    """Tres cifras significativas con la coma decimal del idioma: 131 µm, 65,6 µm, 0,641 µm."""
    return num(f"{x:#.3g}".rstrip(".")) + " " + unidad


def tabla():
    """Una fila por temperatura: T (K) y n, A (m) del ciclotrón, del axial y del magnetrón."""
    filas = []
    for T in T_CUERPO:
        m = tabla_modos(T)
        filas.append((T, *m["ciclotrón"], *m["axial"], *m["magnetrón"]))
    return np.array(filas)


def figura():
    d = tabla()
    r0 = radio_punto_cero(W_1)
    t = np.linspace(0, PERIODOS * 2 * np.pi / W_Z, 800)
    ang = np.linspace(0, 2 * np.pi, 400)
    cos, sen = np.cos(ang), np.sin(ang)
    fig, ax = plt.subplots(3, 4, figsize=tamano(2, 12.5), sharex="row", sharey="row",
                           gridspec_kw={"height_ratios": (0.8, 1, 1)})
    for j, (T, n_p, A_p, n_z, A_z, n_m, A_m) in enumerate(d):
        c = color_T(T)
        a, b, k = ax[:, j]
        a.plot(t * 1e9, A_z * np.sin(W_Z * t) * 1e6, color=c, lw=1.3)
        for s in (1, -1):
            a.axhline(s * A_z * 1e6, color=COLOR["secundario"], ls=":", lw=0.9)
        b.plot(A_m * cos * 1e6, A_m * sen * 1e6, color=c, lw=1.3)
        k.plot(A_p * cos * 1e9, A_p * sen * 1e9, color=c, lw=1.3)
        k.plot(r0 * cos * 1e9, r0 * sen * 1e9, color=COLOR["teoria"], ls="--", lw=0.8)  # encima: a 0,1 K coinciden
        for eje, txt in ((a, r"$A_z$ = " + cifra(A_z * 1e6, "µm")), (b, r"$A_-$ = " + cifra(A_m * 1e6, "µm")),
                         (k, r"$A_+$ = " + cifra(A_p * 1e9, "nm"))):
            eje.text(0.04, 0.96, txt, transform=eje.transAxes, ha="left", va="top", fontsize=7.5)
        a.set_title(num(f"{T:g} K"), color=c, fontsize=10)
        a.set_xlabel(r"$t$ (ns)")
        b.set_xlabel(r"$x$ (µm)")
        k.set_xlabel(L(r"$x - x_{\mathrm{cg}}$ (nm)", r"$x - x_{\mathrm{gc}}$ (nm)"))
        for eje in (b, k):
            eje.set_aspect("equal", adjustable="box")
        for eje in (a, b, k):
            eje.tick_params(labelsize=8)
    # límites comunes en cada fila, fijados por la amplitud mayor (la de 4 K)
    ax[0, 0].set_xlim(0, 300)
    ax[0, 0].set_xticks((0, 100, 200))
    for i, (col, escala, marcas) in enumerate(((4, 1e6, (-100, 0, 100)), (6, 1e6, (-5, 0, 5)),
                                               (2, 1e9, (-100, 0, 100)))):
        lim = HOLGURA * d[:, col].max() * escala
        ax[i, 0].set_ylim(-lim, lim)
        ax[i, 0].set_yticks(marcas)
        if i > 0:
            ax[i, 0].set_xlim(-lim, lim)
            ax[i, 0].set_xticks(marcas)
    # cabecera de cada fila, a la izquierda de su eje y
    filas = (L("(a) axial", "(a) axial"),
             L("(b) magnetrón\n" r"$\bar n_- = \bar n_z$", "(b) magnetron\n" r"$\bar n_- = \bar n_z$"),
             L("(c) ciclotrón", "(c) cyclotron"))
    etiquetas_y = (r"$z$ (µm)", r"$y$ (µm)", L(r"$y - y_{\mathrm{cg}}$ (nm)", r"$y - y_{\mathrm{gc}}$ (nm)"))
    for e, fila, ey in zip(ax[:, 0], filas, etiquetas_y):
        e.set_ylabel(ey)
        e.annotate(fila, xy=(0, 0.5), xytext=(-6, 0), xycoords=e.yaxis.label, textcoords="offset points",
                   ha="right", va="center", rotation=90, fontsize=10)
    fig.align_ylabels(ax[:, 0])
    guias = [Line2D([], [], color=COLOR["secundario"], ls=":", lw=0.9),
             Line2D([], [], color=COLOR["teoria"], ls="--", lw=0.8)]
    fig.legend(guias, [L(r"(a): amplitud efectiva $\pm A_z$", r"(a): effective amplitude $\pm A_z$"),
                       L(r"(c): punto cero del ciclotrón, $\sqrt{\hbar/(m\omega_1)}$ = ",
                         r"(c): cyclotron zero point, $\sqrt{\hbar/(m\omega_1)}$ = ") + cifra(r0 * 1e9, "nm")],
               loc="outside lower center", ncol=2, fontsize=7, frameon=False, handlelength=2.4, columnspacing=1.6)
    guardar(fig, "fig4_06b_trayectorias", CARPETA)
    guardar_datos("trayectorias_amplitudes", "T_K,n_cic,A_cic_m,n_ax,A_ax_m,n_mag,A_mag_m", *d.T)
    return d


if __name__ == "__main__":
    usar_estilo()
    d = figura()
    for T, n_p, A_p, n_z, A_z, n_m, A_m in d:
        print(num(f"T = {T:g} K: A_+ = {A_p * 1e9:.1f} nm (n_+ = {n_p:.3g}); A_z = {A_z * 1e6:.1f} µm; "
                  f"A_- = {A_m * 1e9:.0f} nm (n_- = n_z = {n_z:.4g})"))
