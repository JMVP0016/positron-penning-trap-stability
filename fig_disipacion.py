"""
fig_disipacion.py — Figura del anexo F: tiempo medio de pérdida con y sin la disipación que el teorema
de fluctuación-disipación asocia al ruido térmico (decisión D-T19; ronda 5).

    python fig_disipacion.py                                      # figures/figF_disipacion.pdf y .png
    FIG_IDIOMA=en FIG_CARPETA=figures/en python fig_disipacion.py  # etiquetas en inglés

(a) t̄ frente a la temperatura del sistema, sin disipación (línea continua: el modelo del cuerpo, Ilustr. 4-8)
    y con ella (discontinua), para el filtro de tres etapas + EMI, el filtro de 12 Hz + EMI y el filtro de
    tres etapas entero a la temperatura de la trampa (sin EMI: la versión realizable de «solo su etapa fría»).
(b) Para cualquier fuente de ruido en equilibrio a T_b, t̄(con)/t̄(sin) = Ein(U)/U con U = |E_-(r_wall)|/(k_B T_b):
    por eso los tiempos de solo ruido de las fuentes térmicas son cotas superiores.
Los datos se guardan en datos/figF_disipacion.csv. No modifica ningún CSV de la tesis.
"""
import matplotlib.pyplot as plt
import numpy as np

from disipacion import (Z_nodo, escenarios_con_disipacion, factor_fdt_termico, lambda_disipacion,
                        tiempo_medio_con_disipacion)
from figuras import COLOR, guardar, tamano, usar_estilo
from figuras_tesis import CARPETA, L, guardar_datos, marcar_temperaturas, num
from params import K_B, T_01K, T_4K, UN_ANIO
from ruido import FILTRO_REAL, D_desde_SE, SE_desde_SV, tiempo_medio

T = np.logspace(np.log10(0.05), 1, 120)
CLAVES = ("filtro de tres etapas + EMI", "filtro fuerte + EMI")


def tiempos_filtro_entero(Ti):
    """Filtro de tres etapas con sus tres resistencias a Ti y sin EMI: (t̄ sin, t̄ con) en años."""
    ReZ = Z_nodo(FILTRO_REAL).real
    D = D_desde_SE(SE_desde_SV(4 * K_B * Ti * ReZ))           # Nyquist con todo el filtro a Ti
    return tiempo_medio(D) / UN_ANIO, tiempo_medio_con_disipacion(D, lambda_disipacion(ReZ)) / UN_ANIO


def calcular():
    filas = []
    for Ti in T:
        e = escenarios_con_disipacion(Ti)
        f_sin, f_con = e[CLAVES[0]][0] / UN_ANIO, e[CLAVES[0]][1] / UN_ANIO
        g_sin, g_con = e[CLAVES[1]][0] / UN_ANIO, e[CLAVES[1]][1] / UN_ANIO
        h_sin, h_con = tiempos_filtro_entero(Ti)
        filas.append((Ti, f_sin, f_con, g_sin, g_con, h_sin, h_con, float(factor_fdt_termico(Ti))))
    return np.array(filas)


def figura():
    d = calcular()
    fig, (a, b) = plt.subplots(1, 2, figsize=tamano(2, 8.8))
    # (a) con y sin disipación
    curvas = ((1, 2, COLOR["serie3"], L("filtro de tres etapas + EMI", "three-stage filter + EMI")),
              (3, 4, COLOR["secundario"], L("filtro de 12 Hz + EMI", "12 Hz filter + EMI")),
              (5, 6, COLOR["teoria"], L("filtro entero a $T$ (sin EMI)", "whole filter at $T$ (no EMI)")))
    for i_sin, i_con, c, nombre in curvas:
        a.loglog(d[:, 0], d[:, i_sin], color=c, ls="-", lw=1.6, label=nombre + L(", sin disipación", ", no dissipation"))
        a.loglog(d[:, 0], d[:, i_con], color=c, ls="--", lw=1.6, label=nombre + L(", con disipación", ", with dissipation"))
    a.set_xlabel(L(r"temperatura del sistema $T$ (K)", r"system temperature $T$ (K)"))
    a.set_ylabel(L(r"tiempo medio de pérdida $\bar t$ (años)", r"mean loss time $\bar t$ (years)"))
    a.set_ylim(0.1, 1e4)
    a.set_title("(a)", loc="left", fontsize=10)
    fig.legend(*a.get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=6.5,
               columnspacing=1.4, handlelength=2.4, frameon=False)
    # (b) cociente general para una fuente térmica a T_b
    b.loglog(d[:, 0], d[:, 7], color=COLOR["teoria"], ls="-", lw=1.6)
    for Tk, k, mk in ((T_4K, "4K", "o"), (T_01K, "01K", "s")):
        fk = float(factor_fdt_termico(Tk))
        b.plot(Tk, fk, marker=mk, color=COLOR[k], ls="none", ms=5)
        b.annotate(f"1/{1 / fk:.0f}", (Tk, fk), textcoords="offset points", xytext=(6, -14) if k == "4K" else (6, -12),
                   fontsize=8, color=COLOR[k])
    b.set_xlabel(L(r"temperatura de la fuente de ruido $T_b$ (K)", r"noise-source temperature $T_b$ (K)"))
    b.set_ylabel(L(r"$\bar t$ con disipación / $\bar t$ sin disipación", r"$\bar t$ with / without dissipation"))
    formula = r"$\mathrm{Ein}(U)/U$,  $U = 2901\ \mathrm{K}/T_b$"
    b.text(0.22, 0.80, L("fuente térmica:\n", "thermal source:\n") + formula,
           transform=b.transAxes, fontsize=7.5, va="bottom", ha="left",
           bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5))
    b.set_title("(b)", loc="left", fontsize=10)
    for ax in (a, b):
        marcar_temperaturas(ax)
    guardar(fig, "figF_disipacion", CARPETA)
    guardar_datos("figF_disipacion", "T_K,t_tres_etapas_sin_anios,t_tres_etapas_con_anios,t_12Hz_sin_anios,"
                  "t_12Hz_con_anios,t_entero_sin_anios,t_entero_con_anios,factor_fdt", *d.T)
    return d


if __name__ == "__main__":
    usar_estilo()
    d = figura()
    for Tk in (T_4K, T_01K):
        e = escenarios_con_disipacion(Tk)
        h = tiempos_filtro_entero(Tk)
        print(num(f"T = {Tk:g} K: tres etapas {e[CLAVES[0]][0] / UN_ANIO:.2f} → {e[CLAVES[0]][1] / UN_ANIO:.3f} años; "
                  f"12 Hz {e[CLAVES[1]][0] / UN_ANIO:.0f} → {e[CLAVES[1]][1] / UN_ANIO:.0f} años; "
                  f"filtro entero {h[0]:.0f} → {h[1]:.3f} años; Ein(U)/U = 1/{1 / float(factor_fdt_termico(Tk)):.0f}"))
