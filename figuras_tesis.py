"""
figuras_tesis.py — Genera las 8 figuras finales del capítulo 4 y guarda sus datos.

    python figuras_tesis.py          # las 8 figuras (unos 2 minutos)
    python figuras_tesis.py 5 7      # solo las figuras 5 y 7

Cada figura se guarda en figures/ (PDF para LaTeX y PNG para revisar) y sus datos numéricos
en datos/ (CSV), para que cualquier cifra del texto pueda rastrearse hasta un archivo.

Regla de colores (figuras.py): azul = 4 K y rojo vino = 0,1 K, SOLO cuando la figura compara
esas dos temperaturas. Las figuras sin comparación de temperaturas usan negro (teoría),
gris y verde, para que el azul y el vino no cambien de significado dentro de la tesis.

Figuras
    1  Parche estático: órbita del centro guía y umbral de pérdida       (centro_guia.py)
    2  Apantallamiento: campo en el centro frente a la longitud del parche (superficie.py)
    3  Prueba de la hipótesis de tránsito                                (transito.py)
    4  Difusión del centro guía y supresión por la rotación magnetrón    (ruido.py)
    5  Tiempo medio de pérdida frente al ruido S_E(f_-)                  (ruido.py)
    6  Ocupaciones y amplitudes de los modos frente a la temperatura     (cuantico.py)
    7  Tiempo de pérdida frente a la temperatura del sistema y canal del positrón
    8  Mapa de regímenes                                                  (todos)
    9  Parches que fluctúan: Monte Carlo y tiempo de pérdida frente a T   (parches_fluctuantes.py)
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from figuras import COLOR, LINEA, guardar, tamano, usar_estilo
from params import B0, R_WALL, T_01K, T_4K, U0, UN_ANIO, W_1, W_M, W_P, W_Z

_DIR = os.path.dirname(os.path.abspath(__file__))
E_V11 = 1.0                                   # campo de parche usado en la V11 (V/m)
E_ALPHA = 1.0     # orden del campo de parche medido por ALPHA a 4 K: O(10 mV/cm) (Baker et al. 2024)
D_V11 = {"4K": 1.816e-11, "01K": 1.149e-10}   # coeficientes de difusión de la V11 (m^2/s)


def guardar_datos(nombre, cabecera, *columnas):
    ruta = os.path.join(_DIR, "datos")
    os.makedirs(ruta, exist_ok=True)
    np.savetxt(os.path.join(ruta, f"{nombre}.csv"), np.column_stack(columnas),
               delimiter=",", header=cabecera, comments="", fmt="%.6e")


def marcar_temperaturas(ax):
    """Líneas verticales en 4 K (azul) y 0,1 K (vino), con etiqueta arriba."""
    for T, c, txt in ((T_4K, COLOR["4K"], "4 K"), (T_01K, COLOR["01K"], "0,1 K")):
        ax.axvline(T, color=c, lw=1, ls=LINEA["4K" if T == T_4K else "01K"]["ls"], alpha=0.8)
        ax.text(T, 1.01, txt, color=c, fontsize=8, ha="center", va="bottom",
                transform=ax.get_xaxis_transform())


# ===========================================================================
def fig1_parche_estatico():
    from centro_guia import E_critico, P_M, centro_desplazado, simular_rampa
    from superficie import E_centro_rms

    Ec = E_V11 * np.exp(1j * 0.7)
    fig, (a, b) = plt.subplots(1, 2, figsize=tamano(2, 7.0))
    # (a) órbitas con 1 V/m: el campo aparece de golpe o en 5 periodos magnetrón
    t0, Z0 = simular_rampa(Ec, 0.0, periodos_despues=1.05)
    t1, Z1 = simular_rampa(Ec, 5 * P_M, periodos_despues=1.05)
    Zs = centro_desplazado(Ec)
    a.plot(Z0.real * 1e6, Z0.imag * 1e6, color=COLOR["teoria"], ls="-", label="aparece de golpe")
    a.plot(Z1.real * 1e6, Z1.imag * 1e6, color=COLOR["serie3"], ls="--", label="aparece en 5 $P_-$")
    a.plot(0, 0, "o", color=COLOR["teoria"], ms=3)
    a.plot(Zs.real * 1e6, Zs.imag * 1e6, "+", color=COLOR["teoria"], ms=8, mew=1.2)
    a.annotate("centro de\nla trampa", (0, 0), (6, 4), textcoords="offset points", fontsize=8)
    a.annotate(r"$Z^*$", (Zs.real * 1e6, Zs.imag * 1e6), (5, -11), textcoords="offset points", fontsize=9)
    a.set_aspect("equal")
    a.set_xlabel(r"$X$ (µm)"); a.set_ylabel(r"$Y$ (µm)")
    a.set_xlim(-95, 75)
    a.legend(loc="lower right", fontsize=8)
    a.set_title("(a)", loc="left", fontsize=10)
    # (b) radio máximo frente al campo estático
    E = np.logspace(-4, 3, 200)
    b.loglog(E, 2 * E / U0 * 1e3, color=COLOR["teoria"], ls="-", label=r"de golpe: $2\delta$")
    b.loglog(E, E / U0 * 1e3, color=COLOR["serie3"], ls="--", label=r"lento: $\delta$")
    b.axhline(R_WALL * 1e3, color=COLOR["secundario"], ls=":", lw=1.2)
    b.text(1.2e-4, R_WALL * 1e3 * 0.3, r"pared, $r_{\mathrm{wall}}$ = 5 mm", fontsize=8)
    b.axvspan(E_critico("golpe"), E_critico("lento"), color=COLOR["secundario"], alpha=0.15, lw=0)
    b.text(70, 2e-5, "pérdida:\n50–100 V/m", fontsize=8, ha="center")
    b.axvline(E_ALPHA, color=COLOR["secundario"], lw=0.8, ls="-.")
    b.text(E_ALPHA / 1.4, 3e-6, "medido por ALPHA\na 4 K (~10 mV/cm)\ny usado en la V11", fontsize=7.5,
           ha="right")
    b.set_xlabel(r"campo estático del parche $|E|$ (V/m)")
    b.set_ylabel(r"distancia máxima al centro (mm)")
    b.set_ylim(1e-6, 1e2)
    b.legend(loc="center left", bbox_to_anchor=(0.0, 0.62), fontsize=8)
    b.set_title("(b)", loc="left", fontsize=10)
    guardar(fig, "fig4_01_parche_estatico")
    guardar_datos("fig4_01_umbral", "E_V_m,max_golpe_m,max_lento_m", E, 2 * E / U0, E / U0)


def fig2_apantallamiento():
    from superficie import A_CIL, A1_montecarlo, E_centro_rms

    ells = np.logspace(-7, -2.3, 80)
    fig, ax = plt.subplots(figsize=tamano(1.5 if False else 2, 7.5))
    for Vr, ls in ((1e-3, "-"), (1e-2, "--")):
        ax.loglog(ells * 1e3, [E_centro_rms(Vr, l) for l in ells], color=COLOR["teoria"], ls=ls,
                  label=f"analítico, $V_{{\\mathrm{{rms}}}}$ = {Vr*1e3:.0f} mV")
    ell_mc = np.array([1e-5, 1e-4, 1e-3])
    mc = [A1_montecarlo(1e-3, l, semilla=300 + i) for i, l in enumerate(ell_mc)]
    E_mc = np.array([np.sqrt(2) * m / A_CIL for m, _ in mc])
    E_err = np.array([np.sqrt(2) * e / A_CIL for _, e in mc])
    ax.errorbar(ell_mc * 1e3, E_mc, yerr=E_err, color=COLOR["serie3"], ls="none", marker="o",
                ms=4, capsize=2, label="Monte Carlo, 1 mV (400 electrodos)")
    ax.axhline(E_V11, color=COLOR["secundario"], ls="-.", lw=0.9)
    ax.text(1.3e-4, 1.3, "~1 V/m: orden medido por ALPHA a 4 K (y valor de la V11)", fontsize=8)
    ax.axhline(U0 * R_WALL / 2, color=COLOR["secundario"], ls=":", lw=1.2)
    ax.text(1.3e-4, 65, "50 V/m: umbral de pérdida", fontsize=8)
    ax.set_xlabel(r"longitud de correlación del parche $\ell$ (mm)")
    ax.set_ylabel(r"$|E(0)|_{\mathrm{rms}}$ en el centro (V/m)")
    ax.legend(loc="lower right", fontsize=8)
    guardar(fig, "fig4_02_apantallamiento")
    guardar_datos("fig4_02_montecarlo", "ell_m,E_mc_V_m,error_V_m,E_analitico_V_m",
                  ell_mc, E_mc, E_err, [E_centro_rms(1e-3, l) for l in ell_mc])


def fig3_transito():
    from transito import W_M as W_ESC, ou_de_transito, parche_estatico_muestreado

    t1, R1, th1 = parche_estatico_muestreado(semilla=2026)
    t2, R2, th2, _, _ = ou_de_transito(semilla=4052)
    P = 2 * np.pi / W_ESC
    fig, ax = plt.subplots(figsize=tamano(2, 7.0))
    ax.plot(t2 / P, R2, color=COLOR["secundario"], ls="--", label="hipótesis de tránsito (OU)")
    ax.plot(t2 / P, th2, color=COLOR["teoria"], ls=":", lw=1.2, label=r"$4Dt$")
    ax.plot(t1 / P, R1, color=COLOR["serie3"], ls="-", label="parche estático muestreado")
    ax.plot(t1 / P, th1, color=COLOR["teoria"], ls=(0, (1, 3)), lw=1.0,
            label=r"$2\langle|G_0|^2\rangle(1-\cos\omega_- t)/U_0^2$")
    ax.set_xlabel(r"$t/P_-$")
    ax.set_ylabel(r"$\langle |Z|^2\rangle$ (unidades escaladas)")
    ax.legend(loc="upper left", fontsize=8)
    guardar(fig, "fig4_03_transito")
    guardar_datos("fig4_03_estatico", "t_escalado,R2_sim,R2_teoria", t1, R1, th1)
    guardar_datos("fig4_03_ou", "t_escalado,R2_sim,R2_teoria", t2, R2, th2)


def fig4_difusion():
    from ruido import C50, D_ou, densidad_perdida, mediana_bootstrap, simular_difusion, simular_perdida

    sigma = 1 / np.sqrt(2)
    xs = np.array([0.1, 0.3, 1.0, 3.0, 10.0])
    sup, rel, rel_err, ejemplo = [], [], [], None
    for i, x in enumerate(xs):
        tau = x / W_M
        # Con tau largo el campo cambia despacio: hace falta simular más periodos para
        # muestrear muchos valores independientes del ruido (duración >> tau).
        n_per = 150 if x < 3 else 600
        t, R2, D_sim, D_err = simular_difusion(tau, sigma, n_tray=8000, n_lotes=16,
                                               n_periodos=n_per, semilla=5000 + i)
        D_th = D_ou(sigma, tau)
        D_bl = sigma**2 * tau / B0**2
        sup.append(D_sim / D_bl); rel.append(D_sim / D_th); rel_err.append(D_err / D_th)
        if x == 1.0:
            ejemplo = (t, R2, D_th)
    fig = plt.figure(figsize=tamano(2, 6.2))
    gs = fig.add_gridspec(2, 3, height_ratios=(2.2, 1))
    a0, a1, a2 = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[:, 2])
    a1r = fig.add_subplot(gs[1, 1], sharex=a1)
    t, R2, D_th = ejemplo
    a0.plot(t * 1e3, R2 * 1e6, color=COLOR["serie3"], ls="-", label="simulación")
    a0.plot(t * 1e3, 4 * D_th * t * 1e6, color=COLOR["teoria"], **LINEA["teoria"], label=r"$4Dt$")
    a0.set_xlabel(r"$t$ (ms)"); a0.set_ylabel(r"$\langle |Z|^2\rangle$ (mm$^2$)")
    a0.legend(loc="upper left", fontsize=8); a0.set_title("(a)", loc="left", fontsize=10)
    xx = np.logspace(-1.3, 1.3, 200)
    a1.loglog(xx, 1 / (1 + xx**2), color=COLOR["teoria"], **LINEA["teoria"], label="teoría")
    a1.plot(xs, sup, color=COLOR["serie3"], ls="none", marker="o", ms=4, label="simulación")
    a1.set_ylabel(r"$D/D_{\mathrm{blanco}}$"); a1.legend(loc="lower left", fontsize=8)
    a1.set_title("(b)", loc="left", fontsize=10)
    plt.setp(a1.get_xticklabels(), visible=False)
    a1r.axhline(1.0, color=COLOR["teoria"], ls=":", lw=1.2)
    a1r.errorbar(xs, rel, yerr=rel_err, color=COLOR["serie3"], ls="none", marker="o", ms=4,
                 elinewidth=1, capsize=2.5)
    a1r.set_ylim(0.85, 1.15)
    a1r.set_xlabel(r"$\omega_-\tau_s$"); a1r.set_ylabel(r"$D_{\mathrm{sim}}/D_{\mathrm{teo}}$")
    # (c) Monte Carlo de la pérdida con ruido acelerado frente a la distribución analítica
    tp, t_med = simular_perdida(n_tray=4000, semilla=6001)
    med, med_err = mediana_bootstrap(tp, semilla=6002)
    x = tp[np.isfinite(tp)] / t_med
    D_ac = R_WALL**2 / (4 * t_med)
    a2.hist(x, bins=np.linspace(0, 4, 41), density=True, color=COLOR["serie3"], alpha=0.45,
            lw=0, label=f"Monte Carlo\n({tp.size} positrones)")
    tt = np.linspace(0.005, 4, 400)
    a2.plot(tt, densidad_perdida(tt * t_med, D_ac) * t_med, color=COLOR["teoria"], ls="-", lw=1.2,
            label="analítica")
    a2.axvline(med / t_med, color=COLOR["serie3"], ls="--", lw=1)
    a2.axvline(4 * C50, color=COLOR["teoria"], ls=":", lw=1.2)
    txt = f"mediana:\nMC {med/t_med:.3f} ± {med_err/t_med:.3f}\nanalítica {4*C50:.3f}"
    a2.text(0.98, 0.40, txt.replace(".", ","), transform=a2.transAxes, ha="right", fontsize=7.5)
    a2.set_xlabel(r"$t/\bar t$"); a2.set_ylabel("densidad de probabilidad")
    a2.legend(loc="upper right", fontsize=7.5, handlelength=1.2)
    a2.set_title("(c)", loc="left", fontsize=10)
    guardar(fig, "fig4_04_difusion")
    guardar_datos("fig4_04_supresion", "omega_tau,D_sobre_Dblanco,D_sim_sobre_D_teo,error",
                  xs, sup, rel, rel_err)
    guardar_datos("fig4_04_tiempos_perdida", "t_sobre_t_medio", tp / t_med)


def fig5_tiempo_vs_ruido():
    from ruido import D_desde_SE, SE_desde_D, escenarios, tiempo_medio

    S = np.logspace(-21, -9, 200)
    t_anios = tiempo_medio(D_desde_SE(S)) / UN_ANIO
    fig, ax = plt.subplots(figsize=tamano(2, 7.5))
    ax.loglog(S, t_anios, color=COLOR["teoria"], ls="-",
              label=r"$\bar t = r_{\mathrm{wall}}^2 B_0^2 / S_E(f_-)$")
    ax.axhline(1.0, color=COLOR["secundario"], ls=":", lw=1.2)
    ax.text(1e-18, 1.4, "1 año", fontsize=8)
    ax.axvspan(1e-21, 7.5e-20, color=COLOR["secundario"], alpha=0.15, lw=0)
    ax.text(1.3e-21, 4e-4, "cota de\nBASE\n(a MHz;\nsolo orden\nde magnitud)", fontsize=7, va="bottom")
    for k, T in (("4K", T_4K), ("01K", T_01K)):
        S_v11 = SE_desde_D(D_V11[k])
        ax.plot(S_v11, tiempo_medio(D_V11[k]) / UN_ANIO, color=COLOR[k], ls="none",
                marker=LINEA[k]["marker"], ms=5, label=f"implícito en la V11, {'4 K' if k == '4K' else '0,1 K'}")
        esc = escenarios(T)
        for j, nombre in enumerate(("filtro real (corte 1,2 kHz)", "filtro fuerte (corte 12 Hz) + EMI")):
            S_j = esc[nombre][0]
            ax.plot(S_j, tiempo_medio(D_desde_SE(S_j)) / UN_ANIO, color=COLOR[k], ls="none",
                    marker=LINEA[k]["marker"], ms=5, mfc="none",
                    label=(f"ruido realista, {'4 K' if k == '4K' else '0,1 K'}" if j == 0 else None))
    esc4 = escenarios(T_4K)
    for nombre, txt, off in (("filtro real (corte 1,2 kHz)", "filtro real\n(1,2 kHz)", (10, 2)),
                             ("filtro fuerte (corte 12 Hz) + EMI", "filtro de 12 Hz\n+ EMI (BASE)", (10, 2))):
        S_j = esc4[nombre][0]
        ax.annotate(txt, (S_j, tiempo_medio(D_desde_SE(S_j)) / UN_ANIO), off,
                    textcoords="offset points", fontsize=7.5)
    ax.set_xlabel(r"densidad espectral del campo a $f_-$, $S_E(f_-)$ (V$^2$ m$^{-2}$ Hz$^{-1}$)")
    ax.set_ylabel(r"tiempo medio de pérdida $\bar t$ (años)")
    ax.legend(loc="upper right", fontsize=7.5)
    guardar(fig, "fig4_05_tiempo_vs_ruido")
    guardar_datos("fig4_05_tiempo", "S_E,t_medio_anios", S, t_anios)


def fig6_modos():
    from cuantico import MODOS, amplitud, ocupacion, radio_punto_cero

    T = np.logspace(np.log10(0.02), 1, 300)
    estilos = {"ciclotrón": (COLOR["teoria"], "-"), "axial": (COLOR["secundario"], "--"),
               "magnetrón": (COLOR["serie3"], "-.")}
    fig, (a, b) = plt.subplots(1, 2, figsize=tamano(2, 7.0))
    datos = [T]
    for nombre, (w_e, w_ef) in MODOS.items():
        c, ls = estilos[nombre]
        n = ocupacion(T, w_e)
        A = amplitud(n, w_ef)
        datos += [n, A]
        a.loglog(T, n, color=c, ls=ls, label=nombre if nombre != "magnetrón" else r"magnetrón ($\bar n_- = \bar n_z$)")
        b.loglog(T, A * 1e6, color=c, ls=ls, label=nombre)
    a.axhline(1.0, color=COLOR["secundario"], lw=0.8, ls=":")
    a.set_xlabel(r"temperatura del positrón $T$ (K)"); a.set_ylabel(r"ocupación media $\bar n$")
    a.legend(loc="lower right", fontsize=8); a.set_title("(a)", loc="left", fontsize=10)
    b.axhline(radio_punto_cero(W_1) * 1e6, color=COLOR["teoria"], lw=0.8, ls=":")
    b.text(0.022, radio_punto_cero(W_1) * 1e6 * 1.15, "punto cero del ciclotrón, 36 nm", fontsize=7.5)
    b.set_xlabel(r"temperatura del positrón $T$ (K)"); b.set_ylabel(r"amplitud (µm)")
    b.set_title("(b)", loc="left", fontsize=10)
    for axx in (a, b):
        marcar_temperaturas(axx)
    guardar(fig, "fig4_06_modos")
    guardar_datos("fig4_06_modos", "T_K,n_cic,A_cic_m,n_ax,A_ax_m,n_mag,A_mag_m", *datos)


def fig7_canales_temperatura():
    from cuantico import razon_tiempo_positron
    from ruido import D_desde_SE, escenarios, tiempo_medio

    T = np.logspace(np.log10(0.05), 1, 120)
    fig, (a, b) = plt.subplots(1, 2, figsize=tamano(2, 7.5))
    # (a) tiempo medio frente a la temperatura del sistema (trampa, última etapa del filtro y positrón)
    esc = escenarios(T)
    anios = {k: tiempo_medio(D_desde_SE(v)) / UN_ANIO for k, v in esc.items()}
    estilo = {"filtro real (corte 1,2 kHz)": (COLOR["serie3"], "-", "filtro real, 1,2 kHz"),
              "solo la etapa fría, filtro real": (COLOR["serie3"], ":", "  solo su etapa fría"),
              "filtro fuerte (corte 12 Hz) + EMI": (COLOR["secundario"], "-", "filtro de 12 Hz + EMI"),
              "solo la etapa fría, filtro fuerte": (COLOR["secundario"], ":", "  solo su etapa fría")}
    for k, (c, ls, lab) in estilo.items():
        a.loglog(T, anios[k], color=c, ls=ls, lw=1.6 if ls == "-" else 1.2, label=lab)
    for k, Tk in (("4K", T_4K), ("01K", T_01K)):
        a.plot(Tk, tiempo_medio(D_V11[k]) / UN_ANIO, color=COLOR[k], ls="none",
               marker=LINEA[k]["marker"], ms=5, label="V11, 4 K" if k == "4K" else "V11, 0,1 K")
    a.set_xlabel(r"temperatura del sistema $T$ (K)")
    a.set_ylabel(r"tiempo medio de pérdida $\bar t$ (años)")
    a.set_ylim(1e-3, 1e8)
    a.legend(loc="lower left", bbox_to_anchor=(0.0, 0.07), fontsize=6.5, ncol=2, columnspacing=0.8)
    a.set_title("(a)", loc="left", fontsize=10)
    # (b) temperatura del positrón: canal J0, desviación en partes por 10^4
    dev = (razon_tiempo_positron(T, T_4K, R_WALL) - 1) * 1e4
    b.semilogx(T, dev, color=COLOR["teoria"], ls="-")
    b.axhline(0, color=COLOR["secundario"], lw=0.8, ls=":")
    b.set_xlabel(r"temperatura del positrón $T$ (K)")
    b.set_ylabel(r"$[\bar t(T)/\bar t(4\,\mathrm{K}) - 1] \times 10^4$")
    b.set_title("(b)", loc="left", fontsize=10)
    for axx in (a, b):
        marcar_temperaturas(axx)
    guardar(fig, "fig4_07_canales_temperatura")
    guardar_datos("fig4_07_canales", "T_K," + ",".join(k.replace(",", ".") for k in anios)
                  + ",desviacion_positron_x1e4", T, *anios.values(), dev)


def fig8_mapa():
    from centro_guia import E_critico
    from ruido import SE_desde_D, escenarios
    from superficie import E_centro_rms

    S_anio = SE_desde_D(R_WALL**2 / (4 * UN_ANIO))
    fig, ax = plt.subplots(figsize=tamano(2, 8.0))
    x0, x1, y0, y1 = 1e-21, 1e-8, 1e-5, 1e4
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.fill_between([x0, S_anio], y0, E_critico("golpe"), color=COLOR["serie3"], alpha=0.12, lw=0)
    ax.fill_between([x0, x1], E_critico("golpe"), E_critico("lento"), color=COLOR["secundario"], alpha=0.18, lw=0)
    ax.axhline(E_critico("golpe"), color=COLOR["teoria"], lw=0.9, ls="-")
    ax.axhline(E_critico("lento"), color=COLOR["teoria"], lw=0.9, ls="--")
    ax.axvline(S_anio, color=COLOR["teoria"], lw=0.9, ls=":")
    ax.text(3e-21, 3e-4, "estable:\nórbita acotada y\n" r"$\bar t$ > 1 año", fontsize=8, color=COLOR["teoria"])
    ax.text(3e-21, 400, "pérdida inmediata por el parche estático", fontsize=8)
    ax.text(3e-21, 70, "50–100 V/m: depende de cómo aparece el campo", fontsize=7, va="center")
    ax.text(S_anio * 2, 3e-2, "pérdida por ruido\nen menos de un año", fontsize=8)
    for k, T in (("4K", T_4K), ("01K", T_01K)):
        lab = "4 K" if k == "4K" else "0,1 K"
        ax.plot(SE_desde_D(D_V11[k]), E_V11, color=COLOR[k], ls="none", marker=LINEA[k]["marker"],
                ms=5, label=f"V11, {lab}")
        esc = escenarios(T)
        for j, nombre in enumerate(("filtro real (corte 1,2 kHz)", "filtro fuerte (corte 12 Hz) + EMI")):
            ax.plot(esc[nombre][0], E_ALPHA, color=COLOR[k], ls="none", marker=LINEA[k]["marker"],
                    ms=5, mfc="none", label=(f"modelo corregido, ruido realista, {lab}" if j == 0 else None))
    ax.set_xlabel(r"ruido del campo a $f_-$, $S_E(f_-)$ (V$^2$ m$^{-2}$ Hz$^{-1}$)")
    ax.set_ylabel(r"campo estático del parche $|E|$ (V/m)")
    ax.legend(loc="lower right", fontsize=7.5)
    guardar(fig, "fig4_08_mapa_regimenes")


def fig9_parches_fluctuantes():
    from parches_fluctuantes import (SE_cota_BASE, banda_superficie, simular_parches_fluctuantes)
    from ruido import D_desde_SE, escenarios, tiempo_medio

    fig, (a, b) = plt.subplots(1, 2, figsize=tamano(2, 7.5))
    # (a) Monte Carlo de parches reales que fluctúan, frente a la difusión predicha
    xs, rel, err = (0.3, 1.0, 3.0), [], []
    for i, x in enumerate(xs):
        t, R2, D, e, D_th, sE = simular_parches_fluctuantes(wtau=x, semilla=910 + i,
                                                            n_periodos=150 if x < 3 else 400)
        rel.append(D / D_th); err.append(e / D_th)
        if x == 1.0:
            ej = (t, R2, D_th)
    t, R2, D_th = ej
    a.plot(t * 1e3, R2 * 1e12, color=COLOR["serie3"], ls="-", label="200 parches que fluctúan")
    a.plot(t * 1e3, 4 * D_th * t * 1e12, color=COLOR["teoria"], **LINEA["teoria"], label=r"$4Dt$ (fórmula)")
    a.set_xlabel(r"$t$ (ms)"); a.set_ylabel(r"$\langle |Z|^2\rangle$ (µm$^2$)")
    a.legend(loc="upper left", fontsize=8)
    txt = "\n".join(f"$\\omega_-\\tau_s$ = {x:g}: {r:.3f} ± {e:.3f}" for x, r, e in zip(xs, rel, err))
    a.text(0.97, 0.05, "$D_{sim}/D_{teo}$\n" + txt.replace(".", ","), transform=a.transAxes,
           ha="right", va="bottom", fontsize=7.5)
    a.set_title("(a)", loc="left", fontsize=10)
    # (b) tiempo medio por parches que fluctúan, escenarios A y B, frente a la temperatura
    T = np.logspace(np.log10(0.05), 1, 150)
    for esc, c, txt in (("A", COLOR["secundario"], "A: los parches fluctúan igual"),
                        ("B", COLOR["serie3"], "B: los parches se congelan")):
        lo, hi = banda_superficie(T, esc)
        b.fill_between(T, tiempo_medio(D_desde_SE(hi)) / UN_ANIO, tiempo_medio(D_desde_SE(lo)) / UN_ANIO,
                       color=c, alpha=0.25, lw=0, label=txt)
        b.loglog(T, np.sqrt(tiempo_medio(D_desde_SE(hi)) * tiempo_medio(D_desde_SE(lo))) / UN_ANIO,
                 color=c, ls="-" if esc == "B" else "--", lw=1.4)
    b.axhline(tiempo_medio(D_desde_SE(SE_cota_BASE(1.43))) / UN_ANIO, color=COLOR["teoria"], ls=":", lw=1.2)
    b.text(0.055, tiempo_medio(D_desde_SE(SE_cota_BASE(1.43))) / UN_ANIO * 0.25,
           "mínimo compatible con BASE", fontsize=7.5)
    for k, Tk in (("4K", T_4K), ("01K", T_01K)):
        b.plot(Tk, tiempo_medio(D_V11[k]) / UN_ANIO, color=COLOR[k], ls="none",
               marker=LINEA[k]["marker"], ms=5, label="V11, 4 K" if k == "4K" else "V11, 0,1 K")
    b.set_xscale("log"); b.set_yscale("log")
    b.set_ylim(1e-4, 1e9)
    b.set_xlabel(r"temperatura del sistema $T$ (K)")
    b.set_ylabel(r"tiempo medio de pérdida $\bar t$ (años)")
    b.legend(loc="center left", fontsize=7, bbox_to_anchor=(0.0, 0.35))
    b.set_title("(b)", loc="left", fontsize=10)
    marcar_temperaturas(b)
    guardar(fig, "fig4_09_parches_fluctuantes")
    lo_A, hi_A = banda_superficie(T, "A"); lo_B, hi_B = banda_superficie(T, "B")
    guardar_datos("fig4_09_parches", "T_K,SE_min_A,SE_max_A,SE_min_B,SE_max_B", T, lo_A, hi_A, lo_B, hi_B)


FIGURAS = {1: fig1_parche_estatico, 2: fig2_apantallamiento, 3: fig3_transito, 4: fig4_difusion,
           5: fig5_tiempo_vs_ruido, 6: fig6_modos, 7: fig7_canales_temperatura, 8: fig8_mapa,
           9: fig9_parches_fluctuantes}

if __name__ == "__main__":
    usar_estilo()
    elegidas = [int(a) for a in sys.argv[1:]] or list(FIGURAS)
    for n in elegidas:
        print(f"Figura {n}: {FIGURAS[n].__name__}")
        FIGURAS[n]()
        plt.close("all")
