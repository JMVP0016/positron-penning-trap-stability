"""
figuras_verificacion.py — Figuras de las comprobaciones numéricas del anexo B (versión 2.0.2).

    python figuras_verificacion.py          # las cuatro figuras (unos 5 minutos)
    python figuras_verificacion.py 1 3      # solo las figuras 1 y 3
    FIG_IDIOMA=en python figuras_verificacion.py

Cada figura se guarda en figures/ (PDF y PNG) y sus datos en datos/ (CSV). No cambia ningún
resultado de la tesis: solo muestra, con semillas fijas, comprobaciones que el código ya hace.

Figuras
    1  verif_boris_fase        Integrador de Boris (orbita_completa.py): la transformada de Cayley
                               conserva la norma en la rotación magnética pura, pero gira
                               2 arctan(w_c dt / 2) por paso en lugar de w_c dt. Error relativo de
                               fase y su convergencia como (w_c dt)^2 / 12.
    2  verif_convergencia_dt   Convergencia temporal: D simulado / D teórico frente al paso dt del
                               integrador del centro guía con ruido OU (ruido.py), w_- tau = 1.
    3  verif_disipacion        Tiempo medio de primer paso con disipación: Monte Carlo frente a
                               Ein(U) / (2 lambda), y límite sin disipación R^2 / (4 D) (disipacion.py).
    4  verif_orbita_completa   Órbita completa (fuerza de Lorentz) frente al centro guía: umbral
                               ante un campo súbito, ritmos con disipación, difusión con ruido OU y
                               tiempo medio con ruido térmico y disipación (orbita_completa.py).
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from figuras import COLOR, LINEA, guardar, tamano, usar_estilo

_DIR = os.path.dirname(os.path.abspath(__file__))
IDIOMA = os.environ.get("FIG_IDIOMA", "es")


def L(es, en):
    """Devuelve el texto en el idioma elegido."""
    return en if IDIOMA == "en" else es


def coma(ax, eje="y"):
    """Separador decimal de las marcas de un eje lineal: coma en español."""
    if IDIOMA == "en":
        return
    f = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    (ax.yaxis if eje == "y" else ax.xaxis).set_major_formatter(f)


def guardar_datos(nombre, cabecera, *columnas):
    ruta = os.path.join(_DIR, "datos")
    os.makedirs(ruta, exist_ok=True)
    np.savetxt(os.path.join(ruta, f"{nombre}.csv"), np.column_stack(columnas),
               delimiter=",", header=cabecera, comments="", fmt="%.6e")


# ===========================================================================
# 1. Integrador de Boris: norma y fase de la rotación magnética pura
# ===========================================================================
def fig1_boris_fase():
    """Matriz de Cayley exactamente como en orbita_completa.boris con G = Gz = 0 y q B0 / m = w_c = 1."""
    wdt = np.logspace(-3, 0, 25)
    theta, norma = [], []
    for x in wdt:
        Mv = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        I3 = np.eye(3)
        Cay = np.linalg.inv(I3 - 0.5 * x * Mv) @ (I3 + 0.5 * x * Mv)
        theta.append(np.arctan2(Cay[0, 1], Cay[0, 0]))       # ángulo que gira v en un paso
        v = np.array([1.0, 0.0, 0.0])
        for _ in range(1000):                                # mil pasos seguidos
            v = Cay @ v
        norma.append(abs(np.linalg.norm(v) - 1.0))
    theta, norma = np.array(theta), np.array(norma)
    err_num = (wdt - theta) / wdt
    err_teo = 1 - 2 * np.arctan(wdt / 2) / wdt
    guardar_datos("verif_boris_fase", "wc_dt,theta_num,error_rel_num,error_rel_teo,desvio_norma_1000pasos",
                  wdt, theta, err_num, err_teo, norma)
    fig, (a, b) = plt.subplots(1, 2, figsize=tamano(2, 6.0))
    a.loglog(wdt, err_teo, color=COLOR["teoria"], ls="-", lw=1.0,
             label=L(r"$1-2\arctan(\omega_c\Delta t/2)/(\omega_c\Delta t)$", r"$1-2\arctan(\omega_c\Delta t/2)/(\omega_c\Delta t)$"))
    a.loglog(wdt, wdt**2 / 12, color=COLOR["teoria"], **LINEA["teoria"], label=r"$(\omega_c\Delta t)^2/12$")
    a.loglog(wdt, err_num, "o", color=COLOR["serie3"], ms=4, label=L("matriz de Cayley del código", "Cayley matrix of the code"))
    a.set_xlabel(L(r"paso adimensional $\omega_c\Delta t$", r"dimensionless step $\omega_c\Delta t$"))
    a.set_ylabel(L("error relativo de fase por paso", "relative phase error per step"))
    a.legend(fontsize=7.5, loc="lower right"); a.set_title("(a)", loc="left", fontsize=10)
    b.loglog(wdt, np.maximum(norma, 1e-17), "s", color=COLOR["secundario"], ms=4)
    b.axhline(np.finfo(float).eps * 1000, color=COLOR["teoria"], **LINEA["teoria"])
    b.text(1.2e-3, np.finfo(float).eps * 1000 * 1.6, L(r"$10^3\,\epsilon_{\mathrm{máq}}$", r"$10^3\,\epsilon_{\mathrm{mach}}$"), fontsize=8)
    b.set_ylim(1e-17, 1e-10)
    b.set_xlabel(L(r"paso adimensional $\omega_c\Delta t$", r"dimensionless step $\omega_c\Delta t$"))
    b.set_ylabel(L(r"$\left|\,|\mathbf{v}_{1000}|/|\mathbf{v}_0|-1\right|$", r"$\left|\,|\mathbf{v}_{1000}|/|\mathbf{v}_0|-1\right|$"))
    b.set_title("(b)", loc="left", fontsize=10)
    fig.tight_layout()
    guardar(fig, "verif_boris_fase")
    print(f"  Boris: error relativo de fase {err_num[0]:.2e} con w_c dt = 1e-3 y {err_num[-1]:.3f} con w_c dt = 1;"
          f" desvío máximo de la norma tras 1000 pasos = {norma.max():.1e}")


# ===========================================================================
# 2. Convergencia temporal del integrador del centro guía con ruido OU
# ===========================================================================
def fig2_convergencia_dt():
    from params import B0, T_M, W_M
    from ruido import D_ou, simular_difusion
    sigma, tau = 1 / np.sqrt(2), 1.0 / W_M                    # w_- tau = 1
    divisores = np.array([12.5, 25, 50, 100, 200, 400])        # dt = T_- / divisor
    rel, err = [], []
    for k, n in enumerate(divisores):
        _, _, D, e = simular_difusion(tau, sigma, n_tray=8000, n_lotes=16, n_periodos=150,
                                      semilla=7000 + k, dt=T_M / n)
        rel.append(D / D_ou(sigma, tau)); err.append(e / D_ou(sigma, tau))
    rel, err = np.array(rel), np.array(err)
    guardar_datos("verif_convergencia_dt", "dt_sobre_T_menos,D_sim_sobre_D_teo,error", 1 / divisores, rel, err)
    fig, a = plt.subplots(figsize=tamano(1, 6.0))
    a.axhline(1.0, color=COLOR["teoria"], **LINEA["teoria"], label=L("teoría", "theory"))
    a.errorbar(1 / divisores, rel, yerr=err, fmt="o", color=COLOR["serie3"], ms=4, capsize=2,
               label=L("simulación", "simulation"))
    a.axvline(1 / 200, color=COLOR["secundario"], lw=0.8, ls="--")
    a.text(1 / 200 * 1.08, 0.955, L("paso usado\nen la tesis", "step used\nin the thesis"), fontsize=7.5, color=COLOR["secundario"])
    a.set_xscale("log")
    a.set_xticks(1 / divisores)
    a.set_xticklabels([("1/" + f"{n:g}").replace(".", "," if IDIOMA != "en" else ".") for n in divisores])
    a.minorticks_off()
    a.set_ylim(0.93, 1.07)
    a.set_xlabel(L(r"paso $\Delta t/T_-$", r"step $\Delta t/T_-$"))
    a.set_ylabel(r"$D_{\mathrm{sim}}/D_{\mathrm{teo}}$")
    coma(a, "y")
    a.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    guardar(fig, "verif_convergencia_dt")
    print("  convergencia: D_sim/D_teo = " + ", ".join(f"{r:.3f}±{e:.3f} (dt=T/{n:g})" for r, e, n in zip(rel, err, divisores)))


# ===========================================================================
# 3. Tiempo medio con disipación: Monte Carlo frente a Ein(U)/(2 lambda)
# ===========================================================================
def fig3_disipacion():
    from disipacion import Ein, simular_primer_paso
    Us = np.array([1.0, 3.0, 10.0, 30.0, 100.0])
    media, err = [], []
    for k, U in enumerate(Us):
        tp = simular_primer_paso(U, n_tray=4000, dt=2e-3, semilla=8000 + k)
        media.append(np.mean(tp)); err.append(np.std(tp) / np.sqrt(tp.size))
    media, err = np.array(media), np.array(err)
    teo = Ein(Us) / 2.0                                         # unidades lambda = 1, R = 1
    guardar_datos("verif_disipacion", "U,t_medio_sim_lambda,error,t_medio_teo_lambda", Us, media, err, teo)
    UU = np.logspace(-0.3, 2.3, 200)
    fig, a = plt.subplots(figsize=tamano(1, 6.0))
    a.loglog(UU, Ein(UU) / 2, color=COLOR["teoria"], ls="-", lw=1.0, label=r"$\mathrm{Ein}(U)/(2\lambda)$")
    a.loglog(UU, UU / 2, color=COLOR["teoria"], **LINEA["teoria"],
             label=L(r"sin disipación, $R^2/(4D)$", r"no dissipation, $R^2/(4D)$"))
    a.errorbar(Us, media, yerr=err, fmt="o", color=COLOR["serie3"], ms=4, capsize=2,
               label=L("Monte Carlo, 4000 trayectorias", "Monte Carlo, 4000 trajectories"))
    a.set_xlabel(L(r"$U=\lambda R^2/(2D)$", r"$U=\lambda R^2/(2D)$"))
    a.set_ylabel(L(r"tiempo medio $\lambda\,\bar t$", r"mean time $\lambda\,\bar t$"))
    a.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    guardar(fig, "verif_disipacion")
    print("  disipación: sim/teo = " + ", ".join(f"{m/t:.3f} (U={U:g})" for m, t, U in zip(media, teo, Us)))


# ===========================================================================
# 4. Órbita completa frente al centro guía
# ===========================================================================
def fig4_orbita_completa():
    from orbita_completa import Ein, Trampa, discretizar, matriz_lineal, propagar, u_v
    TR = Trampa(w_z=0.07)
    KAPPA = 2 / np.pi
    fig, ((a, b), (c, d)) = plt.subplots(2, 2, figsize=tamano(2, 11.0))

    # (a) campo uniforme que aparece de golpe
    E0, dt = 1e-4, 0.05
    A = matriz_lineal(TR, n_extra=1); A[3, 6] = TR.q / TR.m * E0
    Phi, _ = discretizar(A, np.zeros_like(A), dt)
    X0 = np.zeros((1, 7)); X0[0, 6] = 1.0
    n, Xs = propagar(Phi, None, X0, int(1.2 * TR.P_m / dt), None, registrar=20)
    Xs = np.array(Xs)[:, 0, :]
    r = np.hypot(Xs[:, 0], Xs[:, 1]) / (E0 / TR.U0)
    a.plot(n * dt / TR.P_m, r, color=COLOR["serie3"], lw=0.8, label=L("órbita completa", "full orbit"))
    a.axhline(2.0, color=COLOR["teoria"], **LINEA["teoria"], label=L(r"centro guía, $2\delta$", r"guiding centre, $2\delta$"))
    a.set_xlabel(L(r"tiempo $t/T_-$", r"time $t/T_-$")); a.set_ylabel(r"$|u|/\delta$")
    coma(a, "x"); coma(a, "y"); a.legend(fontsize=7, loc="lower right"); a.set_title("(a)", loc="left", fontsize=10)
    max_a = r.max()

    # (b) ritmos con disipación E_x = -G v_x
    Gs = np.logspace(-4, -1, 7)
    cre, dec = [], []
    for G in Gs:
        ev = np.linalg.eigvals(matriz_lineal(TR, G=G))
        re = lambda w: ev[np.argmin(np.abs(np.abs(ev.imag) - w))].real
        cre.append(re(TR.w_m)); dec.append(-re(TR.w_p))
    cre, dec = np.array(cre), np.array(dec)
    b.loglog(Gs, TR.q * Gs * TR.w_m / (2 * TR.m * TR.w_1), color=COLOR["teoria"], ls="-", lw=1.0,
             label=L(r"teoría, $qG\omega_-/(2m\omega_1)$", r"theory, $qG\omega_-/(2m\omega_1)$"))
    b.loglog(Gs, TR.q * Gs * TR.w_p / (2 * TR.m * TR.w_1), color=COLOR["teoria"], **LINEA["teoria"],
             label=L(r"teoría, $qG\omega_+/(2m\omega_1)$", r"theory, $qG\omega_+/(2m\omega_1)$"))
    b.loglog(Gs, cre, "o", color=COLOR["serie3"], ms=4, label=L("crecimiento del magnetrón", "magnetron growth"))
    b.loglog(Gs, dec, "s", color=COLOR["secundario"], ms=4, label=L("decaimiento del ciclotrón", "cyclotron decay"))
    b.set_xlabel(L(r"coeficiente de reacción $G$", r"reaction coefficient $G$"))
    b.set_ylabel(L("ritmo (unidades escaladas)", "rate (scaled units)"))
    b.set_ylim(5e-8, 30); b.legend(fontsize=6.5, loc="upper left"); b.set_title("(b)", loc="left", fontsize=10)

    # (c) difusión con ruido OU isótropo, w_- tau = 1
    sigma, tau, sub, N = 1e-4, 1 / TR.w_m, 50, 4000
    A = matriz_lineal(TR, n_extra=2); Qc = np.zeros_like(A)
    A[3, 6] = A[4, 7] = TR.q / TR.m
    A[6, 6] = A[7, 7] = -1 / tau
    Qc[6, 6] = Qc[7, 7] = 2 * sigma**2 / tau
    dtc = TR.P_m / sub
    Phi, Lc = discretizar(A, Qc, dtc)
    rng = np.random.default_rng(123)
    X0 = np.zeros((N, 8)); X0[:, 6:] = sigma * rng.standard_normal((N, 2))
    idx, Xs = propagar(Phi, Lc, X0, 80 * sub, rng, registrar=sub // 2)
    t = idx * dtc
    R2 = np.array([np.mean(np.abs(TR.modos(*u_v(X))[1]) ** 2) for X in Xs])
    D_cg = sigma**2 * tau / (TR.B0**2 * (1 + (TR.w_m * tau) ** 2))
    D_oc = D_cg * (TR.w_c / TR.w_1) ** 2
    c.plot(t / TR.P_m, R2 / (4 * D_oc * TR.P_m), color=COLOR["serie3"], lw=0.9, label=L("órbita completa", "full orbit"))
    c.plot(t / TR.P_m, t / TR.P_m, color=COLOR["teoria"], **LINEA["teoria"],
           label=L(r"$4D\,(\omega_c/\omega_1)^2\,t$", r"$4D\,(\omega_c/\omega_1)^2\,t$"))
    c.set_xlabel(L(r"tiempo $t/T_-$", r"time $t/T_-$"))
    c.set_ylabel(L(r"$\langle|u_-|^2\rangle$ en unidades de $4D_{\mathrm{oc}}T_-$", r"$\langle|u_-|^2\rangle$ in units of $4D_{\mathrm{fo}}T_-$"))
    c.legend(fontsize=7, loc="upper left"); c.set_title("(c)", loc="left", fontsize=10)

    # (d) ruido térmico de las dos mitades del anillo + disipación, U = 30
    lam_obj, U, N = 0.005, 30.0, 4000
    ReZ = lam_obj * TR.a**2 * TR.B0 / (TR.q * KAPPA**2)
    G = 2 * TR.q * KAPPA**2 * ReZ / TR.a**2
    lam = TR.q * G * TR.w_m / (2 * TR.m * TR.w_1)
    kT = -TR.energia_magnetron(1.0) / U
    A = matriz_lineal(TR, G=G); Qc = np.zeros_like(A)
    Qc[3, 3] = (TR.q / TR.m) ** 2 * KAPPA**2 * 4 * kT * ReZ / TR.a**2
    dtd = TR.P_m / 256
    Phi, Ld = discretizar(A, Qc, dtd)
    rng = np.random.default_rng(321)
    tp = np.full(N, np.inf); vivo = np.ones(N, bool)

    def vigilar(k, X):
        f = vivo & (np.abs(TR.modos(*u_v(X))[1]) >= 1.0)
        tp[f] = k * dtd; vivo[f] = False
        return not vivo.any()

    t_teo = Ein(U) / (2 * lam)
    propagar(Phi, Ld, np.zeros((N, 6)), int(15 * t_teo / dtd), rng, registrar=10**9, alfin=vigilar)
    ts = np.sort(tp / t_teo)
    S = 1 - np.arange(1, N + 1) / N
    d.plot(ts, S, color=COLOR["serie3"], lw=0.9, label=L("órbita completa", "full orbit"))
    d.axvline(np.mean(tp) / t_teo, color=COLOR["serie3"], lw=0.8, ls="--", label=L("media simulada", "simulated mean"))
    d.axvline(1.0, color=COLOR["teoria"], **LINEA["teoria"], label=r"$\mathrm{Ein}(U)/(2\lambda)$")
    d.set_xlim(0.5, 1.6)
    d.set_xlabel(L(r"tiempo $t/\bar t_{\mathrm{teo}}$ ($U=30$)", r"time $t/\bar t_{\mathrm{theo}}$ ($U=30$)"))
    d.set_ylabel(L("fracción confinada", "fraction confined"))
    coma(d, "x"); coma(d, "y")
    d.legend(fontsize=6.5, loc="lower left"); d.set_title("(d)", loc="left", fontsize=10)

    fig.tight_layout()
    guardar(fig, "verif_orbita_completa")
    guardar_datos("verif_orbita_completa_ritmos", "G,crecimiento_magnetron,teoria_magnetron,decaimiento_ciclotron,teoria_ciclotron",
                  Gs, cre, TR.q * Gs * TR.w_m / (2 * TR.m * TR.w_1), dec, TR.q * Gs * TR.w_p / (2 * TR.m * TR.w_1))
    guardar_datos("verif_orbita_completa_difusion", "t_sobre_T_menos,R2_sobre_4DT", t / TR.P_m, R2 / (4 * D_oc * TR.P_m))
    pend = np.polyfit(t[t > t[-1] / 2], R2[t > t[-1] / 2], 1)[0] / 4
    print(f"  órbita completa: max|u|/delta = {max_a:.5f} (teoría {1 + TR.w_c / TR.w_1:.5f}; centro guía 2);"
          f" ritmos sim/teo = {np.mean(cre / (TR.q * Gs * TR.w_m / (2 * TR.m * TR.w_1))):.5f} y"
          f" {np.mean(dec / (TR.q * Gs * TR.w_p / (2 * TR.m * TR.w_1))):.5f};"
          f" D_oc/(D_cg (w_c/w_1)^2) = {pend / D_oc:.3f}; media/teoría (U=30) = {np.mean(tp) / t_teo:.3f}")


FIGURAS = {1: fig1_boris_fase, 2: fig2_convergencia_dt, 3: fig3_disipacion, 4: fig4_orbita_completa}

if __name__ == "__main__":
    usar_estilo()
    pedidas = [int(a) for a in sys.argv[1:]] or list(FIGURAS)
    for k in pedidas:
        print(f"Figura de verificación {k}:")
        FIGURAS[k]()
