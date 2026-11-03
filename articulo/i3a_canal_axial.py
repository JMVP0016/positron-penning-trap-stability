"""
i3a_canal_axial.py — A1, ronda 2, tarea A1-2 (I3a): canal T2.

Mecanismo: un campo de parche ESTÁTICO E0 que varía a lo largo de z (número de onda k) llega al
centro guía promediado sobre la oscilación axial: G0 = E0 J0(k A_z). Si la energía axial E_z
fluctúa térmicamente (el resonador de detección la amortigua y la excita), G0(t) fluctúa, y eso
es un ruido de UNA componente (la dirección de E0) a la frecuencia magnetrón.

FORMA CERRADA (supuestos: oscilador axial clásico, n_z >> 1, en equilibrio térmico con un baño a T;
cuadraturas OU, de modo que la energía tiene distribución exponencial con media k_B T y
autocorrelación (k_B T)^2 exp(-t/tau_z), donde tau_z = 1/gamma_z es el tiempo de relajación de la energía):
    x = E_z/(k_B T),  A_z^2 = A_th^2 x,  A_th^2 = 2 k_B T/(m w_z^2),  beta = k A_th
    G0 = E0 J0(beta sqrt(x)) = E0 sum_n a_n L_n(x),   a_n = (beta^2/4)^n exp(-beta^2/4) / n!
    (expansión de Laguerre: los L_n son las autofunciones del proceso de la energía, con decaimiento n/tau_z)
    S_G(f) = E0^2 sum_{n>=1} a_n^2 * 4 (tau_z/n) / [1 + (2 pi f tau_z/n)^2]     (unilateral)
    D = S_G(f_-) / (8 B0^2)          (una sola componente; véase R5 de la ronda 1)
    t = r_wall^2/(4D) = 2 r_wall^2 B0^2 / S_G(f_-)
    Límite beta << 1 y w_- tau_z >> 1:  S_G ~ 4 E0^2 (beta^2/4)^2 / (w_-^2 tau_z)  ->  t ∝ tau_z / (E0^2 k^4 T^2)

Uso: MPLBACKEND=Agg python3 i3a_canal_axial.py [--codigo RUTA] [--salida CARPETA]
Es determinista (semillas fijas) y tarda alrededor de 1 min.
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
from math import factorial  # noqa: E402
from scipy.special import j0  # noqa: E402

from params import B0, F_M, K_B, M_E, R_WALL, UN_ANIO, W_M, W_Z, Z0  # noqa: E402
from ruido import D_desde_SE, escenarios, tiempo_medio  # noqa: E402

E0 = 1.0          # V/m: campo de parche estático medido por ALPHA (orden de magnitud, Baker et al. 2024)


def coef_laguerre(beta, nmax=40):
    b = beta**2 / 4
    return np.array([b**n * np.exp(-b) / factorial(n) for n in range(nmax + 1)])


def S_G(f, beta, tau, E0=E0, nmax=40):
    a = coef_laguerre(beta, nmax)
    n = np.arange(1, nmax + 1)
    return E0**2 * np.sum(a[1:] ** 2 * 4 * (tau / n) / (1 + (2 * np.pi * f * tau / n) ** 2))


def beta_de(T, k):
    return k * np.sqrt(2 * K_B * T / (M_E * W_Z**2))


def t_anios(T, k, tau, E0=E0):
    S = S_G(F_M, beta_de(T, k), tau, E0)
    return 2 * R_WALL**2 * B0**2 / S / UN_ANIO


# --------------------------------------------------------------------------- simulación (escalada)
def simular(beta, wtau, n_tray=4000, t_fin=400.0, dt=0.05, n_lotes=8, semilla=0):
    """Unidades escaladas: w_- = 1, B = 1, E0 = 1. Cuadraturas X, Y ~ OU con tasa 1/(2 tau_z):
    x = (X^2 + Y^2)/2 tiene distribución exponencial (media 1) y Cov(x(0), x(t)) = exp(-t/tau_z).
    El campo es G0 - <G0>, a lo largo de x. Devuelve (D_sim, err, D_teo)."""
    rng = np.random.default_rng(semilla)
    tau = wtau
    q = np.exp(-dt / (2 * tau))
    s = np.sqrt(1 - q * q)
    X, Y = rng.normal(size=n_tray), rng.normal(size=n_tray)
    media = np.exp(-beta**2 / 4)
    G = lambda X, Y: j0(beta * np.sqrt((X * X + Y * Y) / 2)) - media
    E = G(X, Y)
    Z = np.zeros(n_tray, complex)
    rot = np.exp(-1j * dt)
    ker = 1 - rot
    ts, R2 = [], []
    n_pasos = int(t_fin / dt)
    for i in range(1, n_pasos + 1):
        X = q * X + s * rng.normal(size=n_tray)
        Y = q * Y + s * rng.normal(size=n_tray)
        En = G(X, Y)
        Z = rot * Z - ker * 0.5 * (E + En)
        E = En
        if i % 20 == 0:
            ts.append(i * dt)
            R2.append(np.mean(np.abs(Z.reshape(n_lotes, -1)) ** 2, axis=1))
    ts, R2 = np.array(ts), np.array(R2)
    m = ts > ts[-1] / 2
    Dl = np.polyfit(ts[m], R2[m], 1)[0] / 4
    D_teo = S_G(1 / (2 * np.pi), beta, tau, 1.0) / 8           # f_- = 1/(2 pi) en estas unidades
    return Dl.mean(), Dl.std(ddof=1) / np.sqrt(n_lotes), D_teo


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cif = []
    print("1) Validación: simulación del centro guía con G0(t) estocástico frente a la forma cerrada")
    for i, (beta, wtau) in enumerate(((0.3, 1.0), (1.0, 1.0), (1.0, 3.0), (2.0, 1.0))):
        D, e, Dt = simular(beta, wtau, semilla=100 + i)
        print(f"   beta = {beta}, w_- tau_z = {wtau}: D_sim/D_teo = {D/Dt:.3f} ± {e/Dt:.3f}")
        cif.append((f"Dsim_sobre_Dteo_beta{beta}_wtau{wtau}", D / Dt, "1", f"±{e/Dt:.3f}; escalado", "i3a_canal_axial.py", "VERIFICADO"))

    ks = {"1/a": 1 / R_WALL, "pi/z0": np.pi / Z0}
    taus = np.logspace(-2, 1, 7)
    Ts = np.logspace(np.log10(0.05), 1, 25)
    filas = []
    for kn, k in ks.items():
        for tau in taus:
            for T in Ts:
                filas.append((kn, k, tau, T, beta_de(T, k), t_anios(T, k, tau)))
    with open(os.path.join(args.salida, "i3a_canal_axial.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["k_nombre", "k_1_m", "tau_z_s", "T_K", "beta_kA_th", "t_medio_anios"])
        for f in filas:
            w.writerow([f[0], f"{f[1]:.4e}", f"{f[2]:.4e}", f"{f[3]:.4e}", f"{f[4]:.4e}", f"{f[5]:.4e}"])

    print("2) Pendiente d log t / d log T (0,05-10 K)")
    for kn, k in ks.items():
        for tau in (0.01, 1.0, 10.0):
            tt = np.array([t_anios(T, k, tau) for T in Ts])
            p = np.polyfit(np.log(Ts), np.log(tt), 1)[0]
            print(f"   k = {kn:6s}, tau_z = {tau:5.2f} s: pendiente = {p:.3f}")
            cif.append((f"pendiente_logt_logT_k{kn}_tau{tau}", p, "1", "T 0,05-10 K", "i3a", "VERIFICADO (cálculo)"))

    print("3) Comparación con el filtro real (E0 = 1 V/m)")
    esc = escenarios([4.0, 0.1])
    tf = tiempo_medio(D_desde_SE(esc["filtro real (corte 1,2 kHz)"])) / UN_ANIO
    print(f"   filtro real (código V12): 4 K -> {tf[0]:.2f} años; 0,1 K -> {tf[1]:.2f} años")
    for kn, k in ks.items():
        for tau in (0.01, 0.1, 1.0, 10.0):
            t4, t01 = t_anios(4.0, k, tau), t_anios(0.1, k, tau)
            print(f"   k = {kn:6s}, tau_z = {tau:5.2f} s: canal axial 4 K -> {t4:.3g} años, 0,1 K -> {t01:.3g} años"
                  f"   (canal/filtro a 4 K = {t4/tf[0]:.3g})")
            cif.append((f"t_canal_axial_4K_k{kn}_tau{tau}", t4, "años", "E0=1 V/m, una componente", "i3a", "INFERIDO (modelo)"))
            cif.append((f"t_canal_axial_01K_k{kn}_tau{tau}", t01, "años", "E0=1 V/m, una componente", "i3a", "INFERIDO (modelo)"))
    # E0 que haría el canal igual al filtro real a 4 K (peor caso: k = pi/z0, tau = 0,01 s)
    E_igual = np.sqrt(t_anios(4.0, np.pi / Z0, 0.01) / tf[0])
    print(f"   E0 necesario para igualar el filtro real a 4 K (k = pi/z0, tau_z = 0,01 s): {E_igual:.2g} V/m")
    cif.append(("E0_igual_filtro_4K_peor_caso", E_igual, "V/m", "k=pi/z0, tau_z=0,01 s", "i3a", "INFERIDO"))

    with open(os.path.join(args.salida, "i3a_cifras.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["magnitud", "valor", "unidad", "condiciones", "script_o_fuente", "estado"])
        for r in cif:
            w.writerow([r[0], f"{r[1]:.5g}", *r[2:]])

    fig, ax = plt.subplots(figsize=(6.3, 3.9))
    for kn, ls in (("1/a", "-"), ("pi/z0", "--")):
        for tau, c in ((0.01, "tab:red"), (1.0, "tab:blue"), (10.0, "tab:green")):
            ax.loglog(Ts, [t_anios(T, ks[kn], tau) for T in Ts], ls=ls, color=c,
                      label=f"k = {kn}, τ_z = {tau:g} s")
    TT = np.logspace(np.log10(0.05), 1, 60)
    tfr = tiempo_medio(D_desde_SE(escenarios(TT)["filtro real (corte 1,2 kHz)"])) / UN_ANIO
    ax.loglog(TT, tfr, color="k", lw=2, label="filtro real (código V12)")
    ax.set_xlabel("temperatura T (K)")
    ax.set_ylabel("tiempo medio de pérdida (años)")
    ax.set_title("Canal axial T2 (E0 = 1 V/m)", fontsize=9)
    ax.legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(args.salida, "i3a_canal_axial.pdf"))
    fig.savefig(os.path.join(args.salida, "i3a_canal_axial.png"), dpi=130)
