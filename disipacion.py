"""
disipacion.py — La disipación que acompaña al ruido del circuito (teorema de fluctuación-disipación)
y su efecto sobre el magnetrón.

FÍSICA
------
La resistencia del filtro que produce el ruido de Johnson (ruido.py) también absorbe la energía de
las corrientes que el positrón induce en los electrodos (Shockley-Ramo, con el mismo kappa que
acopla el ruido). El magnetrón tiene energía NEGATIVA, E_- ~ -(1/2) q B0 w_- rho^2: perder energía
aumenta su radio. Con el anillo en dos mitades, cada una con su línea, el radio crece como e^{lambda t}:
    lambda = q kappa^2 Re Z(w_-) w_- / (a^2 B0)
Con deriva y difusión a la vez, el tiempo medio de primer paso desde el centro hasta r_wall es
    t_medio = Ein(U) / (2 lambda),   Ein(U) = gamma_E + ln U + E1(U),   U = lambda r_wall^2 / (2 D)
U = |E_-(r_wall)| / (k_B T_ef), con T_ef = S_V / (4 k_B Re Z). Sin disipación (U -> 0), t_medio = r^2/(4D).

GENERALIZACIÓN (auditoría de la ronda 5, E3-01): no es exclusivo del filtro. Toda fuente de ruido en
equilibrio a T_b (resistencia, superficie en equilibrio, eje con su detector) lleva su disipación con
    U = |E_-(r_wall)| / (k_B T_b) = 0,25 eV / (k_B T_b) ~ 2901 K / T_b,
sea cual sea el acoplamiento, y entonces t_medio = t0 * Ein(U)/U con t0 = r^2/(4D) el de solo ruido:
los tiempos de solo ruido de las fuentes térmicas son cotas superiores (factor 1/101 a 4 K).
Con disipación la distribución del primer paso es estrecha (tipo Gumbel): mediana ~ 0,96 t_medio.
Referencia: Brown y Gabrielse, Rev. Mod. Phys. 58, 233 (1986), ec. (3.86) y sec. IV.

Uso:  python disipacion.py   (tabla de la tesis y una comprobación Monte Carlo rápida)
"""
import numpy as np
from scipy.special import exp1

from params import B0, K_B, Q_E, R_WALL, UN_ANIO, F_M, W_M
from ruido import (FILTRO_FUERTE, FILTRO_REAL, KAPPA_MEDIO_ANILLO, SV_EMI_BASE, SV_filtro,
                   D_desde_SE, SE_desde_SV, tiempo_medio)


def Z_nodo(filtro=FILTRO_REAL, f=F_M):
    """Impedancia (compleja) del nodo del electrodo a la frecuencia f: escalera R-C del filtro."""
    w = 2 * np.pi * f
    R, C = filtro["R"], filtro["C"]
    Zc = [1 / (1j * w * c) for c in C]
    Zn = 1 / (1 / R[0] + 1 / Zc[0])                 # primer nodo: R1 a la fuente (tierra en AC) || C1
    for k in range(1, len(R)):
        Zn = 1 / (1 / (R[k] + Zn) + 1 / Zc[k])      # (R_k + nodo anterior) || C_k
    return Zn


def ReZ_nyquist(filtro=FILTRO_REAL, f=F_M, T_ultima=4.0):
    """Re Z como suma de R_i |H_i|^2 (teorema de Nyquist-Twiss): debe coincidir con Z_nodo().real."""
    partes = SV_filtro(T_ultima, filtro, f=f, por_etapa=True)
    T = list(filtro["T_calientes"]) + [T_ultima]
    return sum(p / (4 * K_B * t) for p, t in zip(partes, T))


def lambda_disipacion(ReZ, kappa=KAPPA_MEDIO_ANILLO, a=R_WALL):
    """Ritmo de crecimiento del radio magnetrón por la disipación (1/s)."""
    return Q_E * kappa**2 * ReZ * W_M / (a**2 * B0)


def Ein(U):
    """Ein(U) = int_0^U (1 - e^-u)/u du = gamma_E + ln U + E1(U)."""
    U = np.asarray(U, dtype=float)
    return np.euler_gamma + np.log(U) + exp1(U)


def tiempo_medio_con_disipacion(D, lam, R=R_WALL):
    """Tiempo medio de primer paso con difusión D y anti-amortiguamiento lam (s)."""
    if lam <= 0:
        return tiempo_medio(D, R)
    U = lam * R**2 / (2 * D)
    if U < 1e-6:                                    # límite sin disipación
        return tiempo_medio(D, R)
    return float(Ein(U) / (2 * lam))


def U_termico(T_b, R=R_WALL):
    """U = |E_-(R)|/(k_B T_b) de una fuente en equilibrio a T_b; |E_-(R)| = (1/2) q B0 w_- R^2."""
    return 0.5 * Q_E * B0 * W_M * R**2 / (K_B * np.asarray(T_b, dtype=float))


def factor_fdt_termico(T_b, R=R_WALL):
    """Cociente t_medio(con disipación)/t_medio(solo ruido) = Ein(U)/U para una fuente térmica a T_b."""
    U = U_termico(T_b, R)
    return Ein(U) / U


def cociente_mediana_media(U):
    """Mediana/media del tiempo de pérdida con disipación (ley de Gumbel, válida para U >> 1)."""
    U = np.asarray(U, dtype=float)
    return (np.log(U) - np.log(np.log(2))) / (np.log(U) + np.euler_gamma)


def T_efectiva(S_V, ReZ):
    """Temperatura efectiva del ruido en el electrodo (K)."""
    return S_V / (4 * K_B * ReZ)


def escenarios_con_disipacion(T, f=F_M):
    """{escenario: (t_medio sin disipación, t_medio con disipación, U, T_ef)} en segundos."""
    out = {}
    for nombre, filtro in (("filtro de tres etapas + EMI", FILTRO_REAL), ("filtro fuerte + EMI", FILTRO_FUERTE)):
        ReZ = Z_nodo(filtro, f).real
        lam = lambda_disipacion(ReZ)
        S_V = SV_filtro(T, filtro, f=f) + SV_EMI_BASE
        D = D_desde_SE(SE_desde_SV(S_V))
        U = lam * R_WALL**2 / (2 * D)
        out[nombre] = (tiempo_medio(D), tiempo_medio_con_disipacion(D, lam), U, T_efectiva(S_V, ReZ))
    return out


def simular_primer_paso(U, n_tray=4000, dt=2e-3, semilla=None):
    """Monte Carlo en unidades lambda = 1, R = 1: tiempos de primer paso (paso exacto de la ecuación lineal)."""
    rng = np.random.default_rng(semilla)
    lam, R = 1.0, 1.0
    D = lam * R**2 / (2 * U)
    a = np.exp(lam * dt)
    s = np.sqrt(2 * D * (a * a - 1) / (2 * lam))
    Z = np.zeros(n_tray, dtype=complex)
    tp = np.full(n_tray, np.nan)
    vivos = np.ones(n_tray, bool)
    k = 0
    while vivos.any() and k < 10**6:
        k += 1
        Z = a * Z + s * (rng.normal(size=n_tray) + 1j * rng.normal(size=n_tray))
        fuera = vivos & (np.abs(Z) >= R)
        tp[fuera] = k * dt
        vivos &= ~fuera
    return tp


if __name__ == "__main__":
    Zn = Z_nodo()
    print(f"Re Z(f_-) del filtro de tres etapas = {Zn.real:.1f} Ω (Nyquist-Twiss: {ReZ_nyquist():.1f} Ω)")
    lam = lambda_disipacion(Zn.real)
    print(f"lambda = {lam:.3e} 1/s: rho^2 crece un factor e cada {1/(2*lam)/86400:.1f} días")
    for T in (4.0, 1.0, 0.3, 0.1):
        for nombre, (t0, t1, U, Tef) in escenarios_con_disipacion(T).items():
            print(f"  T = {T:>4} K, {nombre:28s}: T_ef = {Tef:7.1f} K, U = {U:7.1f};"
                  f"  sin disipación {t0/UN_ANIO:8.3g} años; con disipación {t1/UN_ANIO:8.3g} años")
    print("Fuente térmica genérica a T_b (U = |E_-(r_wall)|/k_B T_b): factor t_con/t_sin = Ein(U)/U")
    for T in (4.0, 1.0, 0.3, 0.1):
        print(f"  T_b = {T:>4} K: U = {float(U_termico(T)):9.1f}; Ein(U)/U = {float(factor_fdt_termico(T)):.3e}"
              f" (1/{1/float(factor_fdt_termico(T)):.0f})")
    print(f"Mediana/media con disipación (U = 92): {float(cociente_mediana_media(92.0)):.3f} (sin disipación: 0,802)")
    tp = simular_primer_paso(30.0, semilla=7)
    print(f"Monte Carlo (U = 30): media simulada {np.nanmean(tp):.3f} frente a Ein(U)/2 = {float(Ein(30.0))/2:.3f}")
