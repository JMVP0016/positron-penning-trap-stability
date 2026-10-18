"""
centro_guia.py — Parte estática de los potenciales de parche: el centro guía con un campo fijo.

FÍSICA
------
El centro guía (el centro de la órbita magnetrón) se mueve con la deriva E x B, perpendicular
al campo, es decir, A LO LARGO de las equipotenciales del potencial total
psi = -U0 r^2 / 2 + phi_parche. Si el campo no cambia, recorre una curva cerrada: un parche
estático no hace difundir al positrón, solo le cambia el centro de giro.

ECUACIÓN (Z = X + iY,  Ec = Ex + iEy)
    dZ/dt = -i w Z - i Ec / B0,      w = U0 / B0  (coincide con omega_- a una parte en 10^6)

CAMPO UNIFORME Y CONSTANTE
    nuevo centro:  Z* = -Ec / U0  (ahí la fuerza del parche compensa la de la trampa)
    desplazamiento: delta = |E| / U0
    órbita:        Z(t) = Z* + (Z0 - Z*) exp(-i w t)   (círculo alrededor de Z*)

UMBRAL DE PÉRDIDA: depende de lo rápido que aparece el campo frente al periodo magnetrón P_-
    de golpe:  el positrón queda en el centro y gira alrededor de Z* con radio delta -> llega a 2 delta
               E_crit = U0 r_wall / 2 = 50 V/m
    despacio:  el radio alrededor de Z* es un invariante adiabático: el centro guía acompaña a Z*
               E_crit = U0 r_wall = 100 V/m
    rampa lineal de duración t_r:  max|Z| / delta = 1 + |sen(pi x)| / (pi x),  x = t_r / P_-

Uso:  python centro_guia.py
"""
import numpy as np

from params import B0, R_WALL, U0

W_GC = U0 / B0                 # frecuencia de rotación del centro guía (rad/s)
P_M = 2 * np.pi / W_GC         # periodo magnetrón (s), unos 157 µs


# ===========================================================================
# 1. Integrador del centro guía (lo usan también ruido.py y transito.py)
# ===========================================================================
def paso_centro_guia(Z, E_ini, E_fin, dt):
    """Integrador exponencial: la rotación se integra de forma exacta y el campo se toma
    como el promedio del paso. Sale de resolver la ecuación con Ec constante durante dt.
    Con Euler el factor de rotación tendría módulo sqrt(1 + w^2 dt^2) > 1 y el radio
    crecería artificialmente; aquí |rot| = 1 exacto."""
    rot = np.exp(-1j * W_GC * dt)
    ker = (1 - rot) / (W_GC * B0)
    return rot * Z - ker * 0.5 * (E_ini + E_fin)


# ===========================================================================
# 2. Campo estático uniforme: solución exacta y umbral
# ===========================================================================
def centro_desplazado(Ec):
    """Nuevo centro de la órbita, Z* = -Ec / U0 (m, complejo)."""
    return -Ec / U0


def orbita_exacta(Ec, t, Z0=0.0):
    """Órbita del centro guía con campo uniforme constante desde t = 0."""
    Zs = centro_desplazado(Ec)
    return Zs + (Z0 - Zs) * np.exp(-1j * W_GC * t)


def E_critico(rapidez="golpe", r_wall=R_WALL):
    """Campo estático que lleva el centro guía a la pared.
    'golpe': el campo aparece en mucho menos de un periodo magnetrón -> U0 r_wall / 2.
    'lento': el campo aparece o cambia en mucho más de un periodo  -> U0 r_wall."""
    return U0 * r_wall / (2 if rapidez == "golpe" else 1)


def max_Z_rampa_teorico(x):
    """max|Z| / delta para una rampa lineal de duración x periodos magnetrón."""
    x = np.asarray(x, dtype=float)
    sinc = np.where(x > 0, np.abs(np.sin(np.pi * x)) / (np.pi * np.maximum(x, 1e-300)), 1.0)
    return 1 + sinc


def simular_rampa(Ec_final, t_rampa, periodos_despues=10, pasos_por_periodo=400):
    """Integra el centro guía mientras el campo sube linealmente de 0 a Ec_final en t_rampa
    (t_rampa = 0: aparece de golpe). Devuelve (t, Z)."""
    dt = P_M / pasos_por_periodo
    t = np.arange(0.0, t_rampa + periodos_despues * P_M, dt)
    rampa = np.ones_like(t) if t_rampa == 0 else np.clip(t / t_rampa, 0.0, 1.0)
    E = Ec_final * rampa
    Z = np.zeros(t.size, dtype=complex)
    for n in range(1, t.size):
        Z[n] = paso_centro_guia(Z[n - 1], E[n - 1], E[n], dt)
    return t, Z


# ===========================================================================
# Validación y figura de control
# ===========================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from figuras import COLOR, LINEA, guardar, tamano, usar_estilo

    Ec = 1.0 * np.exp(1j * 0.7)          # 1 V/m (el valor de la V11) en una dirección cualquiera
    delta = abs(centro_desplazado(Ec))
    print("1) Campo uniforme de 1 V/m que aparece de golpe")
    t, Z = simular_rampa(Ec, 0.0, periodos_despues=3)
    err = np.max(np.abs(Z - orbita_exacta(Ec, t)))
    print(f"   delta = E/U0 = {delta*1e6:.1f} µm; radio máximo = {np.abs(Z).max()*1e6:.1f} µm"
          f" (2 delta = {2*delta*1e6:.1f} µm); r_wall = {R_WALL*1e6:.0f} µm")
    print(f"   integrador frente a solución exacta: diferencia máxima = {err*1e9:.2e} nm")

    print("2) Umbral de pérdida")
    print(f"   de golpe: E_crit = U0 r_wall / 2 = {E_critico('golpe'):.0f} V/m")
    print(f"   despacio: E_crit = U0 r_wall     = {E_critico('lento'):.0f} V/m")

    print("3) Rampa lineal: cuánto se aleja el centro guía según lo rápido que aparece el campo")
    xs = (0.0, 0.25, 0.5, 1.0, 1.5, 3.0, 10.0)
    sim = []
    for x in xs:
        t, Z = simular_rampa(Ec, x * P_M)
        sim.append(np.abs(Z).max() / delta)
        print(f"   rampa de {x:5.2f} periodos ({x*P_M*1e6:7.1f} µs): max|Z|/delta = {sim[-1]:.3f}"
              f"   (teoría: {float(max_Z_rampa_teorico(x)):.3f})")

    # Figura de control
    usar_estilo()
    fig, ax = plt.subplots(1, 2, figsize=tamano(2, 7.5))
    t0, Z0 = simular_rampa(Ec, 0.0, periodos_despues=1.05)
    t1, Z1 = simular_rampa(Ec, 5 * P_M, periodos_despues=1.05)
    Zs = centro_desplazado(Ec)
    ax[0].plot(Z0.real * 1e6, Z0.imag * 1e6, color=COLOR["4K"], ls="-", label="de golpe")
    ax[0].plot(Z1.real * 1e6, Z1.imag * 1e6, color=COLOR["01K"], ls="--", label="en 5 periodos")
    ax[0].plot(0, 0, "o", color=COLOR["teoria"], ms=4)
    ax[0].plot(Zs.real * 1e6, Zs.imag * 1e6, "+", color=COLOR["teoria"], ms=9, mew=1.5)
    ax[0].annotate("centro de\nla trampa", (0, 0), (8, 6), textcoords="offset points", fontsize=8)
    ax[0].annotate(r"$Z^*$", (Zs.real * 1e6, Zs.imag * 1e6), (6, -12), textcoords="offset points", fontsize=9)
    ax[0].set_aspect("equal")
    ax[0].set_xlabel(r"$X$ (µm)"); ax[0].set_ylabel(r"$Y$ (µm)")
    ax[0].legend(loc="lower left", fontsize=8)
    xx = np.linspace(0, 10, 600)
    ax[1].plot(xx, max_Z_rampa_teorico(xx), color=COLOR["teoria"], **LINEA["teoria"], label="teoría")
    ax[1].plot(xs, sim, color=COLOR["4K"], ls="none", marker="o", ms=4, label="integración")
    ax[1].set_xlabel(r"duración de la rampa $t_r/P_-$")
    ax[1].set_ylabel(r"$\max|Z|/\delta$")
    ax[1].set_ylim(0.9, 2.1)
    ax[1].legend(loc="upper right", fontsize=8)
    guardar(fig, "control_centro_guia")
