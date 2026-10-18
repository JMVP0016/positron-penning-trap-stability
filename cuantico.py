"""
cuantico.py — Los tres modos del positrón vistos como osciladores cuánticos, y cómo entra la
temperatura DEL POSITRÓN en el confinamiento.

FÍSICA
------
Cada modo (ciclotrón, axial, magnetrón) es un oscilador armónico con niveles separados por
hbar*omega. A una temperatura T, la ocupación media sigue la distribución de Bose-Einstein:
    n = 1 / (exp(hbar w / k_B T) - 1)
- Si k_B T >> hbar w, n ~ k_B T / (hbar w) >> 1: el modo es clásico.
- Si k_B T << hbar w, n -> 0: el modo queda en su estado fundamental ("congelado").
Para el ciclotrón, hbar w_+ / k_B = 0,67 K: a 4 K es clásico (n ~ 5,5) y a 0,1 K queda
congelado (n ~ 1e-3). El axial (hbar w_z / k_B = 0,64 mK) y el magnetrón siguen clásicos.

AMPLITUD COHERENTE CON LA ENERGÍA (corrige la V11)
    A = sqrt( 2 hbar (n + 1/2) / (m w_ef) ),   w_ef = w_1 (ciclotrón y magnetrón), w_z (axial)
El término 1/2 es el punto cero: ningún oscilador cuántico tiene amplitud menor que
sqrt(hbar / (m w_1)) = 36 nm en el ciclotrón. La V11 daba 1,786 nm a 0,1 K, que es imposible.

MAGNETRÓN: n_- = n_z es una HIPÓTESIS: es el límite del enfriamiento por bandas laterales
axial-magnetrón (Brown y Gabrielse, 1986). Sin ese acoplamiento el magnetrón no se termaliza.

CANAL DE LA TEMPERATURA DEL POSITRÓN EN LA PÉRDIDA
El positrón siente el campo de los parches promediado sobre su oscilación axial. Para una
variación axial de número de onda k:  E_promedio = E * J0(k A_z). Como S_E va con el cuadrado
del campo, el tiempo medio de pérdida va como 1 / J0(k A_z)^2. Con k = 1/a:
    1 - J0(k A_z) ~ (k A_z)^2 / 4 = k^2 k_B T / (2 m w_z^2)   -> del orden de 1e-4 a 4 K.

PUENTE CON LA DIFUSIÓN (ruido.py)
    <rho_-^2> = 2 hbar (n_- + 1/2) / (m w_1)  ->  dn_-/dt = (m w_1 / 2 hbar) d<rho^2>/dt
                                               = 2 m w_1 D / hbar = q^2 S_E(f_-) / (2 m hbar w_1)
La pérdida es un aumento continuo del número cuántico del magnetrón (su energía, que es
negativa, se hace cada vez más negativa).

Uso:  python cuantico.py
"""
import numpy as np
from scipy.special import j0

from params import HBAR, K_B, M_E, Q_E, W_1, W_M, W_P, W_Z

MODOS = {           # nombre: (frecuencia que fija la energía, frecuencia que fija la amplitud)
    "ciclotrón": (W_P, W_1),
    "axial": (W_Z, W_Z),
    "magnetrón": (W_Z, W_1),    # n_- = n_z (hipótesis de bandas laterales)
}


# ===========================================================================
# 1. Ocupaciones y amplitudes
# ===========================================================================
def ocupacion(T, w):
    """Ocupación media de Bose-Einstein de un modo de frecuencia w a temperatura T."""
    x = HBAR * w / (K_B * np.asarray(T, dtype=float))
    return 1.0 / np.expm1(x)


def amplitud(n, w_ef):
    """Amplitud efectiva con el punto cero incluido (m)."""
    return np.sqrt(2 * HBAR * (n + 0.5) / (M_E * w_ef))


def amplitud_clasica(T, w_e, w_ef):
    """Amplitud clásica (sin punto cero), la que usaría un modelo puramente térmico."""
    return np.sqrt(2 * K_B * np.asarray(T, dtype=float) / (HBAR * w_e) * HBAR / (M_E * w_ef))


def radio_punto_cero(w_ef):
    return np.sqrt(HBAR / (M_E * w_ef))


def tabla_modos(T):
    """Devuelve {modo: (n, amplitud en m)} a la temperatura T."""
    out = {}
    for nombre, (w_e, w_ef) in MODOS.items():
        n = ocupacion(T, w_e)
        out[nombre] = (n, amplitud(n, w_ef))
    return out


# ===========================================================================
# 2. Canal de la temperatura del positrón: promedio axial del campo
# ===========================================================================
def factor_promedio_axial(T, a):
    """J0(k A_z) con k = 1/a: fracción del campo que sobrevive al promedio axial."""
    n = ocupacion(T, W_Z)
    return j0(amplitud(n, W_Z) / a)


def razon_tiempo_positron(T, T_ref, a):
    """t_medio(T) / t_medio(T_ref) debido SOLO a la temperatura del positrón."""
    return (factor_promedio_axial(T_ref, a) / factor_promedio_axial(T, a)) ** 2


# ===========================================================================
# 3. Puente con la difusión: cuánto sube n_- por unidad de tiempo
# ===========================================================================
def dn_menos_dt(S_E):
    """Ritmo de aumento del número cuántico magnetrón (1/s) para un ruido S_E(f_-)."""
    return Q_E**2 * S_E / (2 * M_E * HBAR * W_1)


def energia_magnetron(rho):
    """Energía del magnetrón (J) con radio rho: E_- ~ -(1/4) m w_z^2 rho^2 (negativa)."""
    return -0.25 * M_E * W_Z**2 * rho**2


if __name__ == "__main__":
    from params import B0, R_WALL
    from ruido import D_desde_SE

    print("1) Escalas de energía de cada modo (hbar w / k_B)")
    for nombre, w in (("ciclotrón", W_P), ("axial", W_Z), ("magnetrón", W_M)):
        print(f"   {nombre:10s}: {HBAR*w/K_B:.3e} K")

    print("2) Ocupaciones y amplitudes (con punto cero)")
    for T in (4.0, 0.1):
        print(f"   T = {T} K")
        for nombre, (n, A) in tabla_modos(T).items():
            w_e, w_ef = MODOS[nombre]
            Acl = amplitud_clasica(T, w_e, w_ef)
            print(f"     {nombre:10s}: n = {n:10.3e}   A = {A*1e9:10.1f} nm   (clásica: {Acl*1e9:10.1f} nm)")
    print(f"   radio de punto cero del ciclotrón: {radio_punto_cero(W_1)*1e9:.1f} nm"
          f" (la V11 daba 1,786 nm a 0,1 K)")

    print("3) Canal de la temperatura del positrón (promedio axial, k = 1/r_wall)")
    for T in (4.0, 0.1):
        print(f"   T = {T} K: 1 - J0(k A_z) = {1 - factor_promedio_axial(T, R_WALL):.2e}")
    print(f"   t_medio(0,1 K) / t_medio(4 K) por este canal = {razon_tiempo_positron(0.1, 4.0, R_WALL):.6f}")

    print("4) Puente con la difusión")
    S = 1.98e-13       # el ruido que da una media de un año
    print(f"   S_E = {S:.2e}: dn_-/dt = {dn_menos_dt(S):.3e} 1/s;"
          f" comprobación 2 m w_1 D / hbar = {2*M_E*W_1*D_desde_SE(S)/HBAR:.3e} 1/s")
    print(f"   energía del magnetrón: {energia_magnetron(radio_punto_cero(W_1)*np.sqrt(2*156.5))/K_B:.2e} K"
          f" (0,1 K, al inicio) -> {energia_magnetron(R_WALL)/K_B:.0f} K en la pared")
