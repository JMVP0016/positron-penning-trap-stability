"""
superficie.py — Qué campo de los potenciales de parche llega al positrón (apantallamiento).

FÍSICA
------
La superficie de los electrodos no es un equipotencial perfecto: tiene parches cuyo potencial
varía unos milivoltios (V_rms) sobre distancias de tamaño l. El positrón está a unos 5 mm de
la pared. En el vacío el potencial cumple la ecuación de Laplace, y una variación de tamaño l
se atenúa como exp(-2 pi d / l) al alejarse: las estructuras pequeñas se anulan entre sí.
En el centro de la trampa solo sobrevive la parte más suave del patrón, el modo dipolar m = 1,
que da un campo uniforme. Ese campo es el que desplaza el centro guía (centro_guia.py).

MODELO (2D: cilindro de radio a con potencial de pared aleatorio V(theta))
    phi(r, theta) = sum_m (r/a)^m [A_m cos(m theta) + B_m sen(m theta)]
    en r = 0 solo queda m = 1:   E(0) = -(A_1, B_1) / a
    <A_1^2> = (1/pi) int_{-pi}^{pi} C(Delta) cos(Delta) dDelta,   C = V_rms^2 exp[-(a Delta)^2 / (2 l^2)]
            ~ sqrt(2/pi) (l/a) V_rms^2        si l << a
    |E(0)|_rms = sqrt(2) A_1,rms / a          (A_1 y B_1 aportan lo mismo)

El cilindro aproxima la trampa hiperbólica. En 3D el patrón también varía a lo largo de z y se
promedia más, así que el campo de este modelo es una COTA SUPERIOR.

Uso:  python superficie.py
"""
import numpy as np
from scipy.integrate import quad

from params import R_WALL, U0

A_CIL = R_WALL          # radio del cilindro que aproxima la trampa (m)


# ===========================================================================
# 1. Resultado analítico
# ===========================================================================
def A1_rms(V_rms, ell, a=A_CIL):
    """Amplitud rms del modo dipolar A_1 (V) para un patrón con correlación gaussiana."""
    s = ell / a
    integrando = lambda d: np.exp(-d**2 / (2 * s**2)) * np.cos(d)
    # El integrando es par y gaussiano: basta integrar en [0, min(pi, 12 s)] y duplicar.
    val = 2 * quad(integrando, 0.0, min(np.pi, 12 * s), limit=400)[0]
    return V_rms * np.sqrt(max(val, 0.0) / np.pi)


def A1_rms_aprox(V_rms, ell, a=A_CIL):
    """Fórmula asintótica para l << a: A_1,rms = V_rms (2/pi)^(1/4) sqrt(l/a)."""
    return V_rms * (2 / np.pi) ** 0.25 * np.sqrt(ell / a)


def E_centro_rms(V_rms, ell, a=A_CIL):
    """Campo rms en el centro de la trampa (V/m), sumando sus dos componentes."""
    return np.sqrt(2) * A1_rms(V_rms, ell, a) / a


def V_rms_necesario(E_objetivo, ell, a=A_CIL):
    """Qué V_rms haría falta para que llegue E_objetivo al centro (el campo es lineal en V_rms)."""
    return E_objetivo / E_centro_rms(1.0, ell, a)


# ===========================================================================
# 2. Comprobación Monte Carlo: patrones aleatorios generados por FFT
# ===========================================================================
def generar_patron(V_rms, ell, rng, a=A_CIL, N=2**16):
    """Un patrón V(theta) aleatorio con correlación gaussiana de longitud ell sobre la pared."""
    k = np.fft.rfftfreq(N, d=1.0 / N)                       # índice de modo m
    espectro = np.sqrt(2 * np.pi) * (ell / a) * np.exp(-(k * ell / a) ** 2 / 2)
    fase = (rng.normal(size=k.size) + 1j * rng.normal(size=k.size)) / np.sqrt(2)
    ruido = np.fft.irfft(np.sqrt(espectro * N) * fase, n=N)
    return V_rms * ruido / np.sqrt(np.mean(ruido**2))       # normalizado a V_rms exacto


def A1_montecarlo(V_rms, ell, n_patrones=400, semilla=None, a=A_CIL, N=2**16):
    """A_1,rms medido sobre n_patrones electrodos distintos (cada uno con su patrón fijo)."""
    rng = np.random.default_rng(semilla)
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    A1 = [2 * np.mean(generar_patron(V_rms, ell, rng, a, N) * np.cos(theta)) for _ in range(n_patrones)]
    return np.std(A1), np.std(A1) / np.sqrt(2 * (n_patrones - 1))   # valor y su error estadístico


# ===========================================================================
# Validación y figura de control
# ===========================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from figuras import COLOR, guardar, usar_estilo

    V = 1e-3
    print("1) Campo en el centro con V_rms = 1 mV (modelo 2D, cota superior)")
    for ell in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3):
        print(f"   l = {ell*1e6:8.1f} µm:  |E(0)|_rms = {E_centro_rms(V, ell):.2e} V/m"
              f"   (fórmula asintótica: {np.sqrt(2)*A1_rms_aprox(V, ell)/A_CIL:.2e})")

    print("2) Comprobación Monte Carlo (400 electrodos con patrón aleatorio fijo)")
    for i, ell in enumerate((5e-5, 2e-4)):
        mc, err = A1_montecarlo(V, ell, semilla=100 + i)
        print(f"   l = {ell*1e3:.2f} mm:  A_1,rms analítico = {A1_rms(V, ell)*1e6:.1f} µV,"
              f"  Monte Carlo = {mc*1e6:.1f} ± {err*1e6:.1f} µV")

    print("3) Qué patrón haría falta")
    for ell in (1e-7, 1e-3):
        print(f"   l = {ell*1e6:7.1f} µm: para 1 V/m (valor de la V11) V_rms = {V_rms_necesario(1.0, ell):.2e} V;"
              f" para 50 V/m (pérdida) V_rms = {V_rms_necesario(U0*R_WALL/2, ell):.2e} V")

    # Figura de control
    usar_estilo()
    ells = np.logspace(-7, -2.3, 60)
    fig, ax = plt.subplots()
    for Vr, c, ls in ((1e-3, COLOR["serie1"], "-"), (1e-2, COLOR["serie2"], "--")):
        ax.loglog(ells * 1e3, [E_centro_rms(Vr, l) for l in ells], color=c, ls=ls, label=f"$V_{{rms}}$ = {Vr*1e3:.0f} mV")
    ax.axhline(1.0, color=COLOR["teoria"], ls="--", lw=1)
    ax.text(1.3e-4, 1.25, "1 V/m: valor usado en la V11", fontsize=8)
    ax.axhline(U0 * R_WALL / 2, color=COLOR["teoria"], ls=":", lw=1.2)
    ax.text(1.3e-4, 62, "50 V/m: umbral de pérdida", fontsize=8)
    ax.set_xlabel(r"longitud de correlación del parche $\ell$ (mm)")
    ax.set_ylabel(r"$|E(0)|_{rms}$ en el centro (V/m)")
    ax.legend(loc="lower right")
    guardar(fig, "control_superficie")
