"""
params.py — Parámetros físicos de la tesis: única fuente para todos los módulos.

Sistema: un positrón en una trampa de Penning hiperbólica ideal.
    B0 = 0,5 T   (campo magnético uniforme: confinamiento radial)
    V0 = 1 V     (potencial entre anillo y tapas: confinamiento axial)
    d  = 5 mm    (dimensión característica de la trampa)
    rho0 = r_wall = 5 mm (radio interior del anillo; el positrón se pierde al llegar ahí)

Todas las magnitudes en unidades SI. Los demás módulos importan desde aquí
(from params import ...) para que nadie use un valor distinto por error.
"""
import numpy as np

# ---------------------------------------------------------------------------
# Constantes fundamentales (CODATA 2018)
# ---------------------------------------------------------------------------
Q_E = 1.602176634e-19      # carga del positrón (C)
M_E = 9.1093837015e-31     # masa del positrón = masa del electrón (kg)
HBAR = 1.054571817e-34     # constante de Planck reducida (J s)
K_B = 1.380649e-23         # constante de Boltzmann (J/K)

# ---------------------------------------------------------------------------
# Geometría y campos de la trampa
# ---------------------------------------------------------------------------
B0 = 0.5                   # T
V0 = 1.0                   # V
D_TRAP = 5e-3              # d (m), definida por d^2 = (z0^2 + rho0^2/2) / 2
RHO0 = 5e-3                # radio interior del anillo (m)
R_WALL = RHO0              # radio de pérdida usado en la tesis (m)

# Semialtura hasta las tapas. En una trampa hiperbólica no es libre: sale de d y rho0.
# Con d = rho0 = 5 mm da 6,12 mm (la V11 fijaba z_wall = 10 mm, que no es coherente).
Z0 = np.sqrt(2 * D_TRAP**2 - RHO0**2 / 2)

# Curvatura del potencial cuadrupolar: phi(r, z) = U0 (z^2 - r^2/2)
U0 = V0 / (2 * D_TRAP**2)  # 2e4 V/m^2

# ---------------------------------------------------------------------------
# Frecuencias propias de los tres modos
# ---------------------------------------------------------------------------
W_C = Q_E * B0 / M_E                   # ciclotrón libre (rad/s)
W_Z = np.sqrt(2 * Q_E * U0 / M_E)      # axial: oscilación en el pozo parabólico
W_1 = np.sqrt(W_C**2 - 2 * W_Z**2)     # W_P - W_M
W_P = 0.5 * (W_C + W_1)                # ciclotrón modificado (omega_+)
W_M = 0.5 * (W_C - W_1)                # magnetrón (omega_-), casi igual a U0/B0
T_M = 2 * np.pi / W_M                  # periodo magnetrón (s), unos 157 µs
F_M = W_M / (2 * np.pi)                # frecuencia magnetrón (Hz), unos 6,4 kHz

# ---------------------------------------------------------------------------
# Regímenes de temperatura de la tesis y escalas de tiempo útiles
# ---------------------------------------------------------------------------
T_4K = 4.0                 # régimen clásico (K)
T_01K = 0.1                # extrapolación con contexto cuántico (K)
UN_ANIO = 365.25 * 24 * 3600   # s


if __name__ == "__main__":
    print("Parámetros de la trampa")
    print(f"  U0 = {U0:.3e} V/m^2   z0 = {Z0*1e3:.3f} mm")
    print(f"  omega_c = {W_C:.4e} rad/s   omega_z = {W_Z:.4e} rad/s")
    print(f"  omega_+ = {W_P:.4e} rad/s   omega_- = {W_M:.4e} rad/s (U0/B0 = {U0/B0:.4e})")
    print(f"  nu_z = {W_Z/2/np.pi/1e6:.2f} MHz   nu_- = {F_M/1e3:.2f} kHz   T_- = {T_M*1e6:.1f} µs")
    print(f"  hbar*omega_+/k_B = {HBAR*W_P/K_B:.3f} K   (temperatura de congelamiento del ciclotrón)")
