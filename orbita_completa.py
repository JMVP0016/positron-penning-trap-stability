"""
orbita_completa.py — Órbita completa (fuerza de Lorentz) de una carga en una trampa de Penning ideal,
en unidades escaladas, para validar de forma independiente el modelo de centro guía de la tesis.

Auditoría E1, ronda 5 (31-oct-2026). Independiente del código V12: no importa nada de él.

UNIDADES ESCALADAS
    q = m = B0 = 1  ->  omega_c = 1 ;  a = r_wall = 1 ;  U0 = omega_z^2 / 2
    Jerarquía por defecto: omega_z = 0,07  ->  omega_+ = 0,99754, omega_z = 0,07, omega_- = 0,0024562
    (omega_+/omega_z = 14,3 ; omega_z/omega_- = 28,5). Con los parámetros reales: 1048 y 2097.

ECUACIONES (carga positiva, B = B0 z^, phi = U0 (z^2 - r^2/2), u = x + i y)
    u''  = (omega_z^2/2) u - i omega_c u' + (q/m) E_c(t)          (E_c = E_x + i E_y externo)
    z''  = - omega_z^2 z + (q/m) E_z(t)
    Reacción resistiva de un anillo partido en dos mitades (cada una con impedancia real Rz):
    E_x = - G v_x ,   G = 2 q kappa^2 Rz / a^2   (Shockley-Ramo + reciprocidad)

DOS INTEGRADORES INDEPENDIENTES
  1) Boris (con la parte dependiente de la velocidad tratada por punto medio implícito: transformada
     de Cayley, que para G = 0 es exactamente el Boris estándar).
  2) Propagador EXACTO de la parte lineal: el sistema (x, y, z, vx, vy, vz, [estados de ruido]) es
     lineal, dX = A X dt + Sigma dW; su discretización exacta para cualquier dt es
     X_{n+1} = Phi X_n + eta_n,  Phi = expm(A dt),  eta ~ N(0, Q),  Q por el método de Van Loan.
     No hay error de discretización: solo la vigilancia de la pared en instantes discretos.

DESCOMPOSICIÓN EN MODOS (exacta para la trampa ideal sin disipación)
    u_- = (u' + i omega_+ u) / (i omega_1) ,   u_+ = -(u' + i omega_- u) / (i omega_1) ,  u = u_+ + u_-
    con omega_1 = omega_+ - omega_- = sqrt(omega_c^2 - 2 omega_z^2).
    La amplitud magnetrón cumple EXACTAMENTE  u_-' = -i omega_- u_- - i (q / (m omega_1)) E_c(t):
    es la ecuación del centro guía de la tesis con omega_- en lugar de U0/B0 y q/(m omega_1) en lugar
    de 1/B0 (factor omega_c/omega_1 = 1 + omega_z^2/omega_c^2 + ...).
"""
import numpy as np
from scipy.linalg import expm, eigh


class Trampa:
    """Parámetros de la trampa en unidades escaladas (q = m = B0 = 1 por defecto)."""

    def __init__(self, w_z=0.07, q=1.0, m=1.0, B0=1.0, a=1.0):
        self.q, self.m, self.B0, self.a = q, m, B0, a
        self.w_c = q * B0 / m
        self.w_z = w_z
        self.U0 = m * w_z**2 / (2 * q)
        self.w_1 = np.sqrt(self.w_c**2 - 2 * w_z**2)
        self.w_p = 0.5 * (self.w_c + self.w_1)
        self.w_m = 0.5 * (self.w_c - self.w_1)
        self.P_m = 2 * np.pi / self.w_m
        self.P_p = 2 * np.pi / self.w_p
        self.w_gc = self.U0 / B0          # la frecuencia que usa el código V12 (centro_guia.W_GC)

    def resumen(self):
        return (f"w_c = {self.w_c:.6g}, w_+ = {self.w_p:.6g}, w_z = {self.w_z:.6g}, w_- = {self.w_m:.6g}"
                f" (U0/B0 = {self.w_gc:.6g}); w_+/w_z = {self.w_p/self.w_z:.2f}, w_z/w_- = {self.w_z/self.w_m:.2f};"
                f" w_c/w_1 = {self.w_c/self.w_1:.6f}; P_- = {self.P_m:.2f}")

    # ------------------------------------------------------------------ modos
    def modos(self, u, v):
        """(u_+, u_-) a partir de posición compleja u y velocidad compleja v."""
        u_m = (v + 1j * self.w_p * u) / (1j * self.w_1)
        u_p = -(v + 1j * self.w_m * u) / (1j * self.w_1)
        return u_p, u_m

    def energia_magnetron(self, rho):
        """E_- = -(1/2) m w_- w_1 rho^2 (exacta para una órbita magnetrón pura de radio rho)."""
        return -0.5 * self.m * self.w_m * self.w_1 * rho**2


# ===========================================================================
# 1. Integrador de Boris (con amortiguamiento lineal en v_x por punto medio implícito)
# ===========================================================================
def boris(tr, x0, v0, dt, n_pasos, E_ext=None, G=0.0, Gz=0.0, registrar=1, callback=None):
    """Integra n_pasos de Boris para un conjunto de partículas.

    x0, v0: arreglos (N, 3). E_ext(t, x) -> (N, 3) campo externo (o None). G: E_x = -G v_x.
    Gz: E_z = -Gz v_z (electrodo con campo de peso axial; solo para la comprobación del modo axial).
    La velocidad vive en semipasos (leapfrog); se devuelve v sincronizada (promedio de semipasos).
    Devuelve (t, X, V) registrados cada 'registrar' pasos.
    """
    q_m = tr.q / tr.m
    x = np.array(x0, dtype=float)
    N = x.shape[0]
    # v en t = -dt/2: medio paso hacia atrás con la fuerza en t = 0 (inicialización estándar)
    v = np.array(v0, dtype=float) - 0.5 * dt * _aceleracion(tr, x, np.array(v0, float), 0.0, E_ext, G, Gz)
    # Matriz de la parte lineal en v:  dv/dt = M v + q/m E(x)   (magnético + amortiguamiento)
    Mv = np.array([[-q_m * G, q_m * tr.B0, 0.0],
                   [-q_m * tr.B0, 0.0, 0.0],
                   [0.0, 0.0, -q_m * Gz]])
    I3 = np.eye(3)
    Ainv = np.linalg.inv(I3 - 0.5 * dt * Mv)
    Cay = Ainv @ (I3 + 0.5 * dt * Mv)      # transformada de Cayley (rotación exacta si G = Gz = 0)
    t_reg, X_reg, V_reg = [0.0], [x.copy()], [np.array(v0, float).copy()]
    for n in range(n_pasos):
        t = n * dt
        Et = _campo_total(tr, x, t, E_ext)
        v_new = v @ Cay.T + dt * q_m * (Et @ Ainv.T)
        x = x + dt * v_new
        if (n + 1) % registrar == 0:
            t_reg.append((n + 1) * dt)
            X_reg.append(x.copy())
            # velocidad sincronizada aproximada en t_{n+1}: v_{n+1/2} + (dt/2) a(t_{n+1})
            V_reg.append(v_new + 0.5 * dt * _aceleracion(tr, x, v_new, (n + 1) * dt, E_ext, G, Gz))
        if callback is not None:
            callback(n + 1, x, v_new)
        v = v_new
    return np.array(t_reg), np.array(X_reg), np.array(V_reg)


def _campo_total(tr, x, t, E_ext):
    E = np.empty_like(x)
    E[:, 0] = tr.U0 * x[:, 0]
    E[:, 1] = tr.U0 * x[:, 1]
    E[:, 2] = -2 * tr.U0 * x[:, 2]
    if E_ext is not None:
        E = E + E_ext(t, x)
    return E


def _aceleracion(tr, x, v, t, E_ext, G, Gz):
    q_m = tr.q / tr.m
    E = _campo_total(tr, x, t, E_ext)
    E[:, 0] -= G * v[:, 0]
    E[:, 2] -= Gz * v[:, 2]
    a = q_m * E
    a[:, 0] += q_m * v[:, 1] * tr.B0
    a[:, 1] += -q_m * v[:, 0] * tr.B0
    return a


# ===========================================================================
# 2. Propagador exacto de la parte lineal (con estados de ruido OU o ruido blanco)
# ===========================================================================
def matriz_lineal(tr, G=0.0, Gz=0.0, n_extra=0):
    """Matriz A (6 + n_extra) de la dinámica lineal de (x, y, z, vx, vy, vz, extra...)."""
    q_m = tr.q / tr.m
    n = 6 + n_extra
    A = np.zeros((n, n))
    A[0, 3] = A[1, 4] = A[2, 5] = 1.0
    A[3, 0] = q_m * tr.U0
    A[4, 1] = q_m * tr.U0
    A[5, 2] = -2 * q_m * tr.U0
    A[3, 4] = q_m * tr.B0
    A[4, 3] = -q_m * tr.B0
    A[3, 3] = -q_m * G
    A[5, 5] = -q_m * Gz
    return A


def discretizar(A, Qc, dt):
    """Van Loan: Phi = expm(A dt) y covarianza del ruido Q = int_0^dt e^{As} Qc e^{A^T s} ds.
    Devuelve (Phi, L) con L L^T = Q (raíz por autovalores, robusta si Q es singular)."""
    n = A.shape[0]
    M = np.zeros((2 * n, 2 * n))
    M[:n, :n] = -A
    M[:n, n:] = Qc
    M[n:, n:] = A.T
    F = expm(M * dt)
    Phi = F[n:, n:].T
    Q = Phi @ F[:n, n:]
    Q = 0.5 * (Q + Q.T)
    w, V = eigh(Q)
    L = V * np.sqrt(np.clip(w, 0.0, None))
    return Phi, L


def propagar(Phi, L, X0, n_pasos, rng, registrar=1, alfin=None):
    """X_{n+1} = Phi X_n + L xi_n para N partículas (X0: (N, d)). alfin(n, X) se llama cada paso
    (para vigilar la pared) y puede devolver True para parar. Devuelve (índices, lista de X)."""
    X = np.array(X0, dtype=float)
    reg_n, reg_X = [0], [X.copy()]
    PhiT = Phi.T
    LT = None if L is None else L.T
    d = X.shape[1]
    for n in range(1, n_pasos + 1):
        X = X @ PhiT
        if LT is not None:
            X += rng.standard_normal(size=(X.shape[0], d)) @ LT
        if n % registrar == 0:
            reg_n.append(n)
            reg_X.append(X.copy())
        if alfin is not None and alfin(n, X):
            break
    return np.array(reg_n), reg_X


def u_v(X):
    """Posición y velocidad complejas (plano) a partir del estado (N, d)."""
    return X[:, 0] + 1j * X[:, 1], X[:, 3] + 1j * X[:, 4]


def Ein(U):
    """Ein(U) = int_0^U (1 - e^-s)/s ds."""
    from scipy.special import exp1
    U = np.asarray(U, dtype=float)
    return np.euler_gamma + np.log(U) + exp1(U)
