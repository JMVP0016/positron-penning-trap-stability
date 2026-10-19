"""
ruido.py — Parte fluctuante de los potenciales de parche:
difusión del centro guía por ruido eléctrico y tiempo de pérdida.

FÍSICA
------
De los tres movimientos del positrón, el que decide la pérdida es el magnetrón: su centro
(el centro guía) es lo que se desplaza hacia la pared. Un campo eléctrico uniforme E en la
zona del positrón (la parte *dipolar* del ruido de los electrodos) arrastra el centro guía
con la deriva E x B. Como el centro guía además rota a omega_-, en su sistema giratorio el
ruido aparece modulado a omega_-: solo la potencia del ruido cercana a f_- ~ 6,4 kHz empuja
siempre en la misma dirección y se acumula. Lo más lento o lo más rápido se promedia.

ECUACIÓN (Z = X + iY,  Ec = Ex + iEy)
    dZ/dt = -i omega_- Z - i Ec(t) / B0

RESULTADOS QUE ESTE MÓDULO COMPRUEBA
    <|Z|^2> = 4 D t,        D = S_E(f_-) / (4 B0^2)
    ruido OU (varianza sigma^2 por componente, tiempo de correlación tau_s):
        D = sigma^2 tau_s / [B0^2 (1 + omega_-^2 tau_s^2)]
    primer paso por r_wall partiendo del centro (pérdida):
        media = r_wall^2 / (4 D),   mediana ~ 0,80 de la media,   supervivencia: serie de Bessel

Uso:  python ruido.py   (valida, imprime los valores de referencia y guarda una figura de control)
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import j1, jn_zeros

from centro_guia import paso_centro_guia      # una sola copia del integrador del centro guía
from params import B0, D_TRAP, F_M, K_B, R_WALL, T_M, UN_ANIO, W_M

# ===========================================================================
# 1. Ruido de Ornstein–Uhlenbeck (OU)
#    Campo con memoria: correlación <E(t) E(t+u)> = sigma^2 exp(-|u|/tau) por componente.
# ===========================================================================
def paso_ou(E, dt, tau, sigma, rng):
    """Un paso EXACTO del OU complejo (Ex y Ey independientes), válido para cualquier dt."""
    a = np.exp(-dt / tau)                                   # memoria que queda tras dt
    xi = rng.normal(size=E.shape) + 1j * rng.normal(size=E.shape)
    return a * E + sigma * np.sqrt(1 - a**2) * xi


def SE_ou(f, sigma, tau):
    """Densidad espectral unilateral de una componente del OU, en V^2 m^-2 Hz^-1."""
    return 4 * sigma**2 * tau / (1 + (2 * np.pi * f * tau) ** 2)


# ===========================================================================
# 2. Centro guía con rotación magnetrón (el integrador está en centro_guia.py)
# ===========================================================================
def D_ou(sigma, tau):
    """Coeficiente de difusión teórico para ruido OU con rotación magnetrón (m^2/s)."""
    return sigma**2 * tau / (B0**2 * (1 + (W_M * tau) ** 2))


def D_desde_SE(S_E):
    """D a partir de la densidad espectral del campo a f_- (cualquier tipo de ruido)."""
    return S_E / (4 * B0**2)


def SE_desde_D(D):
    return 4 * B0**2 * D


def simular_difusion(tau, sigma=1 / np.sqrt(2), n_tray=4000, n_lotes=8, n_periodos=150, semilla=None):
    """Simula n_tray centros guía con ruido OU y estima D con la pendiente de <|Z|^2>.

    sigma = 1/sqrt(2) V/m por componente equivale a E_rms = 1 V/m. La ecuación es lineal,
    así que D/D_teoría no depende de sigma.
    Las trayectorias se reparten en n_lotes independientes: D sale de cada lote y la
    dispersión entre lotes da la barra de error. semilla=None usa una semilla distinta en
    cada llamada; nunca reutilices la misma semilla para comparar dos casos.
    Devuelve (t, <|Z|^2>, D_simulado, error_de_D).
    """
    rng = np.random.default_rng(semilla)
    dt = min(T_M / 200, tau / 20)            # resuelve la rotación y la memoria del ruido
    n_pasos = int(n_periodos * T_M / dt)
    E = sigma * (rng.normal(size=n_tray) + 1j * rng.normal(size=n_tray))   # ruido ya estacionario
    Z = np.zeros(n_tray, dtype=complex)      # todos parten del centro de la trampa
    cada = max(n_pasos // 300, 1)
    t_reg, R2_lotes = [], []
    for n in range(1, n_pasos + 1):
        E_nuevo = paso_ou(E, dt, tau, sigma, rng)
        Z = paso_centro_guia(Z, E, E_nuevo, dt)
        E = E_nuevo
        if n % cada == 0:
            t_reg.append(n * dt)
            R2_lotes.append(np.mean(np.abs(Z.reshape(n_lotes, -1)) ** 2, axis=1))
    t_reg, R2_lotes = np.array(t_reg), np.array(R2_lotes)       # R2_lotes: (tiempos, lotes)
    mitad = t_reg > t_reg[-1] / 2            # régimen difusivo: se descarta el transitorio
    D_lote = np.polyfit(t_reg[mitad], R2_lotes[mitad], 1)[0] / 4
    return t_reg, R2_lotes.mean(axis=1), D_lote.mean(), D_lote.std(ddof=1) / np.sqrt(n_lotes)


# ===========================================================================
# 3. Primer paso: cuándo llega el centro guía a r_wall (pérdida)
#    Difusión 2D en un disco con borde absorbente, partiendo del centro:
#    S(t) = sum_n 2 / (j_n J1(j_n)) exp(-j_n^2 D t / R^2),  j_n = ceros de J0.
# ===========================================================================
_J = jn_zeros(0, 200)
_COEF = 2.0 / (_J * j1(_J))


def _S_adim(x):
    """Supervivencia en función del tiempo adimensional x = D t / R^2."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    S = (_COEF[None, :] * np.exp(-(_J[None, :] ** 2) * x[:, None])).sum(axis=1)
    # Para x < 1e-3 la serie truncada no converge, pero ahí S = 1 a precisión de máquina:
    # en tan poco tiempo el centro guía no puede haber llegado a la pared.
    return np.where(x < 1e-3, 1.0, S)


def supervivencia(t, D, R=R_WALL):
    """Probabilidad de que el positrón siga atrapado en el tiempo t."""
    return _S_adim(D * np.asarray(t) / R**2)


def tiempo_medio(D, R=R_WALL):
    """Tiempo medio de pérdida: R^2 / (4D)."""
    return R**2 / (4 * D)


C50 = brentq(lambda x: _S_adim(x)[0] - 0.5, 0.05, 1.0)   # S(x) = 1/2  ->  x ~ 0,20


def mediana(D, R=R_WALL):
    """Tiempo en que se ha perdido la mitad: C50 R^2 / D (unas 0,80 veces la media)."""
    return C50 * R**2 / D


# ===========================================================================
# 4. De dónde sale S_E: canal Johnson de un electrodo segmentado
#    S_V = 4 k_B T R / (1 + (2 pi f R C)^2)  (resistencia R del filtro y capacidad C)
#    S_E ~ kappa^2 S_V / d^2  (kappa: fracción del voltaje que llega como campo uniforme)
#    R, C y kappa son SUPUESTOS: hay que justificarlos con la literatura antes de usarlos.
#    El Johnson es el ruido MÍNIMO de un circuito a temperatura T. En trampas reales suele
#    dominar ruido técnico o electromagnético externo, que no baja al enfriar; por eso
#    t_medio proporcional a 1/T es la mejora MÁXIMA que puede dar el enfriamiento.
# ===========================================================================
def SE_johnson(T, R=1e6, C=1e-9, kappa=0.5, f=F_M, d=D_TRAP):
    S_V = 4 * K_B * T * R / (1 + (2 * np.pi * f * R * C) ** 2)
    return kappa**2 * S_V / d**2


# ===========================================================================
# Validación
# ===========================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.integrate import trapezoid

    from figuras import COLOR, LINEA, guardar, tamano, usar_estilo

    sigma = 1 / np.sqrt(2)
    print("1) D con rotación magnetrón: simulación frente a fórmula (semillas independientes)")
    xs, ratios, errores, rel, rel_err, ejemplo = (0.3, 1.0, 3.0, 10.0), [], [], [], [], None
    for i, x in enumerate(xs):
        tau = x / W_M
        t, R2, D_sim, D_err = simular_difusion(tau, sigma, semilla=1000 + i)
        D_th = D_ou(sigma, tau)
        D_blanco = sigma**2 * tau / B0**2
        ratios.append(D_sim / D_blanco)
        errores.append(D_err / D_blanco)
        rel.append(D_sim / D_th)
        rel_err.append(D_err / D_th)
        if x == 1.0:
            ejemplo = (t, R2, D_th)
        print(f"   omega_- tau = {x:5.1f}:  D_sim/D_teoría = {D_sim/D_th:.3f} ± {D_err/D_th:.3f}"
              f"   (supresión por rotación: D/D_blanco = {1/(1+x**2):.3f})")

    print("2) Primer paso (solución analítica)")
    x = np.linspace(0, 3, 30001)
    integral = trapezoid(_S_adim(x), x)      # media = integral de S; debe dar 1/4 en estas unidades
    print(f"   S(t -> 0) = {_S_adim(1e-4)[0]:.4f}  (debe ser 1)")
    print(f"   media por integración = {integral:.4f} R^2/D  (exacta: 0,2500)")
    print(f"   mediana = {C50:.4f} R^2/D ;  mediana/media = {4*C50:.3f}")

    print("3) Valores de referencia del plan")
    D_ref = D_desde_SE(1e-15)
    print(f"   S_E = 1e-15 V^2 m^-2 Hz^-1 -> media = {tiempo_medio(D_ref):.2e} s"
          f" = {tiempo_medio(D_ref)/UN_ANIO:.0f} años")
    print(f"   S_E que da una media de un año: {SE_desde_D(R_WALL**2/(4*UN_ANIO)):.2e} V^2 m^-2 Hz^-1")
    for etiqueta, D_v11, med_v11 in (("4 K", 1.816e-11, 78.20), ("0,1 K", 1.149e-10, 12.37)):
        print(f"   V11 {etiqueta}: S_E implícito = {SE_desde_D(D_v11):.2e};"
              f" media = {tiempo_medio(D_v11)/3600:.1f} h;"
              f" mediana analítica = {mediana(D_v11)/3600:.2f} h (Monte Carlo de la V11: {med_v11} h)")

    print("4) Canal Johnson con valores supuestos (R = 1 MOhm, C = 1 nF, kappa = 0,5)")
    for T in (4.0, 0.1):
        S = SE_johnson(T)
        print(f"   T = {T:>3} K: S_E = {S:.2e} -> media = {tiempo_medio(D_desde_SE(S))/UN_ANIO:.0f} años")

    # Figura de control
    # Panel central en escala log-log: la supresión llega a 0,01 y en escala lineal esos puntos
    # quedarían pegados al cero con barras invisibles. Debajo, el cociente simulación/teoría
    # con sus barras de error, que es donde se ve si la simulación coincide con la fórmula.
    usar_estilo()
    fig = plt.figure(figsize=tamano(2, 6.2))
    gs = fig.add_gridspec(2, 3, height_ratios=(2.2, 1))
    a0 = fig.add_subplot(gs[:, 0])
    a1 = fig.add_subplot(gs[0, 1])
    a1r = fig.add_subplot(gs[1, 1], sharex=a1)
    a2 = fig.add_subplot(gs[:, 2])
    t, R2, D_th = ejemplo
    a0.plot(t * 1e3, R2 * 1e6, color=COLOR["4K"], ls="-", label="simulación")
    a0.plot(t * 1e3, 4 * D_th * t * 1e6, color=COLOR["teoria"], **LINEA["teoria"], label=r"$4Dt$")
    a0.set_xlabel(r"$t$ (ms)"); a0.set_ylabel(r"$\langle |Z|^2\rangle$ (mm$^2$)")
    a0.legend(loc="upper left", fontsize=8)
    xx = np.logspace(-1.3, 1.3, 200)
    a1.loglog(xx, 1 / (1 + xx**2), color=COLOR["teoria"], **LINEA["teoria"], label="teoría")
    a1.plot(xs, ratios, color=COLOR["4K"], ls="none", marker="o", ms=4, label="simulación")
    a1.set_ylabel(r"$D/D_{\mathrm{blanco}}$")
    a1.legend(loc="lower left", fontsize=8)
    plt.setp(a1.get_xticklabels(), visible=False)
    a1r.axhline(1.0, color=COLOR["teoria"], ls=":", lw=1.2)
    a1r.errorbar(xs, rel, yerr=rel_err, color=COLOR["4K"], ls="none", marker="o", ms=4,
                 elinewidth=1, capsize=2.5, capthick=1)
    a1r.set_ylim(0.9, 1.1)
    a1r.set_xlabel(r"$\omega_-\tau_s$"); a1r.set_ylabel(r"$D_{\mathrm{sim}}/D_{\mathrm{teo}}$")
    tt = np.linspace(0, 3, 400)              # en unidades de la media
    a2.plot(tt, _S_adim(tt / 4), color=COLOR["4K"], ls="-")
    a2.axvline(4 * C50, color=COLOR["secundario"], ls="--", lw=1)
    a2.axvline(1.0, color=COLOR["teoria"], ls=":", lw=1.2)
    a2.text(4 * C50 - 0.05, 0.92, "mediana", ha="right", fontsize=8)
    a2.text(1.05, 0.92, "media", fontsize=8)
    a2.set_xlabel(r"$t/\bar t$"); a2.set_ylabel(r"$S(t)$")
    guardar(fig, "control_ruido")
