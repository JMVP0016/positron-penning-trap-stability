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
# 4. De dónde sale S_E(f_-): el ruido que llega a un electrodo segmentado
#
#    a) Del voltaje al campo. Si una mitad del electrodo anular fluctúa con V(t) y la otra no,
#       el modelo 2D del cilindro (superficie.py, coeficiente B_1 = 2V/pi) da en el centro
#           E(0) = kappa V / a,   kappa = 2/pi   ->   S_E = kappa^2 S_V / a^2
#       Un cilindro de longitud finita da menos campo: kappa = 2/pi es una COTA SUPERIOR del
#       acoplamiento, así que los tiempos que salen son COTAS INFERIORES.
#
#    b) El ruido de voltaje S_V en el electrodo a f_- tiene dos partes:
#       - Johnson de las resistencias del filtro RC por el que se polariza el electrodo. Cada
#         etapa está a su temperatura (ambiente, 60 K, la de la trampa). Se calcula con el
#         circuito completo: cada resistencia es una fuente 4 k_B T R en serie y el filtro la
#         atenúa hasta el electrodo.
#         Valores reales de un filtro de 3 etapas (von Boehn et al., Commun. Phys. 8, 107, 2025):
#         R = 5,0 / 5,2 / 7,1 kOhm a 300 / 60 / 5 K, C = 4,8 nF, corte ~1,2 kHz.
#       - Interferencia electromagnética (EMI) que se acopla a los electrodos. BASE midió en el
#         electrodo un ruido de 225 pV/sqrt(Hz) y lo atribuyó a EMI (Borchert et al., PRL 122,
#         043201, 2019). Se midió a ~18 MHz: a 6,4 kHz se usa SOLO como orden de magnitud.
#       Solo la última etapa del filtro está a la temperatura de la trampa. Esa es la única
#       parte del ruido que baja al enfriar el sistema de 4 K a 0,1 K.
# ===========================================================================
KAPPA_MEDIO_ANILLO = 2 / np.pi
FILTRO_REAL = {"R": (5.0e3, 5.2e3, 7.1e3), "C": (4.8e-9, 4.8e-9, 4.8e-9), "T_calientes": (300.0, 60.0)}
FILTRO_FUERTE = {"R": (5.0e3, 5.2e3, 7.1e3), "C": (4.8e-7, 4.8e-7, 4.8e-7), "T_calientes": (300.0, 60.0)}
#   FILTRO_FUERTE: el mismo con C x 100 (corte ~12 Hz). Es una variante SUPUESTA para representar
#   montajes con filtrado más agresivo; no es un filtro publicado.
SV_EMI_BASE = (225e-12) ** 2           # V^2/Hz (orden de magnitud)


def SV_filtro(T_trampa, filtro=FILTRO_REAL, f=F_M, por_etapa=False):
    """Ruido Johnson de un filtro RC en escalera visto en el electrodo (V^2/Hz).
    La primera etapa está en la entrada (puesta a tierra por la fuente) y la última junto al
    electrodo, a la temperatura de la trampa."""
    R, C = filtro["R"], filtro["C"]
    T = list(filtro["T_calientes"]) + [T_trampa]
    w, n = 2 * np.pi * f, len(R)
    Y = np.zeros((n, n), dtype=complex)            # matriz de admitancias de los nodos
    for k in range(n):
        g = 1 / R[k]
        Y[k, k] += g + 1j * w * C[k]
        if k > 0:
            Y[k - 1, k - 1] += g
            Y[k - 1, k] -= g
            Y[k, k - 1] -= g
    partes = []
    for i in range(n):                             # fuente de 1 V en serie con R_i
        I = np.zeros(n, dtype=complex)
        I[i] += 1 / R[i]
        if i > 0:
            I[i - 1] -= 1 / R[i]
        H = np.linalg.solve(Y, I)[-1]              # lo que llega al electrodo
        partes.append(4 * K_B * T[i] * R[i] * abs(H) ** 2)
    return partes if por_etapa else sum(partes)


def SE_desde_SV(S_V, kappa=KAPPA_MEDIO_ANILLO, a=R_WALL):
    return kappa**2 * S_V / a**2


def escenarios(T_trampa):
    """S_E(f_-) de los escenarios de la tesis para una temperatura de la trampa (o array)."""
    T = np.atleast_1d(np.asarray(T_trampa, dtype=float))
    real = np.array([SV_filtro(x, FILTRO_REAL) for x in T])
    fuerte = np.array([SV_filtro(x, FILTRO_FUERTE) for x in T])
    frio_real = np.array([SV_filtro(x, FILTRO_REAL, por_etapa=True)[-1] for x in T])
    frio_fuerte = np.array([SV_filtro(x, FILTRO_FUERTE, por_etapa=True)[-1] for x in T])
    return {
        "filtro real (corte 1,2 kHz)": SE_desde_SV(real + SV_EMI_BASE),
        "filtro fuerte (corte 12 Hz) + EMI": SE_desde_SV(fuerte + SV_EMI_BASE),
        "solo la etapa fría, filtro real": SE_desde_SV(frio_real),
        "solo la etapa fría, filtro fuerte": SE_desde_SV(frio_fuerte),
    }


def SE_johnson(T, R=1e6, C=1e-9, kappa=0.5, f=F_M, d=D_TRAP):
    """Ejemplo simple de una sola resistencia a temperatura T (usado en la primera versión)."""
    S_V = 4 * K_B * T * R / (1 + (2 * np.pi * f * R * C) ** 2)
    return kappa**2 * S_V / d**2


# ===========================================================================
# 5. Monte Carlo de la pérdida: positrones seguidos hasta la pared
#    Con el ruido real la pérdida tarda años y no se puede simular paso a paso. Como la
#    supervivencia solo depende de t/t_medio, se acelera el ruido (se elige sigma para que
#    t_medio sea de pocos cientos de periodos magnetrón) y se compara la DISTRIBUCIÓN de tiempos
#    con la solución analítica. Lo que se valida es la forma, que vale para cualquier S_E.
# ===========================================================================
def simular_perdida(n_tray=4000, t_medio_periodos=300, wtau=1.0, pasos_por_periodo=50,
                    t_max_medios=4.0, semilla=None):
    """Devuelve (tiempos de pérdida en s, t_medio teórico en s); np.inf si no se perdió."""
    rng = np.random.default_rng(semilla)
    tau = wtau / W_M
    t_med = t_medio_periodos * T_M
    D = R_WALL**2 / (4 * t_med)
    sigma = np.sqrt(D * B0**2 * (1 + wtau**2) / tau)     # despejado de D = sigma^2 tau / [B^2 (1 + w^2 tau^2)]
    dt = T_M / pasos_por_periodo
    E = sigma * (rng.normal(size=n_tray) + 1j * rng.normal(size=n_tray))
    Z = np.zeros(n_tray, dtype=complex)
    t_perdida = np.full(n_tray, np.inf)
    vivos = np.ones(n_tray, dtype=bool)
    for n in range(1, int(t_max_medios * t_med / dt) + 1):
        E_n = paso_ou(E, dt, tau, sigma, rng)
        Z = paso_centro_guia(Z, E, E_n, dt)
        E = E_n
        fuera = vivos & (np.abs(Z) >= R_WALL)
        t_perdida[fuera] = n * dt
        vivos &= ~fuera
        if not vivos.any():
            break
    return t_perdida, t_med


def densidad_perdida(t, D, R=R_WALL):
    """Densidad de probabilidad del tiempo de pérdida, p(t) = -dS/dt (1/s)."""
    x = np.atleast_1d(D * np.asarray(t, dtype=float) / R**2)
    return (D / R**2) * (_COEF[None, :] * _J[None, :] ** 2 * np.exp(-(_J[None, :] ** 2) * x[:, None])).sum(axis=1)


def mediana_bootstrap(tiempos, n_boot=2000, semilla=None):
    """Mediana y su error (desviación estándar bootstrap)."""
    rng = np.random.default_rng(semilla)
    meds = np.median(rng.choice(tiempos, size=(n_boot, tiempos.size), replace=True), axis=1)
    return np.median(tiempos), meds.std(ddof=1)


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

    print("4) Ruido real en el electrodo a f_- (V^2/Hz) y tiempo medio con kappa = 2/pi")
    for T in (4.0, 0.1):
        partes = SV_filtro(T, FILTRO_REAL, por_etapa=True)
        print(f"   T = {T:>3} K, filtro real: etapas 300 K / 60 K / trampa = "
              + " / ".join(f"{x:.1e}" for x in partes) + f";  EMI (BASE) = {SV_EMI_BASE:.1e}")
    for nombre, S in escenarios([4.0, 0.1]).items():
        t4, t01 = tiempo_medio(D_desde_SE(S)) / UN_ANIO
        print(f"   {nombre:36s}: 4 K -> {t4:9.3g} años, 0,1 K -> {t01:9.3g} años  (cociente {t01/t4:.2f})")

    print("5) Monte Carlo de la pérdida (ruido acelerado, 4000 positrones)")
    tp, t_med = simular_perdida(semilla=777)
    perdidos = np.isfinite(tp)
    med, err = mediana_bootstrap(tp, semilla=778)        # los no perdidos (inf) cuentan como "más tarde"
    print(f"   perdidos: {perdidos.mean()*100:.1f} %;  mediana / t_medio = {med/t_med:.3f} ± {err/t_med:.3f}"
          f"  (analítica: {4*C50:.3f})")

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
