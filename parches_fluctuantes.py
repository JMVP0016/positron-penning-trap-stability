"""
parches_fluctuantes.py — Potenciales de parche que cambian en el tiempo: la fuente de pérdida
del modelo (trampa ideal + parches), y cómo entra en ella la temperatura.

IDEA
----
La trampa ideal (modelo de Crimin et al.) confina para siempre. La tesis toma los potenciales
de parche como la única fuente de pérdida. Un parche fijo solo desplaza la órbita
(centro_guia.py); para que haya pérdida, el potencial de los parches tiene que FLUCTUAR en el
tiempo. Este módulo junta las dos piezas que ya existen:
  - apantallamiento (Henry, superficie.py): qué parte del potencial de la pared llega al centro;
  - difusión con rotación magnetrón (ruido.py): qué hace ese campo fluctuante al centro guía.

1) DEL PARCHE QUE FLUCTÚA AL RUIDO EN EL CENTRO (modelo 2D, parches de ancho l_p en la pared)
   Cada parche i tiene un potencial V_i(t), independiente de los demás, con <V_i^2> = V_f^2
   (la parte que fluctúa) y tiempo de correlación tau_s. En el centro solo llega el modo m = 1:
       E_x = -A_1/a,  A_1 = (1/pi) sum_i V_i [sen(th_{i+1}) - sen(th_i)]
   de donde, por componente:  sigma_E^2 = V_f^2 l_p / (pi a^3)
   y el centro guía difunde con D = sigma_E^2 tau_s / [B0^2 (1 + w_-^2 tau_s^2)]   (ruido.py).
   En términos espectrales:  S_E(f) = [l_p / (pi a^3)] S_V(f).

2) CUÁNTO FLUCTÚAN LOS PARCHES EN UNA SUPERFICIE REAL
   Las trampas de iones miden este ruido de superficie directamente. Labaziewicz et al.
   (PRL 101, 180602, 2008): ion a 75 µm de electrodos de oro, ~1 MHz, entre 7 y 100 K:
       S_E(T) = S_0 [1 + (T/T_0)^beta],  S_0 = 1,8e-14 a 3,3e-12 V^2 m^-2 Hz^-1,
       T_0 = 17-73 K, beta = 1,8-4,1   (plano por debajo de ~40 K)
   Para llevarlo a esta trampa se escala con la distancia (d^-4, el exponente medido va de
   3,5 a 3,8) y con la frecuencia (S_E ~ f^-alpha, alpha medido de ~0,6 a 1,4; Brownnutt et al.,
   RMP 87, 1419, 2015). Son extrapolaciones de casi dos décadas en distancia y en frecuencia:
   el resultado es un ORDEN DE MAGNITUD.
   Cota independiente: BASE (Borchert et al., PRL 2019), S_E < 7,5e-20 a 17,8 MHz y 1,8 mm.

3) DÓNDE ENTRA LA TEMPERATURA
   Por debajo de 7 K no hay medidas. Labaziewicz et al. dicen que el valor plano S_0 indica
   "o fluctuaciones a temperatura cero, o un proceso térmicamente activado con energía de
   activación menor que 7 K". Eso da dos escenarios para 0,1-4 K:
     A) los parches fluctúan igual a cualquier temperatura (S_0 constante): t no cambia;
     B) las fluctuaciones son térmicamente activadas y se congelan al enfriar. Con una
        distribución plana de energías de activación (modelo de Dutta y Horn, RMP 53, 497,
        1981) el ruido baja linealmente con T: S_0 (T / 7 K), y t crece como 1/T.
   La tesis no puede decidir entre A y B: acota el efecto de la temperatura entre ambos.

Uso:  python parches_fluctuantes.py
"""
import numpy as np

from centro_guia import paso_centro_guia
from params import B0, R_WALL, T_M, UN_ANIO, W_M
from ruido import D_ou, mediana_bootstrap, tiempo_medio, D_desde_SE

# ---------------------------------------------------------------------------
# Datos de la literatura
# ---------------------------------------------------------------------------
LAB_D = 75e-6                  # distancia ion-superficie (m)
LAB_F = 1e6                    # frecuencia de referencia (Hz)
LAB_S0 = np.array([65, 42, 167, 120, 54, 60, 18, 3300]) * 1e-15   # V^2 m^-2 Hz^-1, trampas I-IV
LAB_T0 = np.array([73, 46, 46, 45, 44, 49, 17, 73.0])            # K
LAB_BETA = np.array([3.0, 4.1, 3.6, 3.5, 3.2, 2.1, 1.8, 3.2])
LAB_TMIN = 7.0                 # temperatura más baja medida (K)
BASE_SE, BASE_F, BASE_D = 7.5e-20, 17.845e6, 1.8e-3   # cota de BASE (radio interior 1,8 mm)


# ===========================================================================
# 1. Monte Carlo: parches reales que fluctúan, uno por uno
# ===========================================================================
def geometria_parches(n_parches, a=R_WALL):
    """Proyección de cada parche sobre el modo m = 1 (para E_x y E_y) y sigma_E por V_f = 1 V."""
    th = np.linspace(0, 2 * np.pi, n_parches + 1)
    px = (np.sin(th[1:]) - np.sin(th[:-1])) / (np.pi * a)     # E_x = -sum V_i px_i
    py = (np.cos(th[:-1]) - np.cos(th[1:])) / (np.pi * a)     # E_y = -sum V_i py_i
    sigma_por_V = np.sqrt(np.sum(px**2))                      # igual para E_y
    return px, py, sigma_por_V


def simular_parches_fluctuantes(n_parches=200, V_f=1e-3, wtau=1.0, n_tray=1000, n_periodos=150,
                                pasos_por_periodo=50, n_lotes=8, semilla=None, a=R_WALL):
    """Cada trayectoria es un electrodo con n_parches independientes; cada V_i(t) es un OU.
    Devuelve (t, <|Z|^2>, D_simulado, error de D, D_teórico, sigma_E)."""
    rng = np.random.default_rng(semilla)
    px, py, s1 = geometria_parches(n_parches, a)
    sigma_E = V_f * s1
    tau = wtau / W_M
    dt = T_M / pasos_por_periodo
    q = np.exp(-dt / tau)
    V = V_f * rng.normal(size=(n_tray, n_parches))            # parches ya estacionarios
    campo = lambda V: -(V @ px) - 1j * (V @ py)
    E = campo(V)
    Z = np.zeros(n_tray, dtype=complex)
    n_pasos = n_periodos * pasos_por_periodo
    t, R2 = [0.0], [0.0]
    for n in range(1, n_pasos + 1):
        V = q * V + V_f * np.sqrt(1 - q**2) * rng.normal(size=V.shape)
        E_n = campo(V)
        Z = paso_centro_guia(Z, E, E_n, dt)
        E = E_n
        if n % pasos_por_periodo == 0:
            t.append(n * dt)
            R2.append(np.mean(np.abs(Z.reshape(n_lotes, -1)) ** 2, axis=1))
    t = np.array(t)
    R2 = np.array([np.zeros(n_lotes)] + R2[1:])
    mitad = t > t[-1] / 2
    D_lote = np.polyfit(t[mitad], R2[mitad], 1)[0] / 4
    return (t, R2.mean(axis=1), D_lote.mean(), D_lote.std(ddof=1) / np.sqrt(n_lotes),
            D_ou(sigma_E, tau), sigma_E)


# ===========================================================================
# 2. Ruido de superficie llevado a esta trampa (orden de magnitud)
# ===========================================================================
def SE_superficie(T, escenario="A", S0=42e-15, T0=46.0, beta=4.1, alpha=1.0, exp_d=4.0,
                  d=R_WALL, f=W_M / (2 * np.pi)):
    """S_E(f_-) del ruido de superficie a la distancia d, escalado desde Labaziewicz et al.
    escenario A: S_0 constante por debajo de 7 K;  escenario B: S_0 congelándose como T / 7 K."""
    T = np.asarray(T, dtype=float)
    escala = (LAB_D / d) ** exp_d * (LAB_F / f) ** alpha
    forma = 1 + (T / T0) ** beta
    if escenario == "B":
        forma = np.where(T < LAB_TMIN, (T / LAB_TMIN) * (1 + (LAB_TMIN / T0) ** beta), forma)
    return S0 * forma * escala


def SE_cota_BASE(alpha=1.0, d=R_WALL, f=W_M / (2 * np.pi)):
    """La cota de BASE llevada a esta trampa con las mismas leyes de escala."""
    return BASE_SE * (BASE_D / d) ** 4 * (BASE_F / f) ** alpha


def banda_superficie(T, escenario):
    """Mínimo y máximo de S_E sobre las 8 trampas de Labaziewicz y alpha entre 1,0 y 1,43."""
    todos = [SE_superficie(T, escenario, S0, T0, b, al)
             for S0, T0, b in zip(LAB_S0, LAB_T0, LAB_BETA) for al in (1.0, 1.43)]
    return np.min(todos, axis=0), np.max(todos, axis=0)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from figuras import COLOR, LINEA, guardar, tamano, usar_estilo

    print("1) Monte Carlo de parches que fluctúan (200 parches por electrodo, V_f = 1 mV)")
    for i, x in enumerate((0.3, 1.0, 3.0)):
        t, R2, D, err, D_th, sE = simular_parches_fluctuantes(wtau=x, semilla=900 + i,
                                                              n_periodos=150 if x < 3 else 400)
        print(f"   w_- tau_s = {x:3.1f}: D_sim/D_teoría = {D/D_th:.3f} ± {err/D_th:.3f}"
              f"   (sigma_E = {sE*1e3:.2f} mV/m)")
    s_teo = 1e-3 * np.sqrt((2 * np.pi * R_WALL / 200) / (np.pi * R_WALL**3))
    print(f"   sigma_E por la fórmula V_f sqrt(l_p / (pi a^3)) = {s_teo*1e3:.2f} mV/m")

    print("2) Tiempo medio por ruido de superficie en esta trampa (orden de magnitud)")
    for esc in ("A", "B"):
        lo, hi = banda_superficie(np.array([4.0, 0.1]), esc)
        t_hi = tiempo_medio(D_desde_SE(lo)) / UN_ANIO
        t_lo = tiempo_medio(D_desde_SE(hi)) / UN_ANIO
        print(f"   escenario {esc}: 4 K -> {t_lo[0]:.1e} a {t_hi[0]:.1e} años;"
              f"  0,1 K -> {t_lo[1]:.1e} a {t_hi[1]:.1e} años")
    for al in (1.0, 1.43):
        S = SE_cota_BASE(al)
        print(f"   cota de BASE llevada a 5 mm y 6,4 kHz (alpha = {al}): S_E < {S:.1e}"
              f" -> t_medio > {tiempo_medio(D_desde_SE(S))/UN_ANIO:.1e} años")

    print("3) ¿Podrían los parches de la V11 (1 V/m) fluctuar a la escala del magnetrón?")
    sig = 1 / np.sqrt(2)
    tau = 1 / W_M
    S_alta = 4 * sig**2 * tau / (1 + (2 * np.pi * BASE_F * tau) ** 2)
    print(f"   OU con E_rms = 1 V/m y tau_s = 1/w_-: a 17,8 MHz daría S_E = {S_alta:.1e},"
          f" frente a la cota de BASE llevada a 5 mm: {BASE_SE*(BASE_D/R_WALL)**4:.1e}")
