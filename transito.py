"""
transito.py — Prueba numérica de la hipótesis de tránsito de la V11.

LA HIPÓTESIS (V11)
------------------
El positrón "recorre" parches de tamaño l_c a su velocidad térmica v0, así que encontraría un
campo nuevo cada tau = l_c / v0. El campo que siente sería un ruido de memoria corta (OU) y
el centro guía haría una caminata aleatoria: <|Z|^2> = 4 D t, sin límite.

POR QUÉ FALLA PARA UN POSITRÓN ATRAPADO
---------------------------------------
El positrón no avanza por la superficie: oscila en z y vuelve al MISMO punto cada periodo
axial T_z. Si el parche no cambia, el campo que siente es periódico, E(t + T_z) = E(t), y se
puede escribir como serie de Fourier (derivación de Henry, sección 3.2):
    E(t) = G_0 + sum_{n != 0} G_n exp(i n w_z t)
- G_0 (el promedio sobre una oscilación) es un campo fijo: solo desplaza la órbita a Z* = -G_0/U0
  (centro_guia.py). El centro guía describe un círculo: <|Z|^2> = 2 <|G_0|^2> (1 - cos w_- t) / U0^2.
- Los G_n producen oscilaciones de tamaño |G_n| / (B0 n w_z), unas w_-/w_z ~ 5e-4 veces menores,
  que no se acumulan.
Resultado: NO hay difusión. La misma superficie, tratada como OU de tránsito, sí difunde.

UNIDADES ESCALADAS
------------------
Con los parámetros reales, w_z / w_- ~ 2000 y habría que integrar millones de pasos. Aquí se usa
w_- = 1, B0 = 1, U0 = 1 y w_z = 50 * w_-. El resultado no depende de esa razón mientras sea
mucho mayor que 1 (la teoría lo dice y Henry lo deriva en 3.2).

Uso:  python transito.py
"""
import numpy as np

W_M = 1.0          # frecuencia magnetrón (escalada)
B = 1.0            # campo magnético (escalado); U0 = W_M * B = 1
U0 = W_M * B
W_Z = 50.0         # frecuencia axial: 50 veces la magnetrón
A_Z = 1.0          # amplitud axial (unidad de longitud)
ELL = 0.1          # longitud de correlación del parche a lo largo de z: el positrón cruza ~20 parches
N_MODOS = 64       # modos de Fourier para generar cada patrón


def paso(Z, E_ini, E_fin, dt):
    """Integrador exponencial del centro guía (el mismo de centro_guia.py, en unidades escaladas)."""
    rot = np.exp(-1j * W_M * dt)
    return rot * Z - (1 - rot) / (W_M * B) * 0.5 * (E_ini + E_fin)


def patrones_axiales(n, rng):
    """n patrones fijos E(z), complejos, con <|E|^2> = 1 y correlación gaussiana de longitud ELL."""
    k = rng.normal(0.0, 1.0 / ELL, size=(n, N_MODOS))
    fase = rng.uniform(0, 2 * np.pi, size=(n, N_MODOS))
    xi = rng.normal(size=(n, N_MODOS)) + 1j * rng.normal(size=(n, N_MODOS))
    return lambda z: (xi * np.cos(k * z[:, None] + fase)).sum(axis=1) / np.sqrt(N_MODOS)


def parche_estatico_muestreado(n_tray=1000, n_periodos=40, pasos_axial=64, semilla=None):
    """Cada trayectoria: un patrón fijo distinto y una fase axial distinta. Devuelve (t, <|Z|^2>, teoría)."""
    rng = np.random.default_rng(semilla)
    E_de_z = patrones_axiales(n_tray, rng)
    fase0 = rng.uniform(0, 2 * np.pi, n_tray)
    T_z = 2 * np.pi / W_Z
    dt = T_z / pasos_axial
    # El campo es periódico: basta calcularlo en un periodo axial.
    E_periodo = np.array([E_de_z(A_Z * np.cos(W_Z * j * dt + fase0)) for j in range(pasos_axial)])
    G0 = E_periodo.mean(axis=0)                       # promedio axial de cada trayectoria
    n_pasos = int(n_periodos * 2 * np.pi / W_M / dt)
    Z = np.zeros(n_tray, dtype=complex)
    t, R2 = [0.0], [0.0]
    for n in range(1, n_pasos + 1):
        Z = paso(Z, E_periodo[(n - 1) % pasos_axial], E_periodo[n % pasos_axial], dt)
        if n % pasos_axial == 0:
            t.append(n * dt)
            R2.append(np.mean(np.abs(Z) ** 2))
    t = np.array(t)
    teoria = 2 * np.mean(np.abs(G0) ** 2) / U0**2 * (1 - np.cos(W_M * t))
    return t, np.array(R2), teoria


def ou_de_transito(n_tray=4000, n_periodos=40, n_lotes=8, semilla=None):
    """La hipótesis de la V11: el mismo campo (<|E|^2> = 1) como OU con tau = ELL / v_axial."""
    rng = np.random.default_rng(semilla)
    v = A_Z * W_Z / np.sqrt(2)                        # velocidad axial cuadrática media
    tau = ELL / v
    sigma = 1 / np.sqrt(2)                            # por componente
    dt = tau / 4                                      # el paso del OU es exacto
    E = sigma * (rng.normal(size=n_tray) + 1j * rng.normal(size=n_tray))
    Z = np.zeros(n_tray, dtype=complex)
    n_pasos = int(n_periodos * 2 * np.pi / W_M / dt)
    cada = max(n_pasos // 400, 1)
    a = np.exp(-dt / tau)
    t, R2 = [0.0], [0.0]
    for n in range(1, n_pasos + 1):
        E_n = a * E + sigma * np.sqrt(1 - a**2) * (rng.normal(size=n_tray) + 1j * rng.normal(size=n_tray))
        Z = paso(Z, E, E_n, dt)
        E = E_n
        if n % cada == 0:
            t.append(n * dt)
            R2.append(np.mean(np.abs(Z.reshape(n_lotes, -1)) ** 2, axis=1))
    t, R2 = np.array(t), np.array([np.zeros(n_lotes)] + R2[1:])
    mitad = t > t[-1] / 2
    pend = np.polyfit(t[mitad], R2[mitad], 1)[0]      # una pendiente por lote
    D = sigma**2 * tau / (B**2 * (1 + (W_M * tau) ** 2))
    return t, R2.mean(axis=1), 4 * D * t, pend.mean(), pend.std(ddof=1) / np.sqrt(n_lotes)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from figuras import COLOR, guardar, tamano, usar_estilo

    print("Unidades escaladas: w_- = 1, w_z = 50, B = U0 = 1, A_z = 1, l = 0,1, <|E|^2> = 1")
    t1, R1, th1 = parche_estatico_muestreado(semilla=2026)
    t2, R2, th2, pend, err = ou_de_transito(semilla=4052)   # semilla independiente
    P = 2 * np.pi / W_M

    print("1) Parche estático muestreado por la oscilación axial")
    print(f"   <|Z|^2> máximo = {R1.max():.4f};  teoría 4<|G0|^2>/U0^2 = {th1.max():.4f}"
          f"  (diferencia {abs(R1.max()/th1.max()-1)*100:.2f} %)")
    print(f"   <|Z|^2> al final de {t1[-1]/P:.0f} periodos = {R1[-1]:.4f}: acotado, no crece")
    print(f"   <|G0|^2> = {th1.max()/4:.4f}: el promedio axial deja {th1.max()/4*100:.1f} % de la potencia del campo")

    print("2) Hipótesis de tránsito: el mismo campo como OU con tau = l / v")
    print(f"   pendiente simulada / 4D teórico = {pend/(th2[-1]/t2[-1]):.3f} ± {err/(th2[-1]/t2[-1]):.3f}")
    print(f"   <|Z|^2> al final = {R2[-1]:.3f}: crece sin límite ({R2[-1]/R1.max():.1f} veces el máximo del caso estático)")

    usar_estilo()
    fig, ax = plt.subplots(figsize=tamano(2, 7.0))
    ax.plot(t2 / P, R2, color=COLOR["secundario"], ls="--", label="hipótesis de tránsito (OU)")
    ax.plot(t2 / P, th2, color=COLOR["teoria"], ls=":", lw=1.2, label=r"$4Dt$")
    ax.plot(t1 / P, R1, color=COLOR["serie3"], ls="-", label="parche estático muestreado")
    ax.plot(t1 / P, th1, color=COLOR["teoria"], ls=(0, (1, 3)), lw=1.0,
            label=r"$2\langle|G_0|^2\rangle(1-\cos\omega_- t)/U_0^2$")
    ax.set_xlabel(r"$t/P_-$")
    ax.set_ylabel(r"$\langle |Z|^2\rangle$ (unidades escaladas)")
    ax.legend(loc="upper left", fontsize=8)
    guardar(fig, "control_transito")
