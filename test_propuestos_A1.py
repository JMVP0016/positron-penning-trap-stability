"""
test_propuestos_A1.py — Pruebas PROPUESTAS por el agente A1 para que cada cifra citada en la
tesis V12 (y en las tablas de temperaturas intermedias) quede respaldada por una prueba.

Integrada en el repositorio (tarea WP1.7) junto a test_referencia.py.
    CODIGO_V12=/ruta/al/codigo python3 -m pytest -q test_propuestos_A1.py          (rápidas, ~5 s)
    CODIGO_V12=/ruta/al/codigo python3 -m pytest -q -m lento test_propuestos_A1.py  (Monte Carlo, ~1-2 min)

Cada prueba indica entre corchetes la cifra y el lugar de la tesis V12 que respalda
(página del PDF Tesis_V12_marcas_revision.pdf o archivo .tex).
"""
import os
import sys

import numpy as np
import pytest

sys.dont_write_bytecode = True
_CODIGO = os.environ.get("CODIGO_V12", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _CODIGO)

from params import (B0, F_M, HBAR, K_B, M_E, Q_E, R_WALL, U0, UN_ANIO, W_1, W_M, W_P,  # noqa: E402
                    W_Z, Z0)

lento = pytest.mark.lento


def anios(S):
    from ruido import D_desde_SE, tiempo_medio
    return tiempo_medio(D_desde_SE(np.asarray(S, dtype=float))) / UN_ANIO


# =========================================================================== params / escalas
def test_escalas_citadas_en_el_texto():
    # [ω−/ωz ≈ 4,8e-4 y |K| ≲ 24 nm para 1 V/m; main.tex V12-115/V12-116, p. ~83]
    assert W_M / W_Z == pytest.approx(4.77e-4, rel=2e-3)
    assert 1.0 / (B0 * W_Z) * 1e9 == pytest.approx(23.8, abs=0.1)
    # [ħω+/kB ≈ 0,67 K; ħωz/kB = 6,4e-4 K; ħω−/kB ≈ 3e-7 K; V12-120]
    assert HBAR * W_Z / K_B == pytest.approx(6.407e-4, rel=1e-3)
    assert HBAR * W_M / K_B == pytest.approx(3.055e-7, rel=1e-3)
    # [ν_z = 13,35 MHz; f− = 6,37 kHz]
    assert W_Z / (2 * np.pi) / 1e6 == pytest.approx(13.35, abs=0.01)
    assert F_M / 1e3 == pytest.approx(6.37, abs=0.01)
    # identidad ω+ ω− = ωz²/2 que usan las fórmulas de amplitud del cap. 3
    assert W_P * W_M == pytest.approx(W_Z**2 / 2, rel=1e-10)
    # longitud magnética = radio de punto cero (p. 151)
    assert np.sqrt(HBAR / (Q_E * B0)) == pytest.approx(np.sqrt(HBAR / (M_E * W_1)), rel=1e-6)


def test_tabla_41_debe_usar_z0_no_zwall_10mm():
    # [Tabla 4.1 y 5.1 dicen z_wall = 10 mm; params.Z0 = 6,124 mm (V12-110, p. 80)]
    assert Z0 * 1e3 == pytest.approx(6.124, abs=1e-3)
    assert Z0 * 1e3 != pytest.approx(10.0, abs=0.5)


# =========================================================================== parte estática
def test_umbral_por_diferencia_de_potencial():
    # [U0 r_wall²/2 = 0,25 V, ec. (2.36), p. 54]
    assert U0 * R_WALL**2 / 2 == pytest.approx(0.25)


def test_apantallamiento_con_l_de_1mm():
    # La Tabla 4.1 pone V = 1 mV, d_patch = 1 mm y E = 1 V/m (= V/d). El modelo de
    # superficie.py (cota superior 2D) da 0,11 V/m para ese patrón: la tabla es incoherente.
    from superficie import E_centro_rms
    assert E_centro_rms(1e-3, 1e-3) == pytest.approx(0.112, rel=0.01)
    assert E_centro_rms(1e-3, 1e-7) == pytest.approx(1.13e-3, rel=0.01)   # [p. 44]


# =========================================================================== ruido de circuitos
def _Y(filtro, f):
    R, C = filtro["R"], filtro["C"]
    w, n = 2 * np.pi * f, len(R)
    Y = np.zeros((n, n), complex)
    for k in range(n):
        g = 1 / R[k]
        Y[k, k] += g + 1j * w * C[k]
        if k > 0:
            Y[k - 1, k - 1] += g; Y[k - 1, k] -= g; Y[k, k - 1] -= g
    return Y


@pytest.mark.parametrize("nombre", ["FILTRO_REAL", "FILTRO_FUERTE"])
def test_nyquist_del_filtro(nombre):
    """Con todas las etapas a la misma T, el ruido en el electrodo debe ser 4 kT Re Z_out."""
    import ruido
    filtro = dict(getattr(ruido, nombre))
    for T in (300.0, 4.0, 0.1):
        filtro["T_calientes"] = (T, T)
        Zout = np.linalg.inv(_Y(filtro, F_M))[-1, -1]
        assert ruido.SV_filtro(T, filtro) == pytest.approx(4 * K_B * T * Zout.real, rel=1e-10)


def test_frecuencias_de_corte_citadas():
    # [«corte ~1,2 kHz» y «corte ~12 Hz», ruido.py y fig. 4.5]
    from scipy.optimize import brentq
    import ruido

    def H(f, filtro):
        I = np.zeros(3, complex); I[0] = 1 / filtro["R"][0]
        return abs(np.linalg.solve(_Y(filtro, f), I)[-1])
    assert brentq(lambda f: H(f, ruido.FILTRO_REAL) - 2**-0.5, 1, 1e6) == pytest.approx(1207, rel=0.01)
    assert brentq(lambda f: H(f, ruido.FILTRO_FUERTE) - 2**-0.5, 0.1, 1e4) == pytest.approx(12.1, rel=0.01)


@pytest.mark.parametrize("T, real, frio", [(4.0, 4.48, 44.2), (1.0, 4.85, 177), (0.3, 4.94, 589), (0.1, 4.97, 1770)])
def test_tiempos_de_circuitos_por_temperatura(T, real, frio):
    # [CONTEXTO_COMUN tabla y propuesta de 1 K y 0,3 K]
    from ruido import escenarios
    e = escenarios(T)
    assert anios(e["filtro real (corte 1,2 kHz)"])[0] == pytest.approx(real, rel=0.01)
    assert anios(e["solo la etapa fría, filtro real"])[0] == pytest.approx(frio, rel=0.01)
    assert anios(e["filtro fuerte (corte 12 Hz) + EMI"])[0] == pytest.approx(241, rel=0.01)


# =========================================================================== ruido de superficie
@pytest.mark.parametrize("T, lo, hi", [(4.0, 1.50e3, 2.01e6), (1.0, 6.0e3, 8.05e6), (0.3, 2.0e4, 2.68e7), (0.1, 6.0e4, 8.05e7)])
def test_superficie_escenario_B(T, lo, hi):
    from parches_fluctuantes import banda_superficie
    S_lo, S_hi = banda_superficie(np.array([T]), "B")
    assert anios(S_hi)[0] == pytest.approx(lo, rel=0.01)
    assert anios(S_lo)[0] == pytest.approx(hi, rel=0.01)


def test_superficie_escenario_A_y_cota_BASE():
    from parches_fluctuantes import SE_cota_BASE, banda_superficie
    for T in (4.0, 1.0, 0.3, 0.1):
        S_lo, S_hi = banda_superficie(np.array([T]), "A")
        assert anios(S_hi)[0] == pytest.approx(858, rel=0.01)
    # el máximo NO es igual a toda T: 1,29e6 a 4 K y 1,38e6 por debajo de 1 K
    assert anios(banda_superficie(np.array([4.0]), "A")[0])[0] == pytest.approx(1.29e6, rel=0.01)
    assert anios(SE_cota_BASE(1.43)) == pytest.approx(1850, rel=0.01)


def test_escenario_B_lineal_en_T_bajo_7K():
    from parches_fluctuantes import SE_superficie
    assert SE_superficie(0.1, "B") / SE_superficie(1.0, "B") == pytest.approx(0.1, rel=1e-12)
    # continuidad en 7 K
    assert SE_superficie(7.0 - 1e-9, "B") == pytest.approx(SE_superficie(7.0 + 1e-9, "B"), rel=1e-6)


def test_sigma_E_de_parches_discretos():
    # σ_E² = V_f² ℓ_p/(π a³) (parches_fluctuantes.py, cap. 2)
    from parches_fluctuantes import geometria_parches
    for N in (50, 200, 2000):
        _, _, s = geometria_parches(N)
        assert s == pytest.approx(np.sqrt((2 * np.pi * R_WALL / N) / (np.pi * R_WALL**3)), rel=1e-3)


# =========================================================================== primer paso
def test_primer_paso_normalizacion():
    from scipy.integrate import quad
    from ruido import densidad_perdida
    D = R_WALL**2 / 4.0                                  # t_medio = 1 s
    assert quad(lambda t: densidad_perdida(t, D)[0], 1e-6, 20, limit=200)[0] == pytest.approx(1.0, abs=1e-4)
    assert quad(lambda t: t * densidad_perdida(t, D)[0], 1e-6, 40, limit=200)[0] == pytest.approx(1.0, abs=1e-4)


# =========================================================================== cuántico
@pytest.mark.parametrize("T, n_p, A_p, A_z, A_m", [(4.0, 5.469, 125.4, 131.28, 4054.4),
                                                   (1.0, 1.04, 63.8, 65.64, 2027),
                                                   (0.3, 0.119, 40.4, 35.95, 1110),
                                                   (0.1, 1.211e-3, 36.3, 20.757, 641.1)])
def test_amplitudes_por_temperatura(T, n_p, A_p, A_z, A_m):
    from cuantico import amplitud, ocupacion
    assert ocupacion(T, W_P) == pytest.approx(n_p, rel=5e-3)
    assert amplitud(ocupacion(T, W_P), W_1) * 1e9 == pytest.approx(A_p, rel=5e-3)
    assert amplitud(ocupacion(T, W_Z), W_Z) * 1e6 == pytest.approx(A_z, rel=5e-4)
    assert amplitud(ocupacion(T, W_Z), W_1) * 1e9 == pytest.approx(A_m, rel=5e-4)


def test_energias_modales_citadas():
    from cuantico import ocupacion
    # [E+/kB = 0,337 K y E−/kB = −4,769e-5 K a 0,1 K, fig. 4.10; E−/kB = −1,908e-3 K a 4 K, fig. 4.5]
    n_p, n_z = ocupacion(0.1, W_P), ocupacion(0.1, W_Z)
    assert HBAR * W_P * (n_p + 0.5) / K_B == pytest.approx(0.337, abs=1e-3)
    assert -HBAR * W_M * (n_z + 0.5) / K_B == pytest.approx(-4.769e-5, rel=1e-3)
    assert -(W_M / W_Z) * 4.0 == pytest.approx(-1.908e-3, rel=1e-3)
    # [Tabla 4.3: 0,1252 µm es la amplitud CLÁSICA a 4 K; con punto cero es 0,1254 µm]
    from cuantico import amplitud_clasica
    assert amplitud_clasica(4.0, W_P, W_1) * 1e9 == pytest.approx(125.2, abs=0.05)


def test_canal_J0_citado():
    # [1,7e-4 a 4 K y 4,3e-6 a 0,1 K, p. 148]
    from cuantico import factor_promedio_axial
    assert 1 - factor_promedio_axial(4.0, R_WALL) == pytest.approx(1.72e-4, rel=0.01)
    assert 1 - factor_promedio_axial(0.1, R_WALL) == pytest.approx(4.31e-6, rel=0.01)


# =========================================================================== reproducibilidad
def test_misma_semilla_mismo_resultado_y_semillas_distintas_difieren():
    from ruido import simular_difusion
    a = simular_difusion(1 / W_M, n_tray=200, n_periodos=10, semilla=1)[2]
    b = simular_difusion(1 / W_M, n_tray=200, n_periodos=10, semilla=1)[2]
    c = simular_difusion(1 / W_M, n_tray=200, n_periodos=10, semilla=2)[2]
    assert a == b and a != c


def test_csv_analiticos_coinciden_con_el_codigo():
    """Los CSV de datos/ que no dependen de números aleatorios deben poder regenerarse."""
    ruta = os.path.join(_CODIGO, "datos", "fig4_06_modos.csv")
    if not os.path.exists(ruta):
        pytest.skip("sin datos/")
    from cuantico import MODOS, amplitud, ocupacion
    d = np.loadtxt(ruta, delimiter=",", skiprows=1)
    T = d[:, 0]
    cols = []
    for _, (w_e, w_ef) in MODOS.items():
        n = ocupacion(T, w_e)
        cols += [n, amplitud(n, w_ef)]
    assert np.allclose(np.column_stack(cols), d[:, 1:], rtol=1e-6)


# =========================================================================== Monte Carlo (lentas)
@lento
def test_convencion_S_E_ruido_polarizado():
    """Convención: D = S_E/(4B²) supone que las DOS componentes tienen S_E. Con ruido polarizado
    (solo E_x) D = S_Ex/(8B²). Con la geometría adoptada (D-T6: dos mitades con líneas
    independientes) S_Ex = 2 kappa² S_V / a², de modo que D = kappa² S_V / (4 a² B²), que es lo que
    calcula el código (ver test_wp1.py::test_geometria_dos_mitades_independientes)."""
    from ruido import SE_ou, paso_centro_guia
    rng = np.random.default_rng(31)
    tau, s, n, dt = 1 / W_M, 1.0, 3000, 2 * np.pi / W_M / 100
    E = s * rng.normal(size=n)
    Z = np.zeros(n, complex)
    a = np.exp(-dt / tau)
    t, R2 = [], []
    for k in range(1, 100 * 100 + 1):
        En = a * E + s * np.sqrt(1 - a * a) * rng.normal(size=n)
        Z = paso_centro_guia(Z, E, En, dt); E = En
        if k % 50 == 0:
            t.append(k * dt); R2.append(np.mean(abs(Z) ** 2))
    t, R2 = np.array(t), np.array(R2)
    m = t > t[-1] / 2
    D = np.polyfit(t[m], R2[m], 1)[0] / 4
    assert D / (SE_ou(F_M, s, tau) / (8 * B0**2)) == pytest.approx(1.0, abs=0.08)


@lento
def test_mediana_montecarlo_de_la_perdida():
    from ruido import C50, mediana_bootstrap, simular_perdida
    tp, t_med = simular_perdida(n_tray=2000, semilla=4242)
    med, err = mediana_bootstrap(tp, semilla=4243)
    # Sesgo conocido (+2 %): memoria del ruido acelerado de la simulación; extrapolada a ruido sin memoria, 0,808 ± 0,014.
    assert med / t_med == pytest.approx(4 * C50, abs=max(3 * err / t_med, 0.03))


@lento
@pytest.mark.parametrize("razon", [10, 200])
def test_transito_no_depende_de_wz_sobre_wm(razon):
    import transito as tr
    viejo = tr.W_Z
    try:
        tr.W_Z = float(razon)
        t, R2, th = tr.parche_estatico_muestreado(n_tray=300, n_periodos=6, semilla=11)
        assert R2.max() == pytest.approx(th.max(), rel=0.03)
        assert R2[-1] < 0.02 * R2.max()
    finally:
        tr.W_Z = viejo
