"""
Tabla de valores de referencia de la tesis como pruebas automáticas.

    python3 -m pytest -q          (desde la carpeta del proyecto; tarda unos segundos)

Cada prueba fija un número que aparece en el texto o en una figura. Si un cambio en el código
mueve alguno de estos valores, la prueba falla y se sabe exactamente qué cifra de la tesis
habría que revisar. Las tolerancias son las de la precisión con que se cita cada valor.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from params import B0, HBAR, K_B, R_WALL, U0, W_1, W_M, W_P, W_Z, Z0  # noqa: E402


# ---------------------------------------------------------------- params.py
def test_parametros_de_la_trampa():
    assert U0 == pytest.approx(2.0e4)
    assert Z0 * 1e3 == pytest.approx(6.124, abs=1e-3)
    assert W_Z == pytest.approx(8.3877e7, rel=1e-4)
    assert W_M == pytest.approx(U0 / B0, rel=1e-5)          # omega_- ~ U0/B0
    assert 2 * np.pi / W_M * 1e6 == pytest.approx(157.1, abs=0.1)
    assert HBAR * W_P / K_B == pytest.approx(0.672, abs=1e-3)


# ---------------------------------------------------------------- superficie.py
def test_apantallamiento():
    from superficie import A1_rms, A1_rms_aprox, E_centro_rms
    assert E_centro_rms(1e-3, 1e-7) == pytest.approx(1.13e-3, rel=0.01)
    assert A1_rms(1e-3, 1e-5) == pytest.approx(A1_rms_aprox(1e-3, 1e-5), rel=1e-3)   # l << a
    # escala con sqrt(l): 100 veces más largo -> 10 veces más campo
    assert E_centro_rms(1e-3, 1e-5) / E_centro_rms(1e-3, 1e-7) == pytest.approx(10, rel=1e-3)


# ---------------------------------------------------------------- centro_guia.py
def test_parche_estatico():
    from centro_guia import E_critico, P_M, max_Z_rampa_teorico, orbita_exacta, simular_rampa
    assert E_critico("golpe") == pytest.approx(50.0)
    assert E_critico("lento") == pytest.approx(100.0)
    Ec = 1.0 * np.exp(0.7j)
    t, Z = simular_rampa(Ec, 0.0, periodos_despues=2)
    assert np.abs(Z).max() * 1e6 == pytest.approx(100.0, rel=1e-4)        # 2 delta con 1 V/m
    assert np.max(np.abs(Z - orbita_exacta(Ec, t))) < 1e-12               # integrador exacto
    for x, v in ((0.5, 1.637), (1.0, 1.0), (1.5, 1.212)):
        assert float(max_Z_rampa_teorico(x)) == pytest.approx(v, abs=1e-3)
    t, Z = simular_rampa(Ec, 1.5 * P_M)
    assert np.abs(Z).max() / 50e-6 == pytest.approx(1.212, abs=2e-3)


# ---------------------------------------------------------------- ruido.py
def test_primer_paso_y_v11():
    from ruido import C50, D_desde_SE, SE_desde_D, mediana, tiempo_medio
    from params import UN_ANIO
    assert C50 == pytest.approx(0.2005, abs=1e-4)
    assert mediana(1.816e-11) / 3600 == pytest.approx(76.68, abs=0.01)
    assert mediana(1.149e-10) / 3600 == pytest.approx(12.12, abs=0.01)
    assert SE_desde_D(R_WALL**2 / (4 * UN_ANIO)) == pytest.approx(1.98e-13, rel=0.01)
    assert SE_desde_D(1.816e-11) / 1.98e-13 == pytest.approx(92, rel=0.01)
    assert tiempo_medio(D_desde_SE(1e-15)) / UN_ANIO == pytest.approx(198, rel=0.01)


def test_johnson_proporcional_a_T():
    from ruido import SE_johnson
    assert SE_johnson(4.0) / SE_johnson(0.1) == pytest.approx(40.0)
    assert SE_johnson(4.0) == pytest.approx(1.38e-15, rel=0.01)


def test_difusion_con_rotacion():
    """Simulación corta: D simulado frente a la fórmula con omega_- tau = 1."""
    from ruido import D_ou, simular_difusion
    s = 1 / np.sqrt(2)
    _, _, D, err = simular_difusion(1 / W_M, s, n_tray=2000, n_periodos=80, semilla=7)
    assert D / D_ou(s, 1 / W_M) == pytest.approx(1.0, abs=max(4 * err / D_ou(s, 1 / W_M), 0.06))


# ---------------------------------------------------------------- cuantico.py
def test_modos_cuanticos():
    from cuantico import amplitud, dn_menos_dt, ocupacion, radio_punto_cero, razon_tiempo_positron
    from ruido import D_desde_SE
    from params import M_E
    assert radio_punto_cero(W_1) * 1e9 == pytest.approx(36.3, abs=0.05)
    assert ocupacion(0.1, W_P) == pytest.approx(1.211e-3, rel=1e-3)
    assert ocupacion(0.1, W_Z) == pytest.approx(155.6, abs=0.1)
    assert amplitud(ocupacion(4.0, W_P), W_1) * 1e9 == pytest.approx(125.4, abs=0.1)
    assert amplitud(ocupacion(0.1, W_P), W_1) * 1e9 == pytest.approx(36.3, abs=0.1)
    assert amplitud(ocupacion(4.0, W_Z), W_Z) * 1e6 == pytest.approx(131.3, abs=0.1)
    assert 1 - razon_tiempo_positron(0.1, 4.0, R_WALL) == pytest.approx(3.36e-4, rel=0.01)
    S = 1.98e-13                       # puente: dos caminos dan el mismo ritmo
    assert dn_menos_dt(S) == pytest.approx(2 * M_E * W_1 * D_desde_SE(S) / HBAR, rel=1e-3)


# ---------------------------------------------------------------- transito.py
def test_parche_estatico_muestreado_no_difunde():
    from transito import parche_estatico_muestreado
    t, R2, th = parche_estatico_muestreado(n_tray=300, n_periodos=6, semilla=11)
    assert R2.max() == pytest.approx(th.max(), rel=0.01)    # coincide con 2<|G0|^2>(1 - cos)
    assert R2[-1] < 0.02 * R2.max()                          # tras periodos enteros vuelve al centro


# ---------------------------------------------------------------- ruido.py: ruido realista
def test_ruido_realista_en_el_electrodo():
    from ruido import KAPPA_MEDIO_ANILLO, SV_filtro, D_desde_SE, escenarios, tiempo_medio
    from params import UN_ANIO
    assert KAPPA_MEDIO_ANILLO == pytest.approx(2 / np.pi)
    partes = SV_filtro(4.0, por_etapa=True)                 # 300 K / 60 K / trampa
    assert partes[0] == pytest.approx(1.7e-18, rel=0.05)    # domina la etapa a temperatura ambiente
    assert SV_filtro(4.0) == pytest.approx(2.68e-18, rel=0.01)
    esc = escenarios([4.0, 0.1])
    anios = {k: tiempo_medio(D_desde_SE(v)) / UN_ANIO for k, v in esc.items()}
    assert anios["filtro real (corte 1,2 kHz)"][0] == pytest.approx(4.48, rel=0.01)
    assert anios["filtro real (corte 1,2 kHz)"][1] / anios["filtro real (corte 1,2 kHz)"][0] == pytest.approx(1.11, abs=0.01)
    assert anios["filtro fuerte (corte 12 Hz) + EMI"][0] == pytest.approx(241, rel=0.01)
    assert anios["solo la etapa fría, filtro real"][1] / anios["solo la etapa fría, filtro real"][0] == pytest.approx(40.0)
