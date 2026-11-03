"""Pruebas de disipacion.py (cifras de la subsección «Disipación asociada al ruido del circuito»)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_impedancia_del_nodo_y_nyquist():
    from disipacion import Z_nodo, ReZ_nyquist
    assert Z_nodo().real == pytest.approx(1562.6, rel=1e-3)
    assert Z_nodo().real == pytest.approx(ReZ_nyquist(), rel=1e-6)      # teorema de Nyquist-Twiss


def test_ritmo_y_tiempos():
    from disipacion import Z_nodo, lambda_disipacion, escenarios_con_disipacion
    from params import UN_ANIO
    assert lambda_disipacion(Z_nodo().real) == pytest.approx(3.25e-7, rel=0.01)
    t4 = escenarios_con_disipacion(4.0)["filtro de tres etapas + EMI"]
    t01 = escenarios_con_disipacion(0.1)["filtro de tres etapas + EMI"]
    assert t4[0] / UN_ANIO == pytest.approx(4.48, rel=0.01)              # sin disipación: cifra de la tesis
    assert t4[1] / UN_ANIO == pytest.approx(0.249, rel=0.01)             # con disipación
    assert t01[1] / t4[1] == pytest.approx(1.02, abs=0.01)               # casi no depende de T
    assert t01[1] >= t4[1]                                               # enfriar no empeora


def test_limite_sin_disipacion():
    from disipacion import tiempo_medio_con_disipacion
    from ruido import tiempo_medio
    D = 1e-13
    assert tiempo_medio_con_disipacion(D, 0.0) == pytest.approx(tiempo_medio(D))
    assert tiempo_medio_con_disipacion(D, 1e-15) == pytest.approx(tiempo_medio(D), rel=1e-3)


def test_formula_frente_a_monte_carlo():
    from disipacion import Ein, simular_primer_paso
    tp = simular_primer_paso(30.0, n_tray=4000, semilla=11)
    media, err = np.nanmean(tp), np.nanstd(tp) / np.sqrt(tp.size)
    assert media == pytest.approx(float(Ein(30.0)) / 2, abs=max(4 * err, 0.03))


def test_fdt_general_fuente_termica():
    """Ronda 5 (E3-01): U = |E_-(r_wall)|/(k_B T_b) para toda fuente térmica; el filtro lo cumple con T_ef."""
    from disipacion import U_termico, factor_fdt_termico, escenarios_con_disipacion, Ein
    assert float(U_termico(4.0)) == pytest.approx(725.3, rel=1e-3)
    assert float(factor_fdt_termico(4.0)) == pytest.approx(1 / 101.2, rel=5e-3)
    assert float(factor_fdt_termico(0.1)) == pytest.approx(3.741e-4, rel=5e-3)
    t_sin, t_con, U, T_ef = escenarios_con_disipacion(4.0)["filtro de tres etapas + EMI"]
    assert U == pytest.approx(float(U_termico(T_ef)), rel=1e-3)            # U del circuito = |E_-|/(k_B T_ef)
    assert t_con == pytest.approx(t_sin * float(Ein(U)) / U, rel=1e-6)     # t_con = t0 * Ein(U)/U
    # fuente térmica con t0 ∝ 1/T: t_con(0,1 K)/t_con(4 K) = 40·Ein(U)/U|0,1 / Ein(U)/U|4 ≈ 1,5 (logarítmico, ≥ 1)
    cociente = 40 * float(factor_fdt_termico(0.1)) / float(factor_fdt_termico(4.0))
    assert cociente == pytest.approx(1.515, rel=0.01) and cociente > 1


def test_mediana_con_disipacion():
    from disipacion import cociente_mediana_media, simular_primer_paso
    assert float(cociente_mediana_media(92.0)) == pytest.approx(0.959, abs=0.005)
    tp = simular_primer_paso(92.0, n_tray=4000, semilla=5)
    assert np.nanmedian(tp) / np.nanmean(tp) == pytest.approx(0.958, abs=0.02)
