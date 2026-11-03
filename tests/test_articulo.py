"""Fast checks of the round-3 article analyses (A1). Run: python3 -m pytest -q tests/test_articulo.py (~30 s)."""
import csv
import os
import subprocess
import sys

import numpy as np
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(RAIZ, "articulo")
sys.path.insert(0, RAIZ)


def correr(script, salida, *extra):
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONPATH=RAIZ)
    r = subprocess.run([sys.executable, "-B", os.path.join(A, script), "--salida", str(salida), *extra],
                       env=env, capture_output=True, text=True, cwd=str(salida))
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def cifras(ruta):
    with open(ruta) as fh:
        return {r["magnitud"]: float(r["valor"]) for r in csv.DictReader(fh)}


def test_narimannezhad_y_primer_paso(tmp_path):
    correr("i10_microtrampa_r3.py", tmp_path)
    c = cifras(tmp_path / "i10_cifras_r3.csv")
    assert c["t_Narimannezhad_ec25"] == pytest.approx(4042.5, rel=1e-4)
    assert c["t_primer_paso_2D_Narimannezhad"] == pytest.approx(4042.5, rel=1e-4)   # not 1011 s
    assert c["k2_ef_L100mm_W5eV"] == pytest.approx(1.80e6, rel=0.02)
    assert c["k2_umbral_subito_eps20"] == pytest.approx(8e5)


def test_rebote_plano_acotado(tmp_path):
    correr("i10_rebote_real.py", tmp_path)
    with open(tmp_path / "i10_rebote_real.csv") as fh:
        for r in csv.DictReader(fh):
            assert float(r["max_R2_sobre_teoria"]) == pytest.approx(1.0, abs=0.01)
            assert float(r["R2_en_kP_sobre_max"]) < 0.02


def test_cota_BASE_B0_y_variantes(tmp_path):
    correr("cota_BASE_independiente.py", tmp_path)
    c = cifras(tmp_path / "cota_BASE_cifras.csv")
    assert c["B0_AT"] == pytest.approx(1.1714, abs=2e-4)
    assert c["nu_menos_AT"] == pytest.approx(12766, rel=1e-3)
    assert c["Dnu_z_mas_con_B0 de nu_c"] == pytest.approx(62, abs=4)        # published 62(4) mHz
    assert c["G_AT"] == pytest.approx(3.554e10, rel=1e-3)
    assert c["S_E_max_conservadora (deriva, tau = 250 s, sigma = 0,05 Hz)"] == pytest.approx(7.72e-15, rel=0.01)
    assert c["S_E_max_intermedia (deriva, tau = 1400 s, sigma = 0,075 Hz)"] == pytest.approx(2.07e-15, rel=0.01)


def test_canal_axial_T2(tmp_path):
    correr("i3a_canal_axial.py", tmp_path)
    c = cifras(tmp_path / "i3a_cifras.csv")
    assert c["t_canal_axial_4K_kpi/z0_tau0.01"] == pytest.approx(1.23, rel=0.02)
    for k, v in c.items():
        if k.startswith("pendiente_logt_logT"):
            assert v == pytest.approx(-2.0, abs=0.01)
        if k.startswith("Dsim_sobre_Dteo"):
            assert v == pytest.approx(1.0, abs=0.06)


def test_kappa3d_validacion_cilindro_finito():
    import importlib.util
    sys_argv = sys.argv
    sys.argv = ["i4_kappa3d.py"]
    try:
        spec = importlib.util.spec_from_file_location("i4", os.path.join(A, "i4_kappa3d.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
    finally:
        sys.argv = sys_argv
    k, ka = m.kappa_cilindro_finito(m.A / 20, 0.6)
    assert k == pytest.approx(ka, rel=0.005)
    assert m.kappa_hiperbolica(m.A / 20) == pytest.approx(0.519, abs=0.002)


def test_i5_r3_propagacion(tmp_path):
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONPATH=RAIZ)
    r = subprocess.run([sys.executable, "-B", os.path.join(A, "i5_incertidumbre_r3.py"), RAIZ, str(tmp_path), "20000"],
                       env=env, capture_output=True, text=True, cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    with open(tmp_path / "i5_resumen_r3.csv") as fh:
        filas = {(f["canal"], f["EMI"], f["ultima_etapa"], f["con_T2"]): f for f in csv.DictReader(fh)}
    f = filas[("total A", "plana", "T", "True")]
    assert float(f["P_ge_1"]) == 1.0                                         # P[t(0,1 K)/t(4 K) >= 1] = 1
    assert float(f["frac_T2_domina_4K"]) == pytest.approx(0.138, abs=0.002)   # canal axial dominante: 13,8 %
    assert float(filas[("circuitos", "plana", "T", "False")]["cociente_p95"]) == pytest.approx(2.07, abs=0.01)


def test_i2_r3_cota_BASE(tmp_path):
    correr("i2_observable_r3.py", tmp_path)
    with open(tmp_path / "i2_trampas_reales.csv") as fh:
        filas = {f["trampa"]: f for f in csv.DictReader(fh)}
    assert float(filas["BASE AT"]["tasa_um2_h"]) == pytest.approx(183, rel=0.01)
    assert "x34" in filas["BASE AT"]["veredicto"]
    assert float(filas["BASE reservorio (RT)"]["tasa_um2_h"]) == pytest.approx(43.5, rel=0.01)
