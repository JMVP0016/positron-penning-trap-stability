"""Controles de WP1.9: física, estado compartido y archivos de salida."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import centro_guia as cg
import contraste_experimental as ce


def test_alpha_22_ms_y_peil_0229():
    assert ce.tiempo_a_pared(1, ce.K2_SOMERO)*1e3 == pytest.approx(22, abs=0.2)
    assert float(ce.ocupacion_ciclotron_peil(4.2)) == pytest.approx(0.229, abs=0.001)


@pytest.mark.parametrize("campo", [0.35, 0.7, 1.0, 2.0])
def test_primer_cruce_frente_a_solucion_exacta(campo):
    exacto = float(ce.tiempo_a_pared_analitico(campo))
    dt = 2*np.pi/ce.K2_SOMERO/4000
    numerico = ce.tiempo_a_pared(campo, ce.K2_SOMERO)
    assert exacto-1e-12 <= numerico <= exacto+dt+1e-12


def test_sin_perdida_bajo_umbral():
    assert np.isnan(ce.tiempo_a_pared(0.05, ce.K2_SOMERO))
    assert np.isnan(ce.tiempo_a_pared_analitico(0))
    assert np.isnan(ce.tiempo_a_pared_analitico(0.05))


def test_restaura_globales_con_trampa_previa_distinta(monkeypatch):
    monkeypatch.setattr(cg, "U0", 12345.0)
    monkeypatch.setattr(cg, "B0", 0.8)
    monkeypatch.setattr(cg, "W_GC", 15431.25)
    antes = cg.U0, cg.B0, cg.W_GC
    ce.tiempo_a_pared(1, ce.K2_SOMERO)
    assert (cg.U0, cg.B0, cg.W_GC) == antes
    ce.tiempo_a_pared(0.05, ce.K2_SOMERO)
    assert (cg.U0, cg.B0, cg.W_GC) == antes


def test_restaura_globales_incluso_tras_error(monkeypatch):
    antes = cg.U0, cg.B0, cg.W_GC
    def falla(*args):
        raise RuntimeError("fallo deliberado del integrador")
    monkeypatch.setattr(cg, "paso_centro_guia", falla)
    with pytest.raises(RuntimeError, match="fallo deliberado"):
        ce.tiempo_a_pared(1, ce.K2_SOMERO)
    assert (cg.U0, cg.B0, cg.W_GC) == antes


def test_umbral_y_escalado_con_campo_magnetico():
    umbral = ce.K2_SOMERO*ce.RW_A/2
    assert float(ce.tiempo_a_pared_analitico(umbral)) == pytest.approx(np.pi/ce.K2_SOMERO)
    assert float(ce.tiempo_a_pared_analitico(1, B=2)) == pytest.approx(
        2*float(ce.tiempo_a_pared_analitico(1, B=1)))


def test_importar_no_dibuja_ni_crea_archivos(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(Path(ce.__file__).parent), env.get("PYTHONPATH", "")])
    subprocess.run([sys.executable, "-c", "import contraste_experimental; "
                    "import sys; assert 'matplotlib.pyplot' not in sys.modules"],
                   cwd=tmp_path, env=env, check=True, capture_output=True)
    assert not list(tmp_path.iterdir())


def test_figuras_bilingues_y_csv_desde_otro_directorio(tmp_path, monkeypatch):
    monkeypatch.setattr(ce, "_DIR", tmp_path / "modulo")
    otro = tmp_path / "directorio_actual"
    otro.mkdir()
    monkeypatch.chdir(otro)
    monkeypatch.setenv("FIG_IDIOMA", "es")
    monkeypatch.setenv("FIG_CARPETA", "figures")
    rutas_es = ce.figura()
    contenido_es = rutas_es[0].read_bytes()
    csv_es = rutas_es[-1].read_bytes()
    monkeypatch.setenv("FIG_IDIOMA", "en")
    monkeypatch.setenv("FIG_CARPETA", "figures/en")
    rutas_en = ce.figura()
    assert all(p.is_absolute() and p.stat().st_size > 100 for p in (*rutas_es, *rutas_en))
    assert rutas_es[0].read_bytes() == contenido_es
    assert rutas_en[-1].read_bytes() == csv_es
    datos = np.genfromtxt(rutas_es[-1], delimiter=",", names=True)
    assert len(datos) >= 65
    assert np.all(np.abs(datos['t_integrador_ms']-datos['t_analitico_ms']) <= 0.158)
    assert not list(otro.iterdir())


@pytest.mark.parametrize("func,args", [
    (ce.tiempo_a_pared, (-1, 10)),
    (ce.tiempo_a_pared_analitico, (1, 0)),
    (ce.ocupacion_ciclotron_peil, (0,)),
])
def test_rechaza_parametros_no_fisicos(func, args):
    with pytest.raises(ValueError):
        func(*args)
