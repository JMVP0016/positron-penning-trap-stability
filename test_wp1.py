"""
test_wp1.py — Pruebas de las tareas del paquete WP1 (implementación de referencia de la ronda 4).

    python3 -m pytest -q test_wp1.py          (desde la carpeta del proyecto; unos 10 s)

1.2  Geometría del acoplamiento (D-T6): anillo en dos mitades con líneas independientes.
1.3  Variante de la última etapa del filtro a 5 K (D-T8).
1.5  Marcas de 4; 1; 0,3 y 0,1 K en las figuras que barren la temperatura.
1.6  temperaturas.py: tablas 4.2, 4.3, 4.4 y A.4 con 1 K y 0,3 K.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from params import B0, F_M, R_WALL, UN_ANIO, W_M  # noqa: E402


def _simular_D(campo, n_tray=3000, periodos=100, pasos=100, semilla=31):
    """Difusión del centro guía con un campo E = campo(V1, V2) construido con dos voltajes OU
    independientes de varianza 1 V^2 y tau = 1/w_- (uno por mitad del anillo). Devuelve D (m^2/s)."""
    from centro_guia import paso_centro_guia
    rng = np.random.default_rng(semilla)
    tau, dt = 1 / W_M, 2 * np.pi / W_M / pasos
    a = np.exp(-dt / tau)
    V = rng.normal(size=(2, n_tray))
    E = campo(*V)
    Z = np.zeros(n_tray, complex)
    t, R2 = [], []
    for k in range(1, periodos * pasos + 1):
        V = a * V + np.sqrt(1 - a * a) * rng.normal(size=V.shape)
        En = campo(*V)
        Z = paso_centro_guia(Z, E, En, dt)
        E = En
        if k % 50 == 0:
            t.append(k * dt)
            R2.append(np.mean(np.abs(Z) ** 2))
    t, R2 = np.array(t), np.array(R2)
    m = t > t[-1] / 2
    return np.polyfit(t[m], R2[m], 1)[0] / 4


# ---------------------------------------------------------------- 1.2
def test_alias_kappa_dipolo():
    from ruido import KAPPA_DIPOLO, KAPPA_MEDIO_ANILLO, SE_desde_SV
    assert KAPPA_DIPOLO == pytest.approx(2 / np.pi)
    assert KAPPA_MEDIO_ANILLO == KAPPA_DIPOLO
    assert SE_desde_SV(1.0) == pytest.approx(KAPPA_DIPOLO**2 / R_WALL**2)


def test_geometria_dos_mitades_independientes():
    """D-T6: con dos mitades de ruidos independientes (mismo S_V), E_x = kappa (V2 - V1)/a y
    D = kappa^2 S_V / (4 a^2 B0^2), que es lo que calcula el código: D_desde_SE(SE_desde_SV(S_V))."""
    from ruido import KAPPA_DIPOLO, D_desde_SE, SE_desde_SV, SE_ou
    k = KAPPA_DIPOLO / R_WALL
    D_sim = _simular_D(lambda V1, V2: k * (V2 - V1) + 0j)
    D_codigo = D_desde_SE(SE_desde_SV(SE_ou(F_M, 1.0, 1 / W_M)))
    assert D_sim / D_codigo == pytest.approx(1.0, abs=0.08)


def test_una_sola_mitad_duplica_los_tiempos():
    """Si solo una mitad fuera ruidosa, D sería la mitad (tiempos x2): la tesis es conservadora."""
    from ruido import KAPPA_DIPOLO, D_desde_SE, SE_desde_SV, SE_ou
    k = KAPPA_DIPOLO / R_WALL
    D_sim = _simular_D(lambda V1, V2: -k * V1 + 0j, semilla=32)
    D_codigo = D_desde_SE(SE_desde_SV(SE_ou(F_M, 1.0, 1 / W_M)))
    assert D_sim / (D_codigo / 2) == pytest.approx(1.0, abs=0.08)


# ---------------------------------------------------------------- 1.3
def test_ultima_etapa_a_5K():
    from ruido import D_desde_SE, escenarios, tiempo_medio
    clave = "filtro real (corte 1,2 kHz)"
    a5 = tiempo_medio(D_desde_SE(escenarios([4.0, 1.0, 0.3, 0.1], T_ultima_etapa=5.0)[clave])) / UN_ANIO
    assert a5 == pytest.approx([4.37] * 4, abs=0.005)          # 4,37 años a toda T
    aT = tiempo_medio(D_desde_SE(escenarios([4.0, 0.1])[clave])) / UN_ANIO
    assert aT == pytest.approx([4.48, 4.97], abs=0.005)         # por defecto no cambia nada


# ---------------------------------------------------------------- 1.5
def test_marcas_de_las_cuatro_temperaturas():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import figuras_tesis as ft
    fig, ax = plt.subplots()
    ax.set_xscale("log")
    marcadas = ft.marcar_temperaturas(ax)
    xs = sorted(round(float(l.get_xdata()[0]), 6) for l in ax.get_lines())
    textos = {t.get_text() for t in ax.texts}
    plt.close(fig)
    assert sorted(marcadas) == [0.1, 0.3, 1.0, 4.0]
    assert xs == [0.1, 0.3, 1.0, 4.0]
    assert {"4 K", "1 K", "0,3 K", "0,1 K"} <= textos


# ---------------------------------------------------------------- 1.6
@pytest.mark.parametrize("T, n_p, A_p, A_z, A_m, t_filtro, t_fria", [
    (1.0, 1.04, 63.8, 65.6, 2027, 4.85, 177),
    (0.3, 0.119, 40.4, 36.0, 1110, 4.94, 589)])
def test_temperaturas_intermedias(T, n_p, A_p, A_z, A_m, t_filtro, t_fria):
    import temperaturas
    f = temperaturas.fila(T)
    assert f["n_cic"] == pytest.approx(n_p, rel=5e-3)
    assert f["A_cic_nm"] == pytest.approx(A_p, abs=0.05)
    assert f["A_ax_um"] == pytest.approx(A_z, abs=0.05)
    assert f["A_mag_nm"] == pytest.approx(A_m, abs=0.5)
    assert f["t_filtro_anios"] == pytest.approx(t_filtro, abs=0.005)
    assert f["t_filtro_ult5K_anios"] == pytest.approx(4.37, abs=0.005)
    assert f["t_etapa_fria_anios"] == pytest.approx(t_fria, rel=5e-3)


def test_temperaturas_tex_deterministas(tmp_path):
    import temperaturas
    temperaturas.escribir(str(tmp_path / "d1"), str(tmp_path / "t1"))
    temperaturas.escribir(str(tmp_path / "d2"), str(tmp_path / "t2"))
    for n in ("tabla_4_2.tex", "tabla_4_3.tex", "tabla_4_4.tex", "tabla_A_4.tex"):
        assert (tmp_path / "t1" / n).read_bytes() == (tmp_path / "t2" / n).read_bytes()
    t43 = (tmp_path / "t1" / "tabla_4_3.tex").read_text()
    for cifra in ("4{,}48", "4{,}85", "4{,}94", "4{,}97", "4{,}37", "1{,}77\\times10^{3}"):
        assert cifra in t43

