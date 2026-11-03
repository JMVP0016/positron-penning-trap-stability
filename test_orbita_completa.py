"""
test_orbita_completa.py — Propuesta de prueba para el repositorio de la tesis (auditoría E1, ronda 5).

Valida el modelo de centro guía de la V12 contra la ÓRBITA COMPLETA (fuerza de Lorentz) en unidades
escaladas (omega_c/omega_z = 14,3; omega_z/omega_- = 28,5) con el propagador exacto de la parte lineal.
Copiar junto con orbita_completa.py; corre en unos segundos:  python3 -m pytest -q test_orbita_completa.py
"""
import numpy as np
import pytest

from orbita_completa import Ein, Trampa, discretizar, matriz_lineal, propagar, u_v

TR = Trampa(w_z=0.07)
KAPPA = 2 / np.pi


def test_umbral_subito_orbita_completa():
    """Campo uniforme de golpe: max|u| = delta (1 + w_c/w_1) (centro guía: 2 delta)."""
    E0 = 1e-4
    A = matriz_lineal(TR, n_extra=1)
    A[3, 6] = TR.q / TR.m * E0
    Phi, _ = discretizar(A, np.zeros_like(A), 0.05)
    X0 = np.zeros((1, 7)); X0[0, 6] = 1.0
    _, Xs = propagar(Phi, None, X0, int(1.2 * TR.P_m / 0.05), None)
    Xs = np.array(Xs)[:, 0, :]
    r = np.hypot(Xs[:, 0], Xs[:, 1])
    assert r.max() / (E0 / TR.U0) == pytest.approx(1 + TR.w_c / TR.w_1, rel=1e-5)


def test_ritmos_con_disipacion():
    """E_x = -G v_x: magnetrón crece a qG w_-/(2 m w_1), ciclotrón decae a qG w_+/(2 m w_1), axial no cambia."""
    G = 0.01
    ev = np.linalg.eigvals(matriz_lineal(TR, G=G))
    re = lambda w: ev[np.argmin(np.abs(np.abs(ev.imag) - w))].real
    assert re(TR.w_m) == pytest.approx(TR.q * G * TR.w_m / (2 * TR.m * TR.w_1), rel=1e-3)
    assert re(TR.w_p) == pytest.approx(-TR.q * G * TR.w_p / (2 * TR.m * TR.w_1), rel=1e-3)
    assert abs(re(TR.w_z)) < 1e-12
    # la fórmula de la revisión (lambda = q kappa^2 Re Z w_-/(a^2 B0), G = 2 q kappa^2 Re Z / a^2)
    # coincide con la exacta salvo w_c/w_1 (1 + 9e-7 con los parámetros reales)
    ReZ = G * TR.a**2 / (2 * TR.q * KAPPA**2)
    lam_V12 = TR.q * KAPPA**2 * ReZ * TR.w_m / (TR.a**2 * TR.B0)
    assert re(TR.w_m) / lam_V12 == pytest.approx(TR.w_c / TR.w_1, rel=1e-6)


def test_difusion_ou_orbita_completa():
    """D de la órbita completa = D_V12 (w_c/w_1)^2 con ruido OU isótropo, w_- tau = 1."""
    sigma, tau, sub, N = 1e-4, 1 / TR.w_m, 50, 4000
    A = matriz_lineal(TR, n_extra=2)
    Qc = np.zeros_like(A)
    A[3, 6] = A[4, 7] = TR.q / TR.m
    A[6, 6] = A[7, 7] = -1 / tau
    Qc[6, 6] = Qc[7, 7] = 2 * sigma**2 / tau
    dt = TR.P_m / sub
    Phi, L = discretizar(A, Qc, dt)
    rng = np.random.default_rng(123)
    X0 = np.zeros((N, 8)); X0[:, 6:] = sigma * rng.standard_normal((N, 2))
    idx, Xs = propagar(Phi, L, X0, 80 * sub, rng, registrar=sub // 2)
    t = idx * dt
    R2 = np.array([np.mean(np.abs(TR.modos(*u_v(X))[1].reshape(8, -1)) ** 2, axis=1) for X in Xs])
    mitad = t > t[-1] / 2
    Dl = np.polyfit(t[mitad], R2[mitad], 1)[0] / 4
    D_V12 = sigma**2 * tau / (TR.B0**2 * (1 + (TR.w_m * tau) ** 2))
    err = Dl.std(ddof=1) / np.sqrt(8)
    assert Dl.mean() / D_V12 == pytest.approx((TR.w_c / TR.w_1) ** 2, abs=4 * err / D_V12)


def test_tiempo_medio_con_disipacion_orbita_completa():
    """Ruido FDT de dos mitades + disipación: media del primer paso = Ein(U)/(2 lambda) (U = 30)."""
    lam_obj, U, N = 0.005, 30.0, 4000
    ReZ = lam_obj * TR.w_m * TR.a**2 * TR.B0 / (TR.q * KAPPA**2 * TR.w_m)
    G = 2 * TR.q * KAPPA**2 * ReZ / TR.a**2
    lam = TR.q * G * TR.w_m / (2 * TR.m * TR.w_1)
    kT = -TR.energia_magnetron(1.0) / U
    A = matriz_lineal(TR, G=G)
    Qc = np.zeros_like(A)
    Qc[3, 3] = (TR.q / TR.m) ** 2 * KAPPA**2 * 4 * kT * ReZ / TR.a**2
    dt = TR.P_m / 64
    Phi, L = discretizar(A, Qc, dt)
    rng = np.random.default_rng(321)
    tp = np.full(N, np.inf)
    vivo = np.ones(N, bool)

    def vigilar(n, X):
        f = vivo & (np.abs(TR.modos(*u_v(X))[1]) >= 1.0)
        tp[f] = n * dt
        vivo[f] = False
        return not vivo.any()

    propagar(Phi, L, np.zeros((N, 6)), int(15 * Ein(U) / (2 * lam) / dt), rng, registrar=10**9, alfin=vigilar)
    assert np.isfinite(tp).all()
    t_teo = Ein(U) / (2 * lam)
    err = np.std(tp) / np.sqrt(N)
    assert np.mean(tp) == pytest.approx(t_teo, abs=4 * err + 0.01 * t_teo)
    assert np.median(tp) / np.mean(tp) > 0.9          # con disipación t_50 no es 0,802 t_medio
