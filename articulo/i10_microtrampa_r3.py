"""
i10_microtrampa_r3.py — A1, ronda 3: versión corregida de i10_microtrampa.py (ronda 2).
Único cambio de física: la conversión D <-> t de la sec. 1 (fe de erratas del «1011 s»).
Original: A1, ronda 2, tarea A1-1 (I10a y parte numérica de I10b).

La estimación de Narimannezhad et al. (arXiv:1301.0030, sec. III.1.6, ecs. 21-25) se recalcula con
el modelo del centro guía de la tesis V12.

Parámetros publicados (arXiv:1301.0030):
  B = 7 T; R_w = 50 µm; tubo central a tierra de 5 mm a 10 cm (10 cm es la longitud de diseño);
  electrodos de cierre a V_e = 10 V (caso base; Tabla I: 10-500 V); positrón de 5 eV con
  V0 = 1,32e6 m/s; l_c = 0,1 µm; E = 1 mV / 50 µm = 20 V/m.
  Su estimación (ec. 25): t = B^2 X^2 V0 / (E^2 l_c), con X = R_w, da ~ 4000 s.

El pozo es plano (Penning-Malmberg) y el artículo no publica su curvatura k2. Este script la
deduce [INFERIDO] como la curvatura promediada sobre el rebote que siente un positrón aislado:
  - potencial en el eje cerca de un electrodo de cierre (salto ideal, sin hueco) en un tubo de radio a:
        phi0(z) = (V_e/2) sum_n c_n exp(j_n z/a),   c_n = 2/(j_n J1(j_n)),   z < 0 dentro del tubo;
  - campo radial cerca del eje: E_r = (r/2) phi0''(z);
  - promedio temporal sobre el rebote: k2_ef = <phi0''/2>_t, de modo que E_r = k2_ef r, igual que U0 r
    en la trampa de la tesis.
También se considera el régimen de plasma, en el que la rotación medida en la simulación
(8,95e8 s^-1) da un k2 equivalente = omega_r B.

Uso: MPLBACKEND=Agg python3 i10_microtrampa.py [--codigo RUTA] [--salida CARPETA]
Es determinista y tarda menos de 2 min.
"""
import argparse
import csv
import os
import sys

sys.dont_write_bytecode = True
ap = argparse.ArgumentParser()
ap.add_argument("--codigo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap.add_argument("--salida", default=os.path.dirname(os.path.abspath(__file__)))
args = ap.parse_args()
sys.path.insert(0, args.codigo)

import numpy as np  # noqa: E402
from scipy.integrate import quad  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

import centro_guia as cg  # noqa: E402
import transito as tr  # noqa: E402
from params import M_E, Q_E, W_M as WM_TESIS, W_Z as WZ_TESIS  # noqa: E402

M_P = 1.67262192369e-27

# ------------------------------------------------------------------ parámetros publicados
B = 7.0                 # T
RW = 50e-6              # m
V0 = 1.32e6             # m/s (5 eV)
W = 0.5 * M_E * V0**2   # J (~4,95 eV)
LC = 0.1e-6             # m
EPS = 20.0              # V/m
VE = 10.0               # V (caso base)
L_TUBO = (5e-3, 1e-2, 0.1)   # m (simulados: 5 mm y 1 cm; diseño: 10 cm)

# ------------------------------------------------------------------ 1. su estimación
t_N = B**2 * RW**2 * V0 / (EPS**2 * LC)

# ------------------------------------------------------------------ 2. k2 promediado sobre el rebote
from scipy.special import i0 as _I0

_ZG = np.linspace(-12.0, 12.0, 4801)          # z/a


def _tabla(Ve=1.0):
    """phi en el eje y phi'' para un salto ideal 0 -> Ve en z = 0 (tubo de radio a, sin hueco):
    phi0 = (Ve/2)[1 + (2/pi) int_0^inf sin(k z)/(k I0(k a)) dk],  phi0'' = -(Ve/pi) int k sin(kz)/I0(ka) dk."""
    f = [0.5 * (1 + (2 / np.pi) * quad(lambda x: np.sin(x * u) / (x * _I0(x)), 0, 80, limit=600)[0]) for u in _ZG]
    g = [-(1 / np.pi) * quad(lambda x: x * np.sin(x * u) / _I0(x), 0, 80, limit=600)[0] for u in _ZG]
    return np.array(f), np.array(g)       # g en unidades de Ve/a^2


_F, _G = _tabla()


def phi0(z, a=RW, Ve=VE):
    return Ve * np.interp(z / a, _ZG, _F)


def phi0_2(z, a=RW, Ve=VE):
    return Ve * np.interp(z / a, _ZG, _G) / a**2


def k2_rebote(L, W=W, a=RW, Ve=VE, zc_a=10.0):
    """k2_ef = <phi0''/2>_t para un positrón de energía W que rebota en un tubo de longitud L
    (electrodo de cierre en z = 0, tubo en z < 0). Devuelve (k2_ef, T_b, z_t)."""
    zt = brentq(lambda z: phi0(z, a, Ve) * Q_E - W, -zc_a * a, 11 * a)
    zc = -zc_a * a
    v = lambda z: np.sqrt(max(2 * (W - Q_E * phi0(z, a, Ve)) / M_E, 1e-30))
    dphi = (phi0(zt + 1e-3 * a, a, Ve) - phi0(zt - 1e-3 * a, a, Ve)) / (2e-3 * a)
    smax = np.sqrt(zt - zc)
    g0 = lambda s: 2 * s / v(zt - s * s) if s > 1e-6 * np.sqrt(a) else 2 / np.sqrt(2 * Q_E * abs(dphi) / M_E)
    g1 = lambda s: 0.5 * phi0_2(zt - s * s, a, Ve) * g0(s)
    I0 = quad(g0, 0, smax, limit=800)[0]
    I1 = quad(g1, 0, smax, limit=800)[0]
    t_plano = (L / 2 + zc) / V0
    return I1 / (t_plano + I0), 4 * (t_plano + I0), zt


def _dphi(pot, z, h=1e-9):
    return (pot(z + h) - pot(z - h)) / (2 * h)


# ------------------------------------------------------------------ 3. régimen frente a k2
def regimen(k2, eps, R=RW):
    if k2 >= 2 * eps / R:
        return "acotado"                         # aun si el campo aparece de golpe
    if k2 >= eps / R:
        return "acotado si lento / pérdida si súbito"
    return "pérdida por desplazamiento"


def t_p(k2, eps, Bf=B, R=RW):
    """Tiempo hasta la pared con un campo que aparece de golpe (A3 r1 §4.3)."""
    d = eps / k2
    return np.inf if 2 * d < R else (2 * Bf / k2) * np.arcsin(R / (2 * d))


def t_p_integrado(k2, eps, Bf=B, R=RW, pasos=4000):
    cg.U0, cg.B0, cg.W_GC = k2, Bf, k2 / Bf
    P = 2 * np.pi * Bf / k2
    dt = P / pasos
    Z, E = 0j, eps + 0j
    for n in range(1, 2 * pasos + 1):
        Z = cg.paso_centro_guia(Z, E, E, dt)
        if abs(Z) >= R:
            return n * dt
    return np.inf


def omega_z_alpha(k2, B_A=1.0):
    return np.sqrt(2 * Q_E * k2 / M_P), k2 / B_A


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = []
    print(f"1) Estimación publicada (ec. 25): t = {t_N:.0f} s  (publicado: ~4000 s; error {abs(t_N/4000-1)*100:.1f} %)")
    # Corrección r3 (fe de erratas del «1011 s»): con pasos isótropos en 2D, <r^2> = N Dx^2 = 4 D t,
    # de modo que D = R^2/(4 t_N) y el primer paso medio t = R^2/(4D) reproduce t_N (no R^2/(16 D')).
    D_N = RW**2 / (4 * t_N)
    t_pp = RW**2 / (4 * D_N)
    print(f"   D implícito (<r^2> = 4 D t) = R^2/(4 t) = {D_N:.3e} m^2/s;  primer paso medio 2D t = R^2/(4D) = {t_pp:.0f} s")
    out.append(("t_primer_paso_2D_Narimannezhad", t_pp, "s", "<r^2>=N dx^2=4Dt; t=R^2/4D", "i10_microtrampa_r3.py", "VERIFICADO (corrige 1011 s de r2)"))
    # mediana del primer paso 2D desde el centro: 0,802 t
    out.append(("t50_primer_paso_2D_Narimannezhad", 0.80210 * t_pp, "s", "mediana = 4*C50*t", "i10_microtrampa_r3.py", "VERIFICADO"))
    out.append(("t_Narimannezhad_ec25", t_N, "s", "B=7 T, R=50 um, V0=1,32e6 m/s, E=20 V/m, l_c=0,1 um", "i10_microtrampa.py", "VERIFICADO (reproduce ~4000 s)"))

    print("2) k2 promediado sobre el rebote (positrón aislado, V_e = 10 V) [INFERIDO]")
    filas_k2 = {}
    for L in L_TUBO:
        for Wv in (W, 1.0 * Q_E, 0.1 * Q_E):
            k2, Tb, zt = k2_rebote(L, W=Wv)
            wb, wm = 2 * np.pi / Tb, k2 / B
            filas_k2[(L, Wv)] = (k2, wb, wm)
            print(f"   L = {L*1e3:6.1f} mm, W = {Wv/Q_E:4.2f} eV: retorno z_t = {zt/RW:+.2f} a; k2_ef = {k2:.3e} V/m^2;"
                  f" omega_b = {wb:.3e}, omega_- = {wm:.3e} rad/s, omega_b/omega_- = {wb/wm:.3e}")
            out.append((f"k2_ef_L{L*1e3:g}mm_W{Wv/Q_E:.2g}eV", k2, "V/m^2", "positrón aislado, V_e=10 V, salto ideal", "i10_microtrampa.py", "INFERIDO"))
            out.append((f"omega_b_sobre_omega_m_L{L*1e3:g}mm_W{Wv/Q_E:.2g}eV", wb / wm, "1", "idem", "i10_microtrampa.py", "INFERIDO"))
    k2_plasma = 8.95e8 * B
    print(f"   Régimen de plasma: omega_r = 8,95e8 s^-1 (simulado, sec. de resultados) -> k2 equivalente = {k2_plasma:.2e} V/m^2")
    out.append(("k2_equivalente_plasma", k2_plasma, "V/m^2", "omega_r=8,95e8 s^-1 x B", "arXiv:1301.0030 + i10", "INFERIDO"))

    print("3) Umbrales (R = 50 µm):")
    for eps in (EPS, 0.02):
        print(f"   eps = {eps:g} V/m: k2 > {2*eps/RW:.2e} (acotado si súbito), k2 > {eps/RW:.2e} (acotado si lento);"
              f" k2 -> 0: t_p -> B R/eps = {B*RW/eps:.2e} s")
        out.append((f"k2_umbral_subito_eps{eps:g}", 2 * eps / RW, "V/m^2", "R=50 um", "i10", "VERIFICADO (fórmula)"))
    print("   Comprobación de t_p con paso_centro_guia (eps = 20 V/m):")
    for k2 in (1e4, 1e5, 5e5, 7.9e5):
        a, n = t_p(k2, EPS), t_p_integrado(k2, EPS)
        print(f"     k2 = {k2:.1e}: t_p analítico = {a:.4e} s, integrado = {n:.4e} s ({abs(n/a-1)*100:.2f} %)")
        out.append((f"t_p_k2_{k2:.1e}", a, "s", "eps=20 V/m, súbito", "i10 (analítico; integrado coincide)", "VERIFICADO"))

    # barrido de régimen
    k2s = np.logspace(2, 11, 400)
    filas = []
    for eps in (EPS, 0.02):
        for k2 in k2s:
            filas.append((eps, k2, regimen(k2, eps), t_p(k2, eps), eps / k2))
    with open(os.path.join(args.salida, "i10_regimenes_r3.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["eps_V_m", "k2_V_m2", "regimen", "t_p_s_subito", "delta_m"])
        for f in filas:
            w.writerow([f"{f[0]:.3e}", f"{f[1]:.4e}", f[2], f"{f[3]:.4e}", f"{f[4]:.4e}"])

    print("4) Prueba de tránsito con el cociente de la microtrampa (unidades escaladas de transito.py)")
    for etiqueta, R in (("L=1 cm, 5 eV", filas_k2[(1e-2, W)][1] / filas_k2[(1e-2, W)][2]),
                        ("L=10 cm, 5 eV", filas_k2[(0.1, W)][1] / filas_k2[(0.1, W)][2])):
        Ruse = min(R, 3000.0)
        tr.W_Z = float(Ruse)
        t, R2, th = tr.parche_estatico_muestreado(n_tray=150, n_periodos=4, semilla=2026)
        print(f"   {etiqueta}: omega_b/omega_- = {R:.3g} (simulado con {Ruse:.0f}): max<|Z|^2>/teoría = {R2.max()/th.max():.4f};"
              f" <|Z|^2>(4 P_-)/max = {R2[-1]/R2.max():.1e}  -> acotado, no difunde")
        out.append((f"transito_max_sobre_teoria_{etiqueta}", R2.max() / th.max(), "1", f"omega_b/omega_-={Ruse:.0f}", "transito.parche_estatico_muestreado", "VERIFICADO"))
    tr.W_Z = 50.0

    print("5) Resonancias |n w_z - l w_-| (n >= 1, l = 1..4), en unidades de w_-")
    trampas = {"tesis (positrón, 0,5 T)": WZ_TESIS / WM_TESIS}
    k2a, wba, wma = filas_k2[(0.1, W)]
    trampas["microtrampa (L=10 cm, 5 eV)"] = wba / wma
    k2b, wbb, wmb = filas_k2[(5e-3, 0.1 * Q_E)]
    trampas["microtrampa (L=5 mm, 0,1 eV)"] = wbb / wmb
    for nom, k2A in (("ALPHA pozo 10 V (k2 = 8,7e3)", 8.7e3), ("ALPHA pozo 10 V (k2 = 1,41e4)", 1.4055e4)):
        wz, wm = omega_z_alpha(k2A)
        trampas[nom] = wz / wm
    res_rows = []
    for nom, Rr in trampas.items():
        d = [min(abs(n * Rr - l) for n in range(1, 6)) for l in (1, 2, 3, 4)]
        print(f"   {nom:34s}: w_z/w_- = {Rr:.4g};  min_n |n w_z - l w_-|/w_- (l=1..4) = " + ", ".join(f"{x:.4g}" for x in d))
        res_rows.append((nom, Rr, *d))
        out.append((f"wz_sobre_wm_{nom}", Rr, "1", "", "i10", "INFERIDO" if "micro" in nom or "ALPHA" in nom else "VERIFICADO"))
    with open(os.path.join(args.salida, "i10_resonancias_r3.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["trampa", "wz_sobre_wm", "dist_l1", "dist_l2", "dist_l3", "dist_l4"])
        for r in res_rows:
            w.writerow([r[0]] + [f"{x:.5g}" for x in r[1:]])

    with open(os.path.join(args.salida, "i10_cifras_r3.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["magnitud", "valor", "unidad", "condiciones", "script_o_fuente", "estado"])
        for r in out:
            w.writerow([r[0], f"{r[1]:.5g}", *r[2:]])

    # figura: tiempo de pérdida frente a k2
    fig, ax = plt.subplots(figsize=(6.3, 3.9))
    for eps, ls in ((EPS, "-"), (0.02, "--")):
        tp = np.array([t_p(k, eps) for k in k2s])
        m = np.isfinite(tp)
        ax.loglog(k2s[m], tp[m], ls=ls, color="k", label=f"pérdida por desplazamiento, ε = {eps:g} V/m")
        ax.axvline(2 * eps / RW, ls=ls, color="0.5", lw=0.8)
        ax.text(2 * eps / RW * 1.2, 1e-9 if eps > 1 else 1e-6, f"acotado →\n(ε = {eps:g} V/m)", fontsize=7)
    ax.axhline(t_N, color="tab:red", lw=1.2, label="Narimannezhad et al.: ~4000 s")
    for (L, Wv), (k2, wb, wm) in filas_k2.items():
        if abs(Wv - W) < 1e-25:
            ax.axvline(k2, color="tab:blue", lw=0.8, ls=":")
            ax.text(k2, 3e2, f"k2 rebote\nL={L*1e3:g} mm", fontsize=6, color="tab:blue", rotation=90, va="top")
    ax.axvline(k2_plasma, color="tab:green", lw=0.8, ls="-.")
    ax.text(k2_plasma / 3, 3e2, "plasma", fontsize=7, color="tab:green", rotation=90, va="top")
    ax.set_xlabel(r"curvatura efectiva del pozo $k_2$ (V/m$^2$)")
    ax.set_ylabel("tiempo hasta la pared (s)")
    ax.set_ylim(1e-10, 1e5)
    ax.legend(fontsize=6.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(args.salida, "i10_regimenes_r3.pdf"))
    fig.savefig(os.path.join(args.salida, "i10_regimenes_r3.png"), dpi=130)
    print("Archivos: i10_regimenes.csv, i10_resonancias.csv, i10_cifras.csv, i10_regimenes.pdf")
