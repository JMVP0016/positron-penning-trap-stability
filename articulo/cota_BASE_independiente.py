"""
cota_BASE_independiente.py — A1, ronda 3, tarea A1-1(b).

Recálculo independiente de la cota de S_E(nu_-) que sale de la estabilidad axial de la trampa de
análisis (AT) de BASE (Borchert et al. 2019, arXiv:1901.09860). Lo hice sin ver los resultados de
A3 en la ronda 3. Solo reutilizo de la ronda 2 la definición de las 4 variantes (A3 r2,
i2_cota_BASE.md), y escribo las fórmulas desde cero.

Datos publicados [VERIFICADO en arXiv:1901.09860 (ar5iv), 30 oct. 2026]:
  nu_+ = 17,845 MHz; nu_z ≈ 675 kHz; B2 = 272 kT/m^2; Dnu_z,+ = 62(4) mHz; Dnu_z,- = 40(3) µHz;
  imán de 1,945 T; radio interior de la AT 1,8 mm; Dnu_z = h nu_+/(4 pi^2 m nu_z) (B2/B0) [(n+ + 1/2) + (nu_-/nu_+)(n- + 1/2) + ...].
Datos leídos de las figuras (A3 r2; [VERIFICAR] contra las figuras, no están en el texto):
  sigma_nuz(250 s) ≈ 0,05 Hz; sigma_nuz(1400 s) ≈ 0,075 Hz; Fig. 4a: ≈ 0,48 Hz con rho_- = 65 µm (recta).

Uso: python3 cota_BASE_independiente.py [--codigo RUTA] [--salida CARPETA]   (determinista, < 5 s)
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

import ruido as rd  # noqa: E402
from params import B0 as B_TESIS, HBAR, Q_E, R_WALL, UN_ANIO  # noqa: E402

H_PL = 2 * np.pi * HBAR
M_P = 1.67262192369e-27
HORA = 3600.0

NUP, NUZ, B2 = 17.845e6, 675e3, 272e3
DNU_P_PUB, DNU_M_PUB = 62e-3, 40e-6
B_IMAN, A_AT, T_AT = 1.945, 1.8e-3, 6.0     # T_AT ≈ 6 K: la misma suposición de A3 r2 [INFERIDO]
cif = []


def c(mag, val, uni, cond, est):
    cif.append((mag, val, uni, cond, "cota_BASE_independiente.py", est))


print("1) Campo B0 en el centro de la AT")
num = NUZ**2 / (2 * NUP)                       # nu_- = nu_z^2/(2 nu_+)
nuc = NUP + num                                 # nu_c = nu_+ + nu_-
B0 = 2 * np.pi * nuc * M_P / Q_E
print(f"   nu_- = nu_z^2/(2 nu_+) = {num:.1f} Hz;  nu_c = nu_+ + nu_- = {nuc/1e6:.4f} MHz;  B0 = 2 pi nu_c m/q = {B0:.4f} T")
c("nu_menos_AT", num, "Hz", "nu_z^2/(2 nu_+)", "VERIFICADO (cálculo)")
c("B0_AT", B0, "T", "desde nu_+ y nu_z", "VERIFICADO (cálculo)")
for B, nom in ((B0, "B0 de nu_c"), (B_IMAN, "1,945 T (imán)")):
    dp = H_PL * NUP / (4 * np.pi**2 * M_P * NUZ) * B2 / B
    dm = dp * num / NUP
    print(f"   con {nom:15s}: Dnu_z,+ = {dp*1e3:5.1f} mHz (publicado 62(4));  Dnu_z,- = {dm*1e6:5.1f} µHz (publicado 40(3));"
          f"  nu_c = {Q_E*B/(2*np.pi*M_P)/1e6:.3f} MHz")
    c(f"Dnu_z_mas_con_{nom}", dp * 1e3, "mHz", f"B = {B:.4g} T", "VERIFICADO (cálculo)")
    c(f"Dnu_z_menos_con_{nom}", dm * 1e6, "µHz", f"B = {B:.4g} T", "VERIFICADO (cálculo)")

print("2) Escala del observable: rho^2 por cuanto magnetrón y deriva de nu_z")
w1 = 2 * np.pi * (NUP - num)                    # omega_1 = omega_+ - omega_- (solo frecuencias medidas)
rq2 = 2 * HBAR / (M_P * w1)
G = DNU_M_PUB / rq2                              # Hz por m^2 de rho^2
print(f"   omega_1 = 2 pi (nu_+ - nu_-) = {w1:.5e} rad/s;  rho_q^2 = 2 hbar/(m omega_1) = {rq2:.4e} m^2;  G = {G:.4e} Hz/m^2")
print("   omega_1 sale de frecuencias medidas, no de B0: G y D no dependen de B0. S_E = 4 B0^2 D sí depende.")
c("rho_q2_AT", rq2, "m^2", "2 hbar/(m omega_1)", "VERIFICADO (cálculo)")
c("G_AT", G, "Hz/m^2", "Dnu_z,- / rho_q^2", "VERIFICADO (cálculo)")

print("3) Cotas de D (4 variantes de A3 r2; dos convenciones para sigma)")
pend = 0.48 / 65e-6
var = [("conservadora (deriva, tau = 250 s, sigma = 0,05 Hz)", "deriva", 0.05, 250.0, 1.0),
       ("intermedia (deriva, tau = 1400 s, sigma = 0,075 Hz)", "deriva", 0.075, 1400.0, 1.0),
       ("optimista (estocástico, 100 % de la pendiente de la Fig. 4a)", "esto", pend, 250.0, 1.0),
       ("optimista (estocástico, 10 % de la varianza)", "esto", pend, 250.0, 0.1)]


def D_cota(tipo, s, tau, frac, conv):
    """conv = 'dif': sigma = desviación de la diferencia de medias adyacentes (convención de A3 r2);
       conv = 'allan': sigma = desviación de Allan (diferencia / sqrt 2).
    Deriva: nu_z sube a = 4 G D por segundo; diferencia de medias adyacentes = a tau.
    Estocástico: d(rho^2) = 4D dt + 2 rho sqrt(2D) dW -> var(dif. de medias) = (2/3) tau G^2 8 D rho^2."""
    if tipo == "deriva":
        a = s / tau * (np.sqrt(2) if conv == "allan" else 1.0)
        return frac * a / (4 * G)
    k = 1.0 if conv == "dif" else 2.0
    return frac * k * 3 * s**2 / (16 * G**2 * tau)    # s = sigma/rho


H_f = None
filas = []
for nom, tipo, s, tau, frac in var:
    D = D_cota(tipo, s, tau, frac, "dif")
    Da = D_cota(tipo, s, tau, frac, "allan")
    SE = 4 * B0**2 * D
    tasa = 4 * D * HORA * 1e12
    filas.append((nom, D, Da, SE, tasa))
    print(f"   {nom:62s}: D <= {D:.3e} m^2/s (Allan: {Da:.3e});  4D = {tasa:.3g} µm^2/h;  S_E = 4 B0^2 D <= {SE:.3e}")
    c(f"D_max_{nom}", D, "m^2/s", "convención de diferencias (A3 r2)", "INFERIDO (cota)")
    c(f"S_E_max_{nom}", SE, "V^2 m^-2 Hz^-1", f"B0 = {B0:.4f} T, nu_- = {num:.0f} Hz", "INFERIDO (cota)")
    c(f"S_E_max_Allan_{nom}", 4 * B0**2 * Da, "V^2 m^-2 Hz^-1", "si sigma es desviación de Allan", "INFERIDO (cota)")

print("4) Escenario «filtro real + EMI» de la tesis llevado a la AT (kappa = 2/pi, a = 1,8 mm, T = 6 K)")


def H_mod(f, filtro=rd.FILTRO_REAL):
    R, C = filtro["R"], filtro["C"]
    n, w = len(R), 2 * np.pi * f
    Y = np.zeros((n, n), complex)
    for k in range(n):
        g = 1 / R[k]
        Y[k, k] += g + 1j * w * C[k]
        if k > 0:
            Y[k - 1, k - 1] += g; Y[k - 1, k] -= g; Y[k, k - 1] -= g
    I = np.zeros(n, complex); I[0] = 1 / R[0]
    return abs(np.linalg.solve(Y, I)[-1])


for f in (num, 10e3):
    SV = rd.SV_filtro(T_AT, rd.FILTRO_REAL, f=f) + rd.SV_EMI_BASE
    print(f"   f = {f/1e3:5.2f} kHz: |H| = {H_mod(f):.4f};  S_V = {SV:.3e} V^2/Hz")
SVp = rd.SV_filtro(T_AT, rd.FILTRO_REAL, f=num) + rd.SV_EMI_BASE
S_pred = rd.KAPPA_MEDIO_ANILLO**2 * SVp / A_AT**2
D_pred = S_pred / (4 * B0**2)
print(f"   S_E predicho en la AT = {S_pred:.3e};  tasa = {4*D_pred*HORA*1e12:.3g} µm^2/h;  t_medio(AT, a = 1,8 mm) = {A_AT**2/(4*D_pred)/UN_ANIO:.3g} años")
c("H_filtro_nu_menos_AT", H_mod(num), "1", f"f = {num:.0f} Hz", "VERIFICADO (cálculo)")
c("S_E_escenario_filtro_real_AT", S_pred, "V^2 m^-2 Hz^-1", "kappa 2/pi, 6 K, EMI", "INFERIDO")
c("tasa_escenario_filtro_real_AT", 4 * D_pred * HORA * 1e12, "µm^2/h", "B0 1,171 T", "INFERIDO")
c("t_medio_escenario_filtro_real_AT", A_AT**2 / (4 * D_pred) / UN_ANIO, "años", "a = 1,8 mm", "INFERIDO")

print("5) Tensión, kappa_eff y traslado a la tesis (0,5 T, 5 mm)")
tabla = []
for nom, D, Da, SE, tasa in filas:
    F = S_pred / SE
    keff = rd.KAPPA_MEDIO_ANILLO / np.sqrt(F)
    t2 = R_WALL**2 * B_TESIS**2 / (SE * (A_AT / R_WALL) ** 2) / UN_ANIO
    t4 = R_WALL**2 * B_TESIS**2 / (SE * (A_AT / R_WALL) ** 4) / UN_ANIO
    tabla.append((nom, SE, F, keff, t2, t4))
    print(f"   {nom:62s}: tensión x{F:.3g};  kappa_eff <= {keff:.3f};  t_tesis >= {t2:.3g} a (∝a^-2) / {t4:.3g} a (∝a^-4)")
    c(f"tension_{nom}", F, "1", "S_pred/S_cota", "INFERIDO")
    c(f"kappa_eff_{nom}", keff, "1", "", "INFERIDO")
    c(f"t_tesis_a2_{nom}", t2, "años", "S_E ∝ a^-2", "INFERIDO")
    c(f"t_tesis_a4_{nom}", t4, "años", "S_E ∝ a^-4", "INFERIDO")
print("   Nota: el traslado no reescala la frecuencia (12,8 kHz en la AT frente a 6,37 kHz en la tesis).")

print("6) Comparación con A6 r2 §1.1")
A6 = {"conservadora": (7.7e-15, 16, 0.16, 2.0e2, 1.5e3), "intermedia": (2.1e-15, 60, 0.08, 7.4e2, 5.7e3),
      "optimista (estocástico, 100": (1.8e-16, 7e2, 0.02, 8.6e3, 6.6e4)}
comp = []
for nom, SE, F, keff, t2, t4 in tabla:
    for k, v in A6.items():
        if nom.startswith(k):
            d = [SE / v[0] - 1, F / v[1] - 1, keff / v[2] - 1, t2 / v[3] - 1, t4 / v[4] - 1]
            print(f"   {k:28s}: S_E {SE:.2e} vs {v[0]:.1e} ({d[0]*100:+.0f} %); tensión {F:.3g} vs {v[1]:g} ({d[1]*100:+.0f} %);"
                  f" kappa_eff {keff:.3f} vs {v[2]} ({d[2]*100:+.0f} %); t {t2:.3g}/{t4:.3g} vs {v[3]:.2g}/{v[4]:.2g}")
            comp.append((k, SE, v[0], F, v[1], keff, v[2], t2, v[3], t4, v[4]))

with open(os.path.join(args.salida, "cota_BASE_cifras.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["magnitud", "valor", "unidad", "condiciones", "script_o_fuente", "estado"])
    for r in cif:
        w.writerow([r[0], f"{r[1]:.5g}", *r[2:]])
with open(os.path.join(args.salida, "cota_BASE_comparacion_A6.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["variante", "S_E_A1", "S_E_A6", "tension_A1", "tension_A6", "keff_A1", "keff_A6",
                "t_a2_A1", "t_a2_A6", "t_a4_A1", "t_a4_A6"])
    for r in comp:
        w.writerow([r[0]] + [f"{x:.4g}" for x in r[1:]])
