"""
i2_observable_r3.py — Agente 3, ronda 3 (corrige B0 y nu_- de BASE AT; ver A6 r2 §1.1) (I2, I10(b) numérico, I3(b)). Determinista, < 5 s.

Importa los módulos V12 (copia verificada con md5) sin modificarlos.
Uso:  MPLBACKEND=Agg python3 i2_observable.py   -> imprime todo y escribe
      i2_trampas_reales.csv, i10b_criterio.csv, cifras_A3_r2.csv en la carpeta de salida.

(a) Observable común: S_E(nu_-) -> D, d<rho^2>/dt (µm^2/h), dn_-/dt, t_medio.
(b) Cota de S_E(nu_-) a partir de la estabilidad axial de BASE AT (Borchert et al. 2019).
(c) Cuatro trampas reales con el escenario «filtro real + EMI» y lo observado.
(d) Escenario calibrado contra BASE.
I10(b) Cocientes de frecuencia del criterio de validez.
I3(b) Vida por aniquilación con el gas residual acotado por BASE.
"""
import csv
import os
import sys

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ruido as rd  # noqa: E402
from params import B0, F_M, HBAR, K_B, M_E, Q_E, R_WALL, U0, UN_ANIO, W_1, W_M, W_Z

_A = sys.argv[1:]
OUT = _A[_A.index("--salida") + 1] if "--salida" in _A else (_A[0] if _A else ".")
M_P = 1.67262192369e-27
C_LUZ = 299792458.0
R0_E = 2.8179403262e-15          # radio clásico del electrón (m)
HORA, DIA = 3600.0, 86400.0
CIFRAS = []


def cifra(magnitud, valor, unidad, condiciones, fuente, estado):
    CIFRAS.append([magnitud, f"{valor:.4g}" if isinstance(valor, float) else valor, unidad, condiciones, fuente, estado])


# =============================================================================================
# (a) Observable común
# =============================================================================================
def observable(S_E, B, r, w1, m):
    """S_E por componente a nu_- (V^2 m^-2 Hz^-1) -> dict con D, tasa en µm^2/h, dn/dt (1/s), t_medio (s).
    dn_-/dt = (m w1 / 2 hbar) d<rho^2>/dt = 2 m w1 D / hbar   (rho^2 = 2 hbar (n+1/2)/(m w1))."""
    D = S_E / (4 * B**2)
    return {"D": D, "um2_h": 4 * D * HORA * 1e12, "dn_dt": 2 * m * w1 * D / HBAR, "t_medio": r**2 / (4 * D)}


def S_E_desde_tasa(um2_h, B):
    return 4 * B**2 * (um2_h * 1e-12 / HORA) / 4


print("=" * 90 + "\n(a) Observable común en la trampa de la tesis (B = 0,5 T, r = 5 mm)\n" + "=" * 90)
for T in (4.0, 1.0, 0.3, 0.1):
    for nombre, S in rd.escenarios(T).items():
        o = observable(S[0], B0, R_WALL, W_1, M_E)
        if T in (4.0, 0.1) or "real (" in nombre:
            print(f"  T={T:4} K {nombre:36s}: t = {o['t_medio']/UN_ANIO:9.3g} a | {o['um2_h']:9.3g} µm²/h | dn/dt = {o['dn_dt']:8.3g} 1/s")
        cifra(f"tasa d<rho^2>/dt, {nombre}", o["um2_h"], "µm²/h", f"tesis, T={T} K, D=S_E/4B²",
              "i2_observable.py (ruido.escenarios)", "INFERIDO (escenario)")
# comprobación independiente del puente Q5 con cuantico.py (amplitud del magnetrón)
import cuantico as cu
rho2 = lambda n: cu.amplitud(n, W_1) ** 2
print(f"  puente: d(rho^2)/dn con cuantico.amplitud = {rho2(1.0)-rho2(0.0):.4e} m²; 2ħ/(m w1) = {2*HBAR/(M_E*W_1):.4e} m²")
for t_obj in (1.0, 10.0, 100.0):
    S = R_WALL**2 * B0**2 / (t_obj * UN_ANIO)
    o = observable(S, B0, R_WALL, W_1, M_E)
    print(f"  criterio: t_medio >= {t_obj:5.0f} a  <=>  S_E(f_-) <= {S:.2e}  <=>  tasa <= {o['um2_h']:.3g} µm²/h")
    cifra(f"S_E máximo para t_medio >= {t_obj:g} a", S, "V²m⁻²Hz⁻¹", "tesis 0,5 T, 5 mm", "t=r²B²/S_E", "VERIFICADO (fórmula)")

# =============================================================================================
# (b) Cota a partir de la estabilidad axial de BASE AT  [Borchert et al. 2019, arXiv:1901.09860]
# =============================================================================================
print("\n" + "=" * 90 + "\n(b) Cota de S_E(nu_-) desde la estabilidad axial de BASE AT\n" + "=" * 90)
A_AT, NUP_AT, NUZ_AT = 1.8e-3, 17.845e6, 675e3    # Borchert 2019, p. 2 (diámetro 3,6 mm; nu_+; nu_z)
# R3: B0 de la AT NO es el del imán (1,945 T): se deduce de nu_+ (ninguna publicación da B0 de la AT explícitamente).
NUM_AT = NUZ_AT**2 / (2 * NUP_AT)                 # nu_- ≈ nu_z^2 / (2 nu_+)
B_AT = 2 * np.pi * (NUP_AT + NUM_AT) * M_P / Q_E  # nu_c = nu_+ + nu_-
B2_AT = 272e3                                     # T/m², p. 2
dz_mas = 6.62607015e-34 * NUP_AT * B2_AT / (4 * np.pi**2 * M_P * NUZ_AT * B_AT)
print(f"  B0(AT) deducido de nu_+ = {B_AT:.4f} T; nu_- = nu_z²/2nu_+ = {NUM_AT/1e3:.2f} kHz")
print(f"  comprobación 1: con 1,945 T, nu_c = {Q_E*1.945/(2*np.pi*M_P)/1e6:.2f} MHz (≠ 17,845 MHz publicado)")
print(f"  comprobación 2: Δν_z,+ = hν+B2/(4π²mν_zB0) = {dz_mas*1e3:.1f} mHz (publicado 62(4) mHz); con 1,945 T: "
      f"{dz_mas*B_AT/1.945*1e3:.1f} mHz")
print(f"  comprobación 3: Δν_z,- = Δν_z,+ ν-/ν+ = {dz_mas*NUM_AT/NUP_AT*1e6:.1f} µHz (publicado 40(3) µHz); con 1,945 T y 10 kHz: "
      f"{dz_mas*B_AT/1.945*1e4/NUP_AT*1e6:.1f} µHz")
cifra("B0 trampa de análisis BASE (deducido)", B_AT, "T", "nu_c = nu_+ + nu_-", "Borchert 2019 p. 2 + i2_observable_r3.py", "INFERIDO (3 comprobaciones)")
cifra("nu_- trampa de análisis BASE", NUM_AT, "Hz", "nu_z²/2nu_+", "Borchert 2019 p. 2", "INFERIDO")
W1_AT = 2 * np.pi * (NUP_AT - NUM_AT)
DNU_Q = 40e-6                      # Hz por cuanto magnetrón, p. 2 (tras la ec. 1): 40(3) µHz
RHO2_Q = 2 * HBAR / (M_P * W1_AT)  # m² por cuanto
print(f"  rho² por cuanto magnetrón = {RHO2_Q:.4e} m²  (rho_q = {np.sqrt(RHO2_Q)*1e9:.1f} nm)")
G = DNU_Q / RHO2_Q                 # Hz por m² de rho²
casos_b = []
# 1) Deriva: <rho²> crece 4D t -> nu_z crece G 4D; diferencia de medias adyacentes = a tau.
for etiqueta, sig, tau, frac in (("conservador: deriva, tau=250 s, sigma=0,05 Hz (Fig. 2/4a)", 0.05, 250.0, 1.0),
                                 ("intermedio: deriva, tau=1400 s, sigma=0,075 Hz (Fig. 2)", 0.075, 1400.0, 1.0)):
    D = frac * sig / (4 * G * tau)
    casos_b.append((etiqueta, D))
# 2) Término estocástico: d(rho²) = 4D dt + 2 rho sqrt(2D) dW  -> var(dif. medias) = (2/3) tau G² 8 D rho²
#    sigma_mag(tau) = G rho sqrt(16 D tau / 3)  ->  D <= 3 (sigma/rho)² / (16 G² tau)
pend = 0.48 / 65e-6                  # Hz/m: recta de la Fig. 4a (0,48 Hz en 65 µm), toda atribuida (100 %)
for etiqueta, frac in (("optimista: término estocástico, pendiente Fig. 4a, 100 % de la varianza", 1.0),
                       ("optimista (10 % de la varianza libre)", 0.1)):
    D = frac * 3 * pend**2 / (16 * G**2 * 250.0)
    casos_b.append((etiqueta, D))
# comprobación con el punto de 6 µm (0,05 Hz)
D6 = 3 * (0.05 / 6e-6) ** 2 / (16 * G**2 * 250.0)
print(f"  comprobación con el punto rho = 6 µm, sigma = 0,05 Hz: D <= {D6:.2e} m²/s")
for etiqueta, D in casos_b:
    o = observable(4 * B_AT**2 * D, B_AT, A_AT, W1_AT, M_P)
    print(f"  {etiqueta:72s}: D <= {D:.2e} m²/s | S_E <= {4*B_AT**2*D:.2e} | {o['um2_h']:.3g} µm²/h | "
          f"dn/dt <= {o['dn_dt']:.3g} 1/s | t_medio(AT) >= {o['t_medio']/UN_ANIO:.3g} a")
    cifra(f"cota BASE AT, {etiqueta}", 4 * B_AT**2 * D, "V²m⁻²Hz⁻¹", f"{B_AT:.3f} T, a=1,8 mm, nu_-≈{NUM_AT/1e3:.1f} kHz, ~6 K",
          "Borchert 2019 p.2, Fig.2, Fig.4a; i2_cota_BASE.md", "INFERIDO (cota nueva)")
S_BASE_CONS = 4 * B_AT**2 * casos_b[0][1]
S_BASE_PRINC = 4 * B_AT**2 * casos_b[1][1]   # R3: cota principal = deriva con tau = 1400 s
S_BASE_OPT = 4 * B_AT**2 * casos_b[2][1]
print("  D no depende de B0 (rho_q² usa omega_1 medido); S_E = 4 B0² D sí: factor (1,171/1,945)² =", f"{(B_AT/1.945)**2:.4f}")

# =============================================================================================
# (c) Cuatro trampas reales
# =============================================================================================
print("\n" + "=" * 90 + "\n(c) Cuatro trampas reales: escenario «filtro real + EMI» (κ = 2/π, a = r) frente a lo observado\n" + "=" * 90)


def H_mod(f, filtro=rd.FILTRO_REAL):
    R, C = filtro["R"], filtro["C"]
    n, w = len(R), 2 * np.pi * f
    Y = np.zeros((n, n), dtype=complex)
    for k in range(n):
        g = 1 / R[k]
        Y[k, k] += g + 1j * w * C[k]
        if k > 0:
            Y[k - 1, k - 1] += g; Y[k - 1, k] -= g; Y[k, k - 1] -= g
    I = np.zeros(n, dtype=complex); I[0] = 1 / R[0]
    return abs(np.linalg.solve(Y, I)[-1])


def S_escenario(T, f, a, kappa=rd.KAPPA_MEDIO_ANILLO, emi=True):
    partes = rd.SV_filtro(T, rd.FILTRO_REAL, f=f, por_etapa=True)
    SV = sum(partes) + (rd.SV_EMI_BASE if emi else 0.0)
    return kappa**2 * SV / a**2, partes


def S_max_supervivencia(B, r, t_exp, n_part, P=0.05):
    """S_E máximo tal que la probabilidad de no perder ninguna de n_part partículas, cada una expuesta t_exp
    partiendo del centro, sea >= P (difusión 2D con pared absorbente, ruido.supervivencia)."""
    S_obj = P ** (1.0 / n_part)
    f = lambda lnD: rd.supervivencia(t_exp, np.exp(lnD), R=r)[0] - S_obj
    D = np.exp(brentq(f, np.log(1e-30), np.log(1e-3)))
    return 4 * B**2 * D, D


k2_alpha = 1.41e4      # V/m² (pozo de 10 V, modelo del anillo de 2 cm; ronda 1); empírico 8,7e3
trampas = [
    # nombre, partícula, masa, B, r, nu_-, T, U0/k2, observación, fuente_param
    ("BASE reservorio (RT)", "pbar", M_P, 1.946, 4.5e-3, 7.02e3, 6.0, None,
     "405 d, 18 pbar, 0 pérdidas; exposición 3,15e8 s", "Smorra 2015 arXiv:1507.04147 (9 mm, 1,946 T, 7,02 kHz); Sellner 2017 resumen"),
    ("BASE AT", "pbar", M_P, B_AT, 1.8e-3, NUM_AT, 6.0, None,
     "sigma_nuz(250 s) ≈ 0,05 Hz (estabilidad axial)", "Borchert 2019 p. 2, Fig. 2, Fig. 4"),
    ("Harvard e- (Hanneke 2011)", "e-", M_E, 5.36, 4.5e-3, 133e3, 0.1, None,
     "un electrón «meses»; magnetrón enfriado por bandas laterales", "Hanneke 2011 Tabla I (p. 5) y sec. II.C"),
    ("ALPHA pozo 10 V", "pbar", M_P, 1.0, 0.022, k2_alpha / (2 * np.pi * 1.0), 4.0, k2_alpha,
     "pbar confinados «horas»", "Baker 2024 p. 2 y Fig. 1; k2 del modelo del anillo (ronda 1)"),
]
filas = []
for nombre, part, m, B, r, fm, T, k2, obs, fuente in trampas:
    w1 = 2 * np.pi * (Q_E * B / m) - 2 * 2 * np.pi * fm      # w_c - 2 w_-  ≈ w_+ - w_-
    S_pred, partes = S_escenario(T, fm, r)
    o = observable(S_pred, B, r, w1, m)
    Hm = H_mod(fm)
    frac_emi = rd.SV_EMI_BASE / (sum(partes) + rd.SV_EMI_BASE)
    if nombre.startswith("BASE reservorio"):
        S_max, _ = S_max_supervivencia(B, r, 3.15e8 / 18, 18)
        S_max_txt = f"{S_max:.2e} (P(0 pérdidas) >= 5 %)"
    elif nombre == "BASE AT":
        S_max = S_BASE_PRINC
        S_max_txt = f"{S_BASE_PRINC:.2e} (principal, tau=1400 s); {S_BASE_CONS:.2e} (conservadora); {S_BASE_OPT:.1e} (optimista)"
    elif nombre.startswith("Harvard"):
        S_max, _ = S_max_supervivencia(B, r, 90 * DIA, 1)
        S_max_txt = f"{S_max:.2e} (si 90 d sin enfriamiento; no aplica: hay enfriamiento magnetrón)"
    else:
        S_max, _ = S_max_supervivencia(B, r, 3 * HORA, 1)
        S_max_txt = f"{S_max:.2e} (3 h)"
    cociente = S_pred / S_max
    if nombre.startswith("Harvard"):
        veredicto = "compatible (no restrictivo: enfriamiento magnetrón)"
    elif cociente <= 1:
        veredicto = f"compatible (margen x{1/cociente:.2g})"
    else:
        veredicto = f"tensión: el escenario supera la cota x{cociente:.2g}"
    print(f"  {nombre:26s} B={B} T r={r*1e3:.2f} mm nu_-={fm/1e3:.3g} kHz T={T} K | |H(nu_-)|={Hm:.3g} | "
          f"EMI {frac_emi*100:.0f} % de S_V | S_E pred={S_pred:.2e} -> {o['um2_h']:.3g} µm²/h, t={o['t_medio']/UN_ANIO:.3g} a"
          f" | S_E max={S_max_txt} | {veredicto}")
    filas.append([nombre, part, B, r, fm, T, k2 if k2 else "", fuente, f"{Hm:.3g}", f"{frac_emi:.3f}",
                  f"{S_pred:.3e}", f"{o['D']:.3e}", f"{o['um2_h']:.4g}", f"{o['dn_dt']:.4g}",
                  f"{o['t_medio']/UN_ANIO:.4g}", obs, S_max_txt, veredicto])
    cifra(f"S_E escenario filtro real+EMI, {nombre}", S_pred, "V²m⁻²Hz⁻¹", f"κ=2/π, a=r, nu_-={fm:.3g} Hz",
          "i2_observable.py", "INFERIDO (escenario)")
with open(os.path.join(OUT, "i2_trampas_reales.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["trampa", "particula", "B_T", "r_m", "nu_menos_Hz", "T_K", "k2_V_m2", "fuente_parametros",
                "H_filtro_a_nu_menos", "fraccion_EMI", "S_E_pred", "D_pred_m2_s", "tasa_um2_h", "dn_dt_s-1",
                "t_medio_anios", "observado", "S_E_max_compatible", "veredicto"])
    w.writerows(filas)

# =============================================================================================
# (d) Escenario calibrado contra BASE
# =============================================================================================
print("\n" + "=" * 90 + "\n(d) Escenario calibrado contra BASE\n" + "=" * 90)
S_AT_pred, _ = S_escenario(6.0, NUM_AT, A_AT)
print(f"  |H| del filtro a {NUM_AT/1e3:.1f} kHz = {H_mod(NUM_AT):.4f}; S_E escenario en la AT = {S_AT_pred:.3e}")
for etiqueta, S_cota in (("conservadora tau=250 s", S_BASE_CONS), ("PRINCIPAL tau=1400 s", S_BASE_PRINC), ("optimista", S_BASE_OPT), ("optimista 10 %", 4 * B_AT**2 * casos_b[3][1])):
    F = S_AT_pred / S_cota
    k_eff = rd.KAPPA_MEDIO_ANILLO / np.sqrt(F)
    # Multiplicador de C que reduce S_V del filtro (sin EMI) en el factor F a 10 kHz
    def falta(mult):
        filt = {"R": rd.FILTRO_REAL["R"], "C": tuple(c * mult for c in rd.FILTRO_REAL["C"]), "T_calientes": (300.0, 60.0)}
        S = rd.KAPPA_MEDIO_ANILLO**2 * (rd.SV_filtro(6.0, filt, f=NUM_AT) + rd.SV_EMI_BASE) / A_AT**2
        return S - S_cota
    try:
        mult = brentq(falta, 1.0, 1e4)
        mult_txt = f"C x{mult:.2g}"
    except ValueError:
        mult_txt = "inalcanzable solo con C (la EMI dipolar ya supera la cota)"
    # Trasladar a la tesis: mismo S_V en el electrodo (S_E ∝ 1/a²) o ruido de superficie (∝ a^-4)
    S_tesis_V = S_cota * (A_AT / R_WALL) ** 2
    S_tesis_sup = S_cota * (A_AT / R_WALL) ** 4
    t_V = R_WALL**2 * B0**2 / S_tesis_V / UN_ANIO
    t_s = R_WALL**2 * B0**2 / S_tesis_sup / UN_ANIO
    print(f"  cota {etiqueta}: el escenario a 1,8 mm/12,8 kHz excede x{F:.3g} -> κ_eff <= {k_eff:.3f}, o {mult_txt};"
          f" en la tesis: t_medio >= {t_V:.3g} a (∝ a^-2) o >= {t_s:.3g} a (∝ a^-4)")
    cifra(f"exceso del escenario filtro real sobre la cota BASE ({etiqueta})", F, "-", "BASE AT 1,8 mm, 12,8 kHz, 1,171 T",
          "i2_observable.py", "INFERIDO")
    cifra(f"κ_eff calibrado contra BASE ({etiqueta})", k_eff, "-", "mantiene filtro y EMI", "i2_observable.py", "INFERIDO")
    cifra(f"t_medio tesis, escenario calibrado BASE ({etiqueta}, S_E∝a^-2)", t_V, "años", "0,5 T, 5 mm",
          "i2_observable.py", "INFERIDO")
    cifra(f"t_medio tesis, escenario calibrado BASE ({etiqueta}, S_E∝a^-4)", t_s, "años", "0,5 T, 5 mm",
          "i2_observable.py", "INFERIDO")

# =============================================================================================
# I10(b) Cocientes de frecuencia para el criterio de validez
# =============================================================================================
print("\n" + "=" * 90 + "\nI10(b) Frecuencias: rebote/axial frente a rotación\n" + "=" * 90)
E0 = 8.8541878128e-12
crit = []
wz_t, wm_t = W_Z, W_M
crit.append(["tesis (e+, 0,5 T, U0=2e4)", wz_t, wm_t, wz_t / wm_t, 0.0, "vacío criogénico; partícula aislada"])
for k2, et in ((10.0, "ALPHA pozo somero (pbar, k2=10)"), (k2_alpha, "ALPHA pozo 10 V (pbar, k2=1,4e4)")):
    wz = np.sqrt(2 * Q_E * k2 / M_P); wm = k2 / 1.0
    crit.append([et, wz, wm, wz / wm, 0.0, "E_crit somero 0,11 V/m < eps: pérdida por desplazamiento" if k2 == 10 else ""])
v0, Ltubo = 1.32e6, 0.10                 # Narimannezhad 2014, p. 15 y Tabla II
wb = 2 * np.pi * v0 / (2 * Ltubo)
n_pl = 1.6e17                            # m^-3, resumen
wE = Q_E * n_pl / (2 * E0 * 7.0)
K2_MICRO = 1.8e6   # V/m², k2 efectivo calculado por A1 r2 (zona de reflexión), confirmado por A6 r2 §1.2
crit.append(["microtrampa, partícula aislada (5 eV, 7 T, L=10 cm, k2_ef=1,8e6)", wb, K2_MICRO / 7.0, wb / (K2_MICRO / 7.0), 0.0,
             "R3: k2 efectivo no nulo (A1): desplazamiento acotado, 2δ = 22 µm < 50 µm (V2), no deriva directa"])
crit.append(["microtrampa, plasma 1,6e11 cm^-3 (rotación E×B)", wb, wE, wb / wE, 0.0,
             "resonancias de orden bajo accesibles; régimen colectivo"])
for et, wz, wm, rat, _, nota in crit:
    print(f"  {et:52s}: w_z(o w_b) = {wz:.3e} rad/s, w_R = {wm:.3e} rad/s, cociente = {rat:.4g}  {nota}")
with open(os.path.join(OUT, "i10b_criterio.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["trampa", "omega_z_o_omega_b_rad_s", "omega_R_rad_s", "cociente", "orden_azimutal_minimo_resonancia_n1", "nota"])
    for et, wz, wm, rat, _, nota in crit:
        w.writerow([et, f"{wz:.4e}", f"{wm:.4e}", f"{rat:.4g}", (f"{int(np.ceil(rat))}" if np.isfinite(rat) else "no hay rotación"), nota])
# supresión de la resonancia de menor orden (n = 1) en la tesis: (rho/r)^(l-1) con rho = A_- (4 K)
A_m = 4.054e-6
l_min = int(np.ceil(wz_t / wm_t))
print(f"  tesis: orden azimutal mínimo resonante l ≈ {l_min}; supresión (A_-/r)^(l-1) = 10^{(l_min-1)*np.log10(A_m/R_WALL):.0f}")
cifra("w_z/w_- tesis", wz_t / wm_t, "-", "e+, 0,5 T, U0=2e4 V/m²", "params.py", "VERIFICADO (código)")
cifra("w_b microtrampa Narimannezhad (5 eV, 10 cm)", wb, "rad/s", "v0=1,32e6 m/s, L=10 cm", "Narimannezhad 2014 p.15, Tabla II", "INFERIDO")
cifra("t (ec. 24) Narimannezhad recalculado", 7.0**2 * (50e-6) ** 2 * 1.32e6 / (20.0**2 * 1e-7), "s",
      "B=7 T, X=50 µm, v0=1,32e6, E=20 V/m, lc=0,1 µm", "Narimannezhad 2014 ec. (24) p. 15", "VERIFICADO (aritmética)")

# =============================================================================================
# I3(b) Gas residual: aniquilación del positrón
# =============================================================================================
print("\n" + "=" * 90 + "\nI3(b) Aniquilación con el gas residual acotado por BASE\n" + "=" * 90)
for gas, p_mbar, Z in (("H2", 1.2e-18, 2), ("He", 2.7e-18, 2)):
    for Tg in (6.0, 300.0):
        n = p_mbar * 100 / (K_B * Tg)
        for zr in (1.0, 10.0):
            Gam = np.pi * C_LUZ * R0_E**2 * Z * zr * n
            print(f"  {gas}: p < {p_mbar:.1e} mbar, T_gas={Tg:5.0f} K -> n < {n:.2e} m^-3; Z_eff/Z={zr:4.0f}: "
                  f"tau_aniq > {1/Gam/UN_ANIO:.2e} años")
            if Tg == 6.0:
                cifra(f"vida por aniquilación e+ con {gas} (Z_eff/Z={zr:g})", 1 / Gam / UN_ANIO, "años",
                      f"p<{p_mbar} mbar (Sellner 2017), T_gas=6 K", "Danielson 2015 ec. 78, 86", "INFERIDO (cota inferior)")
print(f"  control: Danielson (p. 32): Γ_D^-1 ~ 100 s con n_e = 1e18 m^-3 -> código: {1/(np.pi*C_LUZ*R0_E**2*1e18):.0f} s")

with open(os.path.join(OUT, "cifras_A3_r2.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["magnitud", "valor", "unidad", "condiciones", "script_o_fuente", "estado"])
    w.writerows(CIFRAS)
print(f"\nEscritos: i2_trampas_reales.csv, i10b_criterio.csv, cifras_A3_r2.csv en {OUT}")
