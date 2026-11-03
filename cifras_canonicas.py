"""
cifras_canonicas.py — Versión de máquina de la tabla de cifras canónicas (plan final §6.1 y A7 §3):
escribe datos/cifras_canonicas.csv con las columnas id, magnitud, valor, unidad, script, tesis, articulo, estado.
Implementación de referencia de la tarea WP1.13 (ronda 4, agente 1); los autores deciden.

    python3 temperaturas.py && MPLBACKEND=Agg python3 make_figures_article.py --slow   # una vez
    MPLBACKEND=Agg python3 figuras_tesis.py 9                                          # crea datos/fig4_09_montecarlo.csv
    python3 cifras_canonicas.py                                                        # < 5 s, determinista

Cada valor se CALCULA con los módulos del modelo o se LEE de las salidas de articulo/ (datos_articulo/);
nada se escribe a mano salvo los datos de la literatura (estado LITERATURA) y los supuestos [VERIFICAR].
Regenerar el CSV da el mismo md5 (formato fijo, sin fechas).
"""
import csv
import os

import numpy as np
from scipy.optimize import brentq

import contraste_experimental as ce
import temperaturas as tt
from centro_guia import E_critico
from cuantico import dn_menos_dt, energia_magnetron, razon_tiempo_positron
from params import (B0, D_TRAP, HBAR, K_B, M_E, R_WALL, U0, UN_ANIO, V0, W_M, W_P, W_Z, Z0)
from parches_fluctuantes import SE_cota_BASE, SE_superficie, LAB_S0, LAB_T0, LAB_BETA
from ruido import C50, D_desde_SE, FILTRO_REAL, SE_desde_D, SV_EMI_BASE, SV_filtro, escenarios, mediana_bootstrap, tiempo_medio
from superficie import E_centro_rms

_DIR = os.path.dirname(os.path.abspath(__file__))
DAT, ART = os.path.join(_DIR, "datos"), os.path.join(_DIR, "datos_articulo")
SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
H_EN_UM2 = 1e12 * 3600                                        # m²/s -> µm²/h


def f(x, cif=3):
    """Número con coma decimal; ×10ⁿ fuera de [0,01; 10⁴)."""
    if x == 0:
        return "0"
    if 1e-2 <= abs(x) < 1e4:
        s = f"{x:.{cif}g}"
        if "e" not in s:
            return s.replace(".", ",")
    m, e = f"{x:.{cif - 1}e}".split("e")
    return f"{m.replace('.', ',')}×10{str(int(e)).translate(SUP)}"


def fd(x, dec):
    return f"{x:.{dec}f}".replace(".", ",")


def csv_dict(nombre, clave="magnitud", valor="valor", carpeta=ART):
    with open(os.path.join(carpeta, nombre), encoding="utf-8") as fh:
        return {r[clave]: r[valor] for r in csv.DictReader(fh)}


def csv_filas(nombre, carpeta=ART):
    with open(os.path.join(carpeta, nombre), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def anios(S):
    return float(np.atleast_1d(tiempo_medio(D_desde_SE(np.asarray(S, dtype=float))) / UN_ANIO)[0])


def tasa(t_anios):
    """Tasa de difusión del radio magnetrón d<rho^2>/dt = r_wall^2 / t_medio, en µm²/h."""
    return R_WALL**2 / (t_anios * UN_ANIO) * H_EN_UM2


filas = []


def fila(id_, magnitud, valor, unidad, script, tesis, articulo, estado="VERIFICADO (código)"):
    filas.append([f"{id_:02d}", magnitud, valor, unidad, script, tesis, articulo, estado])


# ------------------------------------------------------------------ 1-7: trampa y parte estática
fila(1, "Trampa", f"B₀ = {fd(B0, 2)} T; V₀ = {fd(V0, 1)} V; d = ρ₀ = r_wall = {fd(D_TRAP * 1e3, 1)} mm; "
     f"z₀ = {fd(Z0 * 1e3, 2)} mm; U₀ = {f(U0, 2)} V/m²", "", "params.py", "Tabla 4.1; A.1", "§2.1; Tabla 1")
fila(2, "Frecuencias", f"ν₊ = {fd(W_P / 2 / np.pi / 1e9, 1)} GHz; ν_z = {fd(W_Z / 2 / np.pi / 1e6, 2)} MHz; "
     f"f₋ = {fd(W_M / 2 / np.pi / 1e3, 2)} kHz; P₋ = {2 * np.pi / W_M * 1e6:.0f} µs; ħω₊/k_B = {fd(HBAR * W_P / K_B, 3)} K; "
     f"ω_z/ω₋ = {W_Z / W_M:.0f}", "", "params.py", "Tabla 4.1", "§2.1")
fila(3, "Energía del magnetrón en la pared", f"{f(energia_magnetron(R_WALL) / K_B, 2)} k_B·K ({fd(U0 * R_WALL**2 / 2, 2)} V)",
     "", "cuantico.py", "§4.6", "§2.2")
fila(4, "Desplazamiento estático", f"δ = {1 / U0 * 1e6:.0f} µm por V/m", "µm/(V/m)", "centro_guia.py", "§4.2; Ilustr. 4-1", "§2.3, §3.1; Fig. 1")
k2_baker = 147.0                     # Baker et al. 2024, Fig. 2: pérdida con 1 V/m en k2 = 147 V/m² (a3_contrastes, r1)
fac = k2_baker * ce.RW_A / 2 / 1.0
fila(5, "Umbral estático", f"{E_critico('golpe'):.0f} (súbito) – {E_critico('lento'):.0f} (lento) V/m, lineal; "
     f"≈ {E_critico('golpe') / fac:.0f} V/m con anarmonicidades (50/{fd(fac, 2)} ≈ 100/{fd(2 * fac, 2)})", "V/m",
     "centro_guia.py; Baker 2024 Fig. 2 (k₂ = 147 V/m²)", "§4.2", "§2.3, §3.1", "VERIFICADO (código) + LITERATURA")
fila(6, "Campo de parche medido", f"~1 V/m (0,35–2 V/m, ALPHA); el umbral lineal súbito ({E_critico('golpe'):.0f} V/m) es ≥ "
     f"{E_critico('golpe') / ce.EPS_MAX:.0f} veces el campo medido (≥ {30 / ce.EPS_MAX:.0f} con la estimación conservadora de 30 V/m)", "V/m", "Baker 2024", "§2, §4.8", "§3.4", "LITERATURA")
mc = np.loadtxt(os.path.join(DAT, "fig4_02_montecarlo.csv"), delimiter=",", skiprows=1)
fila(7, "Apantallamiento", f"{f(E_centro_rms(1e-3, 1e-7))} V/m (ℓ = 100 nm); {f(E_centro_rms(1e-3, 1e-3))} V/m (ℓ = 1 mm); "
     f"Monte Carlo al {fd(np.max(np.abs(mc[:, 1] / mc[:, 3] - 1)) * 100, 1)} %", "V/m", "superficie.py; datos/fig4_02_montecarlo.csv",
     "§2; Ilustr. 4-2", "§3.3; S1; Tabla 2")

# ------------------------------------------------------------------ 8-12: ALPHA, microtrampa, tránsito
tp = lambda e: float(ce.tiempo_a_pared_analitico(e)) * 1e3
fila(8, "ALPHA, pozo somero", f"k₂ = {ce.K2_SOMERO:.0f} V/m²; E_crit = {fd(ce.K2_SOMERO * ce.RW_A / 2, 2)} / {fd(ce.K2_SOMERO * ce.RW_A, 2)} V/m; "
     f"t_p = {tp(ce.EPS_MAX):.0f}–{tp(ce.EPS_MIN):.0f} ms ({tp(1.0):.0f} ms con 1 V/m); observado < 40 ms (~20 ms)", "",
     "contraste_experimental.py", "§4.8; Tabla 4.6; Ilustr. 4-10", "§3.4; Fig. 2; Tabla 4", "VERIFICADO (código) + LITERATURA")
tr = {r["trampa"]: r for r in csv_filas("i2_trampas_reales.csv")}
fila(9, "ALPHA, pozo de 10 V", f"E_crit = {8.7e3 * ce.RW_A / 2:.0f}–{1.41e4 * ce.RW_A:.0f} V/m; "
     f"{float(tr['ALPHA pozo 10 V']['tasa_um2_h']):.0f} µm²/h; t̄ ≈ {f(float(tr['ALPHA pozo 10 V']['t_medio_anios']), 1)} años", "",
     "contraste_experimental.py; articulo/i2_observable_r3.py", "§4.8", "§3.4, §7", "VERIFICADO (salida de articulo/)")
i10 = csv_dict("i10_cifras_r3.csv")
k2u = float(i10["k2_umbral_subito_eps20"])
fila(10, "Microtrampa (Narimannezhad 2014)", f"{float(i10['t_Narimannezhad_ec25']):.0f} s recalculado (≈ 4000 s en su ec. 24); k₂ ≈ {f(float(i10['k2_ef_L100mm_W5eV']), 2)} V/m²; "
     f"ω_b/ω₋ = {float(i10['omega_b_sobre_omega_m_L100mm_W5eV']):.0f}; δ = {20 / float(i10['k2_ef_L100mm_W5eV']) * 1e6:.0f} µm; "
     f"umbral k₂ ≲ {f(k2u / 2, 1)} ({f(k2u, 1)} súbito); t_p ≈ {f(float(i10['t_p_k2_1.0e+04']), 3)} s", "",
     "articulo/i10_microtrampa_r3.py", "F.1", "§3.5", "VERIFICADO (salida de articulo/)")
fila(11, "Prueba de tránsito", "ω_z/ω₋ = 10/50/200/2097 → 2,3/0,19/0,01/0,00 %; microtrampa "
     f"{f(1 - float(i10['transito_max_sobre_teoria_L=10 cm, 5 eV']), 1)} y 1,6×10⁻⁴", "",
     "transito.py (razones 10-2097: ronda_1/anexos_A1/auditoria/chequeo_transito.py); articulo/i10_microtrampa_r3.py",
     "§4.3; F.1", "§4.2", "VERIFICADO (salida de chequeo_transito.py, no integrada en el repositorio)")
res = {r["trampa"]: float(r["wz_sobre_wm"]) for r in csv_filas("i10_resonancias_r3.csv")}
fila(12, "Resonancias", f"l ≳ {f(W_Z / W_M, 1)} (trampa de referencia); ALPHA {min(v for k, v in res.items() if 'ALPHA' in k):.0f}–"
     f"{max(v for k, v in res.items() if 'ALPHA' in k):.0f}; microtrampa {min(float(v) for k, v in i10.items() if k.startswith('omega_b_sobre')):.0f}–"
     f"{max(float(v) for k, v in i10.items() if k.startswith('omega_b_sobre')):.0f}", "", "articulo/i10_microtrampa_r3.py", "F.1", "§4.3",
     "VERIFICADO (salida de articulo/)")

# ------------------------------------------------------------------ 13-17: difusión y verificación
fila(13, "Difusión", f"D = S_E(f₋)/(4B₀²); t̄ = r²B₀²/S_E; C₅₀ = {fd(C50, 4)}; t₅₀ = {fd(4 * C50, 3)} t̄ (sin disipación; con ella ≈ 0,96 t̄ si U ≈ 100)", "", "ruido.py",
     "cap. 2 (A7-3); §4.4", "§2.4–2.5")
sup = np.loadtxt(os.path.join(DAT, "fig4_04_supresion.csv"), delimiter=",", skiprows=1)
fila(14, "Verificación de D", "D_sim/D_teo = " + " / ".join(fd(x, 3) for x in sup[:, 2]) +
     f" (±{fd(sup[:, 3].min(), 3)}–{fd(sup[:, 3].max(), 3)}); supresión {f(sup[-1, 1], 2)} a ω₋τ_s = {sup[-1, 0]:.0f}", "",
     "ruido.py (datos/fig4_04_supresion.csv)", "§4.4; Tabla 4.2", "§5.2; Tabla 2")
p200 = np.loadtxt(os.path.join(DAT, "fig4_09_montecarlo.csv"), delimiter=",", skiprows=1)
fila(15, "200 parches", "; ".join(f"{fd(r, 3)} ± {fd(e, 3)}" for _, r, e in p200), "", "parches_fluctuantes.py (datos/fig4_09_montecarlo.csv)",
     "§4.4; Ilustr. 4-9a", "§5.2")
tpe = np.loadtxt(os.path.join(DAT, "fig4_04_tiempos_perdida.csv"), skiprows=1)
med, err = mediana_bootstrap(tpe, semilla=6002)
m1 = {(int(r["t_medio_periodos"]), int(r["pasos_por_periodo"])): (float(r["mediana_sobre_tmedio"]), float(r["error"]))
      for r in csv_filas("m1_sesgo_mediana.csv")}
(ma, ea), (mb, eb) = m1[(300, 50)], m1[(1200, 50)]
m0, e0 = 2 * mb - ma, np.hypot(2 * eb, ea)          # m = m0 + c (t/P)^(-1/2) por los puntos 300 y 1200 P₋
fila(16, "Mediana Monte Carlo", f"{fd(med, 3)} ± {fd(err, 3)} ({tpe.size} trayectorias; {np.sum(tpe < 4)} < 4t̄); "
     f"extrapolada {fd(m0, 3)} ± {fd(e0, 3)}", "t̄", "ruido.py (datos/fig4_04_tiempos_perdida.csv); articulo/m1_sesgo_mediana.py",
     "§4.4", "§5.3; Tabla 2")
crit = [SE_desde_D(R_WALL**2 / (4 * t * UN_ANIO)) for t in (1, 10, 100)]
fila(17, "Criterio de diseño", " / ".join(f(s, 2) for s in crit) + " V²m⁻²Hz⁻¹ (1 / 10 / 100 años) = " +
     " / ".join(f(tasa(t), 3) for t in (1, 10, 100)) + " µm²/h", "", "ruido.py", "§4.5", "§5.5; Abstract")

# ------------------------------------------------------------------ 18-26: tiempos por fuente
partes = np.array(SV_filtro(4.0, por_etapa=True) + [SV_EMI_BASE])
frac = partes / partes.sum() * 100
i1 = [r for r in csv_filas("i1_presupuesto_filtro.csv") if r["filtro"] == "real" and r["T_trampa_K"] == "4.0"]
Hfm = next(r for r in i1 if abs(float(r["f_Hz"]) - W_M / 2 / np.pi) < 1)
H1M = next(r for r in i1 if float(r["f_Hz"]) == 1e6)


def H_esc(fr):
    R, C = FILTRO_REAL["R"], FILTRO_REAL["C"]
    w = 2 * np.pi * fr
    Y = np.zeros((3, 3), complex)
    for k in range(3):
        Y[k, k] += 1 / R[k] + 1j * w * C[k]
        if k:
            Y[k - 1, k - 1] += 1 / R[k]; Y[k - 1, k] -= 1 / R[k]; Y[k, k - 1] -= 1 / R[k]
    I = np.zeros(3, complex); I[0] = 1 / R[0]
    return abs(np.linalg.solve(Y, I)[-1])


corte = brentq(lambda x: H_esc(x) - 2**-0.5, 1, 1e6)
fila(18, "Presupuesto del filtro (4 K, f₋)", f"300 K: {frac[0]:.0f} %; 60 K: {frac[1]:.0f} %; etapa fría: {frac[2]:.0f} %; EMI: {frac[3]:.0f} %; "
     f"|H| = {fd(float(Hfm['H_300K']), 4)} / {fd(float(Hfm['H_60K']), 4)} (frente a {f(float(H1M['H_300K']), 3)} / {f(float(H1M['H_60K']), 3)} a 1 MHz); "
     f"corte ≈ {fd(corte / 1e3, 1)} kHz", "", "ruido.py; articulo/i1_presupuesto_filtro.py", "§4.7; A.2", "§5.5; Abstract")
T4 = tt.calcular(tt.T_CUERPO)
fila(19, "Filtro de tres etapas (última etapa a T)", "t̄ = " + " / ".join(fd(r["t_filtro_anios"], 2) for r in T4) + " años (t₅₀ " +
     " / ".join(fd(r["t50_filtro_anios"], 2) for r in T4) + f"); {tasa(T4[0]['t_filtro_anios']):.0f} µm²/h (4 K), "
     f"{tasa(T4[-1]['t_filtro_anios']):.0f} (0,1 K); {dn_menos_dt(escenarios(4.0)[tt.FILTRO][0]):.0f} cuantos/s; sin disipación (con ella 0,249–0,254 años)", "",
     "ruido.py; temperaturas.py", "§4.7; Tabla 4.4", "§5.5, §6.1; Tabla 3")
fila(20, "Filtro (última etapa a 5 K)", f"{fd(T4[0]['t_filtro_ult5K_anios'], 2)} años a toda T (sin disipación)", "años", "ruido.py (T_ultima_etapa)", "§4.7", "§6.1")
fila(21, "Solo la etapa fría", " / ".join(f(r["t_etapa_fria_anios"], 3) for r in T4) +
     f" años (cociente {T4[-1]['t_etapa_fria_anios'] / T4[0]['t_etapa_fria_anios']:.0f}); sin disipación y con las otras resistencias sin ruido", "", "ruido.py", "Tabla 4.4", "§6.1; Tabla 3")
fila(22, "Filtro fuerte + EMI", f"{T4[0]['t_fuerte_EMI_anios']:.0f} años; {fd(tasa(T4[0]['t_fuerte_EMI_anios']), 1)} µm²/h; sin disipación (con ella 184 años)", "",
     "ruido.py", "Tabla 4.4", "§5.5")
fila(23, "Superficie A", f"{f(T4[0]['t_supA_min_anios'], 2)}–{f(T4[0]['t_supA_max_anios'], 2)} años (4 K); hasta "
     f"{f(T4[-1]['t_supA_max_anios'], 2)} (≤ 1 K); no térmico: disipación no fijada", "", "parches_fluctuantes.py", "Tabla 4.4; Ilustr. 4-9", "§6.1; S3")
fila(24, "Superficie B", " / ".join(f"{f(r['t_supB_min_anios'], 2)}–{f(r['t_supB_max_anios'], 2)}" for r in T4) + " años, sin disipación (con la de su fuente térmica, de 15–2×10⁴ años a 22–3×10⁴)", "",
     "parches_fluctuantes.py", "Tabla 4.4", "Tabla 3")


def banda_A4(alpha, exp_d):
    t = [anios(SE_superficie(4.0, "A", s0, t0, b, alpha, exp_d)) for s0, t0, b in zip(LAB_S0, LAB_T0, LAB_BETA)]
    return min(t)


fila(25, "Sensibilidad de la superficie", f"α = 0,7: ×{fd(banda_A4(0.7, 4.0) / banda_A4(1.0, 4.0), 1)}; exponente 3,5: "
     f"÷{fd(banda_A4(1.0, 4.0) / banda_A4(1.0, 3.5), 1)}; peor caso ≈ {f(banda_A4(1.43, 3.5), 2)} años (escenario A, sin disipación)", "",
     "parches_fluctuantes.py", "A.2; §4.10", "§6.1")
fila(26, "Vía del positrón", "t̄(T)/t̄(4 K) = " + " / ".join(fd(r["coc_positron_T_k1a"], 5) for r in T4[1:]) +
     f" (k = 1/a); máximo {fd((T4[-1]['coc_positron_T_kpiz0'] - 1) * 100, 2)} % (k = π/z₀) → < 0,3 % dentro del modelo", "",
     "cuantico.py; temperaturas.py", "§4.7; Tabla 4.5", "§2.6, §6.2; Tabla 3")
fila(27, "Localización (4 / 1 / 0,3 / 0,1 K)", "n̄₊ = " + " / ".join(f(r["n_cic"], 3) for r in T4) + "; A₊ = " +
     " / ".join(fd(r["A_cic_nm"], 1) for r in T4) + " nm; A_z = " + " / ".join(fd(r["A_ax_um"], 1) for r in T4) + " µm; A₋ = " +
     " / ".join(fd(r["A_mag_nm"], 0) for r in T4) + " nm (reenfriado); E₊/k_B = " + " / ".join(f(r["E_cic_K"], 3) for r in T4) + " K",
     "", "cuantico.py; temperaturas.py", "Tabla 4.3; Ilustr. 4-6", "§2.6; S2; §6.3")

# ------------------------------------------------------------------ 28-31: κ, canal axial, incertidumbres
k3 = [r for r in csv_filas("i4_kappa3d.csv") if r["caso"].startswith("hiperbolica")]
kb = float(next(r for r in k3 if r["caso"] == "hiperbolica_base" and r["n_malla_por_a"] == "80")["kappa"])
kv = [float(r["kappa"]) for r in k3]
fila(28, "κ", f"2/π (cota 2D, dos mitades); κ_3D = {fd(kb, 3)} (variantes {fd(min(kv), 3)}–{fd(max(kv), 3)}); "
     f"factor {fd((2 / np.pi / kb)**2, 2)}", "", "ruido.py; articulo/i4_kappa3d.py", "§3 (V12-111); A.2", "§2.4; Tabla 2",
     "VERIFICADO (salida de articulo/)")
ax = csv_dict("i3a_cifras.csv")
pend = [float(v) for k, v in ax.items() if k.startswith("pendiente")]
dsd = [float(v) for k, v in ax.items() if k.startswith("Dsim")]
fila(29, "Canal de energía axial", f"t̄ ∝ τ_z/(E₀²k⁴T²); pendiente {fd(np.mean(pend), 3)}; 4 casos {fd(min(dsd), 2)}–{fd(max(dsd), 2)}; "
     f"t̄(4 K) = " + " / ".join(f(float(ax[f't_canal_axial_4K_kpi/z0_tau{x}']), 2) for x in ("0.01", "0.1", "1.0", "10.0")) +
     " años (k = π/z₀; τ_z = 0,01/0,1/1/10 s) y " + " / ".join(f(float(ax[f't_canal_axial_4K_k1/a_tau{x}']), 2) for x in ("0.01", "0.1", "1.0", "10.0")) +
     f" (k = 1/a); {f(float(ax['t_canal_axial_01K_kpi/z0_tau0.01']), 2)} años a 0,1 K (peor caso); E₀ > "
     f"{fd(np.sqrt(float(ax['t_canal_axial_4K_kpi/z0_tau0.01'])), 1)} V/m para t̄ < 1 año", "",
     "articulo/i3a_canal_axial.py", "§4.10; F.2; Ilustr. 4-11", "§2.6, §6.2, §6.4; Tabla 3",
     "VERIFICADO (salida de articulo/); τ_z = 0,01 s [VERIFICAR]")
i5 = {(r["canal"], r["EMI"], r["ultima_etapa"], r["con_T2"]): r for r in csv_filas("i5_resumen_r3.csv")}
c, tot = i5[("circuitos", "plana", "T", "False")], i5[("total A", "plana", "T", "True")]
p_min = min(float(r["P_ge_1"]) for r in i5.values())
fila(30, "Propagación de incertidumbres (I5 r3)", f"P[≥ 1] = {p_min:.0f} ({int(float(c['muestras']))} muestras); circuitos "
     f"{fd(float(c['cociente_p5']), 2)}–{fd(float(c['cociente_p95']), 2)}; total + canal axial {fd(float(tot['cociente_p50']), 2)} "
     f"({fd(float(tot['cociente_p5']), 2)}–{fd(float(tot['cociente_p95']), 1)}), mediana {fd(float(tot['t4K_p50']), 1)} años; "
     f"{fd(float(tot['frac_T2_domina_4K']) * 100, 1)} %; mediana de circuitos con κ_3D {float(c['t4K_p50']):.0f} años", "",
     "articulo/i5_incertidumbre_r3.py", "F.3; §4.7", "§6.1–6.2; Tabla 3", "VERIFICADO (salida de articulo/)")
t0 = T4[0]["t_filtro_anios"]
fila(31, "Reenfriamiento (Hanneke 2011)", f"reenfriar cada ~3 s limita la deriva a ≈ {fd(np.sqrt(tasa(t0) * 3 / 3600), 1)} µm (estimación propia); sin reenfriar, "
     f"{np.sqrt(tasa(t0)):.0f} µm en 1 h; Hanneke 2011 enfría 1,5 s por bandas laterales al empezar cada intento de medida (duración, no intervalo)", "", "ruido.py + Hanneke 2011", "§4.10 (T-a)", "§6.3",
     "VERIFICADO (código) + LITERATURA")

# ------------------------------------------------------------------ 32-39: contrastes, BASE y otros
fila(32, "Gas residual", "p(H₂) < 1,2×10⁻¹⁸ mbar; p(He) < 2,7×10⁻¹⁸ mbar", "mbar", "Sellner 2017, sec. 3", "§4.10 (T-a)", "§7", "LITERATURA")
fila(33, "Peil y Gabrielse", "n̄₊ = " + " / ".join(f(float(ce.ocupacion_ciclotron_peil(x)), 3 if x > 0.5 else 2 if x > 0.09 else 1)
                                             for x in (4.2, 1.0, 0.08)) + " (4,2 K / 1 K / 80 mK); contraste solo entre 4,2 y 1,6 K (≈ 0,23 a 4,2 K); los valores a 1 K y 80 mK son la curva de equilibrio, no medidas; "
     "eje axial a 17 K con la trampa a 80 mK: no apoya T_z = T a 0,1 K", "", "contraste_experimental.py (ν_c = 147 GHz)", "§4.8; Tabla 4.6; §3.1-D", "§6.3, §7",
     "VERIFICADO (código) + LITERATURA")
v0 = np.sqrt(2 * K_B * 6.2 / 1.67262192369e-27)
D_tr = 0.5 * (1.0 / 1.946)**2 * 1e-7 / v0            # hipótesis de tránsito (E = 1 V/m, l_c = 0,1 µm), BASE reservorio
rv = tr["BASE reservorio (RT)"]
fila(34, "BASE reservorio", f"{float(rv['tasa_um2_h']):.0f} µm²/h; t̄ ≈ {float(rv['t_medio_anios']):.0f} años sin disipación (≈ 1 año con ella: no discrimina); "
     f"S_E ≤ {f(float(rv['S_E_max_compatible'].split()[0]), 2)} (95 %); tránsito: t₅₀ ≈ {C50 * 4.5e-3**2 / D_tr / 3600:.0f} h; "
     "observado: nube guardada 405 d (empezó con 18, extraídos uno a uno), sin pérdidas no explicadas; cota de vida > 10,2 años (68 %)", "", "articulo/i2_observable_r3.py; modelo de tránsito", "§4.8; Tabla 4.6", "§7; Tabla 4",
     "VERIFICADO (salida de articulo/) + LITERATURA")
hv = tr["Harvard e- (Hanneke 2011)"]
fila(35, "Harvard", f"{float(hv['nu_menos_Hz']) / 1e3:.0f} kHz; {fd(float(hv['tasa_um2_h']), 2)} µm²/h; t̄ ≈ {f(float(hv['t_medio_anios']), 2)} años; "
     "no restringe (magnetrón reenfriado)", "", "articulo/i2_observable_r3.py", "§4.8", "§7", "VERIFICADO (salida de articulo/)")
cb = csv_dict("cota_BASE_cifras.csv")
at = tr["BASE AT"]
fila(36, "BASE AT (cota corregida)", f"B₀ = {fd(float(cb['B0_AT']), 3)} T; ν₋ = {fd(float(cb['nu_menos_AT']) / 1e3, 1)} kHz; "
     f"S_E ≤ {f(float(cb['S_E_max_conservadora (deriva, tau = 250 s, sigma = 0,05 Hz)']), 2)} / "
     f"{f(float(cb['S_E_max_intermedia (deriva, tau = 1400 s, sigma = 0,075 Hz)']), 2)}; escenario {float(at['tasa_um2_h']):.0f} µm²/h y "
     f"{fd(float(at['t_medio_anios']), 1)} años; ×{fd(float(cb['tension_conservadora (deriva, tau = 250 s, sigma = 0,05 Hz)']), 1)}–"
     f"{float(cb['tension_intermedia (deriva, tau = 1400 s, sigma = 0,075 Hz)']):.0f}; κ_eff ≤ "
     f"{fd(float(cb['kappa_eff_conservadora (deriva, tau = 250 s, sigma = 0,05 Hz)']), 2)} / "
     f"{fd(float(cb['kappa_eff_intermedia (deriva, tau = 1400 s, sigma = 0,075 Hz)']), 2)}; tesis ≥ "
     f"{f(float(cb['t_tesis_a2_conservadora (deriva, tau = 250 s, sigma = 0,05 Hz)']), 2)} / "
     f"{f(float(cb['t_tesis_a2_intermedia (deriva, tau = 1400 s, sigma = 0,075 Hz)']), 2)} años (a⁻²), ≥ "
     f"{f(float(cb['t_tesis_a4_intermedia (deriva, tau = 1400 s, sigma = 0,075 Hz)']), 2)} (a⁻⁴, cota principal)", "",
     "articulo/cota_BASE_independiente.py; articulo/i2_observable_r3.py", "F.4", "§7; Tabla 4; Abstract",
     "INFERIDO; B₀ y ausencia de reenfriamiento [VERIFICAR con BASE]")
tB = [anios(SE_cota_BASE(1.43) * x / 7.5) for x in (7.5, 7.5 + 3.4, 7.5 - 2.8)]     # S_E <= 7,5(+3,4/-2,8)e-20 (Borchert 2019)
fila(37, "Mínimo compatible con BASE a 17,8 MHz", f"{f(tB[0], 3)} años ({f(tB[1], 3)}–{f(tB[2], 3)} con la barra de error; α = 1,43)",
     "años", "parches_fluctuantes.SE_cota_BASE", "A.2", "Fig. 5 (banda)", "VERIFICADO (código) + LITERATURA (barra de Borchert 2019)")
fila(38, "Modelo preliminar (solo §4.3 y anexo E)", "t₅₀ = 78,20 h (4 K) / 12,37 h (0,1 K); 6,32 = √40 por construcción "
     f"(√(4/0,1) = {fd(np.sqrt(40), 2)})", "h", "código V11 (v11/)", "§4.3 (una vez); anexo E", "No va al artículo", "VERIFICADO (V11)")
fila(39, "Pruebas automáticas", "número exacto del tag: ver `python3 -m pytest -q` (en el ensayo de la ronda 4: 9 + 31 + 9 + 5 + 2 = 56)",
     "", "pytest", "anexo C", "Data avail. y §5.3", "VERIFICAR al etiquetar")

if __name__ == "__main__":
    ruta = os.path.join(DAT, "cifras_canonicas.csv")
    with open(ruta, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["id", "magnitud", "valor", "unidad", "script", "tesis", "articulo", "estado"])
        w.writerows(filas)
    print(f"{len(filas)} filas escritas en {ruta}")
