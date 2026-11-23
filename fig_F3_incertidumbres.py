"""fig_F3_incertidumbres.py — Figura del anexo F.3 de la tesis (propagación de incertidumbres).
Mismo cálculo y misma semilla (20261031) que articulo/i5_incertidumbre_r3.py; solo cambian el estilo
(estilo.mplstyle, 15 cm), los colores (paleta de figuras.py) y los rótulos, con la convención de la tesis.
Uso:  python3 fig_F3_incertidumbres.py [N]        # figures/figF3_incertidumbres.pdf y .png (unos 15 s)
      FIG_IDIOMA=en FIG_CARPETA=figures/en python3 fig_F3_incertidumbres.py"""
import os, sys, csv, copy
from math import factorial
import numpy as np
RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RAIZ)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use(os.path.join(RAIZ, "estilo.mplstyle")); plt.rcParams.update({"figure.constrained_layout.use": False})
IDIOMA = os.environ.get("FIG_IDIOMA", "es")          # "es" para la tesis (anexo F.3), "en" para el artículo
_ES = {"total A + T2, flat EMI": "circuitos (última etapa a $T$) + superficie A\n+ canal axial, EMI plana",
       "total B + T2, flat EMI": "total B + canal axial, EMI plana",
       "total A + T2, 1/f EMI": "total A + canal axial, EMI 1/f",
       "total A, last stage 5 K, no T2": "total A, última etapa a 5 K, sin canal axial",
       "axial channel T2 only": "solo el canal de energía axial",
       "circuits (last stage at T), flat EMI": "circuitos (última etapa a $T$),\nEMI plana",
       "surface B": "superficie B",
       "total A + T2, last stage 5 K": "circuitos (última etapa a 5 K) + superficie A\n+ canal axial, EMI plana",
       "(a) single channels": "(a) canales por separado", "(b) totals": "(b) totales",
       "system temperature $T$ (K)": "temperatura del sistema $T$ (K)",
       r"$\bar t(T)/\bar t$(4 K) (median, 5-95 %)": r"$\bar t(T)/\bar t$(4 K) (mediana, 5-95 %)",
       r"$\bar t$(0.1 K) / $\bar t$(4 K)": r"$\bar t$(0,1 K) / $\bar t$(4 K)", "samples": "muestras"}
def L(texto):
    return _ES.get(texto, texto) if IDIOMA == "es" else texto
from params import B0, R_WALL, UN_ANIO, Z0, F_M, K_B, M_E, W_Z
from ruido import SV_filtro, FILTRO_REAL, SV_EMI_BASE, SE_desde_SV, D_desde_SE, tiempo_medio, escenarios
from parches_fluctuantes import SE_superficie, banda_superficie, LAB_S0, LAB_T0, LAB_BETA
from cuantico import razon_tiempo_positron

out = os.path.join(RAIZ, os.environ.get("FIG_CARPETA", "figures"))
os.makedirs(out, exist_ok=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
rng = np.random.default_rng(20261031)
T_HI, T_LO = 4.0, 0.1
F_EMI = 17.845e6
anios = lambda S: tiempo_medio(D_desde_SE(S)) / UN_ANIO

# ---- canal T2 (copia literal de i3a_canal_axial.py) ----
def coef_laguerre(beta, nmax=40):
    b = beta**2 / 4
    return np.array([b**n * np.exp(-b) / factorial(n) for n in range(nmax + 1)])
def S_G(f, beta, tau, E0, nmax=40):
    a = coef_laguerre(beta, nmax); n = np.arange(1, nmax + 1)
    return E0**2 * np.sum(a[1:] ** 2 * 4 * (tau / n) / (1 + (2 * np.pi * f * tau / n) ** 2))
def beta_de(T, k):
    return k * np.sqrt(2 * K_B * T / (M_E * W_Z**2))
def SE_equiv_T2(T, k, tau, E0):
    """S_E isótropo equivalente: t = 2 r^2 B^2/S_G = r^2 B^2/S_eq  ->  S_eq = S_G/2."""
    return S_G(F_M, beta_de(T, k), tau, E0) / 2

# ---- pruebas previas ----
t_def = anios(escenarios(T_HI)["filtro real (corte 1,2 kHz)"])[0]
lo, hi = banda_superficie(np.array([T_HI]), "A")
t_T2 = R_WALL**2 * B0**2 / SE_equiv_T2(T_HI, np.pi / Z0, 0.01, 1.0) / UN_ANIO
print(f"Prueba: filtro real 4 K = {t_def:.3f} a (4,48); sup. A 4 K = {anios(hi)[0]:.0f}-{anios(lo)[0]:.3g} a (858-1,29e6)")
print(f"Prueba: T2 a 4 K con k = pi/z0, tau_z = 0,01 s, E0 = 1 V/m: {t_T2:.2f} a (i3a: 1,2)")
assert abs(t_def - 4.48) < 0.01 and abs(anios(hi)[0] - 858) < 2 and abs(t_T2 - 1.2) < 0.1

def logu(a, b, n): return np.exp(rng.uniform(np.log(a), np.log(b), n))
alpha = logu(0.6, 1.5, N); expd = rng.uniform(3.5, 4.0, N); trap = rng.integers(0, 8, N)
emi = logu(0.1, 10, N); cf = logu(0.5, 10, N); kap = logu(0.51, 0.54, N)
E0 = logu(0.1, 2.0, N); kk = logu(1 / R_WALL, np.pi / Z0, N); tz = logu(0.01, 10.0, N)

def circ(T, j, ultima, emi_var):
    filt = copy.deepcopy(FILTRO_REAL); filt["C"] = tuple(x * cf[j] for x in filt["C"])
    SV = SV_filtro(T if ultima == "T" else 5.0, filt)
    e = emi[j] * SV_EMI_BASE * (F_EMI / F_M if emi_var == "1/f" else 1.0)
    return SE_desde_SV(SV + e, kappa=kap[j])

S = {}
for T in (T_HI, T_LO):
    for ul in ("T", "5K"):
        for ev in ("plana", "1/f"):
            S[("circ", ul, ev, T)] = np.array([circ(T, j, ul, ev) for j in range(N)])
    for esc in ("A", "B"):
        S[(esc, T)] = np.array([float(SE_superficie(T, esc, LAB_S0[trap[j]], LAB_T0[trap[j]], LAB_BETA[trap[j]], alpha[j], expd[j])) for j in range(N)])
    S[("T2", T)] = np.array([SE_equiv_T2(T, kk[j], tz[j], E0[j]) for j in range(N)])

pos = float(razon_tiempo_positron(T_LO, T_HI, Z0 / np.pi))     # peor caso del canal J0
filas = [["canal", "EMI", "ultima_etapa", "con_T2", "muestras", "t4K_p5", "t4K_p50", "t4K_p95", "t01K_p5", "t01K_p50",
          "t01K_p95", "cociente_p5", "cociente_p50", "cociente_p95", "cociente_min", "P_ge_1", "P_ge_1_con_J0_kpiz0",
          "frac_T2_domina_4K"]]
cocientes = {}
def registra(nombre, ev, ul, conT2, s4, s01):
    t4, t01 = anios(s4), anios(s01); q = t01 / t4
    p = lambda x, k: np.percentile(x, k)
    dom = np.mean(S[("T2", T_HI)] > 0.5 * s4) if conT2 else 0.0
    filas.append([nombre, ev, ul, conT2, N, *[f"{p(t4,k):.4g}" for k in (5, 50, 95)], *[f"{p(t01,k):.4g}" for k in (5, 50, 95)],
                  *[f"{p(q,k):.5g}" for k in (5, 50, 95)], f"{q.min():.6g}", f"{np.mean(q >= 1 - 1e-12):.4f}",
                  f"{np.mean(q * pos >= 1 - 1e-12):.4f}", f"{dom:.3f}"])
    cocientes[(nombre, ev, ul, conT2)] = q
    return q
for ev in ("plana", "1/f"):
    for ul in ("T", "5K"):
        c4, c01 = S[("circ", ul, ev, T_HI)], S[("circ", ul, ev, T_LO)]
        registra("circuitos", ev, ul, False, c4, c01)
        for esc in ("A", "B"):
            registra(f"total {esc}", ev, ul, False, c4 + S[(esc, T_HI)], c01 + S[(esc, T_LO)])
            registra(f"total {esc}", ev, ul, True, c4 + S[(esc, T_HI)] + S[("T2", T_HI)], c01 + S[(esc, T_LO)] + S[("T2", T_LO)])
for esc in ("A", "B"):
    registra(f"superficie {esc}", "-", "-", False, S[(esc, T_HI)], S[(esc, T_LO)])
registra("T2 solo", "-", "-", True, S[("T2", T_HI)], S[("T2", T_LO)])
for f in filas[1:]:
    print(f"{f[0]:12s} EMI={f[1]:5s} ult={f[2]:2s} T2={str(f[3]):5s} t4K p50={f[6]:>9s} a  coc p5/50/95={f[11]}/{f[12]}/{f[13]} min={f[14]} P>=1={f[15]} (con J0 {f[16]}) T2domina={f[17]}")
# kappa se cancela
q1 = circ(T_LO, 0, "T", "plana") / circ(T_HI, 0, "T", "plana"); kap0 = kap[0]; kap[0] = 0.2
q2 = circ(T_LO, 0, "T", "plana") / circ(T_HI, 0, "T", "plana"); kap[0] = kap0
print(f"kappa se cancela: {q1:.12f} vs {q2:.12f}")

# ---- figura del anexo F.3: cociente t(T)/t(4 K) frente a T (mediana y percentiles 5-95) ----
Tg = np.logspace(np.log10(0.05), 1, 25); M = 1500; idx = np.arange(M)
def banda(func):
    r = np.array([[func(T, j) for j in idx] for T in Tg]); r4 = np.array([func(T_HI, j) for j in idx])
    q = (1 / r) / (1 / r4)          # t(T)/t(4 K) = S(4 K)/S(T)
    return np.percentile(q, [5, 50, 95], axis=1)
canal = {"circuits (last stage at T), flat EMI": lambda T, j: circ(T, j, "T", "plana"),
         "surface B": lambda T, j: float(SE_superficie(T, "B", LAB_S0[trap[j]], LAB_T0[trap[j]], LAB_BETA[trap[j]], alpha[j], expd[j])),
         "total A + T2, flat EMI": lambda T, j: circ(T, j, "T", "plana") + float(SE_superficie(T, "A", LAB_S0[trap[j]], LAB_T0[trap[j]], LAB_BETA[trap[j]], alpha[j], expd[j])) + SE_equiv_T2(T, kk[j], tz[j], E0[j]),
         "total A + T2, last stage 5 K": lambda T, j: circ(T, j, "5K", "plana") + float(SE_superficie(T, "A", LAB_S0[trap[j]], LAB_T0[trap[j]], LAB_BETA[trap[j]], alpha[j], expd[j])) + SE_equiv_T2(T, kk[j], tz[j], E0[j])}
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from figuras_tesis import marcar_temperaturas
_ES.update({"mean-time ratio\n" r"$\bar t(T)/\bar t(4\,\mathrm{K})$": "cociente de tiempos medios\n" r"$\bar t(T)/\bar t(4\,\mathrm{K})$",
            r"ratio = 1 ($\bar t$ as at 4 K)": r"cociente = 1 ($\bar t$ igual que a 4 K)",
            "line: median; band: 5th–95th percentiles": "línea: mediana; franja: percentiles 5–95",
            "single channels": "canales por separado", "totals": "totales"})
fig, axs = plt.subplots(1, 2, figsize=(15 / 2.54, 9.0 / 2.54), layout="constrained")
hand = {0: [], 1: []}; labs = {0: [], 1: []}
for (nom, fn), c, ls in zip(canal.items(), ("#1baf7a", "#8a8983", "#1baf7a", "#8a8983"), ("-", "--", "-", "--")):
    b = banda(fn); k = 0 if "total" not in nom else 1; ax = axs[k]
    ax.fill_between(Tg, b[0], b[2], color=c, alpha=0.18, lw=0); ax.loglog(Tg, b[1], color=c, ls=ls)
    hand[k].append((Patch(facecolor=c, alpha=0.18, lw=0), Line2D([], [], color=c, ls=ls))); labs[k].append(L(nom))
for k, (ax, tt) in enumerate(zip(axs, ("(a)", "(b)"))):
    ax.axhline(1, color="0.4", lw=0.8, ls=":")
    hand[k].append(Line2D([], [], color="0.4", lw=0.8, ls=":")); labs[k].append(L(r"ratio = 1 ($\bar t$ as at 4 K)"))
    marcar_temperaturas(ax)
    ax.set_xlabel(L("system temperature $T$ (K)")); ax.set_ylabel(L("mean-time ratio\n" r"$\bar t(T)/\bar t(4\,\mathrm{K})$"), fontsize=9)
    ax.set_title(tt, loc="left", fontsize=10)
axs[0].legend(hand[0], labs[0], fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.3), frameon=False, title=L("single channels") + " (" + L("line: median; band: 5th–95th percentiles") + ")", title_fontsize=7)
axs[1].legend(hand[1], labs[1], fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.3), frameon=False, title=L("totals"), title_fontsize=7)
fig.savefig(os.path.join(out, "figF3_incertidumbres.pdf")); fig.savefig(os.path.join(out, "figF3_incertidumbres.png"))
print("Figura guardada:", os.path.join(out, "figF3_incertidumbres.pdf"))
