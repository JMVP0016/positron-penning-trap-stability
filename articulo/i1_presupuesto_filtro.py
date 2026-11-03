"""I1 (ronda 2): presupuesto del ruido por etapa del filtro RC de 3 etapas a f- y en frecuencia.
Uso: python3 -I i1_presupuesto_filtro.py RUTA_COPIA_CODIGO [carpeta_salida]
Reproduce ruido.SV_filtro (por_etapa=True) y además devuelve |H_i(f)|, la transferencia de cada
fuente de Johnson (en serie con R_i) hasta el electrodo. Determinista."""
import os, sys, csv
import numpy as np
sys.path.insert(0, sys.argv[1])
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from params import K_B, F_M
from ruido import SV_filtro, FILTRO_REAL, FILTRO_FUERTE, SV_EMI_BASE

out = sys.argv[2] if len(sys.argv) > 2 else "."

def transferencias(f, filtro=FILTRO_REAL):
    """Copia de la lógica de ruido.SV_filtro: |H_i| de la fuente en serie con R_i al electrodo."""
    R, C = filtro["R"], filtro["C"]; n = len(R); w = 2 * np.pi * f
    Y = np.zeros((n, n), dtype=complex)
    for k in range(n):
        g = 1 / R[k]; Y[k, k] += g + 1j * w * C[k]
        if k > 0:
            Y[k-1, k-1] += g; Y[k-1, k] -= g; Y[k, k-1] -= g
    H = []
    for i in range(n):
        I = np.zeros(n, dtype=complex); I[i] += 1 / R[i]
        if i > 0: I[i-1] -= 1 / R[i]
        H.append(abs(np.linalg.solve(Y, I)[-1]))
    return np.array(H)

def partes(T, f, filtro=FILTRO_REAL):
    Ts = list(filtro["T_calientes"]) + [T]
    H = transferencias(f, filtro)
    return np.array([4 * K_B * Ts[i] * filtro["R"][i] * H[i]**2 for i in range(3)]), H

# comprobación contra el código
for T in (4.0, 0.1):
    p, _ = partes(T, F_M)
    ref = np.array(SV_filtro(T, FILTRO_REAL, f=F_M, por_etapa=True))
    assert np.allclose(p, ref, rtol=1e-12), "no reproduce SV_filtro"
print("Reproduce ruido.SV_filtro(por_etapa=True) a 4 K y 0,1 K: OK")

filas = [["filtro", "T_trampa_K", "f_Hz", "H_300K", "H_60K", "H_fria", "S_V_300K", "S_V_60K", "S_V_fria",
          "S_V_EMI", "frac_300K_con_EMI", "frac_60K_con_EMI", "frac_fria_con_EMI", "frac_EMI"]]
freqs = [1e3, F_M, 1e4, 1e5, 1e6]
for nombre, filtro in (("real", FILTRO_REAL), ("fuerte", FILTRO_FUERTE)):
    for T in (4.0, 0.1):
        for f in freqs:
            p, H = partes(T, f, filtro)
            tot = p.sum() + SV_EMI_BASE
            filas.append([nombre, T, f, *[f"{h:.4g}" for h in H], *[f"{x:.4g}" for x in p], f"{SV_EMI_BASE:.4g}",
                          *[f"{x/tot:.4f}" for x in p], f"{SV_EMI_BASE/tot:.4f}"])
            if nombre == "real":
                fr = p / tot
                print(f"T={T:4.1f} K f={f:9.1f} Hz |H|=({H[0]:.3g},{H[1]:.3g},{H[2]:.3g}) "
                      f"frac 300/60/fria/EMI = {fr[0]*100:5.1f}/{fr[1]*100:5.1f}/{fr[2]*100:5.1f}/{SV_EMI_BASE/tot*100:4.1f} %"
                      f"   (sin EMI: {p[0]/p.sum()*100:.1f}/{p[1]/p.sum()*100:.1f}/{p[2]/p.sum()*100:.1f})")
with open(os.path.join(out, "i1_presupuesto_filtro.csv"), "w", newline="") as fh:
    csv.writer(fh).writerows(filas)

# figura: |H_i(f)| y S_V por etapa a 4 K y 0,1 K (filtro real)
f = np.logspace(2, 7, 400)
HH = np.array([transferencias(x) for x in f])
fig, (a, b) = plt.subplots(1, 2, figsize=(9.0, 3.6))
for i, (lab, ls) in enumerate((("stage 1 (300 K)", "-"), ("stage 2 (60 K)", "--"), ("stage 3 (trap T)", ":"))):
    a.loglog(f, HH[:, i], ls=ls, color="k", label=lab)
a.axvline(F_M, color="0.5", lw=0.8); a.text(F_M*1.1, 2e-4, r"$f_-$ = 6.37 kHz", fontsize=8)
a.axvspan(5e6, 1e7, color="0.85", lw=0); a.text(1.1e6, 1e-6, "MHz\n(axial)", fontsize=8)
a.set_xlabel("frequency (Hz)"); a.set_ylabel(r"$|H_i(f)|$ to electrode"); a.legend(fontsize=8); a.set_title("(a)", loc="left")
for T, ls in ((4.0, "-"), (0.1, "--")):
    P = np.array([partes(T, x)[0] for x in f])
    for i, c in enumerate(("k", "0.45", "tab:green")):
        b.loglog(f, P[:, i], ls=ls, color=c, label=(["300 K", "60 K", "cold stage"][i] + f", {T:g} K") if True else None)
b.axhline(SV_EMI_BASE, color="tab:red", lw=0.9, ls="-."); b.text(150, SV_EMI_BASE*1.4, "EMI (BASE, order of magnitude)", fontsize=7, color="tab:red")
b.axvline(F_M, color="0.5", lw=0.8)
b.set_xlabel("frequency (Hz)"); b.set_ylabel(r"$S_V$ per stage at electrode (V$^2$/Hz)"); b.legend(fontsize=6.5, ncol=2); b.set_title("(b)", loc="left")
fig.tight_layout(); fig.savefig(os.path.join(out, "i1_presupuesto_filtro.pdf")); print("figura y CSV en", out)
