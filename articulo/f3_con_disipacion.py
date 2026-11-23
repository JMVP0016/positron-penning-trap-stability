"""f3_con_disipacion.py — Propagación del anexo F.3 con disipación (circuitos: filtro de tres etapas + EMI plana).
Uso: python3 -I f3_con_disipacion.py RUTA_COPIA_CODIGO [carpeta_salida] [N]
Semilla fija 20261031 y el mismo orden de muestreo que i5_incertidumbre_r3.py (alpha, exponente de distancia,
trampa de referencia, EMI, capacidades, kappa), así que cada combinación es la misma que la del caso sin
disipación. Para cada una se calcula el cociente t(0,1 K)/t(4 K) sin disipación y con disipación
(t = Ein(U)/(2 lambda), anexo F.5), con la última etapa del filtro a la temperatura de la trampa o a 5 K.
Las distribuciones son independientes entre sí: no se supone correlación entre parámetros.
Salida: f3_con_disipacion.csv (percentiles 5, 50 y 95, mínimo y fracción con cociente >= 1)."""
import copy
import csv
import os
import sys

import numpy as np

sys.path.insert(0, sys.argv[1])
from params import UN_ANIO                                           # noqa: E402
from ruido import SV_filtro, FILTRO_REAL, SV_EMI_BASE, SE_desde_SV, D_desde_SE, tiempo_medio   # noqa: E402
from disipacion import Z_nodo, lambda_disipacion, tiempo_medio_con_disipacion                 # noqa: E402

out = sys.argv[2] if len(sys.argv) > 2 else "."
N = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
rng = np.random.default_rng(20261031)


def logu(a, b, n):
    return np.exp(rng.uniform(np.log(a), np.log(b), n))


# mismo orden que i5_incertidumbre_r3.py; alpha, expd y trap no intervienen en los circuitos, pero se
# muestrean igual para que las combinaciones coincidan una a una
alpha = logu(0.6, 1.5, N)
expd = rng.uniform(3.5, 4.0, N)
trap = rng.integers(0, 8, N)
emi = logu(0.1, 10, N)
cf = logu(0.5, 10, N)
kap = logu(0.51, 0.54, N)

filas = []
for ultima in ("T", "5K"):
    q_sin, q_con, t4c = [], [], []
    for j in range(N):
        filt = copy.deepcopy(FILTRO_REAL)
        filt["C"] = tuple(x * cf[j] for x in filt["C"])
        lam = lambda_disipacion(Z_nodo(filt).real, kappa=kap[j])
        t = {}
        for T in (4.0, 0.1):
            SV = SV_filtro(T if ultima == "T" else 5.0, filt) + emi[j] * SV_EMI_BASE
            D = D_desde_SE(SE_desde_SV(SV, kappa=kap[j]))
            t[T] = (tiempo_medio(D), tiempo_medio_con_disipacion(D, lam))
        q_sin.append(t[0.1][0] / t[4.0][0])
        q_con.append(t[0.1][1] / t[4.0][1])
        t4c.append(t[4.0][1] / UN_ANIO)
    for nombre, x in (("cociente sin disipación", np.array(q_sin)), ("cociente con disipación", np.array(q_con)),
                      ("t(4 K) con disipación, años", np.array(t4c))):
        p5, p50, p95 = np.percentile(x, [5, 50, 95])
        fila = dict(ultima_etapa=ultima, magnitud=nombre, p5=p5, mediana=p50, p95=p95, minimo=x.min(),
                    fraccion_mayor_o_igual_1=float(np.mean(x >= 1 - 1e-12)) if "cociente" in nombre else float("nan"))
        filas.append(fila)
        print(f"última etapa {ultima}: {nombre}: mediana {p50:.4f}; 90 %: {p5:.4f}–{p95:.4f}; mínimo {x.min():.4f}")

os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "f3_con_disipacion.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(filas[0]))
    w.writeheader()
    for fila in filas:
        w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in fila.items()})
print(f"N = {N}; guardado {os.path.join(out, 'f3_con_disipacion.csv')}")
