"""
m1_sesgo_mediana.py — A1, ronda 3, tarea A1-4 (R3-2e): sesgo de la mediana del Monte Carlo de pérdida.

ruido.simular_perdida (copia sin modificar) comprueba la pared solo al final de cada paso, y el ruido
tiene memoria tau = 1/w_- (w_- tau = 1). Hay dos fuentes de sesgo positivo:
  (i) monitoreo discreto: escala como sqrt(dt);
  (ii) capa límite por la memoria del ruido: escala como sqrt(tau/t_medio).
Aquí se varían los pasos por periodo (50, 200, 500) y t_medio (300 y 1200 periodos), con N = 16000 (corrida principal; la de N = 2000 queda como registro)
y semillas fijas, y se compara la mediana con 0,802 t_medio.

Uso: python3 m1_sesgo_mediana.py [--codigo RUTA] [--salida CARPETA]   (determinista, ~5-8 min)
"""
import argparse
import csv
import os
import sys
import time

sys.dont_write_bytecode = True
ap = argparse.ArgumentParser()
ap.add_argument("--codigo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap.add_argument("--salida", default=os.path.dirname(os.path.abspath(__file__)))
args = ap.parse_args()
sys.path.insert(0, args.codigo)

import numpy as np  # noqa: E402

from ruido import C50, mediana_bootstrap, simular_perdida  # noqa: E402

N = 16000
casos = [(300, 25), (300, 50), (300, 200), (1200, 50)]
filas = []
for i, (tm, pp) in enumerate(casos):
    t0 = time.time()
    tp, t_med = simular_perdida(n_tray=N, t_medio_periodos=tm, pasos_por_periodo=pp, semilla=8100 + i)
    med, err = mediana_bootstrap(tp, semilla=8200 + i)
    r, e = med / t_med, err / t_med
    filas.append((tm, pp, r, e, (r - 4 * C50) / e, time.time() - t0))
    print(f"t_medio = {tm:5d} P_-, {pp:4d} pasos/P_-: mediana/t_medio = {r:.4f} ± {e:.4f}"
          f"  (analítica {4*C50:.4f}; desviación {(r-4*C50)/e:+.1f} sigma; {time.time()-t0:.0f} s)", flush=True)
with open(os.path.join(args.salida, "m1_sesgo_mediana.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["t_medio_periodos", "pasos_por_periodo", "mediana_sobre_tmedio", "error", "desviacion_sigma", "segundos"])
    for f in filas:
        w.writerow([f[0], f[1], f"{f[2]:.5f}", f"{f[3]:.5f}", f"{f[4]:.2f}", f"{f[5]:.0f}"])
