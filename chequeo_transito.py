"""¿Depende la conclusión de transito.py de w_z/w_- = 50? Repite la prueba con otras razones."""
import sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
import numpy as np
import transito as tr
from params import W_Z as WZ_REAL, W_M as WM_REAL

razon_real = WZ_REAL / WM_REAL
print(f"razón real w_z/w_- = {razon_real:.0f}")
for razon, n_tray, n_per in ((10, 1000, 40), (50, 1000, 40), (200, 600, 20), (round(razon_real), 150, 4)):
    tr.W_Z = float(razon)
    t0 = time.time()
    t, R2, th = tr.parche_estatico_muestreado(n_tray=n_tray, n_periodos=n_per, semilla=2026)
    P = 2 * np.pi / tr.W_M
    # crecimiento: compara el valor en múltiplos enteros del periodo magnetrón
    idx = [np.argmin(abs(t - k * P)) for k in range(1, n_per + 1)]
    print(f"w_z/w_- = {razon:5d}: max<|Z|^2> = {R2.max():.4f}; teoría = {th.max():.4f} (dif {abs(R2.max()/th.max()-1)*100:.2f} %);"
          f" <|Z|^2> en t = k P_- (k=1..{n_per}): máx {max(R2[idx]):.2e} (acotado si << máx); <|G0|^2> = {th.max()/4:.4f}; {time.time()-t0:.0f} s")
tr.W_Z = 50.0
# hipótesis de tránsito con distintas razones: la pendiente teórica se reproduce
for razon in (10, 50, 200):
    tr.W_Z = float(razon)
    t2, R2b, th2, pend, err = tr.ou_de_transito(n_tray=2000, n_periodos=20, semilla=4052)
    print(f"OU tránsito w_z/w_- = {razon}: pendiente sim / 4D = {pend/(th2[-1]/t2[-1]):.3f} ± {err/(th2[-1]/t2[-1]):.3f}")
