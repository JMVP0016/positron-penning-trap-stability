"""
i4_kappa3d.py — A1, ronda 2, tarea A1-3 (I4): factor de acoplamiento dipolar kappa en 3D.

Componente azimutal m = 1 del potencial: phi = f(r, z) cos(theta), con
    f_rr + f_r/r - f/r^2 + f_zz = 0.
Un anillo partido en dos mitades (una a V y la otra a 0) tiene coeficiente m = 1 igual a (2/pi) V
sobre el anillo. Por tanto: f = (2/pi) V en el anillo, f = 0 en las tapas, f = 0 en la caja exterior
a tierra y f(0, z) = 0 en el eje.
kappa_3D = a * (df/dr)(0, 0) / V, con a = rho0 (cilindro largo: kappa = 2/pi).

Geometría (params.py): anillo rho^2/2 - z^2 = rho0^2/2 (rho0 = 5 mm) y tapas z^2 - rho^2/2 = z0^2
(z0 = 6,12 mm). Los electrodos son sólidos, truncados en |z| <= h_anillo (anillo) y r <= w_tapa (tapas).
El hueco queda entre el borde del anillo y el de la tapa. Simetría z -> -z (Neumann en z = 0).
Diferencias finitas de segundo orden en una malla uniforme, resueltas con scipy.sparse.

Uso: python3 i4_kappa3d.py [--codigo RUTA] [--salida CARPETA]   (determinista, < 3 min)
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
import scipy.sparse as sp  # noqa: E402
from scipy.sparse.linalg import spsolve  # noqa: E402

from params import RHO0, Z0  # noqa: E402

A = RHO0


def resolver(mascara_dir, valor_dir, R, Zt, h, neumann_arriba=False):
    """Resuelve el problema m = 1 en [0, R] x [0, Zt] con paso h.
    mascara_dir(r, z) -> bool: puntos de electrodo (Dirichlet); valor_dir(r, z): su valor.
    Bordes: r = 0 -> f = 0; r = R -> f = 0 (salvo que la máscara diga otra cosa);
    z = 0 -> Neumann; z = Zt -> f = 0, o Neumann si neumann_arriba. Devuelve kappa = a f_r(0,0)."""
    nr, nz = int(round(R / h)) + 1, int(round(Zt / h)) + 1
    r = np.arange(nr) * h
    z = np.arange(nz) * h
    RR, ZZ = np.meshgrid(r, z, indexing="ij")
    fijo = mascara_dir(RR, ZZ)
    val = np.where(fijo, valor_dir(RR, ZZ), 0.0)
    fijo[0, :] = True; val[0, :] = 0.0                      # eje
    borde_r = ~fijo[-1, :]
    fijo[-1, borde_r] = True; val[-1, borde_r] = 0.0         # pared exterior a tierra
    if not neumann_arriba:
        borde = ~fijo[:, -1]
        fijo[borde, -1] = True; val[borde, -1] = 0.0
    idx = -np.ones((nr, nz), dtype=int)
    libres = np.argwhere(~fijo)
    idx[~fijo] = np.arange(len(libres))
    filas, cols, datos = [], [], []
    b = np.zeros(len(libres))
    for k, (i, j) in enumerate(libres):
        ri = r[i]
        cE, cW = 1 / h**2 + 1 / (2 * h * ri), 1 / h**2 - 1 / (2 * h * ri)
        cN = cS = 1 / h**2
        diag = -4 / h**2 - 1 / ri**2
        vec = [(i + 1, j, cE), (i - 1, j, cW)]
        # z: Neumann en j = 0 (punto fantasma j=-1 igual a j=1) y en j = nz-1 si neumann_arriba
        jn = j + 1 if j + 1 < nz else j - 1
        js = j - 1 if j - 1 >= 0 else j + 1
        vec += [(i, jn, cN), (i, js, cS)]
        filas.append(k); cols.append(k); datos.append(diag)
        for (ii, jj, c) in vec:
            if fijo[ii, jj]:
                b[k] -= c * val[ii, jj]
            else:
                filas.append(k); cols.append(idx[ii, jj]); datos.append(c)
    M = sp.csr_matrix((datos, (filas, cols)), shape=(len(libres), len(libres)))
    sol = spsolve(M, b)
    f = val.copy()
    f[~fijo] = sol
    # pendiente en el origen: f/r = c + d r^2 -> Richardson con r1 = h, r2 = 2h
    c = (4 * f[1, 0] / h - f[2, 0] / (2 * h)) / 3
    return c * A, f


def kappa_cilindro(h, L=8.0):
    """Validación: cilindro de radio a, mitades a lo largo de toda la longitud (L*a, Neumann arriba)."""
    m = lambda r, z: r >= A - 1e-12
    v = lambda r, z: np.full_like(r, 2 / np.pi)
    return resolver(m, v, A, L * A, h, neumann_arriba=True)[0]


def kappa_cilindro_finito(h, H=1.0, nmax=400):
    """Validación no trivial: cilindro r = a, |z| <= H*a, mitades a V y tapas planas a 0 en z = ±H a.
    Analítico: f = (2/pi) V sum_n (4/pi)(-1)^n/(2n+1) cos(k_n z) I1(k_n r)/I1(k_n a), k_n = (2n+1) pi/(2 H a).
    Devuelve (kappa numérico, kappa analítico)."""
    from scipy.special import ive
    m = lambda r, z: (r >= A - 1e-12) | (z >= H * A - 1e-12)
    v = lambda r, z: np.where(z >= H * A - 1e-12, 0.0, 2 / np.pi)
    kn = (2 * np.arange(nmax) + 1) * np.pi / (2 * H * A)
    an = (4 / np.pi) * (-1.0) ** np.arange(nmax) / (2 * np.arange(nmax) + 1)
    # I1(k a) = ive(1, k a) e^{k a};  f_r(0,0) = sum an (k/2)/I1(k a)
    pend = (2 / np.pi) * np.sum(an * (kn / 2) * np.exp(-kn * A) / ive(1, kn * A))
    return resolver(m, v, A, H * A, h)[0], pend * A


def kappa_hiperbolica(h, h_anillo=None, w_tapa=None, R=None, Zt=None):
    h_anillo = h_anillo if h_anillo is not None else 0.8 * Z0
    w_tapa = w_tapa if w_tapa is not None else 0.8 * RHO0
    R = R if R is not None else 4 * A
    Zt = Zt if Zt is not None else 4 * A
    anillo = lambda r, z: (r**2 / 2 - z**2 >= RHO0**2 / 2) & (z <= h_anillo)
    tapa = lambda r, z: (z**2 - r**2 / 2 >= Z0**2) & (r <= w_tapa)
    m = lambda r, z: anillo(r, z) | tapa(r, z)
    v = lambda r, z: np.where(anillo(r, z), 2 / np.pi, 0.0)
    return resolver(m, v, R, Zt, h)[0]


if __name__ == "__main__":
    filas = []
    print("1) Validación: cilindro largo con mitades (esperado 2/pi = 0,63662)")
    for n in (20, 40, 80):
        k = kappa_cilindro(A / n)
        print(f"   h = a/{n}: kappa = {k:.5f}  (error {abs(k/(2/np.pi)-1)*100:.2f} %)")
        filas.append(("cilindro_largo", n, "", "", "", "", k))
    print("1b) Validación no trivial: cilindro finito con tapas planas a tierra (solución analítica)")
    for H in (0.6, 1.2):
        for n in (20, 40, 80):
            k, ka = kappa_cilindro_finito(A / n, H)
            print(f"   H = {H} a, h = a/{n}: kappa = {k:.5f}; analítico = {ka:.5f} (error {abs(k/ka-1)*100:.2f} %)")
            filas.append((f"cilindro_finito_H{H}", n, "", "", "", "", k))
    print("2) Trampa hiperbólica (rho0 = 5 mm, z0 = 6,12 mm). Convergencia de la malla (caso base)")
    base = {}
    for n in (20, 40, 80):
        k = kappa_hiperbolica(A / n)
        base[n] = k
        print(f"   h = a/{n}: kappa_3D = {k:.5f}")
        filas.append(("hiperbolica_base", n, 0.8 * Z0, 0.8 * RHO0, 4 * A, 4 * A, k))
    print("3) Sensibilidad al truncamiento y al hueco (h = a/40)")
    for ha in (0.5, 0.8, 1.2, 2.0):
        for wt in (0.5, 0.8, 1.2):
            k = kappa_hiperbolica(A / 40, h_anillo=ha * Z0, w_tapa=wt * RHO0)
            print(f"   anillo |z| <= {ha:.1f} z0, tapa r <= {wt:.1f} rho0: kappa_3D = {k:.5f}")
            filas.append(("hiperbolica_trunc", 40, ha * Z0, wt * RHO0, 4 * A, 4 * A, k))
    for Rb in (3, 6):
        k = kappa_hiperbolica(A / 40, R=Rb * A, Zt=Rb * A)
        print(f"   caja exterior {Rb} a: kappa_3D = {k:.5f}")
        filas.append(("hiperbolica_caja", 40, 0.8 * Z0, 0.8 * RHO0, Rb * A, Rb * A, k))
    kref = base[80]
    print(f"4) kappa_3D (caso base, h = a/80) = {kref:.4f};  (2/pi)/kappa_3D = {(2/np.pi)/kref:.3f};"
          f"  factor en los tiempos de circuitos (2/pi / kappa_3D)^2 = {((2/np.pi)/kref)**2:.3f}")
    with open(os.path.join(args.salida, "i4_kappa3d.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["caso", "n_malla_por_a", "h_anillo_m", "w_tapa_m", "R_caja_m", "Z_caja_m", "kappa"])
        for f in filas:
            w.writerow([f[0], f[1]] + [f"{x:.4e}" if x != "" else "" for x in f[2:6]] + [f"{f[6]:.5f}"])
