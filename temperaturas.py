"""
temperaturas.py — Tablas de la tesis para las temperaturas analizadas (implementación de
referencia de la tarea WP1.6; base: ronda_1/anexos_A1/tabla_temperaturas.py).

    python3 temperaturas.py            # escribe datos/temperaturas_*.csv y tablas/tabla_*.tex (< 2 s)

Solo IMPORTA los módulos del modelo; no los modifica. Determinista (sin números aleatorios):
volver a ejecutarlo produce archivos idénticos bit a bit.

Tablas (convención: T del positrón = T de los electrodos = T de la última etapa del filtro,
salvo donde se indica):
  4.2  Localización: ocupaciones, amplitudes con punto cero y energías (n_- = n_z, magnetrón
       reenfriado por bandas laterales; cuantico.py).
  4.3  Permanencia: t_medio y mediana por fuente de ruido (ruido.py, parches_fluctuantes.py),
       con la variante de la última etapa del filtro a 5 K (D-T8).
  4.4  Separación de las dos temperaturas (2x2): positrón a T con electrodos a 4 K, y positrón
       a 4 K con electrodos a T (escenario del filtro de tres etapas).
  A.4  Barrido de 9 temperaturas: 0,05; 0,1; 0,2; 0,3; 0,5; 1; 2; 4; 10 K.
"""
import csv
import os

import numpy as np

from cuantico import amplitud, ocupacion, razon_tiempo_positron
from params import HBAR, K_B, R_WALL, UN_ANIO, W_1, W_M, W_P, W_Z, Z0
from parches_fluctuantes import banda_superficie
from ruido import C50, D_desde_SE, escenarios, tiempo_medio

_DIR = os.path.dirname(os.path.abspath(__file__))
T_CUERPO = (4.0, 1.0, 0.3, 0.1)
T_ANEXO = (0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 4.0, 10.0)
FILTRO = "filtro real (corte 1,2 kHz)"          # clave interna del escenario del filtro de tres etapas
FRIA = "solo la etapa fría, filtro real"
FUERTE = "filtro fuerte (corte 12 Hz) + EMI"
K_PI_Z0 = Z0 / np.pi                            # escala a tal que k = 1/a = pi/z0


def anios(S_E):
    return tiempo_medio(D_desde_SE(np.asarray(S_E, dtype=float))) / UN_ANIO


def fila(T):
    """Todas las magnitudes de las tablas para una temperatura T (K)."""
    n_p, n_z = float(ocupacion(T, W_P)), float(ocupacion(T, W_Z))
    e_T, e_5 = escenarios(T), escenarios(T, T_ultima_etapa=5.0)
    e_4 = escenarios(4.0)
    A_lo, A_hi = banda_superficie(np.array([T]), "A")
    B_lo, B_hi = banda_superficie(np.array([T]), "B")
    t_filtro = float(anios(e_T[FILTRO])[0])
    return {
        "T_K": T,
        "n_cic": n_p, "n_ax": n_z, "n_mag": n_z,
        "A_cic_nm": float(amplitud(n_p, W_1)) * 1e9,
        "A_ax_um": float(amplitud(n_z, W_Z)) * 1e6,
        "A_mag_nm": float(amplitud(n_z, W_1)) * 1e9,
        "E_cic_K": HBAR * W_P * (n_p + 0.5) / K_B,
        "E_ax_K": HBAR * W_Z * (n_z + 0.5) / K_B,
        "E_mag_K": -HBAR * W_M * (n_z + 0.5) / K_B,
        "t_filtro_anios": t_filtro,
        "t50_filtro_anios": 4 * C50 * t_filtro,
        "t_filtro_ult5K_anios": float(anios(e_5[FILTRO])[0]),
        "t_etapa_fria_anios": float(anios(e_T[FRIA])[0]),
        "t_fuerte_EMI_anios": float(anios(e_T[FUERTE])[0]),
        "t_supA_min_anios": float(anios(A_hi)[0]), "t_supA_max_anios": float(anios(A_lo)[0]),
        "t_supB_min_anios": float(anios(B_hi)[0]), "t_supB_max_anios": float(anios(B_lo)[0]),
        # Tabla 4.4: cocientes respecto de todo a 4 K (escenario del filtro de tres etapas)
        "coc_positron_T_k1a": float(razon_tiempo_positron(T, 4.0, R_WALL)),
        "coc_positron_T_kpiz0": float(razon_tiempo_positron(T, 4.0, K_PI_Z0)),
        "coc_electrodos_T": t_filtro / float(anios(e_4[FILTRO])[0]),
    }


def calcular(temperaturas):
    return [fila(float(T)) for T in temperaturas]


# ---------------------------------------------------------------- formato
def g(x, cif=3):
    """Número con coma decimal; notación científica fuera de [0,01; 10^4)."""
    if x == 0:
        return "0"
    if 1e-2 <= abs(x) < 1e4:
        s = f"{x:.{cif}g}"
        if "e" not in s:
            return s.replace(".", "{,}")
    m, e = f"{x:.{cif - 1}e}".split("e")
    return f"{m.replace('.', '{,}')}\\times10^{{{int(e)}}}"


def m(x, cif=3):
    return f"${g(x, cif)}$"


def fijo(x, dec):
    """Número con un número fijo de decimales y coma decimal (amplitudes de la Tabla 4.2)."""
    return f"${x:.{dec}f}$".replace(".", "{,}")


def rango(a, b):
    return f"${g(a, 2)}$--${g(b, 2)}$"


def tabular(columnas, encabezado, filas_tex):
    lineas = [r"\begin{tabular}{" + columnas + "}", r"\toprule", encabezado + r" \\", r"\midrule"]
    lineas += [f + r" \\" for f in filas_tex]
    lineas += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lineas)


def tex_4_2(filas):
    enc = (r"$T$ (K) & $\bar n_+$ & $\bar n_z=\bar n_-$ & $A_+$ (nm) & $A_z$ (µm) & $A_-$ (nm)"
           r" & $E_+/k_B$ (K) & $E_-/k_B$ (K)")
    cuerpo = [" & ".join([m(f["T_K"]), m(f["n_cic"]), m(f["n_ax"], 4), fijo(f["A_cic_nm"], 1),
                          fijo(f["A_ax_um"], 1), fijo(f["A_mag_nm"], 0), m(f["E_cic_K"]), m(f["E_mag_K"])])
              for f in filas]
    return tabular("rrrrrrrr", enc, cuerpo)


def tex_4_3(filas):
    enc = (r"$T$ (K) & tres etapas, últ. a $T$: $\bar t$ ($t_{50}$) & últ. a 5 K & solo etapa fría"
           r" & 12 Hz + EMI & superficie A & superficie B")
    cuerpo = [" & ".join([m(f["T_K"]), f"{m(f['t_filtro_anios'])} ({m(f['t50_filtro_anios'])})",
                          m(f["t_filtro_ult5K_anios"]), m(f["t_etapa_fria_anios"]), m(f["t_fuerte_EMI_anios"]),
                          rango(f["t_supA_min_anios"], f["t_supA_max_anios"]),
                          rango(f["t_supB_min_anios"], f["t_supB_max_anios"])])
              for f in filas]
    return tabular("rrrrrrr", enc, cuerpo)


def tex_4_4(filas):
    enc = (r"$T$ (K) & positrón a $T$, electrodos a 4 K ($k=1/a$) & ídem ($k=\pi/z_0$)"
           r" & positrón a 4 K, electrodos a $T$")
    cuerpo = [" & ".join([m(f["T_K"]), f"${f['coc_positron_T_k1a']:.5f}$".replace(".", "{,}"),
                          f"${f['coc_positron_T_kpiz0']:.4f}$".replace(".", "{,}"),
                          f"${f['coc_electrodos_T']:.3f}$".replace(".", "{,}")])
              for f in filas]
    return tabular("rrrr", enc, cuerpo)


def tex_A_4(filas):
    enc = (r"$T$ (K) & $\bar n_+$ & $A_+$ (nm) & $A_z$ (µm) & $\bar t$ tres etapas (años)"
           r" & solo etapa fría & superficie B & $\bar t(T)/\bar t(4\,\mathrm{K})$ positrón")
    cuerpo = [" & ".join([m(f["T_K"]), m(f["n_cic"]), m(f["A_cic_nm"]), m(f["A_ax_um"]),
                          m(f["t_filtro_anios"]), m(f["t_etapa_fria_anios"]),
                          rango(f["t_supB_min_anios"], f["t_supB_max_anios"]),
                          f"${f['coc_positron_T_k1a']:.5f}$".replace(".", "{,}")])
              for f in filas]
    return tabular("rrrrrrrr", enc, cuerpo)


def guardar_csv(filas, ruta):
    with open(ruta, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]), lineterminator="\n")
        w.writeheader()
        for f in filas:
            w.writerow({k: f"{v:.6e}" for k, v in f.items()})


def escribir(carpeta_datos=None, carpeta_tablas=None):
    carpeta_datos = carpeta_datos or os.path.join(_DIR, "datos")
    carpeta_tablas = carpeta_tablas or os.path.join(_DIR, "tablas")
    os.makedirs(carpeta_datos, exist_ok=True)
    os.makedirs(carpeta_tablas, exist_ok=True)
    cuerpo, anexo = calcular(T_CUERPO), calcular(T_ANEXO)
    guardar_csv(cuerpo, os.path.join(carpeta_datos, "temperaturas_cuerpo.csv"))
    guardar_csv(anexo, os.path.join(carpeta_datos, "temperaturas_anexo_A4.csv"))
    for nombre, texto in (("tabla_4_2.tex", tex_4_2(cuerpo)), ("tabla_4_3.tex", tex_4_3(cuerpo)),
                          ("tabla_4_4.tex", tex_4_4(cuerpo[1:])), ("tabla_A_4.tex", tex_A_4(anexo))):
        with open(os.path.join(carpeta_tablas, nombre), "w") as fh:
            fh.write(texto)
    return cuerpo, anexo


if __name__ == "__main__":
    cuerpo, _ = escribir()
    for f in cuerpo:
        print(f"T = {f['T_K']:g} K: n+ = {f['n_cic']:.3g}, A+ = {f['A_cic_nm']:.1f} nm, A_z = {f['A_ax_um']:.1f} µm,"
              f" t(tres etapas) = {f['t_filtro_anios']:.2f} a (última a 5 K: {f['t_filtro_ult5K_anios']:.2f} a),"
              f" solo etapa fría = {f['t_etapa_fria_anios']:.3g} a")
    print("Escritos datos/temperaturas_*.csv y tablas/tabla_{4_2,4_3,4_4,A_4}.tex")
