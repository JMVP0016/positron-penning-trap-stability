"""WP1.9 de Henry: contraste estático ALPHA y equilibrio ciclotrón de Peil.

Colocar junto a params.py y centro_guia.py. No genera archivos al importarse.
FIG_IDIOMA=es|en; FIG_CARPETA=carpeta de figuras relativa a este archivo,
o absoluta. Los CSV se guardan en datos/ junto a este archivo.

Entradas documentales: Baker 2024, según A3 y el plan §6.1 del paquete;
Peil y Gabrielse 1999: frecuencia publicada de 147 GHz. No se digitalizan
datos experimentales ni se valida aquí la permanencia frente a temperatura.
"""
from pathlib import Path
import os

import numpy as np

import centro_guia as cg
from params import HBAR, K_B, R_WALL, U0

_DIR = Path(__file__).resolve().parent
B_A, RW_A = 1.0, 0.022
EPS_MIN, EPS_MAX = 0.35, 2.0
K2_SOMERO = 10.0
K2_10V = (8.7e3, 1.41e4)
NU_C_PEIL = 147e9


def _positivo(nombre, valor):
    if not np.isfinite(valor) or valor <= 0:
        raise ValueError(f"{nombre} debe ser positivo y finito")


def tiempo_a_pared(eps, k2, B=B_A, Rw=RW_A, pasos=4000, n_per=2.0):
    """Primer cruce numérico de la pared (s), desde Z=0 y campo constante.

    nan significa que no se detectó cruce en la ventana simulada. El cruce
    se vigila al final de cada paso: sobreestima el tiempo hasta un paso.
    Restaura los tres globales del integrador incluso si ocurre una excepción.
    La reparametrización temporal no es apta para llamadas concurrentes.
    """
    for nombre, valor in (("k2", k2), ("B", B), ("Rw", Rw), ("n_per", n_per)):
        _positivo(nombre, valor)
    if not np.isfinite(eps) or eps < 0:
        raise ValueError("eps debe ser no negativo y finito")
    if isinstance(pasos, bool) or not isinstance(pasos, (int, np.integer)) or pasos < 4:
        raise ValueError("pasos debe ser un entero mayor o igual a 4")
    anterior = cg.U0, cg.B0, cg.W_GC
    try:
        cg.U0, cg.B0, cg.W_GC = k2, B, k2 / B
        dt = 2 * np.pi / cg.W_GC / pasos
        Z, E = 0j, complex(eps)
        for n in range(1, int(np.floor(n_per * pasos)) + 1):
            Z = cg.paso_centro_guia(Z, E, E, dt)
            if abs(Z) >= Rw:
                return n * dt
        return float("nan")
    finally:
        cg.U0, cg.B0, cg.W_GC = anterior


def tiempo_a_pared_analitico(eps, k2=K2_SOMERO, B=B_A, Rw=RW_A):
    """Primer cruce exacto: (2B/k2) asin(k2 Rw / 2 eps), o nan si no alcanza.

    El campo aparece súbitamente. No es la condición de carga adiabática.
    """
    for nombre, valor in (("k2", k2), ("B", B), ("Rw", Rw)):
        _positivo(nombre, valor)
    e = np.asarray(eps, dtype=float)
    if np.any(~np.isfinite(e)) or np.any(e < 0):
        raise ValueError("eps debe ser no negativo y finito")
    umbral = k2 * Rw / 2
    cociente = np.divide(umbral, e, out=np.ones_like(e), where=e > 0)
    return np.where(e >= umbral, 2 * B / k2 * np.arcsin(np.minimum(1, cociente)), np.nan)


def ocupacion_ciclotron_peil(T, nu_c=NU_C_PEIL):
    """Bose-Einstein con 147 GHz, no con B=5,3 T redondeado.

    El valor a 4,2 K es contrastable con la ocupación publicada. Por debajo
    de 1 K la salida es una predicción de equilibrio, no una medida de Peil.
    """
    _positivo("nu_c", nu_c)
    t = np.asarray(T, dtype=float)
    if np.any(~np.isfinite(t)) or np.any(t <= 0):
        raise ValueError("T debe ser positiva y finita")
    with np.errstate(over="ignore"):
        return 1 / np.expm1(HBAR * 2 * np.pi * nu_c / (K_B * t))


def figura(idioma=None, carpeta=None):
    """Genera fig4_10_alpha en PDF/PNG y sus datos. Devuelve rutas absolutas.

    El CSV conserva el nombre usado por el paquete; ambos idiomas producen
    los mismos datos. Sólo cambian los rótulos y la carpeta de las figuras.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    idioma = idioma or os.environ.get("FIG_IDIOMA", "es")
    if idioma not in ("es", "en"):
        raise ValueError("FIG_IDIOMA debe ser es o en")
    destino = Path(carpeta or os.environ.get("FIG_CARPETA", "figures"))
    destino = (destino if destino.is_absolute() else _DIR / destino).resolve()
    datos = _DIR / "datos"
    destino.mkdir(parents=True, exist_ok=True)
    datos.mkdir(parents=True, exist_ok=True)

    def L(es, en):
        return en if idioma == "en" else es

    campos = np.unique(np.r_[np.geomspace(0.15, 4, 65), EPS_MIN, 1.0, EPS_MAX])
    numerico = np.array([tiempo_a_pared(e, K2_SOMERO) for e in campos]) * 1e3
    exacto = tiempo_a_pared_analitico(campos) * 1e3
    csv = (datos / "fig4_10_alpha.csv").resolve()
    np.savetxt(csv, np.c_[campos, numerico, exacto], delimiter=",",
               header="eps_V_m,t_integrador_ms,t_analitico_ms", comments="", fmt="%.9e")

    import figuras                      # estilo y paleta comunes de la tesis (importados aquí: importar
    from figuras import COLOR, tamano   # este módulo no debe cargar matplotlib.pyplot)
    estilo = os.path.join(figuras._DIR, "estilo.mplstyle")
    with plt.style.context(estilo), plt.rc_context({"axes.spines.top": False, "axes.spines.right": False}):
        fig, (a, b) = plt.subplots(1, 2, figsize=tamano(2, 8.8))
        k = np.geomspace(1, 1e5, 200)
        a.axhspan(EPS_MIN, EPS_MAX, color=COLOR["secundario"], alpha=0.14,
                  label=L("Campo de parche medido en ALPHA: 0,35–2 V/m", "Patch field measured in ALPHA: 0.35–2 V/m"))
        for radio, color, nombre in ((RW_A, COLOR["teoria"], "ALPHA"),
                                      (R_WALL, COLOR["serie3"], L("Trampa simulada", "Simulated trap"))):
            a.loglog(k, k * radio / 2, color=color, label=nombre + L(": súbito", ": sudden"))
            a.loglog(k, k * radio, "--", color=color, label=nombre + L(": lento", ": slow"))
        # Marcadores: umbral súbito de cada trampa, con su propia leyenda (abajo a la derecha)
        marcas = [
            a.plot(K2_SOMERO, K2_SOMERO * RW_A / 2, "o", color=COLOR["teoria"], ms=4,
                   label=L(r"ALPHA, pozo somero ($k_2 \approx 10$ V/m$^2$): 0,11 V/m",
                           r"ALPHA, shallow well ($k_2 \approx 10$ V/m$^2$): 0.11 V/m"))[0],
            a.plot(K2_10V, np.array(K2_10V) * RW_A / 2, "s", color=COLOR["teoria"], ms=4, ls="none",
                   label=L("ALPHA, pozo de 10 V: 96–155 V/m", "ALPHA, 10 V well: 96–155 V/m"))[0],
            a.plot(U0, U0 * R_WALL / 2, "^", color=COLOR["serie3"], ms=6,
                   label=L(r"Trampa simulada ($k_2 = U_0$): 50 V/m", r"Simulated trap ($k_2 = U_0$): 50 V/m"))[0],
        ]
        a.set(xlabel=L(r"Curvatura $k_2$ (V/m$^2$)", r"Curvature $k_2$ (V/m$^2$)"),
              ylabel=L("Campo crítico (V/m)", "Critical field (V/m)"), ylim=(0.002, 1e4),
              title=L("(a) Umbral del modelo lineal", "(a) Linear-model threshold"))
        # leyenda del panel (a) fuera de los paneles, como en la figura 4-7: franja, rectas y umbrales
        from matplotlib.patches import Rectangle
        h, lab = a.get_legend_handles_labels()
        lineas = [(x, y) for x, y in zip(h, lab) if x not in marcas]
        hueco = (Rectangle((0, 0), 1, 1, visible=False), " ")      # columna 2 con 3 filas
        orden = lineas[:3] + lineas[3:] + [hueco] + [(m, m.get_label()) for m in marcas]
        fig.legend(*zip(*orden), loc="outside lower center", ncol=3, fontsize=7,
                   frameon=False, columnspacing=1.2, handlelength=2.2,
                   title=L("Panel (a): los marcadores son el umbral súbito de cada trampa",
                           "Panel (a): markers are the sudden threshold of each trap"), title_fontsize=7)
        b.axvspan(EPS_MIN, EPS_MAX, color=COLOR["secundario"], alpha=0.14,
                  label=L("Campo medido en ALPHA", "Field measured in ALPHA"))
        b.loglog(campos, exacto, color=COLOR["teoria"], lw=1.5,
                 label=L("Solución analítica", "Analytical solution"))
        b.loglog(campos[::4], numerico[::4], "o", mfc="white", color=COLOR["serie3"], ms=3.5,
                 label=L("Integrador del centro guía", "Guiding-centre integrator"))
        b.loglog(campos, B_A * RW_A / campos * 1e3, "--", color=COLOR["secundario"],
                 label=r"$BR_w/\varepsilon$")
        b.axhline(40, ls=":", color=COLOR["secundario"], lw=1)
        b.text(0.97, 0.97, L("Referencia publicada: < 40 ms\n(no son puntos digitalizados)",
                             "Published reference: < 40 ms\n(not digitized data points)"),
               transform=b.transAxes, va="top", ha="right", fontsize=7, color=COLOR["secundario"])
        b.set(xlabel=L(r"Campo del parche $\varepsilon$ (V/m)", r"Patch field $\varepsilon$ (V/m)"),
              ylabel=L("Tiempo hasta la pared (ms)", "Time to the wall (ms)"),
              ylim=(3, 200), title=L("(b) ALPHA: pozo somero", "(b) ALPHA: shallow well"))
        b.set_xticks([0.2, 0.5, 1, 2, 4])
        b.set_xticklabels([L("0,2", "0.2"), L("0,5", "0.5"), "1", "2", "4"])
        b.xaxis.set_minor_formatter(NullFormatter())
        b.legend(fontsize=7, loc="lower left", frameon=False)
        rutas = []
        try:
            for ext in ("pdf", "png"):
                ruta = destino / f"fig4_10_alpha.{ext}"
                fig.savefig(ruta)
                rutas.append(ruta)
        finally:
            plt.close(fig)
    return (*rutas, csv)


if __name__ == "__main__":
    print(f"ALPHA: integrador = {tiempo_a_pared(1, K2_SOMERO)*1e3:.4f} ms; "
          f"analítico = {float(tiempo_a_pared_analitico(1))*1e3:.4f} ms")
    print(f"Peil: n(4,2 K; 147 GHz) = {float(ocupacion_ciclotron_peil(4.2)):.6f}")
    for ruta in figura():
        print(ruta)
