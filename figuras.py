"""
figuras.py — Utilidades comunes para las figuras de la tesis.

    from figuras import usar_estilo, guardar, COLOR
    usar_estilo()                  # figuras para la tesis (fondo blanco)
    usar_estilo(oscuro=True)       # figuras para las diapositivas (fondo negro)
    ... (dibujar) ...
    guardar(fig, "fig4_xx_nombre")     # crea figures/fig4_xx_nombre.pdf y .png

COLOR asigna un color fijo a cada entidad: 4 K siempre azul y 0,1 K siempre rojo vino,
en la tesis y en las diapositivas. Los colores pasan la comprobación para daltonismo.
IOP pide que la información no dependa solo del color (la figura debe leerse impresa en
gris), así que cada entidad lleva además su tipo de línea y su marcador:
    4 K: línea continua, círculos   0,1 K: línea discontinua, cuadrados   teoría: punteada
Papel de cada color en TODA la tesis (también en contraste_experimental.py y fig_disipacion.py). Las figuras
nuevas usan solo estas claves, nunca códigos #rrggbb sueltos, para que un color no cambie de significado:
    COLOR["4K"]          azul    4 K, solo cuando la figura compara temperaturas
    COLOR["01K"]         vino    0,1 K, ídem
    COLOR["positron"]    morado  la vía de la temperatura del positrón (fig4_07_canales_temperatura, panel b = Ilustr. 4-8 b)
    COLOR["serie3"]      verde   la simulación o el escenario principal de la tesis (simulación, Monte Carlo,
                                 trampa simulada, filtro de tres etapas)
    COLOR["teoria"]      negro   teoría, solución analítica o referencia experimental (p. ej. ALPHA)
    COLOR["secundario"]  gris    variantes y referencias auxiliares (estimaciones, franjas, líneas guía)
    ax.plot(x, y, color=COLOR["4K"], **LINEA["4K"])
Tamaños de IOP: tamano(2) = 15 cm (dos columnas, por defecto), tamano(1) = 8,5 cm.
"""
import os

import matplotlib.pyplot as plt

_DIR = os.path.dirname(os.path.abspath(__file__))

_CLARO = {"4K": "#2a78d6", "01K": "#9b1b30", "serie3": "#1baf7a",
          "teoria": "#0b0b0b", "secundario": "#8a8983", "positron": "#7b4fa6"}
_OSCURO = {"4K": "#3987e5", "01K": "#d4506a", "serie3": "#199e70",
           "teoria": "#ffffff", "secundario": "#c3c2b7", "positron": "#a98bd6"}

LINEA = {"4K": {"ls": "-", "marker": "o"},
         "01K": {"ls": "--", "marker": "s"},
         "serie3": {"ls": "-.", "marker": "^"},
         "teoria": {"ls": ":", "lw": 1.2, "marker": None}}

CM = 1 / 2.54


def tamano(columnas=2, alto_cm=None):
    """figsize en pulgadas para el ancho de IOP: 15 cm (2 columnas) u 8,5 cm (1 columna)."""
    ancho = 15.0 if columnas == 2 else 8.5
    alto = alto_cm if alto_cm else ancho * 0.62
    return (ancho * CM, alto * CM)


COLOR = dict(_CLARO)
COLOR["serie1"], COLOR["serie2"] = COLOR["4K"], COLOR["01K"]


def usar_estilo(oscuro=False):
    """Carga estilo.mplstyle. Con oscuro=True adapta colores y fondo a diapositivas negras."""
    plt.style.use(os.path.join(_DIR, "estilo.mplstyle"))
    COLOR.clear()
    COLOR.update(_OSCURO if oscuro else _CLARO)
    COLOR["serie1"], COLOR["serie2"] = COLOR["4K"], COLOR["01K"]
    if oscuro:
        plt.rcParams.update({
            "figure.facecolor": "#0b0b0b", "axes.facecolor": "#0b0b0b",
            "savefig.facecolor": "#0b0b0b", "text.color": "#ffffff",
            "axes.labelcolor": "#ffffff", "axes.edgecolor": "#c3c2b7",
            "xtick.color": "#c3c2b7", "ytick.color": "#c3c2b7", "grid.color": "#2c2c2a",
            "axes.prop_cycle": plt.cycler(color=[COLOR["4K"], COLOR["01K"], COLOR["serie3"]]),
        })


def guardar(fig, nombre, carpeta="figures"):
    """Guarda la figura en PDF (para LaTeX) y en PNG (para revisarla o ponerla en diapositivas)."""
    ruta = os.path.join(_DIR, carpeta)
    os.makedirs(ruta, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(ruta, f"{nombre}.{ext}"))
    print(f"Figura guardada: {carpeta}/{nombre}.pdf y .png")
