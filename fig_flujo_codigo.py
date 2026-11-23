"""
fig_flujo_codigo.py — Diagrama de flujo clásico del cálculo (capítulo 3, junto a la tabla de módulos).

Uso:
    python3 fig_flujo_codigo.py

Crea figures/met_01_flujo_codigo.pdf y .png (Ilustración 3-1 de la tesis).

Símbolos (norma ISO 5807): óvalo = inicio o fin; paralelogramo = entrada o salida de datos;
rectángulo = cálculo; rombo = decisión. Se lee de arriba abajo; cada símbolo tiene una sola
entrada por arriba o por la izquierda y sale por abajo, salvo los rombos, que tienen dos salidas
rotuladas «Sí» y «No». A la derecha de cada paso va el módulo que lo realiza; aparecen los
15 módulos de la tabla de módulos de la tesis.

Correspondencia con el código (etiqueta v2.0.0):
  params.py                 datos y frecuencias (se calculan una vez; no dependen de T)
  superficie.py             E_centro_rms: campo de los parches que llega al centro
  centro_guia.py            centro_desplazado (delta) y E_critico (50 V/m súbito, 100 V/m lento)
  temperaturas.py           calcular(): «for T in temperaturas» -> fila(T)
  cuantico.py               ocupacion, amplitud
  ruido.py                  escenarios(T), D_desde_SE, tiempo_medio, C50 (mediana)
  parches_fluctuantes.py    banda_superficie(T)
  disipacion.py             escenarios_con_disipacion(T)
El rombo del campo crítico es la comparación del modelo (cifras_canonicas.py, filas 5 y 6): con
los parámetros de la tesis la respuesta es siempre «No» (el umbral es al menos 25 veces el campo
medido), así que la rama «Sí» describe el caso límite y no se recorre en los resultados.

Estilo: el de las demás figuras (estilo.mplstyle, IOP / Physica Scripta): 15 cm de ancho, texto de
8 pt o más al tamaño final, trazos negros (el color de «teoría» de la paleta) sobre fondo blanco.
No se usan los colores de la paleta porque en la tesis cada uno ya significa algo (4 K, 0,1 K...).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon, Rectangle

AQUI = os.path.dirname(os.path.abspath(__file__))
plt.style.use(os.path.join(AQUI, "estilo.mplstyle"))
plt.rcParams.update({"figure.constrained_layout.use": False, "axes.grid": False})

CM = 1 / 2.54
ANCHO, ALTO = 15.0, 19.3
TINTA = "#0b0b0b"           # COLOR["teoria"] de figuras.py
SUAVE = "#efeee9"           # relleno de inicio/fin y decisiones (se distingue también impreso en gris)
XC, W = 6.55, 6.0           # centro y ancho de la columna principal
XM = 10.25                  # donde empiezan los nombres de los módulos
FS, FM = 8.6, 8.0           # texto de los pasos y nombres de módulos (pt)
LW = 1.0

fig = plt.figure(figsize=(ANCHO * CM, ALTO * CM))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, ANCHO)
ax.set_ylim(0, ALTO)
ax.axis("off")


def est(relleno="white"):
    return dict(fc=relleno, ec=TINTA, lw=LW, zorder=3, joinstyle="miter")


def texto(x, y, s, **k):
    ax.text(x, y, s, fontsize=k.pop("fontsize", FS), ha=k.pop("ha", "center"), va="center", color=TINTA,
            linespacing=1.15, zorder=4, **k)


def modulo(y, s):
    ax.text(XM, y, s, family="monospace", fontsize=FM, ha="left", va="center", color=TINTA, linespacing=1.2, zorder=4)


def ovalo(x, y, w, h, s):
    ax.add_patch(Ellipse((x, y), w, h, **est(SUAVE))); texto(x, y, s)


def proceso(x, y, w, h, s):
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, **est())); texto(x, y, s)


def dato(x, y, w, h, s, inc=0.28):
    ax.add_patch(Polygon([(x - w / 2 + inc, y + h / 2), (x + w / 2 + inc, y + h / 2), (x + w / 2 - inc, y - h / 2),
                          (x - w / 2 - inc, y - h / 2)], closed=True, **est())); texto(x, y, s)


def decision(x, y, w, h, s):
    ax.add_patch(Polygon([(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)], closed=True, **est(SUAVE)))
    texto(x, y, s)


def flecha(p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=LW, color=TINTA, shrinkA=0, shrinkB=0,
                                 zorder=5))


def linea(*pts):
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=TINTA, lw=LW, solid_capstyle="projecting",
            solid_joinstyle="miter", zorder=5)


H, G = 0.9, 0.42                     # alto de un cálculo y separación entre símbolos
WD = W * 0.94                        # ancho de los rombos
y = ALTO - 0.45
pasos = []                           # (nombre, y_centro, alto)


def sig(nombre, h):
    global y
    c = y - h / 2
    pasos.append((nombre, c, h))
    y -= h + G
    return c


c = sig("inicio", 0.65); ovalo(XC, c, 2.5, 0.65, "Inicio")
ax.text(XM, c, "Módulo que lo realiza", fontsize=FS, ha="left", va="center", color=TINTA, style="italic")
c = sig("datos", H);     dato(XC, c, W, H, "Datos: $B_0$, $V_0$, $d$, parámetros del parche\ny temperaturas (4; 1; 0,3 y 0,1 K)"); modulo(c, "params.py")
c = sig("frec", H);      proceso(XC, c, W, H, "Calcular las frecuencias de los modos\n$\\omega_c$, $\\omega_z$, $\\omega_+$ y $\\omega_-$"); modulo(c, "params.py")
c = sig("estat", H);     proceso(XC, c, W, H, "Parche estático: campo que llega al centro\ny desplazamiento $\\delta$ del centro guía"); modulo(c, "superficie.py\ncentro_guia.py")
cE = sig("crit", 1.35);  decision(XC, cE, WD, 1.35, "¿El campo supera\nel campo crítico?"); modulo(cE, "centro_guia.py")
cT = sig("T", 0.75);     proceso(XC, cT, W, 0.75, "Tomar la siguiente temperatura $T$ de la lista"); modulo(cT, "temperaturas.py")
c = sig("loc", H);       proceso(XC, c, W, H, "Localización: ocupaciones medias\ny amplitudes de los tres modos"); modulo(c, "cuantico.py")
c = sig("ruido", H);     proceso(XC, c, W, H, "Ruido del campo $S_E(f_-)$ de cada fuente:\ncircuitos y superficie de los electrodos"); modulo(c, "ruido.py\nparches_fluctuantes.py")
c = sig("perm", H);      proceso(XC, c, W, H, "Permanencia: difusión $D$ y tiempo de\npérdida $\\bar{t}$, sin disipación y con ella"); modulo(c, "ruido.py\ndisipacion.py")
cQ = sig("quedan", 1.25); decision(XC, cQ, WD, 1.25, "¿Quedan\ntemperaturas?"); modulo(cQ, "temperaturas.py")
cV = sig("ver", 1.2);    proceso(XC, cV, W, 1.2, "Verificar: 79 pruebas automáticas, órbita\ncompleta, hipótesis de tránsito y\ncontraste con experimentos"); modulo(cV, "test_*.py, tests/\norbita_completa.py\ntransito.py\ncontraste_experimental.py")
cS = sig("sal", 1.2);    dato(XC, cS, W, 1.2, "Resultados: tablas, figuras\ny cifras citadas en la tesis"); modulo(cS, "temperaturas.py\nfiguras_tesis.py\ntrayectorias.py\ncifras_canonicas.py")
cF = sig("fin", 0.65);   ovalo(XC, cF, 2.5, 0.65, "Fin")

# Flechas de la columna principal (salida por abajo, entrada por arriba)
for (n1, c1, h1), (n2, c2, h2) in zip(pasos[:-1], pasos[1:]):
    flecha((XC, c1 - h1 / 2), (XC, c2 + h2 / 2))
hE, hQ = 1.35, 1.25
texto(XC + 0.36, cE - hE / 2 - G / 2 + 0.03, "No", fontsize=FM)
texto(XC + 0.36, cQ - hQ / 2 - G / 2 + 0.03, "No", fontsize=FM)

# Rama «Sí» del campo crítico: pérdida determinista; se une al flujo en los resultados
xI, wI, hI = 1.62, 2.6, 1.3
yI = cE - 1.75
linea((XC - WD / 2, cE), (xI, cE))
flecha((xI, cE), (xI, yI + hI / 2))
texto(XC - WD / 2 - 0.35, cE + 0.20, "Sí", fontsize=FM)
proceso(xI, yI, wI, hI, "Pérdida determinista\npor el campo\nestático")
linea((xI, yI - hI / 2), (xI, cS))
flecha((xI, cS), (XC - W / 2, cS))

# Lazo «Sí» de las temperaturas: vuelve al paso que toma la siguiente temperatura
xL = 3.05
linea((XC - WD / 2, cQ), (xL, cQ), (xL, cT))
flecha((xL, cT), (XC - W / 2, cT))
texto(XC - WD / 2 - 0.35, cQ + 0.20, "Sí", fontsize=FM)

# Leyenda de símbolos
yl = 0.42
for i, s in enumerate(("inicio o fin", "entrada o salida", "cálculo", "decisión")):
    x0 = 1.25 + i * 3.6
    if i == 0:
        ax.add_patch(Ellipse((x0, yl), 0.86, 0.38, **est(SUAVE)))
    elif i == 1:
        ax.add_patch(Polygon([(x0 - 0.32, yl + 0.18), (x0 + 0.48, yl + 0.18), (x0 + 0.32, yl - 0.18), (x0 - 0.48, yl - 0.18)],
                             closed=True, **est()))
    elif i == 2:
        ax.add_patch(Rectangle((x0 - 0.4, yl - 0.18), 0.8, 0.36, **est()))
    else:
        ax.add_patch(Polygon([(x0, yl + 0.25), (x0 + 0.45, yl), (x0, yl - 0.25), (x0 - 0.45, yl)], closed=True, **est(SUAVE)))
    ax.text(x0 + 0.66, yl, s, fontsize=FM, ha="left", va="center", color=TINTA)

os.makedirs(os.path.join(AQUI, "figures"), exist_ok=True)
fig.savefig(os.path.join(AQUI, "figures", "met_01_flujo_codigo.pdf"), bbox_inches=None)
fig.savefig(os.path.join(AQUI, "figures", "met_01_flujo_codigo.png"), dpi=170, bbox_inches=None)
print("Figura guardada; y final =", round(y, 2), "(debe quedar por encima de la leyenda, y > 0,8)")
