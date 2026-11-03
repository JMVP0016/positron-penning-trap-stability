# Positrones en una trampa de Penning: efecto de la temperatura (código de la tesis, V12)

Código en Python de la tesis «Análisis del efecto de la temperatura en la estabilidad de positrones en trampas de
Penning mediante simulación en Python» (Jonnathan Vazquez y Henry Manobanda, Física, ESPOCH, 2026). Calcula cuánto
tiempo permanece un positrón en la trampa (B₀ = 0,5 T, U₀ = 2×10⁴ V/m², pared a 5 mm) cuando el ruido eléctrico
de los electrodos hace difundir el radio del magnetrón, entre 4 K y 0,1 K.

## Instalación

Entorno de referencia: Python 3.11.3 con las versiones fijadas en `requirements.txt`.

    python3 -m pip install -r requirements.txt

## Uso

| Orden | Qué hace | Salida |
|---|---|---|
| `python3 -m pytest -q` | pruebas automáticas | — |
| `python3 figuras_tesis.py` | figuras del capítulo 4, unos 2 min (`python3 figuras_tesis.py 5 7`: solo la 5 y la 7) | `figures/`, `datos/` |
| `python3 trayectorias.py` | trayectorias del positrón a 4; 1; 0,3 y 0,1 K (Ilustración 4-7) | `figures/fig4_06b_trayectorias.*` |
| `python3 contraste_experimental.py` | contraste con ALPHA y con Peil (Ilustración 4-10) | `figures/fig4_10_alpha.*` |
| `python3 temperaturas.py` | tablas de las cuatro temperaturas y barrido del anexo | `datos/temperaturas_*.csv`, `tablas/` |

Con `FIG_IDIOMA=en FIG_CARPETA=figures/en` delante de la orden, las figuras salen con las etiquetas en inglés.
Con `SOURCE_DATE_EPOCH=1798761600`, los PDF llevan una fecha interna fija y, en el entorno de referencia, se
regeneran idénticos (mismo md5).

## Módulos

| Módulo | Contenido |
|---|---|
| `params.py` | constantes y parámetros de la trampa; única fuente para todos los módulos |
| `centro_guia.py` | órbita del centro guía con un campo de parche fijo y umbral de pérdida |
| `transito.py` | prueba numérica de la hipótesis de tránsito de la V11 |
| `ruido.py` | difusión del centro guía por ruido eléctrico y tiempo medio de pérdida |
| `superficie.py` | qué parte del campo de los parches llega al positrón (apantallamiento) |
| `parches_fluctuantes.py` | parches que cambian en el tiempo |
| `cuantico.py` | ocupaciones y amplitudes de los tres modos, con el punto cero |
| `figuras.py`, `estilo.mplstyle` | estilo y paleta comunes de las figuras |

La carpeta `v11/` guarda los guiones de la versión anterior (V11) solo como referencia: ningún módulo los usa.

## Reproducibilidad

Todas las simulaciones Monte Carlo usan semillas fijas: en el entorno de referencia, regenerar los datos reproduce bit
a bit los CSV de `datos/`. El tag `v12-tesis-cifras` marca la versión del código con la que se escribieron las cifras
de la tesis.
