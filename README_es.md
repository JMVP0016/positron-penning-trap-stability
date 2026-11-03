# Estabilidad del confinamiento de positrones en una trampa de Penning

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22652850.svg)](https://doi.org/10.5281/zenodo.22652850)

Código en Python de la tesis «Análisis del efecto de la temperatura en la estabilidad de positrones en trampas de
Penning mediante simulación en Python» (Escuela Superior Politécnica de Chimborazo, Facultad de Ciencias, carrera de
Física, 2026). Calcula cuánto tiempo permanece un positrón en una trampa de Penning ideal (B₀ = 0,5 T,
U₀ = 2×10⁴ V/m², radio de pérdida de 5 mm) entre 4 K y 0,1 K, cuando el ruido eléctrico de los electrodos hace
difundir el radio del magnetrón.

English version: [README.md](README.md).

## Qué cambia en la versión 2.0.0

La versión 1.0.0 modelaba los potenciales de parche como un campo de Ornstein–Uhlenbeck con un tiempo de correlación
fijado por el tránsito térmico del positrón, lo que daba `D ∝ 1/√T` y un confinamiento más corto al enfriar. La
versión 2 pone a prueba esa hipótesis y la descarta: un parche estático solo desplaza la órbita del magnetrón, que
sigue acotada, y el radio crece solo con el ruido del campo a la frecuencia magnetrón `f₋`. Por eso la versión 2

- calcula el tiempo medio de pérdida a partir de fuentes de ruido realistas: el filtro de los electrodos (ruido de
  Johnson más la interferencia de un montaje como BASE), los parches de superficie que fluctúan y el canal axial;
- incluye la disipación que el teorema de fluctuación–disipación asocia a toda fuente de ruido térmica;
- trata los tres modos como osciladores cuánticos (ocupaciones de Bose–Einstein y amplitudes con el punto cero);
- abarca cuatro temperaturas (4; 1; 0,3 y 0,1 K) y un barrido continuo de 0,05 a 10 K;
- compara el modelo con referencias experimentales (ALPHA, Peil y Gabrielse, BASE).

La carpeta `v11/` guarda los guiones de la versión 1 solo como referencia: ningún módulo los usa.

## El modelo en breve

| Ingrediente | Relación |
|---|---|
| Difusión del centro guía | `D = S_E(f₋)/(4B₀²)`, tiempo medio de pérdida `t̄ = r_wall² B₀² / S_E(f₋)` |
| Parche estático | órbita del magnetrón acotada; pérdida inmediata solo por encima de 50–100 V/m |
| Disipación de las fuentes térmicas | `t̄ = Ein(U)/(2λ)`, con `U = |E₋(r_wall)|/(k_B T)` |
| Modos cuánticos | `n̄ = 1/(e^{ħω/k_BT} − 1)`, `A = √(2ħ(n̄ + ½)/(m ω_ef))` |

## Instalación

Entorno de referencia: Python 3.11.3 con las versiones fijadas en `requirements.txt`.

    python3 -m pip install -r requirements.txt

## Uso

| Orden | Qué hace | Salida |
|---|---|---|
| `python3 -m pytest -q` | 79 pruebas automáticas | — |
| `python3 figuras_tesis.py` | figuras del capítulo 4 (≈ 4 min; `python3 figuras_tesis.py 5 7`: solo la 5 y la 7) | `figures/`, `datos/` |
| `python3 trayectorias.py` | trayectorias del positrón a 4; 1; 0,3 y 0,1 K | `figures/fig4_06b_trayectorias.*` |
| `python3 contraste_experimental.py` | contraste con ALPHA y con Peil y Gabrielse | `figures/fig4_10_alpha.*` |
| `python3 temperaturas.py` | tablas de las cuatro temperaturas y del barrido de 0,05 a 10 K | `datos/temperaturas_*.csv`, `tablas/` |
| `python3 fig_disipacion.py` | tiempo de pérdida con y sin disipación | `figures/figF_disipacion.*` |
| `python3 cifras_canonicas.py` | cada cifra que cita la tesis, con el guion que la produce | `datos/cifras_canonicas.csv` |
| `python3 make_figures_article.py --slow` | análisis del artículo (≈ 4 min) | `datos_articulo/`, `figures_articulo/` |

Con `FIG_IDIOMA=en FIG_CARPETA=figures/en` delante de la orden, las figuras salen con las etiquetas en inglés. Con
`SOURCE_DATE_EPOCH=1798761600`, los PDF llevan una fecha interna fija y, en el entorno de referencia, se regeneran
idénticos.

## Módulos

| Módulo | Contenido |
|---|---|
| `params.py` | constantes y parámetros de la trampa; única fuente para todos los módulos |
| `centro_guia.py` | órbita del centro guía con un campo de parche fijo y umbral de pérdida |
| `transito.py` | prueba numérica de la hipótesis de tránsito de la versión 1 (`chequeo_transito.py` la repite con otras razones `ω_z/ω₋`) |
| `ruido.py` | difusión del centro guía por ruido eléctrico y tiempo medio de pérdida |
| `superficie.py` | qué parte del campo de los parches llega al positrón (apantallamiento) |
| `parches_fluctuantes.py` | parches de superficie que fluctúan |
| `cuantico.py` | ocupaciones y amplitudes de los tres modos, con el punto cero |
| `disipacion.py` | fluctuación–disipación: antiamortiguamiento del magnetrón y el factor `Ein(U)/U` |
| `orbita_completa.py` | órbita completa con la fuerza de Lorentz (integrador de Boris y propagador exacto) para validar el centro guía |
| `figuras.py`, `estilo.mplstyle` | estilo y paleta comunes de las figuras |
| `articulo/` | análisis del artículo, que ejecuta `make_figures_article.py` |

## Reproducibilidad

Todas las simulaciones Monte Carlo usan semillas fijas: en el entorno de referencia, regenerar los datos reproduce bit
a bit los CSV de `datos/` (la última columna de `datos_articulo/m1_sesgo_mediana.csv` guarda tiempos de ejecución y
cambia en cada corrida). El tag `v12-tesis-cifras` marca la versión del código de la que sale cada cifra de la tesis;
`v2.0.0` es esta publicación.

## Autores

- **Jonnathan Marcelo Vazquez Pacurucu** — ORCID [0009-0000-2384-2583](https://orcid.org/0009-0000-2384-2583)
  *Contribución (CRediT): software, metodología, análisis formal, investigación, redacción del borrador original.*
- **Henry David Manobanda Muzo** — coautor de la tesis.
  *Contribución (CRediT): software, validación.*
  **Módulo del contraste experimental (`contraste_experimental.py`) y sus pruebas implementados por este autor.**

**Directora de la tesis:** Biof. Azucena Nataly Bonilla, MSc.

## Cómo citar

**Software (este repositorio, todas las versiones):**

> Vazquez Pacurucu, J. M., y Manobanda Muzo, H. D. (2026). *Positron Confinement Stability in a Penning Trap*
> [Software]. Zenodo. https://doi.org/10.5281/zenodo.22652850

Zenodo asigna un DOI propio a cada versión; el de la 1.0.0 es https://doi.org/10.5281/zenodo.22652851.

**Tesis:**

> Vazquez Pacurucu, J. M., y Manobanda Muzo, H. D. (2026). *Análisis del efecto de la temperatura en la estabilidad
> de positrones en trampas de Penning mediante simulación en Python* [Tesis de grado, Escuela Superior Politécnica de
> Chimborazo].

## Licencia y aviso

El código se publica con la licencia Apache 2.0 (`LICENSE`). La propiedad intelectual del trabajo pertenece a la
Escuela Superior Politécnica de Chimborazo. Se autoriza su reproducción total o parcial **con fines académicos**, por
cualquier medio, **siempre que se reconozca la autoría**.
