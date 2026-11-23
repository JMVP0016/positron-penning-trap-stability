# Estabilidad del confinamiento de positrones en una trampa de Penning

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22652850.svg)](https://doi.org/10.5281/zenodo.22652850)

Código en Python de la tesis «Análisis del efecto de la temperatura en la estabilidad de positrones en trampas de
Penning mediante simulación en Python» (Escuela Superior Politécnica de Chimborazo, Facultad de Ciencias, carrera de
Física, 2026). Calcula cuánto tiempo permanece un positrón en una trampa de Penning ideal (B₀ = 0,5 T,
U₀ = 2×10⁴ V/m², radio de pérdida de 5 mm) entre 4 K y 0,1 K, cuando el ruido eléctrico de los electrodos hace
difundir el radio del magnetrón.

English version: [README.md](README.md).

## Qué cambia en la versión 2.0.4

Los cálculos y los datos no cambian y las 79 pruebas pasan. Cambian los rótulos de las figuras y se añaden dos
guiones de figuras.

- Figuras de control (`centro_guia.py`, `ruido.py`, `superficie.py`), de verificación (`figuras_verificacion.py`) y
  de disipación (`fig_disipacion.py`): cada eje lleva nombre, símbolo y unidad; las leyendas explican todos los
  símbolos sin tapar los datos; los colores siguen la paleta de `figuras.py`, con azul y vino solo para comparar
  4 K con 0,1 K; el periodo magnetrón se escribe P₋ y el radio de pérdida r_wall. `figuras_verificacion.py` acepta
  `FIG_CARPETA`, así la versión en inglés ya no sobrescribe la española.
- `figuras.py`: `coma()` escribe las marcas de los ejes lineales con coma decimal y el mismo número de decimales.
- `fig_F3_incertidumbres.py` (nuevo): cociente t̄(T)/t̄(4 K) con su mediana y los percentiles 5–95, con el mismo
  cálculo y la misma semilla que `articulo/i5_incertidumbre_r3.py`.
- `fig_flujo_codigo.py` (nuevo): diagrama de flujo del cálculo, con el módulo que realiza cada paso.
- `articulo/i5_incertidumbre_r3.py`: el histograma de cocientes llega ahora hasta 10^3,4; antes terminaba en
  10^2,6 y el canal axial, con cocientes cercanos a 1600, quedaba acumulado en el borde.
- `figures/control_transito.*` se regeneró con `transito.py`: la copia anterior venía de una versión previa de la
  paleta de colores; los datos son los mismos.

## Qué cambia en la versión 2.0.3

Los resultados anteriores no cambian y las 79 pruebas pasan. La versión 2.0.2 contiene los mismos cambios.

- `articulo/fig_regimes_extended.py`: el mapa de regímenes marca el criterio temporal t* = 1 año, llama
  determinista a la pérdida por el campo estático, ubica el circuito de tres etapas con disipación en su ruido
  equivalente S_eq y explica cada símbolo en la leyenda (versión en español con `FIG_IDIOMA=es`: Ilustración 4-9
  de la tesis).
- `orbita_completa.py`: el comentario de Boris dice ahora que el paso de Cayley conserva la norma en la rotación
  magnética pura, pero gira 2 arctan(ω_c Δt/2) por paso en lugar de ω_c Δt.
- `figuras_verificacion.py` (nuevo): las cuatro figuras de verificación del anexo B (error de fase de Boris,
  convergencia con el paso temporal, tiempo medio con disipación y órbita completa frente al centro guía).
  `ruido.simular_difusion` acepta un `dt` opcional para la prueba de convergencia; su valor por defecto no cambia.
- `articulo/f3_con_disipacion.py` (nuevo): la propagación de incertidumbres del anexo F.3 repetida con disipación.

## Qué cambia en la versión 2.0.1

Solo se ordenaron los comentarios del código y los nombres de dos archivos de pruebas. Los cálculos y los
resultados son idénticos a los de la versión 2.0.0.

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
| `python3 figuras_verificacion.py` | figuras de verificación del anexo B (≈ 30 s) | `figures/verif_*`, `datos/verif_*.csv` |
| `python3 centro_guia.py` (y `ruido.py`, `superficie.py`) | comprobaciones de cada módulo y su figura de control | `figures/control_*` |
| `python3 fig_F3_incertidumbres.py` | propagación de incertidumbres: t̄(T)/t̄(4 K) con mediana y percentiles 5–95 (≈ 15 s) | `figures/figF3_incertidumbres.*` |
| `python3 fig_flujo_codigo.py` | diagrama de flujo del cálculo | `figures/met_01_flujo_codigo.*` |
| `FIG_IDIOMA=es python3 articulo/fig_regimes_extended.py --salida figures` | mapa de regímenes de la tesis (Ilustración 4-9) | `figures/fig_regimes_extended_es.*` |

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
`v2.0.4` es esta publicación.

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
