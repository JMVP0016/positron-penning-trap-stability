# Article analyses (round-3 package)

| script | analysis | output |
|---|---|---|
| `i10_microtrampa_r3.py` | Narimannezhad et al. (2014) microtrap with the guiding-centre model; corrected first-passage conversion (4042 s) | `i10_*_r3.csv`, `i10_regimenes_r3.pdf` |
| `i10_rebote_real.py` | transit test with a flat-well bounce (w_b/w_- = 161) | `i10_rebote_real.*` |
| `i3a_canal_axial.py` | axial channel T2 (closed form + simulation) | `i3a_canal_axial.*` |
| `i4_kappa3d.py` | 3D dipolar coupling kappa (finite differences) | `i4_kappa3d.csv` |
| `cota_BASE_independiente.py` | S_E bound from BASE AT axial stability, B0 = 1.171 T | `cota_BASE_*.csv` |
| `i5_incertidumbre.py` | uncertainty propagation (agent A2) | `i5_*.csv/pdf` |
| `fig4_10_alpha.py` | static-patch contrast with ALPHA (agent A3) | `fig4_10_alpha.*` |
| `fig_regimes_extended.py` | extended regime map (T2, surface as bound, gas note) | `fig_regimes_extended.pdf` |
| `m1_sesgo_mediana.py` | median bias of the loss Monte Carlo | `m1_sesgo_mediana.csv` |
| `i2_observable_PENDIENTE_R3-1.py` | A3 round-2 version, **known error in B_AT**; replace with the corrected A3 version | — |

Run everything with `python3 make_figures_article.py` (add `--slow` for kappa_3D and the median study).
