# Article analyses

| script | analysis | output |
|---|---|---|
| `i10_microtrampa_r3.py` | Narimannezhad et al. (2014) microtrap with the guiding-centre model; corrected first-passage conversion (4042 s) | `i10_*_r3.csv`, `i10_regimenes_r3.pdf` |
| `i10_rebote_real.py` | transit test with a flat-well bounce (w_b/w_- = 161) | `i10_rebote_real.*` |
| `i3a_canal_axial.py` | axial channel T2 (closed form + simulation) | `i3a_canal_axial.*` |
| `i4_kappa3d.py` | 3D dipolar coupling kappa (finite differences) | `i4_kappa3d.csv` |
| `cota_BASE_independiente.py` | S_E bound from BASE AT axial stability, B0 = 1.171 T | `cota_BASE_*.csv` |
| `i5_incertidumbre_r3.py` | uncertainty propagation | `i5_*.csv/pdf` |
| `i1_presupuesto_filtro.py` | filter noise budget per stage | `i1_*.csv` |
| `fig_regimes_extended.py` | extended regime map (T2, surface as bound, gas note) | `fig_regimes_extended.pdf` |
| `m1_sesgo_mediana.py` | median bias of the loss Monte Carlo | `m1_sesgo_mediana.csv` |
| `i2_observable_r3.py` | BASE AT bound and four real traps | `i2_*.csv` |

Run everything with `python3 make_figures_article.py` (add `--slow` for kappa_3D and the median study).
