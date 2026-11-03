# Data and code availability (draft, A1 round 3)

All results in this article were produced with the Python code that accompanies it. The code is
archived at [Zenodo DOI to be added after deposit] and mirrored at [repository URL to be added].
It contains the model modules, the thesis figures (`figuras_tesis.py`), the article analyses
(`articulo/`, run with `make_figures_article.py`) and automatic tests
(`test_referencia.py`, `tests/test_articulo.py`).

Every Monte Carlo calculation uses fixed random seeds. On the reference environment (Python 3.11.3,
NumPy 1.26.4, SciPy 1.11.1, Matplotlib 3.10.7), regenerating the data reproduces the archived CSV files
bit for bit. The numerical data behind every figure and table are provided as CSV files in `datos/`
(thesis figures) and `datos_articulo/` (article analyses).

The paper uses no experimental data of its own. The published values it takes from the literature
(Borchert et al. 2019; Labaziewicz et al. 2008; Baker et al. 2024; Narimannezhad et al. 2014) are
listed with their sources in the code and in the supplementary tables.

[To complete by the authors: licence (e.g. MIT), version tag, DOI.]
