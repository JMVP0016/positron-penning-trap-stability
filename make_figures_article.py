"""
make_figures_article.py — Regenerates the data and figures of the article analyses (round-3 package).

    MPLBACKEND=Agg python3 make_figures_article.py            # all fast analyses (~1 min)
    MPLBACKEND=Agg python3 make_figures_article.py --slow     # also kappa_3D (~8 min) and the median-bias study (~5 min)

Outputs go to datos_articulo/ and figures_articulo/. The thesis data in datos/ and figures/ are NOT touched
(figuras_tesis.py regenerates those). Every analysis uses fixed seeds and is deterministic.
"""
import os
import subprocess
import sys

RAIZ = os.path.dirname(os.path.abspath(__file__))
DAT = os.path.join(RAIZ, "datos_articulo")
FIG = os.path.join(RAIZ, "figures_articulo")
A = os.path.join(RAIZ, "articulo")

RAPIDOS = [
    ("i10 microtrap (Narimannezhad et al. 2014) with the guiding-centre model", ["i10_microtrampa_r3.py", "--salida", DAT]),
    ("i10 flat-well bounce transit test", ["i10_rebote_real.py", "--salida", DAT]),
    ("I3a axial channel T2", ["i3a_canal_axial.py", "--salida", DAT]),
    ("BASE AT bound (independent, B0 = 1.171 T)", ["cota_BASE_independiente.py", "--salida", DAT]),
    ("I5 uncertainty propagation (round 3: axial channel, 1/f EMI, kappa_3D)", ["i5_incertidumbre_r3.py", RAIZ, DAT, "20000"]),
    ("filter budget per stage (I1)", ["i1_presupuesto_filtro.py", RAIZ, DAT]),
    ("BASE AT bound and four real traps (i2, round 3)", ["i2_observable_r3.py", "--salida", DAT]),
    ("ALPHA static-patch contrast (thesis script, figure to figures/en)", ["../contraste_experimental.py"]),
    ("extended regime map (EN)", ["fig_regimes_extended.py", "--salida", FIG]),
]
LENTOS = [
    ("kappa_3D finite differences", ["i4_kappa3d.py", "--salida", DAT]),
    ("median bias of the loss Monte Carlo", ["m1_sesgo_mediana.py", "--salida", DAT]),
]

if __name__ == "__main__":
    os.makedirs(DAT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONPATH=RAIZ, FIG_IDIOMA=os.environ.get("FIG_IDIOMA", "en"))
    # Los scripts de la tesis (contraste_experimental.py) guardan en figures/ por defecto: sin esto,
    # la versión inglesa SOBRESCRIBIRÍA la figura española de la tesis.
    env["FIG_CARPETA"] = os.path.join(RAIZ, "figures", env["FIG_IDIOMA"])
    tareas = RAPIDOS + (LENTOS if "--slow" in sys.argv else [])
    for nombre, cmd in tareas:
        print(f"== {nombre}", flush=True)
        with open(os.path.join(DAT, os.path.basename(cmd[0]).replace(".py", "_log.txt")), "w") as log:
            r = subprocess.run([sys.executable, "-B", os.path.join(A, cmd[0])] + cmd[1:], cwd=DAT, env=env,
                               stdout=log, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            sys.exit(f"FAILED: {nombre} (see {os.path.basename(cmd[0]).replace('.py', '_log.txt')})")
    print("All analyses finished.")
