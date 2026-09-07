# Positron-Penning-Trap-Stability
Monte Carlo simulation of the dynamic stability of a **single positron** confined in an
idealised Penning trap at **4 K** and **0.1 K**, including effective perturbations
associated with **patch potentials**.

Code developed for the undergraduate thesis *"Análisis del efecto de la temperatura en la
estabilidad de positrones en trampas de Penning mediante simulación en Python"*
(Escuela Superior Politécnica de Chimborazo, Faculty of Sciences, Physics Programme, 2026).

---

## Overview

Penning traps confine charged particles through the combination of a uniform axial magnetic
field and a quadrupolar electric potential. They are the confinement technology used for
antihydrogen research at CERN, which makes the stability of trapped positrons a question of
direct relevance to antimatter physics.

This project asks a specific question: **how does temperature affect how long a single
positron stays confined**, once patch potentials — small stochastic field imperfections on
the electrode surfaces — are taken into account.

## Physical model

- Equations of motion of a positron under a uniform axial magnetic field and a quadrupolar
  electric potential.
- Three normal modes identified: **axial**, **modified cyclotron**, **magnetron**.
- **4 K regime** — classical formulation based on equipartition.
- **0.1 K regime** — quantum modal context, using average occupation numbers.
- **Patch potentials** modelled as an **Ornstein–Uhlenbeck stochastic transverse field**,
  displacing the guiding centre through **E × B drift**.
- Radial transport characterised through the diffusive relation `R² = 4·D_gc·t`.

## Method

Monte Carlo ensemble of **600 realisations** per thermal regime, implemented in Python
(NumPy / SciPy / Matplotlib). Confinement time is defined as the time at which the guiding
centre reaches the trap wall.

## Results

| Regime | Median confinement time |
|---|---|
| 4 K | **78.20 h** |
| 0.1 K | **12.37 h** |

Lower temperature **reduces** confinement stability, in agreement with the scaling relation
`t_conf ∝ √T` **derived from the model**.

The mechanism is that temperature governs the thermal velocity of the positron and, through
it, the rate at which the guiding centre spreads, following `D_gc ∝ 1/√T`. A colder positron
diffuses radially faster under the same patch-potential field, intensifying transport towards
the wall.

> These results hold **within the assumptions and approximations of the model**. This is a
> theoretical and computational study; no experimental measurement is claimed.

## Repository structure

```
src/        simulation code
figures/    figures reproduced from the thesis
data/       simulation outputs
```

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

```bash
python src/montecarlo.py
```

## Authors

- **Jonnathan Marcelo Vazquez Pacurucu** — ORCID [0009-0000-2384-2583](https://orcid.org/0009-0000-2384-2583)
  *Contribution (CRediT): Software, Methodology, Formal analysis, Investigation, Writing – original draft.*
  **Simulation code in this repository implemented by this author.**
- **Henry David Manobanda Muzo** — co-author of the thesis.

**Thesis director:** Biof. Azucena Nataly Bonilla, MSc.

## Citation

> Vazquez Pacurucu, J. M., & Manobanda Muzo, H. D. (2026). *Análisis del efecto de la
> temperatura en la estabilidad de positrones en trampas de Penning mediante simulación en
> Python* [Undergraduate thesis, Escuela Superior Politécnica de Chimborazo].

## Notice

Intellectual property of the underlying work belongs to the Escuela Superior Politécnica de
Chimborazo. Reproduction, in whole or in part, is authorised **for academic purposes**, by any
means, **provided that authorship is acknowledged**.
