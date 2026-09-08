# Positron Confinement Stability in a Penning Trap

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

Equations of motion of a positron under a uniform axial magnetic field `B₀` and a
quadrupolar electric potential, yielding the three normal modes of the system:

| Mode | Frequency |
|---|---|
| Axial | `ω_z = √(2qU₀/m)` |
| Modified cyclotron | `ω₊ = ½(ω_c + ω₁)` |
| Magnetron | `ω₋ = ½(ω_c − ω₁)`, with `ω₁ = √(ω_c² − 2ω_z²)` |

Two thermal regimes are treated with different formalisms:

- **4 K — classical.** Modal amplitudes from equipartition. The magnetron mode carries
  **negative (metastable) energy**, handled through an effective negative modal temperature
  `T₋ = −T_z(ω₋/ω_z)`.
- **0.1 K — quantum modal context.** Modal amplitudes reconstructed from **Bose–Einstein
  average occupation numbers** `n̄ = 1/(e^{βħω} − 1)`. At this temperature the cyclotron
  mode is effectively frozen (`n̄₊ ≈ 0`).

**Patch potentials** are modelled as a stochastic transverse field of RMS amplitude
`E_patch = V_patch/d_patch`, with correlation time `τ_patch = ℓ_c/v₀`. This drives the
guiding centre through **E × B drift** as a 2D random walk with diffusion coefficient

```
D_gc = (E_patch/√2)² · τ_patch / B₀²        with   v₀ = √(2k_BT/m)   ⇒   D_gc ∝ 1/√T
```

Loss occurs when either the axial amplitude reaches `z_wall`, or the guiding-centre
excursion plus the radial envelope reaches `r_wall`:

```
√(X_gc² + Y_gc²) + A₊ + A₋  ≥  r_wall
```

## Method

Monte Carlo ensemble of **600 realisations** per regime, with randomised modal phases,
20 000 time steps, and `t_max = 25·t_char` where `t_char = r_wall²/(4D_gc)`.
Radial transport is verified against the 2D diffusive relation `R_gc² = 4·D_gc·t`.

A **sensitivity analysis over the effective radius** (`r_wall` = 2.5, 5.0, 7.5 mm) is run
for both regimes.

### Default parameters

| Parameter | Value |
|---|---|
| `B₀` | 0.5 T |
| `V₀` / `d` | 1.0 V / 5.0 mm |
| `r_wall` / `z_wall` | 5.0 mm / 10.0 mm |
| `V_patch,rms` | 1.0 mV |
| `d_patch` | 1.0 mm |
| `ℓ_c` | 1.0 × 10⁻⁷ m |
| Ensemble | 600 realisations |
| Steps | 20 000 |
| Seed | 20260411 (reproducible) |

## Results

| Regime | Median confinement time |
|---|---|
| 4 K (classical) | **78.20 h** |
| 0.1 K (quantum modal) | **12.37 h** |

Lower temperature **reduces** confinement stability, in agreement with the scaling relation
`t_conf ∝ √T` **derived from the model**.

The mechanism is that temperature governs the thermal velocity of the positron and, through
it, the rate at which the guiding centre spreads, following `D_gc ∝ 1/√T`. A colder positron
diffuses radially faster under the same patch-potential field, intensifying transport
towards the wall. The measured ratio of median times is compared against `√40 ≈ 6.32`.

> These results hold **within the assumptions and approximations of the model**. This is a
> theoretical and computational study; no experimental measurement is claimed.

## Repository structure

```
penning_clasico_parches.py            4 K classical regime (equipartition)
penning_01K_contexto_cuantico.py      0.1 K quantum modal regime (Bose–Einstein)
figures/                              output figures
requirements.txt
CITATION.cff
```

> Both scripts must remain in the **same directory**: the 0.1 K script imports the 4 K one
> to build the comparative survival figure.

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

```bash
# 4 K classical regime (default)
python penning_clasico_parches.py

# any other temperature, in kelvin
python penning_clasico_parches.py 1.0

# 0.1 K quantum modal regime (also produces the 4 K vs 0.1 K comparison)
python penning_01K_contexto_cuantico.py
```

Each run writes figures (`fig4_*.pdf`, `figA_*.pdf`), a summary `resultados_resumen_*.json`
and a parameter table `tabla_control_parametros_*.csv`.

## Authors

- **Jonnathan Marcelo Vázquez Pacurucu** — ORCID [0009-0000-2384-2583](https://orcid.org/0009-0000-2384-2583)
  *Contribution (CRediT): Software, Methodology, Formal analysis, Investigation, Writing – original draft.*
  **Simulation code in this repository implemented by this author.**
- **Henry David Manobanda Muzo** — co-author of the thesis.

**Thesis director:** Biof. Azucena Nataly Bonilla, MSc.

## Citation

> Vázquez Pacurucu, J. M., & Manobanda Muzo, H. D. (2026). *Análisis del efecto de la
> temperatura en la estabilidad de positrones en trampas de Penning mediante simulación en
> Python* [Undergraduate thesis, Escuela Superior Politécnica de Chimborazo].

## Notice

Intellectual property of the underlying work belongs to the Escuela Superior Politécnica de
Chimborazo. Reproduction, in whole or in part, is authorised **for academic purposes**, by any
means, **provided that authorship is acknowledged**.
