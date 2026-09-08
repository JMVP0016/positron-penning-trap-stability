from __future__ import annotations

"""Regimen de 0.1 K en contexto cuantico para una trampa de Penning.

COHERENCIA CON LA TESIS:
Cada bloque de calculo cita la ecuacion de la tesis que lo respalda. No se
introduce ninguna magnitud que no este formulada en el documento.
  - Frecuencias caracteristicas ......... ec. (3.77), (3.78)  [identicas al caso 4 K]
  - Condicion de estabilidad ideal ...... ec. (3.79) / (3.219)
  - Parametros termicos modales ......... ec. (3.220) beta+ = betaz = 1/(kB T)
  - Temperatura/beta del magnetron ...... ec. (3.221) T- = -Tz(w-/wz), beta- = 1/(kB T-)
  - Numeros de ocupacion (Bose-Einstein)  ec. (3.222) nbar = 1/(exp(beta hbar w) - 1)
  - Ocupacion efectiva del magnetron .... ec. (3.223) nbar- = nbar_z
  - Amplitud coherente efectiva ......... ec. (3.224) |alpha_j| = sqrt(nbar_j)
  - Amplitudes espaciales efectivas ..... ec. (3.225) A_eff
  - Trayectorias efectivas (x,y,z) ...... ec. (3.227), (3.228), (3.229)
  - Campo de parche efectivo ............ ec. (3.231)-(3.233)  [muestreado por el movimiento]
  - Deriva E x B / transporte difusivo .. ec. (3.234), (3.235)  [limite difusivo, t_conf ~ sqrt(T)]
  - Trayectoria total efectiva .......... ec. (3.236)
  - Criterio de perdida (envolvente) .... ec. (3.238)-(3.240)
  - Energias modales cuanticas .......... ec. (3.242)-(3.244)
  - Curva de supervivencia .............. ec. (3.241)

FIGURAS FINALES PARA CAPITULO 4:
  fig4_08_movimiento_axial_01K.pdf
  fig4_09_movimiento_radial_01K.pdf
  fig4_10_energias_modales_01K.pdf
  fig4_11_ocupaciones_modales_01K.pdf
  fig4_12_histograma_tconf_01K.pdf
  fig4_13_supervivencia_01K.pdf
  fig4_14_comparacion_supervivencia.pdf
  fig4_17_deriva_centro_guia_01K.pdf
  fig4_18_transporte_difusivo_01K.pdf
  fig4_20_sensibilidad_rwall_01K.pdf

FIGURAS DE APOYO PARA ANEXO:
  figA_02_deriva_3d_01K.pdf

NOTA SOBRE V_patch_rms Y LA CONVENCION DE TRANSPORTE:
Se usa V_patch_rms = 1.0e-3 V, IDENTICO al codigo clasico de 4 K, en coherencia
con la tesis. La diferencia entre 4 K y 0.1 K se manifiesta en las energias,
ocupaciones y amplitudes modales (especialmente el congelamiento del modo
ciclotron, nbar+ << 1), y tambien en el tiempo de confinamiento a traves del
transporte difusivo de parches. Para mantener coherencia con la forma usada por
Baker, se fija v_0 = sqrt(2 kB T / m); entonces D_gc ~ 1/v_0 y
t_conf ~ sqrt(T).
"""

import json
import csv
from dataclasses import dataclass, replace
from math import exp, expm1, pi, sqrt
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HOUR = 3600.0
R_WALL_SENSITIVITY_VALUES_M = (2.5e-3, 5.0e-3, 7.5e-3)


# ---------------------------------------------------------------------------
# Estructuras de datos
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PhysicalConstants:
    q: float
    m: float
    kB: float
    hbar: float


@dataclass(frozen=True)
class SimulationConfig:
    input_mode: str
    B0: float
    V0: float
    U0: Optional[float]
    d: float
    r_wall: float
    z_wall: float
    T: float
    V_patch_rms: float
    d_patch: float
    l_c: float
    N_ensemble: int
    n_steps: int
    trace_stride: int
    save_figures: bool
    seed_base: int


@dataclass(frozen=True)
class TrapParameters:
    B0: float
    U0: float
    r_wall: float
    z_wall: float
    V_patch_rms: float
    d_patch: float
    l_c: float
    E_patch_rms: float
    v_0: float
    tau_patch: float
    D_gc: float
    t_char: float
    dt: float
    t_max: float
    n_steps: int


@dataclass(frozen=True)
class TrapFrequencies:
    omega_c: float
    omega_z: float
    omega_1: float
    omega_plus: float
    omega_minus: float


@dataclass(frozen=True)
class QuantumThermalState:
    T_plus: float
    T_minus: float
    T_z: float
    beta_plus: float
    beta_minus: float
    beta_z: float
    nbar_plus: float
    nbar_minus: float
    nbar_z: float
    A_eff_plus: float
    A_eff_minus: float
    A_eff_z: float
    E_plus: float
    E_minus: float
    E_z: float


@dataclass
class RealizationResult:
    t_conf: float
    phases: dict[str, float]
    thermal_state: QuantumThermalState
    trace: Optional[dict[str, np.ndarray]]


@dataclass
class EnsembleResult:
    t_conf: np.ndarray
    survival_time: np.ndarray
    survival_prob: np.ndarray
    representative: RealizationResult
    mean_t_conf: float
    median_t_conf: float
    lost_count: int
    survivors_count: int


# ---------------------------------------------------------------------------
# Constantes y configuracion
# ---------------------------------------------------------------------------
def get_constants() -> PhysicalConstants:
    return PhysicalConstants(
        q=1.602176634e-19,
        m=9.1093837015e-31,
        kB=1.380649e-23,
        hbar=1.054571817e-34,
    )


def get_default_config() -> SimulationConfig:
    # Parametros externos IDENTICOS al codigo clasico de 4 K (coherencia con la tesis):
    # V_patch_rms = 1.0e-3 V -> E_patch_rms = V_patch_rms/d_patch = 1.0 V/m.
    return SimulationConfig(
        input_mode="V0_d",
        B0=0.5,
        V0=1.0,
        U0=None,
        d=5.0e-3,
        r_wall=5.0e-3,
        z_wall=10.0e-3,
        T=0.1,
        V_patch_rms=1.0e-3,
        d_patch=1.0e-3,
        l_c=1.0e-7,
        N_ensemble=600,
        n_steps=20000,
        trace_stride=1,
        save_figures=True,
        seed_base=20260411,
    )


def resolve_trap_parameters(config: SimulationConfig, constants: PhysicalConstants) -> TrapParameters:
    if config.input_mode == "V0_d":
        if config.d <= 0.0:
            raise ValueError("d debe ser positivo.")
        U0 = config.V0 / (2.0 * config.d**2)
    elif config.input_mode == "U0":
        if config.U0 is None:
            raise ValueError("Debes definir U0 cuando input_mode='U0'.")
        U0 = config.U0
    else:
        raise ValueError("input_mode debe ser 'V0_d' o 'U0'.")
    if constants.q * U0 <= 0.0:
        raise ValueError("Se requiere q*U0 > 0 para obtener omega_z real.")

    E_patch_rms = config.V_patch_rms / config.d_patch     # ec. (3.230)
    sigma_component = E_patch_rms / sqrt(2.0)             # ec. (3.233)

    # Limite difusivo del proceso de parche (Narimannezhad/Baker 2014, ec. 24):
    # la estocasticidad proviene del movimiento del positron sobre un campo de parche
    # estatico; el tiempo de correlacion es el de transito sobre una mancha l_c a la
    # velocidad termica efectiva v_0 = sqrt(2 kB T / m), convencion Baker.
    # En el limite tau<<dt el OU colapsa
    # a difusion: D_gc = sigma_E^2 tau / B0^2 = (E_patch/B0)^2 l_c/(2 v_0).
    # Aqui T = 0.1 K (modo axial casi clasico, kB T >> hbar wz), v_0 es menor que a 4 K,
    # por lo que D_gc es mayor y t_conf ~ sqrt(T) resulta menor que en el caso clasico.
    v_0 = sqrt(2.0 * constants.kB * config.T / constants.m)
    tau_patch = config.l_c / v_0
    D_gc = sigma_component**2 * tau_patch / config.B0**2
    t_char = config.r_wall**2 / (4.0 * D_gc)
    t_max = 25.0 * t_char
    dt = t_max / config.n_steps

    return TrapParameters(
        B0=config.B0, U0=U0, r_wall=config.r_wall, z_wall=config.z_wall,
        V_patch_rms=config.V_patch_rms, d_patch=config.d_patch, l_c=config.l_c,
        E_patch_rms=E_patch_rms, v_0=v_0, tau_patch=tau_patch, D_gc=D_gc,
        t_char=t_char, dt=dt, t_max=t_max, n_steps=config.n_steps,
    )


# ---------------------------------------------------------------------------
# Fisica: frecuencias (identicas al caso clasico) y estado termico cuantico
# ---------------------------------------------------------------------------
def compute_frequencies(trap: TrapParameters, constants: PhysicalConstants) -> TrapFrequencies:
    """Frecuencias caracteristicas, ec. (3.77)-(3.78). Identicas al caso 4 K:
    dependen solo de los parametros de la trampa (B0, U0), no de la temperatura."""
    omega_c = abs(constants.q) * trap.B0 / constants.m
    omega_z = sqrt(2.0 * constants.q * trap.U0 / constants.m)
    disc = omega_c**2 - 2.0 * omega_z**2
    if disc <= 0.0:
        raise ValueError("Condicion de estabilidad violada: omega_c^2 <= 2*omega_z^2.")
    omega_1 = sqrt(disc)
    return TrapFrequencies(
        omega_c=omega_c, omega_z=omega_z, omega_1=omega_1,
        omega_plus=0.5 * (omega_c + omega_1),
        omega_minus=0.5 * (omega_c - omega_1),
    )


def compute_quantum_thermal_state(config, freqs, constants) -> QuantumThermalState:
    """Estado termico cuantico del positron a T = 0.1 K.

    Parametros termicos: ec. (3.220) beta+ = betaz = 1/(kB T).
    Magnetron:           ec. (3.221) T- = -Tz(w-/wz), beta- = 1/(kB T-).
    Ocupaciones:         ec. (3.222) nbar = 1/(exp(beta hbar w) - 1)  [Bose-Einstein].
    Magnetron efectivo:  ec. (3.223) nbar- = nbar_z.
    Amplitud coherente:  ec. (3.224) |alpha_j| = sqrt(nbar_j).
    Amplitudes efectivas: ec. (3.225) A_eff_+ = A_eff_- = sqrt(2 hbar/(m w1))|alpha|,
                          A_eff_z = sqrt(2 hbar/(m wz))|alpha_z|.
    Energias modales:    ec. (3.242)-(3.244) E_j = +-hbar w_j (nbar_j + 1/2).
    """
    if config.T <= 0.0:
        raise ValueError("La temperatura debe ser positiva.")

    T_plus = T_z = config.T
    T_minus = -T_z * (freqs.omega_minus / freqs.omega_z)            # ec. (3.221)

    beta_plus = 1.0 / (constants.kB * T_plus)                       # ec. (3.220)
    beta_z = 1.0 / (constants.kB * T_z)                             # ec. (3.220)
    beta_minus = 1.0 / (constants.kB * T_minus)                     # ec. (3.221)

    # Bose-Einstein (ec. 3.222); expm1 da precision cuando el argumento es pequeno.
    nbar_plus = 1.0 / expm1(beta_plus * constants.hbar * freqs.omega_plus)
    nbar_z = 1.0 / expm1(beta_z * constants.hbar * freqs.omega_z)
    nbar_minus = nbar_z                                             # ec. (3.223)

    # Amplitud coherente |alpha| = sqrt(nbar)  (ec. 3.224)
    alpha_plus = sqrt(nbar_plus)
    alpha_minus = sqrt(nbar_minus)
    alpha_z = sqrt(nbar_z)

    # Amplitudes espaciales efectivas (ec. 3.225)
    A_eff_plus = sqrt(2.0 * constants.hbar / (constants.m * freqs.omega_1)) * alpha_plus
    A_eff_minus = sqrt(2.0 * constants.hbar / (constants.m * freqs.omega_1)) * alpha_minus
    A_eff_z = sqrt(2.0 * constants.hbar / (constants.m * freqs.omega_z)) * alpha_z

    # Energias modales cuanticas (ec. 3.242-3.244). E_- negativa (magnetron).
    E_plus = constants.hbar * freqs.omega_plus * (nbar_plus + 0.5)
    E_minus = -constants.hbar * freqs.omega_minus * (nbar_minus + 0.5)
    E_z = constants.hbar * freqs.omega_z * (nbar_z + 0.5)

    return QuantumThermalState(
        T_plus=T_plus, T_minus=T_minus, T_z=T_z,
        beta_plus=beta_plus, beta_minus=beta_minus, beta_z=beta_z,
        nbar_plus=nbar_plus, nbar_minus=nbar_minus, nbar_z=nbar_z,
        A_eff_plus=A_eff_plus, A_eff_minus=A_eff_minus, A_eff_z=A_eff_z,
        E_plus=E_plus, E_minus=E_minus, E_z=E_z,
    )


def ideal_oscillation_state(t, ts: QuantumThermalState, freqs, phases) -> dict[str, np.ndarray]:
    """Trayectorias efectivas reconstruidas por valores esperados.
    ec. (3.227)  x_eff(t) = A_eff+ sin(w+ t + f+) + A_eff- sin(w- t + f-)
    ec. (3.228)  y_eff(t) = A_eff+ cos(w+ t + f+) + A_eff- cos(w- t + f-)
    ec. (3.229)  z_eff(t) = A_eff_z sin(wz t + fz)
    """
    t_array = np.asarray(t, dtype=float)
    theta_plus = freqs.omega_plus * t_array + phases["phi_plus"]
    theta_minus = freqs.omega_minus * t_array + phases["phi_minus"]
    theta_z = freqs.omega_z * t_array + phases["phi_z"]
    x_eff = ts.A_eff_plus * np.sin(theta_plus) + ts.A_eff_minus * np.sin(theta_minus)
    y_eff = ts.A_eff_plus * np.cos(theta_plus) + ts.A_eff_minus * np.cos(theta_minus)
    z_eff = ts.A_eff_z * np.sin(theta_z)
    return {"x_osc": x_eff, "y_osc": y_eff, "z_osc": z_eff}


# ---------------------------------------------------------------------------
# Fisica: transporte difusivo del centro guia (limite del OU, identico al clasico)
# ---------------------------------------------------------------------------
# En el limite tau_patch = l_c/v_0 << dt el proceso OU del campo se reduce a ruido
# blanco y la deriva E x B del centro guia es una caminata browniana 2D con
# coeficiente D_gc: dX, dY ~ N(0, sqrt(2 D_gc dt)). El campo no se integra paso a
# paso. Las 600 realizaciones del ensamble = 600 muestreos del desorden de parches.


# ---------------------------------------------------------------------------
# Criterios de perdida, ec. (3.238)-(3.240)
# ---------------------------------------------------------------------------
def check_axial_loss(A_eff_z, z_wall) -> bool:
    return A_eff_z >= z_wall                                         # ec. (3.240)


def check_radial_loss(X_gc, Y_gc, A_eff_plus, A_eff_minus, r_wall) -> bool:
    return sqrt(X_gc**2 + Y_gc**2) + A_eff_plus + A_eff_minus >= r_wall  # ec. (3.238)-(3.239)


def compute_survival_curve(t_conf, t_max) -> tuple[np.ndarray, np.ndarray]:
    """Curva de supervivencia S(t), ec. (3.241). Mismo algoritmo que el clasico."""
    t_conf_array = np.asarray(t_conf, dtype=float)
    total = int(t_conf_array.size)
    eps = max(1.0e-15, 1.0e-12 * max(t_max, 1.0))
    loss_times = np.sort(t_conf_array[t_conf_array < (t_max - eps)])
    survival_time: list[float] = [0.0]
    survival_prob: list[float] = [1.0]
    survivors = total
    if loss_times.size > 0:
        uniq, counts = np.unique(loss_times, return_counts=True)
        for lt, lc in zip(uniq, counts):
            survival_time.append(float(lt)); survival_prob.append(survivors / total)
            survivors -= int(lc)
            survival_time.append(float(lt)); survival_prob.append(survivors / total)
    survival_time.append(float(t_max)); survival_prob.append(survivors / total)
    return np.asarray(survival_time, dtype=float), np.asarray(survival_prob, dtype=float)


# ---------------------------------------------------------------------------
# Realizacion individual y ensamble Monte Carlo (protocolo identico al clasico)
# ---------------------------------------------------------------------------
def run_realization(config, trap, freqs, ts: QuantumThermalState, rng, store_trace) -> RealizationResult:
    phases = {
        "phi_z": float(rng.uniform(0.0, 2.0 * pi)),
        "phi_plus": float(rng.uniform(0.0, 2.0 * pi)),
        "phi_minus": float(rng.uniform(0.0, 2.0 * pi)),
    }

    # Perdida axial inmediata (ec. 3.240).
    if check_axial_loss(ts.A_eff_z, trap.z_wall):
        return RealizationResult(0.0, phases, ts, None)

    # --- Caminata browniana 2D del centro guia (limite difusivo del OU) ---
    n = trap.n_steps
    step_std = sqrt(2.0 * trap.D_gc * trap.dt)
    dX = rng.normal(0.0, step_std, size=n)
    dY = rng.normal(0.0, step_std, size=n)
    X_path = np.empty(n + 1, dtype=float); X_path[0] = 0.0; np.cumsum(dX, out=X_path[1:])
    Y_path = np.empty(n + 1, dtype=float); Y_path[0] = 0.0; np.cumsum(dY, out=Y_path[1:])

    # Radio efectivo envolvente (ec. 3.238-3.239) y primer cruce de la pared.
    R_eff = np.sqrt(X_path**2 + Y_path**2) + (ts.A_eff_plus + ts.A_eff_minus)
    lost_mask = R_eff >= trap.r_wall
    if lost_mask.any():
        idx_loss = int(np.argmax(lost_mask))
        t_conf = idx_loss * trap.dt
    else:
        idx_loss = n
        t_conf = trap.t_max

    trace = None
    if store_trace:
        stride = max(1, (idx_loss + 1) // 2000)
        sel = np.arange(0, idx_loss + 1, stride)
        if sel[-1] != idx_loss:
            sel = np.append(sel, idx_loss)
        Xs = X_path[sel]; Ys = Y_path[sel]
        trace = {
            "t_gc": sel.astype(float) * trap.dt,
            "X_gc": Xs,
            "Y_gc": Ys,
            "R_gc": np.sqrt(Xs**2 + Ys**2),
        }

    return RealizationResult(float(t_conf), phases, ts, trace)


def run_ensemble(config, trap, freqs, ts, seed) -> EnsembleResult:
    """Ensamble de N realizaciones. Semilla por realizacion: seed + idx.
    Realizacion representativa: la mas cercana a la mediana. (Identico al clasico.)"""
    t_conf_values = np.empty(config.N_ensemble, dtype=float)
    for idx in range(config.N_ensemble):
        rng = np.random.default_rng(seed + idx)
        realization = run_realization(config, trap, freqs, ts, rng, store_trace=False)
        t_conf_values[idx] = realization.t_conf

    median_t_conf = float(np.median(t_conf_values))
    closest_idx = int(np.argmin(np.abs(t_conf_values - median_t_conf)))
    rng_repr = np.random.default_rng(seed + closest_idx)
    representative = run_realization(config, trap, freqs, ts, rng_repr, store_trace=True)

    survival_time, survival_prob = compute_survival_curve(t_conf_values, trap.t_max)
    lost_count = int(np.count_nonzero(t_conf_values < trap.t_max))
    return EnsembleResult(
        t_conf=t_conf_values, survival_time=survival_time, survival_prob=survival_prob,
        representative=representative, mean_t_conf=float(np.mean(t_conf_values)),
        median_t_conf=median_t_conf, lost_count=lost_count,
        survivors_count=int(config.N_ensemble - lost_count),
    )


# ---------------------------------------------------------------------------
# Salida grafica (9 figuras homologadas en 4 bloques)
# ---------------------------------------------------------------------------
def ensure_images_dir(save_figures: bool) -> Optional[Path]:
    if not save_figures:
        return None
    output_dir = Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_fig(fig: plt.Figure, filename: str, save_figures: bool) -> None:
    out = ensure_images_dir(save_figures)
    if out is not None:
        fig.savefig(out / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _style(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.18, linewidth=0.6)


def save_rwall_sensitivity_table(rows: list[dict[str, float | int | str]], out_dir: Path, suffix: str) -> None:
    """Tabla de sensibilidad geometrica del tiempo de confinamiento."""
    if not rows:
        return
    csv_path = out_dir / f"tabla_sensibilidad_rwall_{suffix}.csv"
    json_path = out_dir / f"resultados_sensibilidad_rwall_{suffix}.json"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=4)


def save_survival_sensitivity_figure(
    curves: list[tuple[float, EnsembleResult]],
    out_dir: Path,
    filename: str,
    title: str,
) -> None:
    """Figura de sensibilidad S(t) frente a r_wall, en el formato usado en la tesis."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass

    styles = [
        {"color": "tab:blue", "ls": "-", "label": r"$r_{wall}=2.5$ mm"},
        {"color": "tab:orange", "ls": "--", "label": r"$r_{wall}=5.0$ mm"},
        {"color": "tab:green", "ls": "-.", "label": r"$r_{wall}=7.5$ mm"},
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    max_t = 0.0
    for (_, result), style in zip(curves, styles):
        t_h = result.survival_time / HOUR
        s_t = result.survival_prob
        positive_t = t_h[s_t > 0.0]
        if positive_t.size:
            max_t = max(max_t, float(positive_t[-1]))
        ax.step(t_h, s_t, where="post", lw=1.7, **style)

    ax.set_title(title)
    ax.set_xlabel("Tiempo de confinamiento [h]")
    ax.set_ylabel(r"$S(t)=N_{confinadas}(t)/N_{total}$")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(0.0, max(max_t * 1.04, 1.0))
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    fig.savefig(out_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_rwall_sensitivity_01K(
    base_result: EnsembleResult,
    config: SimulationConfig,
    base_trap: TrapParameters,
    base_freqs: TrapFrequencies,
) -> None:
    """Bloque D4: sensibilidad del confinamiento al radio efectivo, caso cuantico 0.1 K."""
    out_dir = ensure_images_dir(config.save_figures)
    if out_dir is None:
        return

    constants = get_constants()
    curves: list[tuple[float, EnsembleResult]] = []
    rows: list[dict[str, float | int | str]] = []

    for r_wall in R_WALL_SENSITIVITY_VALUES_M:
        if abs(r_wall - config.r_wall) <= 1.0e-15:
            trap = base_trap
            freqs = base_freqs
            result = base_result
        else:
            sens_config = replace(config, r_wall=r_wall, save_figures=False)
            trap = resolve_trap_parameters(sens_config, constants)
            freqs = compute_frequencies(trap, constants)
            thermal_state = compute_quantum_thermal_state(sens_config, freqs, constants)
            result = run_ensemble(sens_config, trap, freqs, thermal_state, sens_config.seed_base)

        curves.append((r_wall, result))
        rows.append({
            "regimen": "01K_cuantico",
            "T_K": config.T,
            "r_wall_mm": r_wall * 1e3,
            "v0_m_s": trap.v_0,
            "D_gc_m2_s": trap.D_gc,
            "t_char_horas": trap.t_char / HOUR,
            "t_conf_media_horas": result.mean_t_conf / HOUR,
            "t_conf_mediana_horas": result.median_t_conf / HOUR,
            "lost_count": result.lost_count,
            "survivors_count": result.survivors_count,
        })

    save_survival_sensitivity_figure(
        curves,
        out_dir,
        "fig4_20_sensibilidad_rwall_01K.pdf",
        "Sensibilidad del tiempo de confinamiento al radio efectivo, T = 0,1 K",
    )
    save_rwall_sensitivity_table(rows, out_dir, "01K_cuantico")

    print("=== Sensibilidad de r_wall, regimen cuantico 0.1 K ===")
    for row in rows:
        print(
            f"  r_wall={float(row['r_wall_mm']):g} mm | "
            f"mediana={float(row['t_conf_mediana_horas']):.3f} h | "
            f"media={float(row['t_conf_media_horas']):.3f} h"
        )
    print(f"  Figura de sensibilidad: {out_dir / 'fig4_20_sensibilidad_rwall_01K.pdf'}")


def plot_survival_comparison_4K_01K(quantum_result: EnsembleResult, quantum_config: SimulationConfig) -> None:
    """Comparacion directa para la seccion 4.5, sin crear un script adicional."""
    out_dir = ensure_images_dir(quantum_config.save_figures)
    if out_dir is None:
        return

    import penning_clasico_parches as clasico

    constants_c = clasico.get_constants()
    config_c = clasico.get_default_config()
    config_c = replace(config_c, save_figures=False)
    trap_c = clasico.resolve_trap_parameters(config_c, constants_c)
    freqs_c = clasico.compute_frequencies(trap_c, constants_c)
    classic_result = clasico.run_ensemble(
        config_c.temperature_K,
        config_c,
        trap_c,
        freqs_c,
        constants_c,
        config_c.seed_base,
    )

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass

    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    ax.step(
        quantum_result.survival_time / HOUR,
        quantum_result.survival_prob,
        where="post",
        color="tab:orange",
        lw=1.8,
        label="0,1 K, cuántico",
    )
    ax.step(
        classic_result.survival_time / HOUR,
        classic_result.survival_prob,
        where="post",
        color="tab:blue",
        lw=1.8,
        label="4 K, clásico",
    )
    ax.axvline(
        quantum_result.median_t_conf / HOUR,
        color="tab:orange",
        ls=":",
        lw=1.3,
        label=f"Mediana 0,1 K = {quantum_result.median_t_conf/HOUR:.2f} h",
    )
    ax.axvline(
        classic_result.median_t_conf / HOUR,
        color="tab:blue",
        ls=":",
        lw=1.3,
        label=f"Mediana 4 K = {classic_result.median_t_conf/HOUR:.2f} h",
    )
    ratio = classic_result.median_t_conf / quantum_result.median_t_conf
    ax.text(
        0.98,
        0.72,
        f"$t_{{50,4K}}/t_{{50,0.1K}} = {ratio:.2f}$\n$\\sqrt{{40}} = {sqrt(40.0):.2f}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.75", "alpha": 0.9},
        fontsize=9,
    )
    ax.set_title("Comparación de supervivencia entre 4 K y 0,1 K")
    ax.set_xlabel("Tiempo de confinamiento [h]")
    ax.set_ylabel("Fracción confinada S(t)")
    ax.set_xlim(0.0, min(classic_result.median_t_conf * 5.0, classic_result.survival_time[-1]) / HOUR)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="upper right", fontsize=8.4)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    fig.savefig(out_dir / "fig4_14_comparacion_supervivencia.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figura comparativa: {out_dir / 'fig4_14_comparacion_supervivencia.pdf'}")


def plot_case_results(case_result: EnsembleResult, config, trap, freqs) -> None:
    rep = case_result.representative
    if rep.trace is None:
        return
    ts = rep.thermal_state
    phases = rep.phases
    trace = rep.trace
    constants = get_constants()
    suffix = "01K_cuantico"
    T_label = f"{config.T:g}".replace(".", ",") + " K"
    HOUR = 3600.0  # s -> hora (escala difusiva del tiempo de confinamiento)

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass
    plt.rcParams.update({"axes.titlesize": 12.5, "axes.labelsize": 11,
                         "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
                         "legend.fontsize": 9, "lines.linewidth": 1.8})

    # ===================== BLOQUE A - TRAYECTORIAS =====================
    # figA1: oscilacion axial z(t), ec. (3.229)
    Tz = 2.0 * pi / freqs.omega_z
    t_ax = np.linspace(0.0, 8.0 * Tz, 4000)
    z_ax = ts.A_eff_z * np.sin(freqs.omega_z * t_ax + phases["phi_z"])
    figA1, axA1 = plt.subplots(figsize=(8, 4))
    axA1.plot(t_ax * 1e9, z_ax * 1e6, color="tab:red")
    axA1.axhline(ts.A_eff_z * 1e6, color="gray", ls=":", lw=0.9, label="$\\pm A_{eff,z}$")
    axA1.axhline(-ts.A_eff_z * 1e6, color="gray", ls=":", lw=0.9)
    axA1.set_title(f"Movimiento axial $z(t)$  (T = {T_label}, cuántico)\n"
                   f"$A_{{eff,z}}$ = {ts.A_eff_z*1e6:.4f} µm,  $\\bar{{n}}_z$ = {ts.nbar_z:.2f}")
    axA1.set_xlabel("Tiempo [ns]"); axA1.set_ylabel("z [µm]")
    _style(axA1); axA1.legend(loc="upper right")
    save_fig(figA1, "fig4_08_movimiento_axial_01K.pdf", config.save_figures)

    # figA2: epiciclo radial local (modo magnetron + envolvente), ec. (3.227)-(3.228), (3.239)
    ang = np.linspace(0.0, 2.0 * pi, 600)
    x_mag = ts.A_eff_minus * np.sin(ang)
    y_mag = ts.A_eff_minus * np.cos(ang)
    figA2, axA2 = plt.subplots(figsize=(5.6, 5.6))
    axA2.plot(x_mag * 1e9, y_mag * 1e9, color="tab:blue", label="Órbita magnetrón ($A_{eff,-}$)")
    env = (ts.A_eff_plus + ts.A_eff_minus) * 1e9
    axA2.add_patch(plt.Circle((0, 0), env, fill=False, ls=":", color="black",
                              lw=1.4, label="Envolvente $A_{eff,+} + A_{eff,-}$"))
    axA2.set_title(f"Movimiento radial local  (T = {T_label}, cuántico)\n"
                   f"$A_{{eff,+}}$ = {ts.A_eff_plus*1e9:.3f} nm,  $A_{{eff,-}}$ = {ts.A_eff_minus*1e9:.2f} nm")
    axA2.set_xlabel("$x_{loc}$ [nm]"); axA2.set_ylabel("$y_{loc}$ [nm]")
    axA2.set_aspect("equal", adjustable="box")
    _style(axA2); axA2.legend(loc="upper right")
    save_fig(figA2, "fig4_09_movimiento_radial_01K.pdf", config.save_figures)

    # figA3: vista 3D de presentacion de la deriva del centro guia (plano z=0) en la
    # geometria radial de la trampa. Misma informacion que C1, pero en perspectiva 3D;
    # el angulo, el aspecto y los limites se eligen para que la pared se vea circular y
    # el punto final quede claramente sobre ella. La oscilacion axial real (~10^-8 s) no
    # es resoluble frente a la deriva de horas, por lo que el movimiento axial se presenta
    # por separado en A1.
    X_gc = trace["X_gc"]; Y_gc = trace["Y_gc"]
    rw_mm = trap.r_wall * 1e3
    z_half = (ts.A_eff_z * 1e3 * 1.2) if ts.A_eff_z > 0 else 1e-3
    figA3 = plt.figure(figsize=(7.2, 6.2))
    axA3 = figA3.add_subplot(111, projection="3d")
    axA3.plot(X_gc * 1e3, Y_gc * 1e3, np.zeros_like(X_gc), color="tab:purple",
              lw=1.6, label="Centro guía (deriva)")
    axA3.scatter([X_gc[0]*1e3], [Y_gc[0]*1e3], [0], color="tab:green", s=55,
                 depthshade=False, label="Inicio", zorder=6)
    axA3.scatter([X_gc[-1]*1e3], [Y_gc[-1]*1e3], [0], color="tab:red", s=75,
                 marker="X", depthshade=False, label="Fin", zorder=6)
    th = np.linspace(0, 2*pi, 200)
    axA3.plot(rw_mm*np.cos(th), rw_mm*np.sin(th), np.zeros_like(th),
              color="tab:red", ls="--", lw=1.3, label="Pared radial $r_{wall}$")
    axA3.set_xlim(-rw_mm*1.05, rw_mm*1.05)
    axA3.set_ylim(-rw_mm*1.05, rw_mm*1.05)
    axA3.set_zlim(-z_half, z_half)
    axA3.set_box_aspect((1.0, 1.0, 0.42))
    axA3.view_init(elev=52, azim=-60)
    axA3.set_title(f"Vista 3D de la deriva del centro guía  (T = {T_label}, cuántico)")
    axA3.set_xlabel("$X_{gc}$ [mm]"); axA3.set_ylabel("$Y_{gc}$ [mm]")
    axA3.set_zlabel("z [mm]")
    axA3.legend(loc="upper left", fontsize=8)
    save_fig(figA3, "figA_02_deriva_3d_01K.pdf", config.save_figures)

    # ===================== BLOQUE B - ENERGIAS Y OCUPACIONES MODALES =====================
    # figB1: energias modales E+, E-, Ez en E/kB [K], ec. (3.242)-(3.244)
    Ez_K = ts.E_z / constants.kB
    Ep_K = ts.E_plus / constants.kB
    Em_K = ts.E_minus / constants.kB
    vals = [Ep_K, Em_K, Ez_K]
    labels = ["$E_+$ (ciclotrón)", "$E_-$ (magnetrón)", "$E_z$ (axial)"]
    colors = ["tab:blue", "tab:green", "tab:orange"]
    figB1, (axB1, axB1_zoom) = plt.subplots(
        2, 1, figsize=(7.4, 5.8), gridspec_kw={"height_ratios": [2.4, 1.15]},
        constrained_layout=True,
    )
    bars = axB1.bar(labels, vals, color=colors, alpha=0.85)
    axB1.axhline(0.0, color="black", lw=0.9)
    axB1.set_title(f"Energías modales cuánticas  (T = {T_label})")
    axB1.set_ylabel("Energía / $k_B$  [K]")
    span = max(abs(min(vals)), abs(max(vals)), 1e-30)
    for b, v in zip(bars, vals):
        axB1.text(b.get_x() + b.get_width()/2, v + (0.02 if v >= 0 else -0.02) * span,
                  f"{v:.3g}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    _style(axB1)
    mag_margin = max(abs(Em_K) * 1.45, 1.0e-10)
    axB1_zoom.bar(["$E_-$ (magnetrón)"], [Em_K], color="tab:green", alpha=0.85)
    axB1_zoom.axhline(0.0, color="black", lw=0.9)
    axB1_zoom.set_ylim(-mag_margin, mag_margin * 0.35)
    axB1_zoom.set_ylabel("$E_-/k_B$ [K]")
    axB1_zoom.set_title("Zoom del modo magnetrón: energía negativa metaestable")
    axB1_zoom.text(0.0, Em_K - 0.08 * mag_margin, f"{Em_K:.3e} K",
                   ha="center", va="top", fontsize=9)
    _style(axB1_zoom)
    save_fig(figB1, "fig4_10_energias_modales_01K.pdf", config.save_figures)

    # figB2: numeros de ocupacion nbar+, nbar-, nbar_z (EXCLUSIVA del regimen cuantico)
    # Es la huella del efecto cuantico: nbar+ << 1 (modo ciclotron congelado), ec. (3.222)
    figB2, axB2 = plt.subplots(figsize=(7, 4.4))
    nbars = [ts.nbar_plus, ts.nbar_minus, ts.nbar_z]
    barsn = axB2.bar(["$\\bar{n}_+$ (ciclotrón)", "$\\bar{n}_-$ (magnetrón)", "$\\bar{n}_z$ (axial)"],
                     nbars, color=["tab:blue", "tab:green", "tab:orange"], alpha=0.85)
    axB2.set_yscale("log")
    axB2.set_title(f"Números de ocupación modales (Bose-Einstein)  (T = {T_label})\n"
                   f"$\\bar{{n}}_+$ = {ts.nbar_plus:.2e}  →  modo ciclotrón congelado")
    axB2.set_ylabel("$\\bar{n}$ (escala log)")
    for b, v in zip(barsn, nbars):
        axB2.text(b.get_x() + b.get_width()/2, v * 1.3, f"{v:.2e}",
                  ha="center", va="bottom", fontsize=9)
    _style(axB2)
    save_fig(figB2, "fig4_11_ocupaciones_modales_01K.pdf", config.save_figures)

    # ===================== BLOQUE C - POTENCIALES DE PARCHE =====================
    # figC1: deriva del centro guia con pared radial (ec. 3.235, 3.238)
    X_mm = X_gc * 1e3; Y_mm = Y_gc * 1e3
    figC1, axC1 = plt.subplots(figsize=(5.8, 5.8))
    axC1.plot(X_mm, Y_mm, color="tab:purple", lw=1.8, label="Centro guía")
    axC1.scatter(X_mm[0], Y_mm[0], color="tab:green", s=55, label="Inicio", zorder=5)
    axC1.scatter(X_mm[-1], Y_mm[-1], color="tab:red", s=60, marker="X", label="Fin", zorder=5)
    axC1.add_patch(plt.Circle((0, 0), trap.r_wall*1e3, fill=False, ls="--",
                              color="tab:red", lw=1.4, label="Pared radial $r_{wall}$"))
    axC1.set_title(f"Deriva del centro guía por parches  (T = {T_label}, cuántico)\n"
                   f"$t_{{conf}}$ = {rep.t_conf/HOUR:.3f} horas")
    axC1.set_xlabel("$X_{gc}$ [mm]"); axC1.set_ylabel("$Y_{gc}$ [mm]")
    axC1.set_aspect("equal", adjustable="box")
    _style(axC1); axC1.legend(loc="best")
    save_fig(figC1, "fig4_17_deriva_centro_guia_01K.pdf", config.save_figures)

    # figC2: radio efectivo Reff(t) y caracter difusivo del transporte (R^2 ~ 4 D_gc t).
    # Sustituye al antiguo panel del campo OU (en el limite difusivo no se integra el campo).
    t_gc = trace["t_gc"]
    R_gc_mm = trace["R_gc"] * 1e3
    R_eff_mm = R_gc_mm + (ts.A_eff_plus + ts.A_eff_minus) * 1e3
    figC2, (axC2a, axC2b) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axC2a.plot(t_gc / HOUR, R_eff_mm, color="tab:blue", label="$R_{gc}(t) + A_{eff,+} + A_{eff,-}$")
    axC2a.axhline(trap.r_wall*1e3, color="tab:red", ls="--", label="$r_{wall}$")
    if rep.t_conf < trap.t_max:
        axC2a.axvline(rep.t_conf / HOUR, color="black", ls=":", label="$t_{conf}$")
    axC2a.set_ylabel("Radio efectivo [mm]")
    axC2a.set_title(f"Transporte difusivo del centro guía  (T = {T_label}, cuántico)")
    _style(axC2a); axC2a.legend(loc="best")

    R2_mm2 = (trace["R_gc"] * 1e3) ** 2
    R2_theory = 4.0 * trap.D_gc * t_gc * 1e6   # 4 D_gc t,  m^2 -> mm^2
    axC2b.plot(t_gc / HOUR, R2_mm2, color="tab:purple", alpha=0.7, label="$R_{gc}^2(t)$ (realización)")
    axC2b.plot(t_gc / HOUR, R2_theory, color="tab:red", ls="--", label="$4\\,D_{gc}\\,t$ (difusión 2D)")
    axC2b.set_xlabel("Tiempo [horas]"); axC2b.set_ylabel("$R_{gc}^2$ [mm$^2$]")
    axC2b.set_title("Verificación del régimen difusivo: $R_{gc}^2(t)$ frente a $4D_{gc}t$")
    axC2b.text(0.02, 0.92,
               f"$D_{{gc}}$ = {trap.D_gc:.3e} m²/s\n$\\tau_{{patch}}=\\ell_c/v_0$ = {trap.tau_patch:.2e} s",
               transform=axC2b.transAxes, ha="left", va="top",
               bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "0.75", "alpha": 0.85},
               fontsize=9)
    _style(axC2b); axC2b.legend(loc="best")
    save_fig(figC2, "fig4_18_transporte_difusivo_01K.pdf", config.save_figures)

    # ===================== BLOQUE D - ESTADISTICA DE CONFINAMIENTO =====================
    # figD1: histograma de tiempos de confinamiento
    figD1, axD1 = plt.subplots(figsize=(8, 4))
    axD1.hist(case_result.t_conf / HOUR, bins=30, color="tab:orange", alpha=0.8, edgecolor="black")
    axD1.axvline(case_result.mean_t_conf / HOUR, color="tab:blue", ls="--",
                 label=f"Media = {case_result.mean_t_conf/HOUR:.3f} horas")
    axD1.axvline(case_result.median_t_conf / HOUR, color="tab:red", ls=":",
                 label=f"Mediana = {case_result.median_t_conf/HOUR:.3f} horas")
    axD1.set_title(f"Distribución de $t_{{conf}}$  (T = {T_label}, N = {config.N_ensemble})")
    axD1.set_xlabel("$t_{conf}$ [horas]"); axD1.set_ylabel("Número de realizaciones")
    _style(axD1); axD1.legend(loc="best")
    save_fig(figD1, "fig4_12_histograma_tconf_01K.pdf", config.save_figures)

    # figD2: curva de supervivencia S(t), ec. (3.241)
    figD2, axD2 = plt.subplots(figsize=(8, 4))
    axD2.step(case_result.survival_time / HOUR, case_result.survival_prob, where="post", color="tab:cyan")
    axD2.axhline(0.5, color="gray", ls="--", alpha=0.7, label="S(t) = 0.5")
    axD2.axvline(case_result.median_t_conf / HOUR, color="tab:red", ls=":",
                 label=f"Mediana = {case_result.median_t_conf/HOUR:.3f} horas")
    axD2.set_xlim(0.0, min(trap.t_max, max(5.0 * case_result.median_t_conf, 1.0)) / HOUR)
    axD2.set_ylim(0.0, 1.05)
    axD2.set_title(f"Curva de supervivencia S(t)  (T = {T_label}, cuántico)")
    axD2.set_xlabel("Tiempo [horas]"); axD2.set_ylabel("Fracción confinada S(t)")
    _style(axD2); axD2.legend(loc="best")
    save_fig(figD2, "fig4_13_supervivencia_01K.pdf", config.save_figures)

    save_data(case_result, config, trap, freqs)


def save_data(case_result, config, trap, freqs) -> None:
    ts = case_result.representative.thermal_state
    data = {
        "T_K": config.T,
        "B0_T": trap.B0, "V0_V": config.V0, "d_m": config.d,
        "r_wall_m": trap.r_wall, "z_wall_m": trap.z_wall,
        "V_patch_rms_V": config.V_patch_rms, "d_patch_m": config.d_patch,
        "l_c_m": trap.l_c,
        "E_patch_rms_V_m": trap.E_patch_rms,
        "v0_m_s": trap.v_0,
        "tau_patch_s": trap.tau_patch,
        "D_gc_m2_s": trap.D_gc,
        "t_char_s": trap.t_char, "t_char_horas": trap.t_char / 3600.0,
        "dt_s": trap.dt, "t_max_s": trap.t_max, "n_steps": trap.n_steps,
        "N_ensemble": config.N_ensemble,
        "omega_c_rad_s": freqs.omega_c, "omega_z_rad_s": freqs.omega_z,
        "omega_plus_rad_s": freqs.omega_plus, "omega_minus_rad_s": freqs.omega_minus,
        "nbar_plus": ts.nbar_plus, "nbar_minus": ts.nbar_minus, "nbar_z": ts.nbar_z,
        "A_eff_plus_m": ts.A_eff_plus, "A_eff_minus_m": ts.A_eff_minus, "A_eff_z_m": ts.A_eff_z,
        "E_plus_J": ts.E_plus, "E_minus_J": ts.E_minus, "E_z_J": ts.E_z,
        "E_plus_over_kB_K": ts.E_plus / 1.380649e-23,
        "E_minus_over_kB_K": ts.E_minus / 1.380649e-23,
        "E_z_over_kB_K": ts.E_z / 1.380649e-23,
        "t_conf_media_s": case_result.mean_t_conf,
        "t_conf_media_horas": case_result.mean_t_conf / 3600.0,
        "t_conf_mediana_s": case_result.median_t_conf,
        "t_conf_mediana_horas": case_result.median_t_conf / 3600.0,
        "lost_count": case_result.lost_count,
        "survivors_count": case_result.survivors_count,
    }
    out = ensure_images_dir(config.save_figures)
    if out is None:
        out = Path("resultados_penning_01K_cuantico"); out.mkdir(parents=True, exist_ok=True)
    with open(out / "resultados_resumen_01K_cuantico.json", "w") as f:
        json.dump(data, f, indent=4)
    with open(out / "tabla_control_parametros_01K_cuantico.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=data.keys()); w.writeheader(); w.writerow(data)


def print_summary(case_result, trap, freqs, config) -> None:
    ts = case_result.representative.thermal_state
    kB = 1.380649e-23
    print("=== Resumen del regimen de 0.1 K en contexto cuantico ===")
    print(f"  T               = {config.T:g} K")
    print(f"  omega_c         = {freqs.omega_c:.4e} rad/s")
    print(f"  omega_z         = {freqs.omega_z:.4e} rad/s")
    print(f"  omega_+         = {freqs.omega_plus:.4e} rad/s")
    print(f"  omega_-         = {freqs.omega_minus:.4e} rad/s")
    print(f"  nbar_+          = {ts.nbar_plus:.4e}  (modo ciclotron congelado)")
    print(f"  nbar_z          = {ts.nbar_z:.4f}")
    print(f"  nbar_-          = {ts.nbar_minus:.4f}  (= nbar_z, ec. 3.223)")
    print(f"  A_eff_+         = {ts.A_eff_plus*1e9:.4f} nm")
    print(f"  A_eff_-         = {ts.A_eff_minus*1e9:.4f} nm")
    print(f"  A_eff_z         = {ts.A_eff_z*1e6:.4f} um")
    print(f"  E_+/kB          = {ts.E_plus/kB:.6f} K")
    print(f"  E_-/kB          = {ts.E_minus/kB:.6e} K")
    print(f"  E_z/kB          = {ts.E_z/kB:.6f} K")
    print(f"  V_patch_rms     = {config.V_patch_rms*1e3:.2f} mV")
    print(f"  E_patch_rms     = {trap.E_patch_rms:.4f} V/m")
    print(f"  l_c             = {trap.l_c:.2e} m")
    print(f"  v_0 (Baker)     = {trap.v_0:.4e} m/s")
    print(f"  tau_patch=l_c/v0= {trap.tau_patch:.4e} s")
    print(f"  D_gc            = {trap.D_gc:.4e} m^2/s")
    print(f"  t_caracteristico= {trap.t_char:.4e} s  ({trap.t_char/3600.0:.3f} horas)")
    print(f"  dt              = {trap.dt:.4e} s   t_max = {trap.t_max:.4e} s")
    print(f"  t_conf media    = {case_result.mean_t_conf:.6e} s  ({case_result.mean_t_conf/3600.0:.3f} horas)")
    print(f"  t_conf mediana  = {case_result.median_t_conf:.6e} s  ({case_result.median_t_conf/3600.0:.3f} horas)")
    print(f"  perdidas        = {case_result.lost_count} / {config.N_ensemble}")
    print("  Figuras finales: carpeta actual del script (fig4_08..fig4_20 y figA_02)")


def main() -> None:
    constants = get_constants()
    config = get_default_config()
    trap = resolve_trap_parameters(config, constants)
    freqs = compute_frequencies(trap, constants)
    ts = compute_quantum_thermal_state(config, freqs, constants)
    case_result = run_ensemble(config, trap, freqs, ts, config.seed_base)
    print_summary(case_result, trap, freqs, config)
    plot_case_results(case_result, config, trap, freqs)
    plot_survival_comparison_4K_01K(case_result, config)
    plot_rwall_sensitivity_01K(case_result, config, trap, freqs)


if __name__ == "__main__":
    main()
