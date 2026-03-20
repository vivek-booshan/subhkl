from dataclasses import dataclass, replace, astuple
from typing import Optional, List

import numpy as np

from subhkl.core.crystallography import Lattice
from subhkl.core.models import LATTICE_CONFIG
from subhkl.core.experiment import ExperimentData
from subhkl.core.spacegroup import get_centering
from subhkl.instrument.detector import scattering_vector_from_angles
from subhkl._optimization.optimization import VectorizedObjective 


@dataclass(frozen=True, slots=True)
class Result:
    num_indexed: int
    hkl: np.ndarray
    wavelengths: np.ndarray
    U: np.ndarray
    x: np.ndarray

    def __iter__(self):
        return iter(astuple(self))


def _minimize_scipy(
    state: ExperimentData,
    population_size: int = 1000,
    num_generations: int = 100,
    n_runs: int = 1,
    seed: int = 0,
    tolerance_deg: float = 0.1,
    loss_method: str = "gaussian",
    init_params: Optional[np.ndarray] = None,
    refine_lattice: bool = False,
    lattice_bound_frac: float = 0.05,
    refine_goniometer: bool = False,
    goniometer_bound_deg: float = 5.0,
    refine_goniometer_axes: Optional[List[str]] = None,
    refine_sample: bool = False,
    sample_bound_meters: float = 2.0,
    refine_beam: bool = False,
    beam_bound_deg: float = 1.0,
    d_min: Optional[float] = None,
    d_max: Optional[float] = None,
    hkl_search_range: int = 20,
    B_sharpen: float = 50,
) -> Result:
    """
    SciPy-based fallback for minimize when JAX is not available.
    Uses scipy.optimize.differential_evolution with VectorizedObjective.
    """
    from scipy.optimize import differential_evolution, minimize as scipy_minimize

    gonio = state.goniometer
    gonio_mask = np.array([True]*len(gonio.axes))
    if refine_goniometer and refine_goniometer_axes and gonio.names:
        gonio_mask = np.array(
            [any(r in n for r in refine_goniometer_axes) for n in gonio.names]
        )

    lattice_system = state.lattice.system
    num_lattice_params = LATTICE_CONFIG[lattice_system]["num_params"]
    lattice_system_str = LATTICE_CONFIG[lattice_system]["name"]

    print(f"Lattice System: {lattice_system} ({num_lattice_params} free params)")

    # name, dimension, bound
    components = [
        ("orientation", 3, [(-np.pi, np.pi)] * 3),
        (
            "lattice",
            num_lattice_params if refine_lattice else 0,
            [(0.0, 1.0)] * num_lattice_params,
        ),
        ("sample", 3 if refine_sample else 0, [(0.0, 1.0)] * 3),
        ("beam", 2 if refine_beam else 0, [(0.0, 1.0)] * 2),
        (
            "gonio",
            np.sum(gonio_mask) if refine_goniometer else 0,
            [(0.0, 1.0)] * int(np.sum(gonio_mask)),
        ),
    ]

    bounds = []
    for _, dim, bnd in components:
        if dim > 0:
            bounds.extend(bnd)
    num_dims = len(bounds)

    weights = state.peaks.refine_weights(B_sharpen)
    kf_ki_dir = scattering_vector_from_angles(
        state.peaks.two_theta, state.peaks.azimuthal
    )

    # Initialize VectorizedObjective (works with NumPy shim)
    t = np.linspace(0, np.pi, 1024)
    cdf = (t - np.sin(t)) / np.pi
    _angle_cdf = cdf
    _angle_t = t

    objective_v = VectorizedObjective(
        B=state.lattice.get_b_matrix(),
        kf_ki_dir=kf_ki_dir,
        peak_xyz_lab=state.peaks.xyz.T if state.peaks.xyz is not None else None,
        wavelength=state.wavelength,
        weights=weights,
        tolerance_deg=tolerance_deg,
        cell_params=[
            state.lattice.a,
            state.lattice.b,
            state.lattice.c,
            state.lattice.alpha,
            state.lattice.beta,
            state.lattice.gamma,
        ],
        refine_lattice=refine_lattice,
        lattice_bound_frac=lattice_bound_frac,
        lattice_system=lattice_system_str,  # resolve to actual enum usage
        goniometer_axes=gonio.axes,
        goniometer_angles=gonio.angles.T if gonio.angles is not None else None,
        refine_goniometer=refine_goniometer,
        goniometer_bound_deg=goniometer_bound_deg,
        goniometer_refine_mask=gonio_mask,
        goniometer_nominal_offsets=gonio.base_offsets,
        refine_sample=refine_sample,
        sample_bound_meters=sample_bound_meters,
        sample_nominal=state.base_sample_offset,
        refine_beam=refine_beam,
        beam_bound_deg=beam_bound_deg,
        beam_nominal=state.ki_vec,
        loss_method=loss_method,
        hkl_search_range=hkl_search_range,
        d_min=d_min,
        d_max=d_max or 100.0,
        space_group=state.space_group,
        centering=get_centering(state.space_group),
        static_R=gonio.rotation if gonio.rotation is not None else np.eye(3),
        peak_run_indices=state.run_indices,
        angle_cdf=_angle_cdf,
        angle_t=_angle_t
    )

    x0 = init_params
    if x0 is not None:
        if len(x0) < num_dims:
            x0 = np.concatenate([x0, np.full(num_dims - len(x0), 0.5)])
        x0 = x0[:num_dims]

    # 4. Run Optimization
    print(f"\n--- Starting SciPy Differential Evolution ({n_runs} runs) ---")
    print(f"Population: {population_size}, Generations: {num_generations}")

    def _objective_wrapper(x):
        """Wrapper for SciPy to call VectorizedObjective."""
        # SciPy's vectorized mode passes (dims, members)
        if x.ndim == 1:
            return float(objective_v(x[None, :])[0])
        return np.array(objective_v(x.T))

    best_f, best_x = np.inf, None
    for i in range(n_runs):
        res = differential_evolution(
            _objective_wrapper,
            bounds,
            maxiter=num_generations,
            popsize=max(1, population_size // num_dims),
            seed=seed + i,
            x0=x0 if i == 0 else None,
            vectorized=True,
        )
        if res.fun < best_f:
            best_f, best_x = res.fun, res.x

    print("\n--- Optimization Complete ---")
    print(f"Best DE score: {-best_f:.2f}")

    # --- Local Refinement (BFGS) ---
    print("Polishing solution with BFGS refinement...")

    res_polish = scipy_minimize(_objective_wrapper, best_x, method="L-BFGS-B", bounds=bounds, options={"maxiter": 50})
    final_x = res_polish.x if res_polish.success and res_polish.fun < best_f else best_x

    best_member = final_x[None, :]
    phys = objective_v._get_physical_params_jax(best_member)
    (_, B_batch, sample_batch, ki_batch, offsets_batch, R_batch) = phys

    new_lattice = Lattice.from_b_matrix(np.array(B_batch[0])) if refine_lattice else state.lattice
    print("--- Refined Lattice Parameters ---")
    print(f"a: {new_lattice.a:.4f}, b: {new_lattice.b:.4f}, c: {new_lattice.c:.4f}")
    print(f"alpha: {new_lattice.alpha:.4f}, beta: {new_lattice.beta:.4f}, gamma: {new_lattice.gamma:.4f}")

    new_gonio = replace(
        gonio,
        offsets=np.array(offsets_batch[0]) if refine_goniometer else gonio.base_offsets,
        rotation=np.array(R_batch[0]) if R_batch is not None else gonio.rotation
    )
    print(f"--- Refined Goniometer Offsets (deg): {new_gonio.offsets} ---")

    refined_state = replace(
        state,
        lattice=new_lattice,
        goniometer=new_gonio,
        ki_vec=np.array(ki_batch[0]) if refine_beam else state.ki_vec,
        base_sample_offset=np.array(sample_batch[0]) if refine_sample else state.base_sample_offset
    )


    _, accum_probs, hkl_final, lamda_final = objective_v.get_results(best_member)

    # Squeeze batch dim and convert to array
    hkl_final = np.array(hkl_final[0])
    lamda_final = np.array(lamda_final[0])
    accum_probs = np.array(accum_probs[0])

    mask = accum_probs > 0.5
    num_indexed = int(np.sum(mask))
    # Set non-indexed peaks to 0,0,0 to match JAX behavior
    hkl_final[~mask] = 0

    U_final = np.array(_rotation_matrix_from_rodrigues_numpy(final_x[:3]))

    result = Result(num_indexed, hkl_final, lamda_final, U_final, final_x)
    return result, refined_state


def _rotation_matrix_from_rodrigues_numpy(w):
    """NumPy version of Rodrigues rotation."""
    theta = np.linalg.norm(w) + 1e-9
    k = w / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    I = np.eye(3)  # noqa: E741
    R = I + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R
