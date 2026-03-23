from typing import Optional, List, Tuple
from dataclasses import replace

from evosax.algorithms import CMA_ES, PSO, DifferentialEvolution as DE
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np
from scipy.optimize import minimize as scipy_minimize

from subhkl._optimization.solver import Result
from subhkl.core.experiment import ExperimentData
from subhkl.core.crystallography import Lattice
from subhkl.core.spacegroup import get_centering
from subhkl.instrument.detector import scattering_vector_from_angles
from subhkl._optimization.optimization import VectorizedObjective

def _jax_minimize(
    state: ExperimentData,
    strategy_name: str,
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
    search_window_size: int = 256,
    window_batch_size: int = 32,
    chunk_size: int = 2048,
    num_iters: int = 20,
    top_k: int = 32,
    batch_size: Optional[int] = None,
    sigma_init: Optional[float] = None,
    softness: float = 0.01,
    B_sharpen: float = 50,
) -> Tuple[Result, ExperimentData]:

    refine_flags = {
        "lattice": refine_lattice,
        "goniometer": refine_goniometer,
        "beam": refine_beam,
        "sample": refine_sample
    }

    goniometer_axes = state.goniometer.axes
    goniometer_names = state.goniometer.names
    goniometer_angles = state.goniometer.angles.T if state.goniometer.angles is not None else None

    kf_ki_dir_lab = scattering_vector_from_angles(state.peaks.two_theta, state.peaks.azimuthal)
    num_obs = kf_ki_dir_lab.shape[1]

    goniometer_angles, static_R_input, run_indices =_resolve_goniometer_mapping(state, num_obs, goniometer_angles)
    kf_ki_input = kf_ki_dir_lab

    goniometer_refine_mask = None
    if refine_goniometer and refine_goniometer_axes is not None:
        if goniometer_names is None:
            print(
                "Warning: refine_goniometer_axes provided but goniometer_names not found. Refining ALL."
            )
        else:
            print(f"Refining specific goniometer axes: {refine_goniometer_axes}")
            goniometer_refine_mask = np.array(
                [
                    any(req in name for req in refine_goniometer_axes)
                    for name in goniometer_names
                ],
                dtype=bool,
            )
            print(
                f"Goniometer Mask: {goniometer_refine_mask} (Names: {goniometer_names})"
            )

    weights = state.peaks.refine_weights(B_sharpen)
    lattice_system, num_lattice_params = Lattice.infer_system(
        state.lattice, state.space_group
    )

    if refine_lattice:
        print("Lattice Refinement Enabled.")
        print(
            f"Detected System: {lattice_system} ({num_lattice_params} free parameters)."
        )

    if loss_method == "forward" and (d_min is None or d_max is None):
        raise ValueError(
            "Need to supply --d_min and --d_max for loss_method=='forward'"
        )

    has_xyz = state.peaks.xyz is not None
    refine_sample = refine_sample and has_xyz
    refine_beam = refine_beam and has_xyz

    num_dims = 3
    num_dims += num_lattice_params if refine_lattice else 0
    num_dims += 3 if refine_sample else 0
    num_dims += 2 if refine_beam else 0

    if refine_goniometer:
        num_dims += int(np.sum(goniometer_refine_mask)) if goniometer_refine_mask is not None else len(goniometer_axes)


    start_sol_processed = None
    if init_params is not None:
        start_sol = jnp.array(init_params)
        n_current = start_sol.shape[0]

        if n_current < num_dims:
            padding = jnp.full((num_dims - n_current, ), 0.5)
            start_sol_processed = jnp.concatenate([start_sol, padding])
        else:
            start_sol_processed = start_sol[:num_dims]

    sample_solution = jnp.zeros(num_dims)
    target_sigma = sigma_init or (0.01 if start_sol_processed is not None else 3.14)
    print(f"Strategy: {strategy_name.upper()} | Target Sigma: {target_sigma}")

    if strategy_name.lower() == "de":
        strategy = DE(solution=sample_solution, population_size=population_size)
        strategy_type = "population_based"
    elif strategy_name.lower() == "pso":
        strategy = PSO(solution=sample_solution, population_size=population_size)
        strategy_type = "population_based"
    elif strategy_name.lower() == "cma_es":
        strategy = CMA_ES(solution=sample_solution, population_size=population_size)
        strategy_type = "distribution_based"
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    es_params = strategy.default_params

    t_arr = np.linspace(0, np.pi, 1024)
    angle_cdf = (t_arr - np.sin(t_arr)) / np.pi
    angle_t = t_arr
    objective = VectorizedObjective(
        state.lattice.get_b_matrix(),
        kf_ki_input,
        state.peaks.xyz,
        np.array(state.wavelength),
        angle_cdf,
        angle_t,
        weights=weights,
        tolerance_deg=tolerance_deg,
        space_group=state.space_group,
        centering=get_centering(state.space_group),
        loss_method=loss_method,
        cell_params=state.lattice.to_numpy(),
        peak_radii=state.peaks.radius,
        refine_lattice=refine_lattice,
        lattice_bound_frac=lattice_bound_frac,
        lattice_system=lattice_system,
        goniometer_axes=goniometer_axes,
        goniometer_angles=goniometer_angles,
        refine_goniometer=refine_goniometer,
        goniometer_refine_mask=goniometer_refine_mask,
        goniometer_nominal_offsets=state.goniometer.base_offsets,
        refine_sample=refine_sample,
        sample_bound_meters=sample_bound_meters,
        sample_nominal=state.base_sample_offset,
        refine_beam=refine_beam,
        beam_bound_deg=beam_bound_deg,
        beam_nominal=state.ki_vec,
        goniometer_bound_deg=goniometer_bound_deg,
        hkl_search_range=hkl_search_range,
        d_min=d_min,
        d_max=d_max,
        window_batch_size=window_batch_size,
        search_window_size=search_window_size,
        chunk_size=chunk_size,
        num_iters=num_iters,
        top_k=top_k,
        static_R=static_R_input,
        kf_lab_fixed_vectors=kf_ki_dir_lab,
        peak_run_indices=run_indices,
    )
    print(
        f"Objective initialized with {loss_method} loss. Tolerance: {tolerance_deg} deg"
    )


    def init_single_run(rng, start_sol):
        rng, rng_pop, rng_init = jax.random.split(rng, 3)

        if start_sol is not None:
            if strategy_type == "population_based":
                noise = (
                    jax.random.normal(rng_pop, (population_size, num_dims))
                    * target_sigma
                )
                p_orient = start_sol[:3] + noise[:, :3]
                p_rest = jnp.clip(start_sol[3:] + noise[:, 3:], 0.0, 1.0)
                population_init = jnp.concatenate([p_orient, p_rest], axis=1)
                fitness_init = objective(population_init)
                es_state = strategy.init(
                    rng_init, population_init, fitness_init, es_params
                )
            else:
                es_state = strategy.init(rng_init, start_sol, es_params)
                es_state = es_state.replace(std=target_sigma)
        elif strategy_type == "population_based":
            pop_orient = jax.random.normal(rng_pop, (population_size, 3)) * target_sigma
            rng_rest, _ = jax.random.split(rng_pop)
            pop_rest = jax.random.uniform(
                rng_rest, (population_size, max(0, num_dims - 3))
            )
            population_init = jnp.concatenate([pop_orient, pop_rest], axis=1)
            fitness_init = objective(population_init)
            es_state = strategy.init(rng_init, population_init, fitness_init, es_params)
        else:
            mean_orient = jnp.zeros(3)
            mean_rest = jnp.full((max(0, num_dims - 3),), 0.5)
            solution_init = jnp.concatenate([mean_orient, mean_rest])
            es_state = strategy.init(rng_init, solution_init, es_params)
            es_state = es_state.replace(std=target_sigma)
        return es_state

    mesh = Mesh(np.array(jax.devices()), ("i"))

    def step_single_run(rng, es_state):
        rng, rng_ask, rng_tell = jax.random.split(rng, 3)
        x, state_ask = strategy.ask(rng_ask, es_state, es_params)
        x_orient = x[:, :3]
        x_rest = jnp.clip(x[:, 3:], 0.0, 1.0)
        x_valid = jnp.concatenate([x_orient, x_rest], axis=1)
        fitness = objective(x_valid)

        x_valid = jax.lax.with_sharding_constraint(x_valid, NamedSharding(mesh, P("i")))

        state_tell, metrics = strategy.tell(
            rng_tell, x_valid, fitness, state_ask, es_params
        )
        return rng, state_tell, metrics

    init_batch_jit = jax.jit(jax.vmap(init_single_run, in_axes=(0, None)))
    step_batch_jit = jax.jit(jax.vmap(step_single_run, in_axes=(0, 0)))

    exec_batch_size = batch_size if batch_size is not None else n_runs
    print(f"\n--- Starting {n_runs} Runs (Batch Size: {exec_batch_size}) ---")

    seeds = jnp.arange(seed, seed + n_runs)
    all_keys = jax.vmap(jax.random.PRNGKey)(seeds)

    batch_keys_list = []
    batch_states_list = []

    for b_i in range(int(np.ceil(n_runs / exec_batch_size))):
        start_idx = b_i * exec_batch_size
        end_idx = min((b_i + 1) * exec_batch_size, n_runs)
        b_keys = all_keys[start_idx:end_idx]
        b_state = init_batch_jit(b_keys, start_sol_processed)
        batch_keys_list.append(b_keys)
        batch_states_list.append(b_state)

    try:
        from tqdm import trange

        pbar = trange(num_generations, desc="Optimizing")
    except ImportError:
        pbar = range(num_generations)

    for gen in pbar:
        current_gen_best = np.inf
        for b_i in range(len(batch_keys_list)):
            curr_keys = batch_keys_list[b_i]
            curr_state = batch_states_list[b_i]
            next_keys, next_state, _ = step_batch_jit(curr_keys, curr_state)
            batch_keys_list[b_i] = next_keys
            batch_states_list[b_i] = next_state
            b_min = jnp.min(next_state.best_fitness)
            current_gen_best = min(current_gen_best, b_min)

        if hasattr(pbar, "set_description"):
            if loss_method == "sinkhorn":
                pbar.set_description(f"Gen {gen + 1} | Cost: {current_gen_best:.4f}")
            else:
                pbar.set_description(
                    f"Gen {gen + 1} | Best: {-current_gen_best:.1f}/{num_obs}"
                )

    all_fitness_list = []
    all_solutions_list = []
    for b_state in batch_states_list:
        all_fitness_list.append(b_state.best_fitness)
        all_solutions_list.append(b_state.best_solution)

    all_fitness = jnp.concatenate(all_fitness_list, axis=0)
    all_solutions = jnp.concatenate(all_solutions_list, axis=0)


    best_idx = np.argmin(all_fitness)
    best_overall_fitness = all_fitness[best_idx]
    best_overall_member = np.array(all_solutions[best_idx])

    # --- Local Refinement (BFGS) ---
    print("Polishing solution with BFGS refinement...")

    scipy_obj_wrapper = _get_jitted_value_and_grad(objective)
    res_ref = scipy_minimize(
        scipy_obj_wrapper,
        best_overall_member,
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.0, 1.0) if i >= 3 else (None, None) for i in range(num_dims)],
        options={"maxiter": 50},
    )

    if res_ref.success:
        if res_ref.fun < best_overall_fitness:
            print(f"Refinement successful. Final cost: {res_ref.fun:.4f}")
            best_overall_member = res_ref.x
            best_overall_fitness = res_ref.fun
        else:
            print(
                f"Refinement increased cost from {best_overall_fitness:.4f} "
                f"to {res_ref.fun:.4f}. Reverting to best DE solution."
            )
    else:
        print(
            f"Refinement did not converge: {res_ref.message}. Keeping best DE solution."
        )

    print("\n--- Optimization Complete ---")
    if loss_method == "sinkhorn":
        print(f"Best overall cost: {best_overall_fitness:.4f}")
    else:
        print(f"Best overall peaks: {-best_overall_fitness:.2f}")

    x_batch = jnp.array(best_overall_member[None, :])
    phys_results = objective._get_physical_params_jax(x_batch)
    refined_state = _build_refined_state(state, phys_results, refine_flags)

    idx = 0
    rot_params = best_overall_member[idx : idx + 3]
    idx += 3
    U = objective.orientation_U_jax(rot_params[None])[0]

    _print_refined_parameters(refined_state, refine_flags, goniometer_names)

    # Final Score Recalculation using the unified pipeline
    score, accum_probs, hkl, lamb = objective.get_results(x_batch)

    num_peaks_soft = float(np.sum(accum_probs[0]))
    print(
        f"Final Solution indexed {num_peaks_soft:.2f}/{num_obs} "
        "peaks (unweighted count)."
    )

    mask = np.array(accum_probs[0]) > 0.5
    num_indexed = int(np.sum(mask))

    hkl_final = np.array(hkl[0])
    hkl_final[~mask] = 0

    lamda_final = np.array(lamb[0])

    result = Result(
        num_indexed=num_indexed,
        hkl=hkl_final,
        wavelengths=lamda_final,
        U=np.array(U),
        x=best_overall_member,
        state=refined_state
    )

    return result

def _resolve_goniometer_mapping(state: ExperimentData, num_obs: int, goniometer_angles: Optional[np.ndarray]):
    run_indices = state.run_indices
    static_R_input = (
        state.goniometer.rotation
        if state.goniometer.rotation is not None
        else np.eye(3)
    )
    if run_indices is not None:
        num_runs_range = int(np.max(run_indices)) + 1
        unique_runs, first_indices = np.unique(run_indices, return_index=True)

        # Check for intra-run variations
        def has_variation(data, indices):
            if data is None:
                return False
            for r in unique_runs:
                subset = data[indices == r]
                if len(subset) > 1 and not np.allclose(subset, subset[0]):
                    return True
            return False

        angles_per_peak = (
            goniometer_angles is not None
            and goniometer_angles.shape[1] == num_obs
            and not has_variation(goniometer_angles.T, run_indices)
        )
        R_per_peak = (
            state.goniometer.rotation is not None
            and state.goniometer.rotation.ndim == 3
            and state.goniometer.rotation.shape[0] == num_obs
            and not has_variation(state.goniometer.rotation, run_indices)
        )

        if angles_per_peak:
            new_angles = np.zeros((goniometer_angles.shape[0], num_runs_range))
            new_angles[:] = goniometer_angles[:, first_indices[0:1]]
            new_angles[:, unique_runs] = goniometer_angles[:, first_indices]
            goniometer_angles = new_angles

        if R_per_peak:
            new_R = np.zeros((num_runs_range, 3, 3))
            new_R[:] = state.goniometer.rotation[first_indices[0:1]]
            new_R[unique_runs] = state.goniometer.rotation[first_indices]
            static_R_input = new_R
        elif (
            state.goniometer.rotation is not None
            and state.goniometer.rotation.ndim == 3
            and state.goniometer.rotation.shape[0] == num_obs
        ):
            static_R_input = state.goniometer.rotation
            run_indices = np.arange(num_obs, dtype=np.int32)

        elif goniometer_angles is not None and goniometer_angles.shape[1] == num_obs:
            run_indices = np.arange(num_obs, dtype=np.int32)

        return goniometer_angles, static_R_input, run_indices

def _build_refined_state(state: ExperimentData, phys_results: tuple, refine: dict):    
    (UB_batch, B_batch, s_batch, ki_batch, off_batch, R_batch) = phys_results

    new_lattice = (
        Lattice.from_b_matrix(np.array(B_batch[0]))
        if refine["lattice"] else state.lattice
    )
    new_gonio = replace(
        state.goniometer,
        offsets=np.array(off_batch[0]) if refine["goniometer"] else state.goniometer.offsets,
        rotation=np.array(R_batch[0]) if R_batch is not None else state.goniometer.rotation,
    )
    return replace(
        state,
        lattice=new_lattice,
        goniometer=new_gonio,
        ki_vec=np.array(ki_batch[0]).flatten() if refine["beam"] else state.ki_vec,
        sample_offset=np.array(s_batch[0]) if refine["sample"] else state.sample_offset,
    )

def _print_refined_parameters(
    refined_state: ExperimentData, 
    flags: dict, 
    goniometer_names: list | None = None
):
    """Logs the refined physical parameters to the console."""
    if flags.get("refine_lattice"):
        lat = refined_state.lattice
        print("--- Refined Lattice Parameters ---")
        print(f"a: {lat.a:.4f}, b: {lat.b:.4f}, c: {lat.c:.4f}")
        print(f"alpha: {lat.alpha:.4f}, beta: {lat.beta:.4f}, gamma: {lat.gamma:.4f}")

    if flags.get("refine_sample"):
        print("--- Refined Sample Offset (mm) ---")
        s_off = refined_state.base_sample_offset
        print(f"X: {1000 * s_off[0]:.4f}, Y: {1000 * s_off[1]:.4f}, Z: {1000 * s_off[2]:.4f}")

    if flags.get("refine_beam"):
        print("-- Refined Beam Direction ---")
        ki_v = refined_state.ki_vec
        print(f"(ki_x, ki_y, ki_z): ({ki_v[0]:.3f}, {ki_v[1]:.3f}, {ki_v[2]:.3f})")

    # Tied to both the existence of offsets and the refinement flag
    if flags.get("refine_goniometer") and refined_state.goniometer.offsets is not None:
        print("--- Refined Goniometer Offsets (deg) ---")
        if goniometer_names is not None:
            for name, val in zip(goniometer_names, refined_state.goniometer.offsets, strict=True):
                print(f"{name}: {val:.4f}")
        else:
            print(refined_state.goniometer.offsets)

    return

def _get_jitted_value_and_grad(objective: VectorizedObjective):

    @jax.jit
    def val_and_grad_fn(x_flat):
        def loss_fn(x):
            return objective(x[None, :])[0]
        return jax.value_and_grad(loss_fn)(x_flat)
    
    def scipy_wrapper(x_numpy):
        val, grad = val_and_grad_fn(jnp.array(x_numpy))
        return float(val), np.array(grad)
        
    return scipy_wrapper
