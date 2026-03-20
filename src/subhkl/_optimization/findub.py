from dataclasses import replace
import h5py
import numpy as np
import scipy.interpolate
import scipy.linalg
import scipy.spatial
from tqdm import trange
import warnings

from subhkl.core.crystallography import Lattice
from subhkl.core.experiment import PeaksData, ExperimentData
from subhkl.core.models import LATTICE_CONFIG
from subhkl.core.spacegroup import get_centering
from subhkl.instrument.goniometer import Goniometer
from subhkl.instrument.detector import scattering_vector_from_angles
from subhkl._optimization.optimization import (
    VectorizedObjective,
    get_lattice_system,
    _get_active_lattice_indices,
    _forward_map_lattice,
    _forward_map_param,
    _inverse_map_param,
)

from subhkl.utils.shim import (
    CMA_ES,
    HAS_JAX,
    OPTIMIZATION_BACKEND,
    PSO,
    DifferentialEvolution,
    Mesh,
    NamedSharding,
    P,
    jax,
    jnp,
)


def require_jax():
    """
    Check if JAX is available and raise an informative error if not.

    Raises
    ------
    ImportError
        If JAX and evosax are not installed.
    """
    if not HAS_JAX:
        raise ImportError(
            "JAX and evosax are required for this functionality. "
            'Install with: pip install -e ".[jax]" or pip install jax jaxlib evosax'
        )


class FindUB:
    def __init__(self, data):
        self.lattice: Lattice
        self.goniometer: Goniometer
        self.peaks: PeaksData

        for key, value in vars(data).items():
            setattr(self, key, value)

        t = np.linspace(0, np.pi, 1024)
        cdf = (t - np.sin(t)) / np.pi
        self._angle_cdf = cdf
        self._angle_t = t

    def reciprocal_lattice_B(self):
        warnings.warn(
            "instead of self.reciprocal_lattice_B, use self.lattice.get_b_matrix()",
            category=DeprecationWarning,
            stacklevel=2,
        )
        if self.lattice is not None:
            return self.lattice.get_b_matrix()
        return Lattice(
            self.a, self.b, self.c, self.alpha, self.beta, self.gamma
        ).get_b_matrix()

    def _minimize_scipy(
        self,
        population_size: int = 1000,
        num_generations: int = 100,
        n_runs: int = 1,
        seed: int = 0,
        tolerance_deg: float = 0.1,
        softness: float = 0.01,
        loss_method: str = "gaussian",
        init_params: np.ndarray = None,
        refine_lattice: bool = False,
        lattice_bound_frac: float = 0.05,
        goniometer_axes: list = None,
        goniometer_angles: np.ndarray = None,
        refine_goniometer: bool = False,
        goniometer_bound_deg: float = 5.0,
        goniometer_names: list = None,
        refine_goniometer_axes: list = None,
        refine_sample: bool = False,
        sample_bound_meters: float = 2.0,
        refine_beam: bool = False,
        beam_bound_deg: float = 1.0,
        d_min: float = None,
        d_max: float = None,
        hkl_search_range: int = 20,
        B_sharpen: float = 50,
    ):
        """
        SciPy-based fallback for minimize when JAX is not available.
        Uses scipy.optimize.differential_evolution with VectorizedObjective.
        """
        from scipy.optimize import differential_evolution

        # 1. Prepare Metadata & Parameters
        if goniometer_axes is None and self.goniometer.axes is not None:
            goniometer_axes = self.goniometer.axes
        if goniometer_angles is None and self.goniometer.angles is not None:
            # SciPy path usually expects (num_axes, num_runs)
            goniometer_angles = self.goniometer.angles.T
        if goniometer_names is None and self.goniometer.names is not None:
            goniometer_names = self.goniometer.names

        # NOTE(vivek): lattice should already contained inferred system
        lattice_system = "Triclinic"
        num_lattice_params = 6
        if refine_lattice:
            lattice_system, num_lattice_params = get_lattice_system(
                self.a,
                self.b,
                self.c,
                self.alpha,
                self.beta,
                self.gamma,
                self.space_group,
            )
            print(
                f"Lattice System: {lattice_system} ({num_lattice_params} free params)"
            )

        goniometer_refine_mask = None
        if refine_goniometer:
            if refine_goniometer_axes is not None and goniometer_names is not None:
                mask = []
                for name in goniometer_names:
                    should_refine = any(req in name for req in refine_goniometer_axes)
                    mask.append(should_refine)
                goniometer_refine_mask = np.array(mask, dtype=bool)
            else:
                goniometer_refine_mask = np.ones(len(goniometer_axes), dtype=bool)

        # Determine number of dimensions for bounds setup
        num_dims = 3  # orientation
        if refine_lattice:
            num_dims += num_lattice_params
        if refine_sample:
            num_dims += 3
        if refine_beam:
            num_dims += 2
        if refine_goniometer:
            num_dims += np.sum(goniometer_refine_mask)

        # 2. Prepare Objective Input
        kf_ki_dir_lab = scattering_vector_from_angles(
            self.peaks.two_theta, self.peaks.azimuthal
        )

        # Use per-observation rotation if no refinement
        static_R_input = (
            self.goniometer.rotation
            if self.goniometer.rotation is not None
            else np.eye(3)
        )

        # Prepare weights
        snr = self.peaks.intensity / (self.peaks.sigma + 1e-6)
        if B_sharpen is not None:
            theta_rad = np.deg2rad(self.peaks.two_theta) / 2.0
            sin_sq_theta = np.sin(theta_rad) ** 2
            wilson_correction = np.exp(B_sharpen * sin_sq_theta)
            weights = snr * wilson_correction
            weights = weights / (np.mean(weights) + 1e-9)
        else:
            weights = snr
        weights = np.clip(weights, 0, 10.0)

        # Initialize VectorizedObjective (works with NumPy shim)
        lattice = self.lattice
        objective_v = VectorizedObjective(
            B=self.reciprocal_lattice_B(),
            kf_ki_dir=kf_ki_dir_lab,
            peak_xyz_lab=self.peaks.xyz.T if self.peaks.xyz is not None else None,
            wavelength=self.wavelength,
            angle_cdf=self._angle_cdf,
            angle_t=self._angle_t,
            weights=weights,
            tolerance_deg=tolerance_deg,
            cell_params=[
                lattice.a,
                lattice.b,
                lattice.c,
                lattice.alpha,
                lattice.beta,
                lattice.gamma,
            ],
            refine_lattice=refine_lattice,
            lattice_bound_frac=lattice_bound_frac,
            lattice_system=lattice_system,
            goniometer_axes=goniometer_axes,
            goniometer_angles=goniometer_angles,
            refine_goniometer=refine_goniometer,
            goniometer_bound_deg=goniometer_bound_deg,
            goniometer_refine_mask=goniometer_refine_mask,
            goniometer_nominal_offsets=self.goniometer.base_offsets,
            refine_sample=refine_sample,
            sample_bound_meters=sample_bound_meters,
            sample_nominal=self.base_sample_offset,
            refine_beam=refine_beam,
            beam_bound_deg=beam_bound_deg,
            beam_nominal=self.ki_vec,
            loss_method=loss_method,
            hkl_search_range=hkl_search_range,
            d_min=d_min,
            d_max=d_max if d_max is not None else 100.0,
            space_group=self.space_group,
            centering=get_centering(self.space_group),
            static_R=static_R_input,
            peak_run_indices=self.run_indices,
        )

        # 3. Setup Bounds
        bounds = [(-np.pi, np.pi)] * 3  # Orientation (Rodrigues)
        if refine_lattice:
            bounds += [(0.0, 1.0)] * num_lattice_params
        if refine_sample:
            bounds += [(0.0, 1.0)] * 3
        if refine_beam:
            bounds += [(0.0, 1.0)] * 2
        if refine_goniometer:
            bounds += [(0.0, 1.0)] * np.sum(goniometer_refine_mask)

        # Prepare initial guess
        x0 = None
        if init_params is not None:
            x0 = init_params
            if len(x0) < num_dims:
                x0 = np.concatenate([x0, np.full(num_dims - len(x0), 0.5)])
            elif len(x0) > num_dims:
                x0 = x0[:num_dims]

        # 4. Run Optimization
        print(f"\n--- Starting SciPy Differential Evolution ({n_runs} runs) ---")
        print(f"Population: {population_size}, Generations: {num_generations}")

        def objective_scipy(x):
            """Wrapper for SciPy to call VectorizedObjective."""
            if x.ndim == 1:
                return float(objective_v(x[None, :])[0])
            # SciPy's vectorized mode passes (dims, members)
            return np.array(objective_v(x.T))

        best_overall_fun = np.inf
        best_overall_x = None

        for i_run in range(n_runs):
            curr_seed = seed + i_run
            if n_runs > 1:
                print(f"Run {i_run + 1}/{n_runs} (Seed: {curr_seed})...")

            result = differential_evolution(
                objective_scipy,
                bounds,
                maxiter=num_generations,
                popsize=max(1, population_size // num_dims),
                seed=curr_seed,
                x0=x0 if i_run == 0 else None,
                vectorized=True,
                atol=0,
                tol=0.01,
            )

            if result.fun < best_overall_fun:
                best_overall_fun = result.fun
                best_overall_x = result.x

        print("\n--- Optimization Complete ---")
        print(f"Best DE score: {-best_overall_fun:.2f}")

        # --- Local Refinement (BFGS) ---
        print("Polishing solution with BFGS refinement...")
        from scipy.optimize import minimize as scipy_minimize

        res_ref = scipy_minimize(
            objective_scipy,
            best_overall_x,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 50},
        )

        if res_ref.success:
            if res_ref.fun < best_overall_fun:
                print(f"Refinement successful. Final score: {-res_ref.fun:.2f}")
                self.x = res_ref.x
            else:
                print(
                    f"Refinement increased cost from {best_overall_fun:.4f} "
                    f"to {res_ref.fun:.4f}. Reverting to best DE solution."
                )
                self.x = best_overall_x
        else:
            print(
                f"Refinement did not converge: {res_ref.message}. Keeping best DE solution."
            )
            self.x = best_overall_x

        # 5. Store Results
        best_member = self.x[None, :]

        # Extract physical parameters and update self
        phys_results = objective_v._get_physical_params_jax(best_member)

        (
            UB_batch,
            B_batch,
            sample_batch,
            ki_batch,
            offsets_batch,
            R_batch,
        ) = phys_results

        # Reconstruct lattice parameters
        if B_batch.ndim == 3:
            B_final = np.array(B_batch[0])
        else:
            B_final = np.array(B_batch)

        cell_params = self._cell_from_B_numpy(B_final)
        self.a, self.b, self.c = cell_params[:3]
        self.alpha, self.beta, self.gamma = cell_params[3:]

        if refine_lattice:
            print("--- Refined Lattice Parameters ---")
            print(f"a: {self.a:.4f}, b: {self.b:.4f}, c: {self.c:.4f}")
            print(
                f"alpha: {self.alpha:.4f}, beta: {self.beta:.4f}, gamma: {self.gamma:.4f}"
            )

        if refine_sample:
            self.sample_offset = np.array(sample_batch[0])
            print(f"--- Refined Sample Offset: {self.sample_offset} ---")

        if refine_beam:
            self.ki_vec = np.array(ki_batch[0])
            print(f"--- Refined Beam Vector: {self.ki_vec} ---")

        if refine_goniometer:
            self.goniometer.offsets = np.array(offsets_batch[0])
            print(
                f"--- Refined Goniometer Offsets (deg): {self.goniometer.offsets} ---"
            )

        if R_batch is not None:
            self.goniometer.rotation = np.array(R_batch[0])

        # 6. Final Result Generation
        _, accum_probs, hkl_final, lamda_final = objective_v.get_results(best_member)

        # Squeeze batch dim and convert to array
        hkl_final = np.array(hkl_final[0])
        lamda_final = np.array(lamda_final[0])
        accum_probs = np.array(accum_probs[0])

        # Calculate number of indexed peaks for logging
        mask = accum_probs > 0.5
        num_indexed = np.sum(mask)

        # Set non-indexed peaks to 0,0,0 to match JAX behavior
        hkl_final[~mask] = 0

        # Orientation
        rot_best = self.x[:3]
        U_final = np.array(self._rotation_matrix_from_rodrigues_numpy(rot_best))

        return int(num_indexed), hkl_final, lamda_final, U_final

    def _rotation_matrix_from_rodrigues_numpy(self, w):
        """NumPy version of Rodrigues rotation."""
        theta = np.linalg.norm(w) + 1e-9
        k = w / theta
        K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
        I = np.eye(3)  # noqa: E741
        R = I + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        return R

    def _cell_from_B_numpy(self, B):
        """Extract cell parameters from B matrix."""
        G_star = B.T @ B
        G = np.linalg.inv(G_star)

        a = np.sqrt(G[0, 0])
        b = np.sqrt(G[1, 1])
        c = np.sqrt(G[2, 2])

        alpha = np.rad2deg(np.arccos(G[1, 2] / (b * c)))
        beta = np.rad2deg(np.arccos(G[0, 2] / (a * c)))
        gamma = np.rad2deg(np.arccos(G[0, 1] / (a * b)))

        return np.array([a, b, c, alpha, beta, gamma])

    def minimize(
        self,
        strategy_name: str,
        population_size: int = 1000,
        num_generations: int = 100,
        n_runs: int = 1,
        seed: int = 0,
        tolerance_deg: float = 0.1,
        loss_method: str = "gaussian",
        init_params: np.ndarray | None = None,
        refine_lattice: bool = False,
        lattice_bound_frac: float = 0.05,
        goniometer_axes: list | None = None,
        goniometer_angles: np.ndarray | None = None,
        refine_goniometer: bool = False,
        goniometer_bound_deg: float = 5.0,
        goniometer_names: list | None = None,
        refine_goniometer_axes: list | None = None,
        refine_sample: bool = False,
        sample_bound_meters: float = 2.0,
        refine_beam: bool = False,
        beam_bound_deg: float = 1.0,
        d_min: float | None = None,
        d_max: float | None = None,
        hkl_search_range: int = 20,
        search_window_size: int = 256,
        window_batch_size: int = 32,
        chunk_size: int = 2048,
        num_iters: int = 20,
        top_k: int = 32,
        batch_size: int | None = None,
        sigma_init: float | None = None,
        softness: float = 0.01,
        B_sharpen: float = 50,
    ):
        """
        Minimize the objective using evolutionary strategies.

        When JAX is not available, falls back to SciPy's differential_evolution.
        """
        if not HAS_JAX:
            # Fall back to SciPy-based optimization
            print("JAX not available - using SciPy-based optimization")
            return self._minimize_scipy(
                population_size=population_size,
                num_generations=num_generations,
                n_runs=n_runs,
                seed=seed,
                tolerance_deg=tolerance_deg,
                softness=softness,
                loss_method=loss_method,
                init_params=init_params,
                refine_lattice=refine_lattice,
                lattice_bound_frac=lattice_bound_frac,
                goniometer_axes=goniometer_axes,
                goniometer_angles=goniometer_angles,
                refine_goniometer=refine_goniometer,
                goniometer_bound_deg=goniometer_bound_deg,
                goniometer_names=goniometer_names,
                refine_goniometer_axes=refine_goniometer_axes,
                refine_sample=refine_sample,
                sample_bound_meters=sample_bound_meters,
                refine_beam=refine_beam,
                beam_bound_deg=beam_bound_deg,
                d_min=d_min,
                d_max=d_max,
                hkl_search_range=hkl_search_range,
                B_sharpen=B_sharpen,
            )

        # JAX-based optimization (original code follows)
        if goniometer_axes is None and self.goniometer.axes is not None:
            goniometer_axes = self.goniometer.axes
        if goniometer_angles is None and self.goniometer.angles is not None:
            goniometer_angles = self.goniometer.angles.T
        if goniometer_names is None and self.goniometer.names is not None:
            goniometer_names = self.goniometer.names

        kf_ki_dir_lab = scattering_vector_from_angles(
            self.peaks.two_theta, self.peaks.azimuthal
        )
        num_obs = kf_ki_dir_lab.shape[1]

        # --- Gonio Mapping Fix ---
        # If goniometer data is per-peak, reduce it to per-run (image) IF AND ONLY IF
        # all peaks in a run share the same geometry. This saves memory in the
        # optimizer. If they differ, we MUST use per-peak indexing.
        static_R_input = (
            self.goniometer.rotation
            if self.goniometer.rotation is not None
            else np.eye(3)
        )
        if self.run_indices is not None:
            max_run_id = int(np.max(self.run_indices))
            num_runs_range = max_run_id + 1
            unique_runs, first_indices = np.unique(self.run_indices, return_index=True)

            # Check for intra-run variations
            def has_variation(data, indices):
                if data is None:
                    return False
                for r in unique_runs:
                    mask = indices == r
                    if np.sum(mask) <= 1:
                        continue
                    subset = data[mask] if data.ndim == 2 else data[mask, ...]
                    if not np.allclose(subset, subset[0:1], atol=1e-7):
                        return True
                return False

            can_reduce_angles = (
                goniometer_angles is not None
                and goniometer_angles.shape[1] == num_obs
                and not has_variation(goniometer_angles.T, self.run_indices)
            )
            can_reduce_R = (
                self.goniometer.rotation is not None
                and self.goniometer.rotation.ndim == 3
                and self.goniometer.rotation.shape[0] == num_obs
                and not has_variation(self.goniometer.rotation, self.run_indices)
            )

            if can_reduce_angles:
                # We have per-peak angles. We can reduce them to per-run.
                new_angles = np.zeros((goniometer_angles.shape[0], num_runs_range))
                new_angles[:] = goniometer_angles[:, first_indices[0:1]]
                new_angles[:, unique_runs] = goniometer_angles[:, first_indices]
                goniometer_angles = new_angles

            if can_reduce_R:
                # We have per-peak rotations. Reduce to per-run.
                new_R = np.zeros((num_runs_range, 3, 3))
                new_R[:] = self.goniometer.rotation[first_indices[0:1]]
                new_R[unique_runs] = self.goniometer.rotation[first_indices]
                static_R_input = new_R
            elif (
                self.goniometer.rotation is not None
                and self.goniometer.rotation.ndim == 3
                and self.goniometer.rotation.shape[0] == num_obs
            ):
                # Per-peak variation detected. Use per-peak mapping (peak_run_indices = 0..N)
                static_R_input = self.goniometer.rotation
                # This will trigger VectorizedObjective's per-peak mode (arange)
                self.run_indices = np.arange(num_obs, dtype=np.int32)

            # NEW: If gonio_angles is per-peak, also force per-peak mapping
            elif (
                goniometer_angles is not None and goniometer_angles.shape[1] == num_obs
            ):
                self.run_indices = np.arange(num_obs, dtype=np.int32)

        # Always use Lab frame vectors for Objective initialization.
        kf_ki_input = kf_ki_dir_lab

        goniometer_refine_mask = None
        if refine_goniometer and refine_goniometer_axes is not None:
            if self.goniometer.names is None:
                print(
                    "Warning: refine_goniometer_axes provided but goniometer_names not found. Refining ALL."
                )
            else:
                mask = []
                print(f"Refining specific goniometer axes: {refine_goniometer_axes}")
                for name in self.goniometer.names:
                    should_refine = any(req in name for req in refine_goniometer_axes)
                    mask.append(should_refine)
                goniometer_refine_mask = np.array(mask, dtype=bool)
                print(
                    f"Goniometer Mask: {goniometer_refine_mask} (Names: {self.goniometer.names})"
                )

        snr = self.peaks.intensity / (self.peaks.sigma + 1e-6)

        if B_sharpen is not None:
            theta_rad = np.deg2rad(self.peaks.two_theta) / 2.0
            sin_sq_theta = np.sin(theta_rad) ** 2
            wilson_correction = np.exp(B_sharpen * sin_sq_theta)
            weights = snr * wilson_correction
            weights = weights / np.mean(weights)
        else:
            weights = snr

        weights = np.clip(weights, 0, 10.0)

        lattice = self.lattice
        cell_params_init = np.array(
            [
                lattice.a,
                lattice.b,
                lattice.c,
                lattice.alpha,
                lattice.beta,
                lattice.gamma,
            ]
        )
        lattice_system, num_lattice_params = get_lattice_system(
            lattice.a,
            lattice.b,
            lattice.c,
            lattice.alpha,
            lattice.beta,
            lattice.gamma,
            self.space_group,
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

        objective = VectorizedObjective(
            self.reciprocal_lattice_B(),
            kf_ki_input,
            self.peaks.xyz,
            np.array(self.wavelength),
            self._angle_cdf,
            self._angle_t,
            weights=weights,
            tolerance_deg=tolerance_deg,
            space_group=self.space_group,
            centering=get_centering(self.space_group),
            loss_method=loss_method,
            cell_params=cell_params_init,
            peak_radii=self.peaks.radius,
            refine_lattice=refine_lattice,
            lattice_bound_frac=lattice_bound_frac,
            lattice_system=lattice_system,
            goniometer_axes=goniometer_axes,
            goniometer_angles=goniometer_angles,
            refine_goniometer=refine_goniometer,
            goniometer_refine_mask=goniometer_refine_mask,
            goniometer_nominal_offsets=self.goniometer.base_offsets,
            refine_sample=refine_sample,
            sample_bound_meters=sample_bound_meters,
            sample_nominal=self.base_sample_offset,
            refine_beam=refine_beam,
            beam_bound_deg=beam_bound_deg,
            beam_nominal=self.ki_vec,
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
            kf_lab_fixed_vectors=kf_ki_dir_lab,  # Pass raw Lab vectors
            peak_run_indices=self.run_indices,
        )
        print(
            f"Objective initialized with {loss_method} loss. Tolerance: {tolerance_deg} deg"
        )

        num_dims = 3
        if refine_lattice:
            num_dims += num_lattice_params
        if refine_sample:
            if self.peaks.xyz is None:
                refine_sample = False
            else:
                num_dims += 3
        if refine_beam:
            if self.peaks.xyz is None:
                refine_beam = False
            else:
                num_dims += 2
        if refine_goniometer:
            if goniometer_refine_mask is not None:
                num_dims += np.sum(goniometer_refine_mask)
            else:
                num_dims += len(goniometer_axes)

        start_sol_processed = None
        if init_params is not None:
            start_sol = jnp.array(init_params)
            if start_sol.shape[0] != num_dims:
                if start_sol.shape[0] < num_dims:
                    n_new = num_dims - start_sol.shape[0]
                    start_sol_processed = jnp.concatenate(
                        [start_sol, jnp.full((n_new,), 0.5)]
                    )
                else:
                    start_sol_processed = start_sol[:num_dims]
            else:
                start_sol_processed = start_sol

        sample_solution = jnp.zeros(num_dims)
        target_sigma = sigma_init or (0.01 if start_sol_processed is not None else 3.14)
        print(f"Strategy: {strategy_name.upper()} | Target Sigma: {target_sigma}")

        if strategy_name.lower() == "de":
            strategy = DifferentialEvolution(
                solution=sample_solution, population_size=population_size
            )
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
                    state = strategy.init(
                        rng_init, population_init, fitness_init, es_params
                    )
                else:
                    state = strategy.init(rng_init, start_sol, es_params)
                    state = state.replace(std=target_sigma)
            elif strategy_type == "population_based":
                pop_orient = (
                    jax.random.normal(rng_pop, (population_size, 3)) * target_sigma
                )
                rng_rest, _ = jax.random.split(rng_pop)
                pop_rest = jax.random.uniform(
                    rng_rest, (population_size, max(0, num_dims - 3))
                )
                population_init = jnp.concatenate([pop_orient, pop_rest], axis=1)
                fitness_init = objective(population_init)
                state = strategy.init(
                    rng_init, population_init, fitness_init, es_params
                )
            else:
                mean_orient = jnp.zeros(3)
                mean_rest = jnp.full((max(0, num_dims - 3),), 0.5)
                solution_init = jnp.concatenate([mean_orient, mean_rest])
                state = strategy.init(rng_init, solution_init, es_params)
                state = state.replace(std=target_sigma)
            return state

        mesh = Mesh(np.array(jax.devices()), ("i"))

        def step_single_run(rng, state):
            rng, rng_ask, rng_tell = jax.random.split(rng, 3)
            x, state_ask = strategy.ask(rng_ask, state, es_params)
            x_orient = x[:, :3]
            x_rest = jnp.clip(x[:, 3:], 0.0, 1.0)
            x_valid = jnp.concatenate([x_orient, x_rest], axis=1)
            fitness = objective(x_valid)

            # parallelize population across GPUs
            x_valid = jax.lax.with_sharding_constraint(
                x_valid, NamedSharding(mesh, P("i"))
            )

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

        pbar = range(num_generations)
        if trange is not None:
            pbar = trange(num_generations, desc="Optimizing")

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
            if trange is not None:
                if loss_method == "sinkhorn":
                    pbar.set_description(
                        f"Gen {gen + 1} | Cost: {current_gen_best:.4f}"
                    )
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
        best_overall_member = all_solutions[best_idx]

        # --- Local Refinement (BFGS) ---
        # Polishing the best member using JAX gradients for sub-arcsec precision
        print("Polishing solution with BFGS refinement...")

        # Use a small subset of generations for JAX JIT warm-up if needed,
        # but here we can just use scipy.optimize
        from scipy.optimize import minimize as scipy_minimize

        def ref_func(x_flat):
            # Objective returns a scalar (minimized)
            return float(objective(x_flat[None, :])[0])

        def ref_grad(x_flat):
            # Use jax.grad for the objective
            grad_fn = jax.grad(lambda x: objective(x[None, :])[0])
            return np.array(grad_fn(x_flat))

        res_ref = scipy_minimize(
            ref_func,
            np.array(best_overall_member),
            jac=ref_grad,
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

        self.x = np.array(best_overall_member)

        x_batch = jnp.array(self.x[None, :])
        (
            UB_final_batch,
            B_new_batch,
            s_total_batch,
            ki_vec_batch,
            offsets_total_batch,
            R_batch,
        ) = objective._get_physical_params_jax(x_batch)

        np.array(UB_final_batch[0])
        self.sample_offset = np.array(s_total_batch[0])
        self.ki_vec = np.array(ki_vec_batch[0]).flatten()
        if offsets_total_batch is not None:
            self.goniometer.offsets = np.array(offsets_total_batch[0])
        if R_batch is not None:
            # If R_batch is (S, N_runs, 3, 3), we want (N_runs, 3, 3) for the best member
            self.goniometer.rotation = np.array(R_batch[0])

        idx = 0
        rot_params = self.x[idx : idx + 3]
        idx += 3
        U = objective.orientation_U_jax(rot_params[None])[0]

        if refine_lattice:
            print("--- Refined Lattice Parameters ---")
            idx_lat = 3
            cell_norm = jnp.array(self.x[None, idx_lat : idx_lat + num_lattice_params])
            p_full = np.array(objective.reconstruct_cell_params(cell_norm)[0])
            print(f"a: {p_full[0]:.4f}, b: {p_full[1]:.4f}, c: {p_full[2]:.4f}")
            print(
                f"alpha: {p_full[3]:.4f}, beta: {p_full[4]:.4f}, gamma: {p_full[5]:.4f}"
            )
            self.a, self.b, self.c = p_full[0], p_full[1], p_full[2]
            self.alpha, self.beta, self.gamma = p_full[3], p_full[4], p_full[5]

        if refine_sample:
            print("--- Refined Sample Offset (mm) ---")
            print(
                f"X: {1000 * self.sample_offset[0]:.4f}, "
                f"Y: {1000 * self.sample_offset[1]:.4f}, "
                f"Z: {1000 * self.sample_offset[2]:.4f}"
            )

        if refine_beam:
            print("-- Refined Beam Direction ---")
            print(
                f"(ki_x, ki_y, ki_z): ({self.ki_vec[0]:.3f}, {self.ki_vec[1]:.3f}, {self.ki_vec[2]:.3f})"
            )

        if self.goniometer.offsets is not None:
            print("--- Refined Goniometer Offsets (deg) ---")
            if goniometer_names is not None:
                for name, val in zip(
                    goniometer_names, self.goniometer.offsets, strict=True
                ):
                    print(f"{name}: {val:.4f}")
            else:
                print(self.goniometer.offsets)

        # Final Score Recalculation using the unified pipeline
        score, accum_probs, hkl, lamb = objective.get_results(x_batch)

        num_peaks_soft = float(np.sum(accum_probs[0]))
        print(
            f"Final Solution indexed {num_peaks_soft:.2f}/{num_obs} "
            "peaks (unweighted count)."
        )

        return num_peaks_soft, np.array(hkl[0]), np.array(lamb[0]), np.array(U)

