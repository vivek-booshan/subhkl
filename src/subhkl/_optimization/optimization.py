from dataclasses import replace
from typing import Optional
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from subhkl.core import Lattice, ExperimentData
from subhkl.core.spacegroup import generate_hkl_mask, get_centering
from subhkl.instrument.detector import scattering_vector_from_angles

from ._helpers import get_physical_params
from ._types import RefinementConfig, IndexingConfig


class VectorizedObjective:
    def __init__(
        self,
        state: ExperimentData,
        rcfg: RefinementConfig = RefinementConfig(),
        icfg: IndexingConfig = IndexingConfig(),
    ):
        ## CONFIGURATION & FLAGS
        # Extract data availability
        # Determine actual refinement capability (Data vs Request)
        has_xyz = state.peaks.xyz is not None
        self.refine_lattice = rcfg.refine_lattice
        self.refine_goniometer = rcfg.refine_goniometer
        self.refine_sample = rcfg.refine_sample and has_xyz
        self.refine_beam = rcfg.refine_beam and has_xyz

        self.rcfg = replace(
            rcfg, refine_sample=self.refine_sample, refine_beam=self.refine_beam
        )
        self.icfg = icfg

        # Global Algorithm Settings
        self.tolerance_deg = icfg.tolerance_deg
        self.tolerance_rad = jnp.deg2rad(icfg.tolerance_deg)
        self.loss_method = icfg.loss_method
        self.space_group = state.space_group
        self.centering = get_centering(state.space_group)

        # Mathematical Constants
        t_arr = np.linspace(0, np.pi, 1024)
        self.angle_cdf = (t_arr - np.sin(t_arr)) / np.pi
        self.angle_t = t_arr

        ## LATTICE
        lattice = state.lattice
        self.B = lattice.get_b_matrix()
        self.cell_init = lattice.to_jax()
        self.lattice_system, num_lattice_params = Lattice.infer_system(
            lattice, state.space_group
        )
        self.lattice_bound_frac = rcfg.lattice_bound_frac

        # Initialize lattice search vectors
        active_idx = Lattice.get_active_indices(self.lattice_system)
        self.free_params_init = self.cell_init[jnp.array(active_idx)]

        ## PEAKS and BEAM
        peaks = state.peaks
        self.kf_ki_dir = scattering_vector_from_angles(peaks.two_theta, peaks.azimuthal)
        if self.kf_ki_dir.ndim == 2 and self.kf_ki_dir.shape[0] != 3:
            self.kf_ki_dir = self.kf_ki_dir.T

        self.num_obs = self.kf_ki_dir.shape[1]
        self.k_sq_init = jnp.sum(self.kf_ki_dir**2, axis=0)

        # Detector/Sample Geometry
        self.peak_xyz = jnp.array(peaks.xyz).T if has_xyz else None
        self.sample_nominal = (
            jnp.array(state.base_sample_offset)
            if state.base_sample_offset is not None
            else jnp.zeros(3)
        )
        self.sample_bound = rcfg.sample_bound_meters

        self.beam_nominal = (
            jnp.array(state.ki_vec)
            if state.ki_vec is not None
            else jnp.array([0.0, 0.0, 1.0])
        )
        self.beam_bound_deg = rcfg.beam_bound_deg

        # Peak weighting/physics metadata
        self.weights = jnp.array(peaks.refine_weights(icfg.B_sharpen)).flatten()
        self.peak_radii = (
            jnp.array(peaks.radius).flatten()
            if peaks.radius is not None
            else jnp.zeros(self.num_obs)
        )
        self.max_score = jnp.sum(self.weights)

        ## Goniometer and run mapping
        goniometer = state.goniometer
        raw_angles = goniometer.angles.T if goniometer.angles is not None else None

        # Map Lab -> Sample frames
        resolved_angles, static_R, peak_run_indices = _resolve_goniometer_mapping(
            state, self.num_obs, raw_angles
        )

        self.static_R = jnp.array(static_R) if static_R is not None else jnp.eye(3)
        self.peak_run_indices = (
            jnp.array(peak_run_indices, dtype=jnp.int32)
            if peak_run_indices is not None
            else jnp.zeros(self.num_obs, dtype=jnp.int32)
        )

        # Finalize Static_R Broadcasting and Index Clamping
        if self.static_R.ndim == 3:
            num_rot = self.static_R.shape[0]
            max_run = jnp.max(self.peak_run_indices)
            if max_run >= num_rot and num_rot == 1:
                self.static_R = jnp.tile(self.static_R, (max_run + 1, 1, 1))
            self.peak_run_indices = jnp.clip(
                self.peak_run_indices, 0, self.static_R.shape[0] - 1
            )

        # Goniometer Refinement Logic
        if goniometer.axes is not None:
            self.gonio_axes = jnp.array(goniometer.axes)
            if self.gonio_axes.ndim == 2 and self.gonio_axes.shape[1] == 3:
                self.gonio_axes = jnp.concatenate(
                    [self.gonio_axes, jnp.ones((self.gonio_axes.shape[0], 1))], axis=1
                )

            self.gonio_angles = (
                jnp.array(resolved_angles).T
                if jnp.array(resolved_angles).ndim == 2
                and jnp.array(resolved_angles).shape[0] != self.gonio_axes.shape[0]
                else jnp.array(resolved_angles)
            )
            self.num_gonio_axes = self.gonio_axes.shape[0]
            self.gonio_nominal_offsets = (
                jnp.array(goniometer.base_offsets)
                if goniometer.base_offsets is not None
                else jnp.zeros(self.num_gonio_axes)
            )
            self.goniometer_bound_deg = rcfg.goniometer_bound_deg

            # Masking
            gonio_names = goniometer.names
            if (
                self.refine_goniometer
                and rcfg.refine_goniometer_axes is not None
                and gonio_names
            ):
                self.gonio_mask = np.array(
                    [
                        any(req in n for req in rcfg.refine_goniometer_axes)
                        for n in gonio_names
                    ],
                    dtype=bool,
                )
            else:
                self.gonio_mask = np.ones(self.num_gonio_axes, dtype=bool)
            self.num_active_gonio = int(np.sum(self.gonio_mask))
        else:
            self.gonio_axes = None
            self.num_gonio_axes = self.num_active_gonio = 0
            self.gonio_mask = jnp.array([], dtype=bool)
            self.gonio_nominal_offsets = self.gonio_min = self.gonio_max = jnp.array([])
            self.gonio_angles = jnp.empty((0, self.num_obs))

        ## SOLVER STATE SPACE (# of DIMS)
        dims = 3  # Orientation (Rodrigues)
        dims += num_lattice_params if self.refine_lattice else 0
        dims += 3 if self.refine_sample else 0
        dims += 2 if self.refine_beam else 0
        dims += self.num_active_gonio if self.refine_goniometer else 0
        self.num_dims = dims

        ## OPTICAL RECONSTRUCTION (KF_LAB)
        self.kf_lab_fixed = None
        if self.peak_xyz is not None:
            v = self.peak_xyz - self.sample_nominal[:, None]
            dist = jnp.linalg.norm(v, axis=0)
            self.kf_lab_fixed = v / jnp.where(dist == 0, 1.0, dist[None, :])
        else:
            # Fallback to Lab vectors if XYZ detector geometry is missing
            q_vecs = self.kf_ki_dir
            self.kf_lab_fixed = q_vecs + self.beam_nominal[:, None]
            self.kf_lab_fixed /= jnp.linalg.norm(self.kf_lab_fixed, axis=0)

        ## SEARCH POOL & HKL POOL GENERATION
        self.d_min = icfg.d_min if icfg.d_min is not None else 0.0
        self.d_max = icfg.d_max if icfg.d_max is not None else 1000.0
        self.search_window_size = icfg.search_window_size
        self.window_batch_size = icfg.window_batch_size
        self.chunk_size = icfg.chunk_size
        self.num_iters = icfg.num_iters
        self.top_k = icfg.top_k
        self.wl_min_val, self.wl_max_val = jnp.array(state.wavelength)
        self.num_candidates = 64

        # HKL Range Calculation
        a_real, b_real, c_real = jnp.sqrt(jnp.diag(jnp.linalg.inv(self.B @ self.B.T)))
        q_obs_max = jnp.max(jnp.linalg.norm(self.kf_ki_dir, axis=0))
        d_limit = self.d_min if self.d_min > 0 else 1.0 / (q_obs_max + 1e-9)

        h_max = min(max(icfg.hkl_search_range, int(jnp.ceil(a_real / d_limit))), 64)
        k_max = min(max(icfg.hkl_search_range, int(jnp.ceil(b_real / d_limit))), 64)
        l_max = min(max(icfg.hkl_search_range, int(jnp.ceil(c_real / d_limit))), 64)

        # Pool Generation
        r_h, r_k, r_l = (
            jnp.arange(-h_max, h_max + 1),
            jnp.arange(-k_max, k_max + 1),
            jnp.arange(-l_max, l_max + 1),
        )
        h, k, l = jnp.meshgrid(r_h, r_k, r_l, indexing="ij")
        hkl_pool = jnp.stack([h.flatten(), k.flatten(), l.flatten()], axis=0)

        mask_cpu = generate_hkl_mask(h_max, k_max, l_max, self.space_group)
        self.mask_range_h = h_max
        self.mask_range_k = k_max
        self.mask_range_l = l_max
        self.mask_range = self.mask_range_h 
        self.valid_hkl_mask = jnp.array(mask_cpu)
        allowed = self.valid_hkl_mask[
            hkl_pool[0] + h_max, hkl_pool[1] + k_max, hkl_pool[2] + l_max
        ]
        self.pool_hkl_flat = hkl_pool[:, allowed]

        # Sorting for search efficiency
        q_cart = self.B @ self.pool_hkl_flat
        phis = jnp.arctan2(q_cart[1], q_cart[0])
        sort_idx = jnp.argsort(phis)
        self.pool_phi_sorted = phis[sort_idx]
        self.pool_hkl_sorted = self.pool_hkl_flat[:, sort_idx]

        ## PINNING & INITIAL HEURISTICS
        B_inv = jnp.linalg.inv(self.B)
        h_init = B_inv @ self.kf_ki_dir
        self.hkl_mag_sq_pinned = jnp.maximum(
            jnp.sum(h_init**2, axis=0, keepdims=True), 1e-6
        )

        k_dot_q = jnp.sum(self.kf_ki_dir * self.kf_ki_dir, axis=0)
        self.safe_lamb_pinned = jnp.clip(
            self.k_sq_init / jnp.maximum(k_dot_q, 1e-9),
            self.wl_min_val,
            self.wl_max_val,
        )[None, :]

        pad_len = (
            icfg.chunk_size - (self.pool_hkl_flat.shape[1] % icfg.chunk_size)
        ) % icfg.chunk_size
        hkl_padded = (
            jnp.pad(self.pool_hkl_flat, ((0, 0), (0, pad_len)))
            if pad_len > 0
            else self.pool_hkl_flat
        )
        self.pool_norm_q_pinned = jnp.sqrt(
            jnp.sum((self.B @ hkl_padded) ** 2, axis=0) + 1e-9
        )

    @partial(jax.jit, static_argnames="self")
    def get_results(self, x):
        """Full physical model and indexing pipeline for a batch of solutions x."""
        # --- WORKAROUND for JAX/ROCm S=1 bug ---
        # On some AMD backends (e.g. MI200), JITted functions with lax.scan
        # can produce incorrect results when the leading batch dimension is exactly 1.
        # We force a minimum batch size of 2 by duplicating the input if necessary.
        original_S = x.shape[0]
        pad_size = max(0, 2 - original_S)
        x_pad = jnp.pad(x, ((0, pad_size), (0, 0)), mode="edge")

        UB, _, sample_total, ki_vec, _, R = get_physical_params(
            x_pad,
            self.refine_lattice,
            self.free_params_init,
            self.B,
            self.refine_sample,
            self.sample_bound,
            self.sample_nominal,
            self.refine_beam,
            self.beam_bound_deg,
            self.beam_nominal,
            self.refine_goniometer,
            self.num_gonio_axes,
            self.num_active_gonio,
            self.gonio_mask,
            self.goniometer_bound_deg,
            self.gonio_nominal_offsets,
            self.gonio_angles,
            self.gonio_axes,
        )

        # Determine current rotations (Lab -> Sample)
        R_curr = R  # (S, N_runs, 3, 3) or None
        if R_curr is None:
            # Fallback to static rotations
            R_curr = self.static_R  # (N_runs, 3, 3) or (3, 3)

        # Expand rotations to per-observation if mapping is provided
        if R_curr is not None:
            if R_curr.ndim == 4:
                # (S, N_runs, 3, 3) -> (S, N_peaks, 3, 3)
                R_per_peak = R_curr[:, self.peak_run_indices, :, :]
            elif R_curr.ndim == 3:
                # (N_runs, 3, 3) -> (N_peaks, 3, 3)
                R_per_peak = R_curr[self.peak_run_indices, :, :]
            else:
                # (3, 3)
                R_per_peak = R_curr
        else:
            R_per_peak = None

        # Determine current scattered beam directions (kf) in Lab frame
        if self.peak_xyz is not None:
            # Recalculate kf for every run/peak because the sample position s_lab
            # depends on the goniometer rotation R and the refined sample offset.
            if R_per_peak is not None:
                if R_per_peak.ndim == 4:
                    # (S, N, 3, 3) @ (S, 3, 1) -> (S, N, 3, 1)
                    s_lab = jnp.matmul(
                        R_per_peak, sample_total[:, None, :, None]
                    ).squeeze(-1)
                    s = s_lab.transpose(0, 2, 1)  # (S, 3, N)
                elif R_per_peak.ndim == 3:
                    # Broadcasted Matmul: (1, N, 3, 3) @ (S, 1, 3, 1) -> (S, N, 3, 1)
                    s_lab = jnp.matmul(
                        R_per_peak[None, ...], sample_total[:, None, :, None]
                    ).squeeze(-1)
                    s = s_lab.transpose(0, 2, 1)
                else:
                    # (3, 3) @ (S, 3, 1) -> (S, 3, 1)
                    s_lab = jnp.matmul(
                        R_per_peak[None, ...], sample_total[:, :, None]
                    ).squeeze(-1)
                    s = s_lab[:, :, None]
            else:
                s = sample_total[:, :, None]

            p = self.peak_xyz[None, :, :]  # (1, 3, N)
            v = p - s  # (S, 3, N)
            dist = jnp.sqrt(jnp.sum(v**2, axis=1, keepdims=True))
            kf = v / jnp.where(dist == 0, 1.0, dist)
            ki = ki_vec[:, :, None]  # (S, 3, 1)
            q_lab = kf - ki
            k_sq_dyn = jnp.sum(q_lab**2, axis=1)
        else:
            kf = self.kf_lab_fixed[None, :, :].repeat(x.shape[0], axis=0)
            ki = ki_vec[:, :, None]
            q_lab = kf - ki
            k_sq_dyn = jnp.sum(q_lab**2, axis=1)

        # Rotate to SAMPLE FRAME: q_sample = R^T * q_lab
        # (Inputs are always in Lab frame and require rotation)
        if R_per_peak is not None:
            # q_lab is (S, 3, N). We want (S, N, 3, 1) for matmul
            q_lab_vec = q_lab.transpose(0, 2, 1)[..., None]
            if R_per_peak.ndim == 4:
                # (S, N, 3, 3) and (S, N, 3, 1)
                # q_sample = R.T @ q_lab
                RT = R_per_peak.transpose(0, 1, 3, 2)
                kf_ki_vec_T = jnp.matmul(RT, q_lab_vec).squeeze(-1)
            elif R_per_peak.ndim == 3:
                # (1, N, 3, 3) and (S, N, 3, 1)
                RT = R_per_peak.transpose(0, 2, 1)[None, ...]
                kf_ki_vec_T = jnp.matmul(RT, q_lab_vec).squeeze(-1)
            else:
                # (1, 3, 3) and (S, N, 3, 1)
                RT = R_per_peak.T[None, None, ...]
                kf_ki_vec_T = jnp.matmul(RT, q_lab_vec).squeeze(-1)
            kf_ki_vec = kf_ki_vec_T.transpose(0, 2, 1)
        else:
            kf_ki_vec = q_lab

        if self.loss_method == "forward":
            from .indexers.binary import binary_indexer

            res = binary_indexer(
                UB,
                kf_ki_vec,
                self.B,
                self.k_sq_init,
                self.wl_min_val,
                self.wl_max_val,
                self.d_min,
                self.d_max,
                self.peak_radii,
                self.weights,
                self.search_window_size,
                self.pool_hkl_sorted,
                self.pool_phi_sorted,
                self.centering,
                self.mask_range_h,
                self.mask_range_k,
                self.mask_range_l,
                self.valid_hkl_mask,
                k_sq_override=k_sq_dyn,
                tolerance_rad=self.tolerance_rad,
                window_batch_size=self.window_batch_size,
            )
        elif self.loss_method == "cosine":
            from .indexers.cosine import cosine_indexer

            res = cosine_indexer(
                UB,
                self.wl_min_val,
                self.wl_max_val,
                self.d_min,
                self.d_max,
                self.k_sq_init,
                self.num_candidates,
                self.weights,
                self.centering,
                self.mask_range_h,
                self.mask_range_k,
                self.mask_range_l,
                self.valid_hkl_mask,
                kf_ki_vec,
                k_sq_override=k_sq_dyn,
                tolerance_rad=self.tolerance_rad,
            )
        elif self.loss_method == "sinkhorn":
            from .indexers.sinkhorn import sinkhorn_indexer

            res = sinkhorn_indexer(
                UB,
                self.pool_hkl_flat,
                self.k_sq_init,
                self.wl_min_val,
                self.wl_max_val,
                self.d_min,
                self.d_max,
                self.pool_norm_q_pinned,
                self.weights,
                kf_ki_vec,
                k_sq_override=k_sq_dyn,
                tolerance_rad=self.tolerance_rad,
                chunk_size=self.chunk_size,
                num_iters=self.num_iters,
                top_k=self.top_k,
            )
        else:
            from .indexers.soft import soft_indexer

            res = soft_indexer(
                UB,
                kf_ki_vec,
                self.wl_min_val,
                self.wl_max_val,
                self.k_sq_init,
                self.peak_radii,
                self.d_min,
                self.d_max,
                self.weights,
                self.num_candidates,
                self.centering,
                self.mask_range_h,
                self.mask_range_k,
                self.mask_range_l,
                self.valid_hkl_mask,
                k_sq_override=k_sq_dyn,
                tolerance_rad=self.tolerance_rad,
            )

        # Slice results back to original batch size (Workaround cleanup)
        return jax.tree.map(
            lambda arr: (
                arr[:original_S] if hasattr(arr, "shape") and arr.ndim > 0 else arr
            ),
            res,
        )

    @partial(jax.jit, static_argnames="self")
    def __call__(self, x):
        score, _, _, _ = self.get_results(x)
        return score

def _resolve_goniometer_mapping(
    state: ExperimentData, num_obs: int, goniometer_angles: Optional[np.ndarray]
):
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
