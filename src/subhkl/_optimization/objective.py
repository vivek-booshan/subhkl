import warnings
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from subhkl.core import Lattice
from subhkl.core.spacegroup import generate_hkl_mask

from ._helpers import get_physical_params
from ._types import RefinementConfig, IndexingConfig


class Objective:
    def __init__(
        self,
        B,
        kf_ki_dir,
        peak_xyz_lab,
        wavelength,
        angle_cdf,
        angle_t,
        weights=None,
        cell_params=None,
        rcfg: RefinementConfig = RefinementConfig(),
        icfg: IndexingConfig = IndexingConfig(),
        lattice_system="Triclinic",
        goniometer_axes=None,
        goniometer_angles=None,
        goniometer_refine_mask=None,
        goniometer_nominal_offsets=None,
        sample_nominal=None,
        beam_nominal=None,
        peak_radii=None,
        space_group="P 1",
        centering="P",
        static_R=None,
        kf_lab_fixed_vectors=None,
        peak_run_indices=None,
    ):
        self.B = jnp.array(B)
        self.kf_ki_dir_init = jnp.array(kf_ki_dir)
        if self.kf_ki_dir_init.ndim == 2:
            if self.kf_ki_dir_init.shape[0] != 3 and self.kf_ki_dir_init.shape[1] == 3:
                self.kf_ki_dir_init = self.kf_ki_dir_init.T

        self.k_sq_init = jnp.sum(self.kf_ki_dir_init**2, axis=0)
        num_peaks = self.kf_ki_dir_init.shape[1]

        self.centering = centering

        # Convert tolerance from degrees to radians
        self.tolerance_rad = jnp.deg2rad(icfg.tolerance_deg)

        # FIX: Handle Static Rotation (R) correctly
        if static_R is not None:
            self.static_R = jnp.array(static_R)
        else:
            self.static_R = jnp.eye(3)

        # Handle Peak-to-Run mapping metadata
        if peak_run_indices is not None:
            self.peak_run_indices = jnp.array(peak_run_indices, dtype=jnp.int32)
            # Validation: Ensure R stack is large enough for the max run_index
            if self.static_R.ndim == 3:
                max_run = jnp.max(self.peak_run_indices)
                num_rot = self.static_R.shape[0]
                if max_run >= num_rot:
                    # If we only have ONE rotation, broadcast it to match the peaks
                    if num_rot == 1:
                        self.static_R = jnp.tile(self.static_R, (max_run + 1, 1, 1))
                    else:
                        # Major mismatch: Force everything to run 0 to prevent crash, but warn
                        # (JAX doesn't warn easily in JIT, so we'll just clamp later)
                        pass
        # Default heuristic:
        # 1. If R is a stack of N rotations and we have N peaks, assume 1-to-1 mapping.
        elif self.static_R.ndim == 3:
            num_rotations = self.static_R.shape[0]
            if num_rotations == num_peaks:
                self.peak_run_indices = jnp.arange(num_peaks, dtype=jnp.int32)
            else:
                # Fallback: everything to run 0 if we can't decide
                self.peak_run_indices = jnp.zeros(num_peaks, dtype=jnp.int32)
        else:
            self.peak_run_indices = jnp.zeros(num_peaks, dtype=jnp.int32)

        # Final safety: Clamp run indices to R stack bounds to prevent UB in JAX
        if self.static_R.ndim == 3:
            self.peak_run_indices = jnp.clip(
                self.peak_run_indices, 0, self.static_R.shape[0] - 1
            )

        if peak_xyz_lab is not None:
            # peak_xyz_lab is (N, 3) or (3, N). We want (3, N).
            p_xyz = jnp.array(peak_xyz_lab)
            if p_xyz.shape[0] != 3 and p_xyz.shape[1] == 3:
                p_xyz = p_xyz.T
            self.peak_xyz = p_xyz
        else:
            self.peak_xyz = None

        self.refine_sample = rcfg.refine_sample
        self.sample_bound = rcfg.sample_bound_meters
        if sample_nominal is None:
            self.sample_nominal = jnp.zeros(3)
        else:
            self.sample_nominal = jnp.array(sample_nominal)

        self.refine_beam = rcfg.refine_beam
        self.beam_bound_deg = rcfg.beam_bound_deg
        if beam_nominal is None:
            self.beam_nominal = jnp.array([0.0, 0.0, 1.0])
        else:
            self.beam_nominal = jnp.array(beam_nominal)

        # Reconstruct kf from Q (kf = Q + ki)
        self.kf_lab_fixed = None
        if self.peak_xyz is not None:
            # Always calculate kf from physical detector positions and sample offset
            # peak_xyz is (3, N), sample_nominal is (3,)
            v = self.peak_xyz - self.sample_nominal[:, None]
            dist = jnp.linalg.norm(v, axis=0)
            self.kf_lab_fixed = v / jnp.where(dist == 0, 1.0, dist[None, :])
            # Input is in LAB frame, so it is NOT yet rotated to Sample frame.

        if kf_lab_fixed_vectors is not None and self.kf_lab_fixed is None:
            # Input was Lab Frame. Q_lab = kf_lab - ki_lab.
            q_vecs = jnp.array(kf_lab_fixed_vectors)
            if q_vecs.shape[0] != 3 and q_vecs.shape[1] == 3:
                q_vecs = q_vecs.T
            self.kf_lab_fixed = q_vecs + self.beam_nominal[:, None]
            self.kf_lab_fixed = self.kf_lab_fixed / jnp.linalg.norm(
                self.kf_lab_fixed, axis=0
            )

        if self.kf_lab_fixed is None:
            # Fallback
            q_vecs = self.kf_ki_dir_init
            if q_vecs.shape[0] != 3 and q_vecs.shape[1] == 3:
                q_vecs = q_vecs.T
            self.kf_lab_fixed = q_vecs + self.beam_nominal[:, None]
            self.kf_lab_fixed = self.kf_lab_fixed / jnp.linalg.norm(
                self.kf_lab_fixed, axis=0
            )
            # FIX: Lab angles (two_theta, azimuthal) are ALWAYS in Lab frame.
            # We must ensure the optimizer
            # applies the Lab -> Sample rotation (R^T) during objective evaluation.

        self.tolerance_deg = icfg.tolerance_deg
        self.loss_method = icfg.loss_method
        self.angle_cdf = jnp.array(angle_cdf)
        self.angle_t = jnp.array(angle_t)
        self.space_group = space_group

        self.refine_lattice = rcfg.refine_lattice
        self.lattice_system = lattice_system
        self.lattice_bound_frac = rcfg.lattice_bound_frac
        self.refine_goniometer = rcfg.refine_goniometer
        self.goniometer_bound_deg = rcfg.goniometer_bound_deg

        self.free_params_init = None
        if self.refine_lattice:
            if cell_params is None:
                self.cell_init = jnp.array(cell_params)
                self.free_params_init = self.cell_init[
                    Lattice.get_active_indices(self.lattice_system)
                ]

        if goniometer_axes is not None:
            axes = jnp.array(goniometer_axes)
            if axes.ndim == 2 and axes.shape[1] == 3:
                # Fallback for 3-component axes: add 1.0 orientation (CCW)
                axes = jnp.concatenate([axes, jnp.ones((axes.shape[0], 1))], axis=1)
            self.gonio_axes = axes

            angles = jnp.array(goniometer_angles)
            if angles.ndim == 2:
                # Expecting (num_axes, num_runs). If (num_runs, num_axes), transpose.
                if (
                    angles.shape[0] != self.gonio_axes.shape[0]
                    and angles.shape[1] == self.gonio_axes.shape[0]
                ):
                    angles = angles.T
            self.gonio_angles = angles
            self.num_gonio_axes = self.gonio_axes.shape[0]

            # CRITICAL: If gonio_angles is per-peak, force per-peak run mapping
            # to ensure R_per_peak = R[peak_run_indices] works correctly.
            if self.gonio_angles.shape[1] == num_peaks:
                self.peak_run_indices = jnp.arange(num_peaks, dtype=jnp.int32)

            if goniometer_refine_mask is not None:
                self.gonio_mask = np.array(goniometer_refine_mask, dtype=bool)
            else:
                self.gonio_mask = np.ones(self.num_gonio_axes, dtype=bool)
            self.num_active_gonio = np.sum(self.gonio_mask)

            if goniometer_nominal_offsets is None:
                self.gonio_nominal_offsets = jnp.zeros(self.num_gonio_axes)
            else:
                self.gonio_nominal_offsets = jnp.array(goniometer_nominal_offsets)

            self.gonio_min = jnp.full(self.num_gonio_axes, -rcfg.goniometer_bound_deg)
            self.gonio_max = jnp.full(self.num_gonio_axes, rcfg.goniometer_bound_deg)
        else:
            self.gonio_axes = None
            self.num_gonio_axes = 0
            self.num_active_gonio = 0
            self.gonio_mask = jnp.array([], dtype=bool)
            self.gonio_nominal_offsets = jnp.array([])
            self.gonio_min = jnp.array([])
            self.gonio_max = jnp.array([])
            self.gonio_angles = jnp.empty((0, num_peaks))

        wavelength = jnp.array(wavelength)
        self.wl_min_val = wavelength[0]
        self.wl_max_val = wavelength[1]
        self.num_candidates = 64

        if weights is None:
            self.weights = jnp.ones(num_peaks)
        else:
            self.weights = jnp.array(weights).flatten()
            if self.weights.shape[0] != num_peaks:
                raise ValueError(
                    f"Weights shape {self.weights.shape} does not match num_peaks {num_peaks}"
                )

        if peak_radii is None:
            self.peak_radii = jnp.zeros(num_peaks)
        else:
            self.peak_radii = jnp.array(peak_radii).flatten()
            if self.peak_radii.shape[0] != num_peaks:
                raise ValueError(
                    f"Peak radii shape {self.peak_radii.shape} does not match num_peaks {num_peaks}"
                )

        self.max_score = jnp.sum(self.weights)
        self.d_min = icfg.d_min if icfg.d_min is not None else 0.0
        self.d_max = icfg.d_max if icfg.d_max is not None else 1000.0
        self.search_window_size = icfg.search_window_size
        self.window_batch_size = icfg.window_batch_size

        self.chunk_size = icfg.chunk_size
        self.num_iters = icfg.num_iters
        self.top_k = icfg.top_k

        # --- Search Window Heuristic Warning ---
        if self.loss_method == "forward":
            # Calculate Volume (Real Space) from B matrix (Reciprocal Basis, 2pi included)
            # V_real = (2pi)^3 / det(B)
            det_B = float(np.abs(np.linalg.det(self.B)))
            if det_B > 1e-9:
                vol_real = 1.0 / det_B
                # Peak Density Heuristic: N approx Vol / d^3
                # Factor 0.0025 empirically determined for +/- 2 deg coverage on MANDI
                heuristic_win = int((vol_real / (self.d_min**3)) * 0.0025)
                # Clamp for sanity in warning logic
                heuristic_win = max(64, heuristic_win)

                if self.search_window_size < (heuristic_win * 0.75):
                    warnings.warn(
                        f"\n[WARNING] search_window_size ({self.search_window_size}) is likely too small "
                        f"for resolution {self.d_min:.2f}A and Volume {vol_real:.0f}A^3.\n"
                        f"Binary search indexer may miss valid peaks.\n"
                        f"RECOMMENDED SIZE: >= {heuristic_win}\n",
                        stacklevel=2,
                    )

        # --- HKL Mask Generation ---
        # Robustly determine search range from cell and observed resolution
        # inv(B @ B.T) = inv(G*) = G (Real space metric tensor)
        # sqrt(diag(G)) = [a, b, c]
        a_real, b_real, c_real = jnp.sqrt(jnp.diag(jnp.linalg.inv(self.B @ self.B.T)))

        # Calculate resolution of observed peaks
        q_obs_max = jnp.max(jnp.linalg.norm(self.kf_ki_dir_init, axis=0))
        d_min_obs = 1.0 / (q_obs_max + 1e-9)

        # Determine pool resolution limit
        d_limit = self.d_min if self.d_min > 0 else d_min_obs

        # h_max = a / d_min
        h_max_res = int(jnp.ceil(a_real / d_limit))
        k_max_res = int(jnp.ceil(b_real / d_limit))
        l_max_res = int(jnp.ceil(c_real / d_limit))
        h_max = max(icfg.hkl_search_range, h_max_res)
        k_max = max(icfg.hkl_search_range, k_max_res)
        l_max = max(icfg.hkl_search_range, l_max_res)

        # Clamp to a reasonable maximum to prevent OOM
        h_max = min(h_max, 64)
        k_max = min(k_max, 64)
        l_max = min(l_max, 64)

        print(
            f"Generating HKL pool for Space Group: {self.space_group} (Range: {h_max},{k_max},{l_max})"
        )

        r_h = jnp.arange(-h_max, h_max + 1)
        r_k = jnp.arange(-k_max, k_max + 1)
        r_l = jnp.arange(-l_max, l_max + 1)
        h, k, l = jnp.meshgrid(r_h, r_k, r_l, indexing="ij")  # noqa: E741
        hkl_pool = jnp.stack([h.flatten(), k.flatten(), l.flatten()], axis=0)

        # Apply Symmetry Mask to Pool
        mask_cpu = generate_hkl_mask(h_max, k_max, l_max, self.space_group)
        self.valid_hkl_mask = jnp.array(mask_cpu)
        self.mask_range_h = h_max
        self.mask_range_k = k_max
        self.mask_range_l = l_max
        self.mask_range = self.mask_range_h

        idx_h = hkl_pool[0] + h_max
        idx_k = hkl_pool[1] + k_max
        idx_l = hkl_pool[2] + l_max
        allowed_pool = self.valid_hkl_mask[idx_h, idx_k, idx_l]

        hkl_pool = hkl_pool[:, allowed_pool]
        q_cart = self.B @ hkl_pool

        # NOTE: For Sinkhorn, we keep the flat pool available directly
        self.pool_hkl_flat = hkl_pool

        phis = jnp.arctan2(q_cart[1], q_cart[0])
        sort_idx = jnp.argsort(phis)
        self.pool_phi_sorted = phis[sort_idx]
        self.pool_hkl_sorted = hkl_pool[:, sort_idx]

        # --- PINNING INITIALIZATION ---
        # Pre-calculate reference HKL and Q magnitudes to prevent lattice bias
        # in derivative-free optimization. Assume identity orientation for pinning.
        B_inv_init = jnp.linalg.inv(self.B)
        h_init = B_inv_init @ self.kf_ki_dir_init
        self.hkl_mag_sq_pinned = jnp.sum(h_init**2, axis=0, keepdims=True)
        self.hkl_mag_sq_pinned = jnp.maximum(self.hkl_mag_sq_pinned, 1e-6)

        # Reference Lambda for soft/binary kernels
        # lambda = |k|^2 / (k . Q)
        k_dot_q_init = jnp.sum(self.kf_ki_dir_init * self.kf_ki_dir_init, axis=0)
        self.safe_lamb_pinned = jnp.clip(
            self.k_sq_init / jnp.maximum(k_dot_q_init, 1e-9),
            self.wl_min_val,
            self.wl_max_val,
        )[None, :]

        # Reference Pool Norms for Sinkhorn
        # Use padded pool to match chunked indexing in sinkhorn
        pad_len = (
            icfg.chunk_size - (hkl_pool.shape[1] % icfg.chunk_size)
        ) % icfg.chunk_size
        hkl_pool_padded = (
            jnp.pad(hkl_pool, ((0, 0), (0, pad_len)), constant_values=0)
            if pad_len > 0
            else hkl_pool
        )
        q_pool_init = self.B @ hkl_pool_padded
        self.pool_norm_q_pinned = jnp.sqrt(jnp.sum(q_pool_init**2, axis=0) + 1e-9)

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

        # UB, _, sample_total, ki_vec, _, R = self._get_physical_params_jax(x_pad)
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
            from .indexers import binary_indexer

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
            from .indexers import cosine_indexer

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
            from .indexers import sinkhorn_indexer

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
            from .indexers import soft_indexer

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
