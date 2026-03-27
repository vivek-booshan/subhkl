import jax
import jax.numpy as jnp

def sinkhorn_indexer(
        ub_mat,
        pool_hkl_flat,
        k_sq_init,
        wl_min_val,
        wl_max_val,
        d_min,
        d_max,
        pool_norm_q_pinned,
        weights,
        kf_ki_sample,
        k_sq_override=None,
        tolerance_rad=0.002,
        num_iters=20,
        epsilon=1.0,
        top_k=32,
        chunk_size=256,
    ):
        """
        Robust Memory-Efficient Sinkhorn with Soft-Masking and Log-Stability.
        """
        # 1. Setup Data
        hkl_pool = pool_hkl_flat  # (3, N_hkl)

        # Normalize Obs
        norm_obs = jnp.linalg.norm(kf_ki_sample, axis=1, keepdims=True)
        r_obs_unit = kf_ki_sample / (norm_obs + 1e-9)

        # Re-project Unit Obs into Crystal Frame: (Batch, 3, N_obs) @ (Batch, 3, 3) -> (Batch, 3, N_obs)
        # We need r_obs_unit_crystal = U^T @ r_obs_unit_lab
        # Batched matmul: (S, 3, 3) @ (S, 3, N) -> (S, 3, N)
        r_obs_proj_unit = jnp.matmul(ub_mat.transpose(0, 2, 1), r_obs_unit)

        k_sq_obs = (
            k_sq_override if k_sq_override is not None else k_sq_init[None, :]
        )

        batch_size, _, n_obs = kf_ki_sample.shape
        _, n_hkl = hkl_pool.shape

        # Bandwidth and Resolution Constants for Penalties
        wl_mid = 0.5 * (wl_min_val + wl_max_val)
        wl_half_width = 0.5 * (wl_max_val - wl_min_val)
        res_mid = 0.5 * (d_min + d_max)
        res_half_width = 0.5 * (d_max - d_min)

        # Pad pool for chunking
        pad_len = (chunk_size - (n_hkl % chunk_size)) % chunk_size
        hkl_pool_padded = (
            jnp.pad(hkl_pool, ((0, 0), (0, pad_len)), constant_values=0)
            if pad_len > 0
            else hkl_pool
        )
        n_hkl_padded = n_hkl + pad_len

        num_chunks = n_hkl_padded // chunk_size

        def scan_topk(carry, i):
            curr_vals, curr_idxs = carry
            idx_start = i * chunk_size
            hkl_chunk = jax.lax.dynamic_slice(
                hkl_pool_padded, (0, idx_start), (3, chunk_size)
            )

            # dot_raw = (r_obs @ U) . h
            # r_obs_proj_unit is (Batch, 3, N_obs), hkl_chunk is (3, Chunk)
            # Result (Batch, N_obs, Chunk)
            # (S, 3, N).T @ (3, C) -> (S, N, C)
            dot_raw = jnp.matmul(r_obs_proj_unit.transpose(0, 2, 1), hkl_chunk)

            # cosine = dot_raw / |UB h|

            # Use pinned norms to prevent the optimizer from 'cheating' by
            # enlarging the lattice to reduce the predicted |Q|.
            norm_q_chunk_pinned = jax.lax.dynamic_slice(
                pool_norm_q_pinned, (idx_start,), (chunk_size,)
            )
            dots_chunk = dot_raw / (norm_q_chunk_pinned[None, None, :] + 1e-9)

            # --- FIX: Resolution & Wavelength Aware Top-K ---
            # lambda = k_sq_obs / (norm_obs * dot_raw)
            # norm_obs is (batch, 1, n_obs), dot_raw is (batch, n_obs, chunk)
            safe_dot = jnp.maximum(dot_raw, 1e-6)
            est_lambda = k_sq_obs[:, :, None] / (
                norm_obs.transpose(0, 2, 1) * safe_dot + 1e-9
            )

            wl_penalty = -jnp.abs(est_lambda - wl_mid) / (wl_half_width + 1e-9)

            # norm_q_chunk_pinned is (chunk,)
            d_chunk = 1.0 / (norm_q_chunk_pinned + 1e-9)
            res_penalty = -jnp.abs(d_chunk - res_mid) / (res_half_width + 1e-9)

            selection_metric = (
                dots_chunk + 0.1 * wl_penalty + 0.1 * res_penalty[None, None, :]
            )

            # Handle padded 0,0,0 vectors (norm 0) by ensuring they have low metrics
            selection_metric = jnp.where(
                norm_q_chunk_pinned[None, None, :] < 1e-6, -1e9, selection_metric
            )

            global_idxs = (jnp.arange(chunk_size) + idx_start).astype(jnp.int32)
            combined_vals = jnp.concatenate([curr_vals, selection_metric], axis=2)
            combined_idxs = jnp.concatenate(
                [
                    curr_idxs,
                    jnp.tile(global_idxs[None, None, :], (batch_size, n_obs, 1)),
                ],
                axis=2,
            )

            vals, top_k_indices = jax.lax.top_k(combined_vals, top_k)
            idxs = jnp.take_along_axis(combined_idxs, top_k_indices, axis=2)
            return (vals, idxs), None

        (top_vals, top_idxs), _ = jax.lax.scan(
            scan_topk,
            (
                jnp.full((batch_size, n_obs, top_k), -1e9),
                jnp.zeros((batch_size, n_obs, top_k), dtype=jnp.int32),
            ),
            jnp.arange(num_chunks),
        )

        # 3. Log-Kernel with Soft Penalties
        # Gather HKL vectors and re-calculate full geometry for top-k
        hkl_selected = jnp.take(hkl_pool_padded.T, top_idxs, axis=0)
        # ub_mat: (S, 3, 3), hkl_selected: (S, N, K, 3)
        # We want (S, N, K, 3)
        q_selected = jnp.matmul(
            ub_mat[:, None, None, ...], hkl_selected[..., None]
        ).squeeze(-1)
        q_sq_selected = jnp.sum(q_selected**2, axis=3)
        norm_q_selected = jnp.sqrt(q_sq_selected + 1e-9)

        # Actual cosines for top-k HKLs
        # (Batch, 3, N_obs) -> (Batch, N_obs, 3)
        k_obs_unit = r_obs_unit.transpose(0, 2, 1)[:, :, None, :]
        dot_selected = jnp.sum(k_obs_unit * q_selected, axis=3)
        top_cosines = dot_selected / (norm_q_selected + 1e-9)

        # Wavelength penalty
        k_obs_aligned = kf_ki_sample.transpose(0, 2, 1)[:, :, None, :]
        k_dot_q = jnp.sum(k_obs_aligned * q_selected, axis=-1)
        lambda_sparse = k_sq_obs[:, :, None] / (k_dot_q + 1e-9)

        # Soft Lambda Penalty (Gaussian penalty for being outside bandwidth)
        # Using a broader width (10% of bandwidth) to prevent numerical
        # drowning of angular signal
        bw_width = wl_max_val - wl_min_val
        dist_wl = jnp.maximum(0.0, wl_min_val - lambda_sparse) + jnp.maximum(
            0.0, lambda_sparse - wl_max_val
        )
        # Scale: lambda penalty should be comparable to angular penalty (order of 1-10)
        # dist_wl of 0.1A should not give 1e5 cost.
        log_P_wl = -0.5 * (dist_wl / (0.1 * bw_width + 1e-9)) ** 2

        # Soft Resolution Penalty
        d_sparse = 1.0 / norm_q_selected
        dist_res = jnp.maximum(0.0, d_min - d_sparse) + jnp.maximum(
            0.0, d_sparse - d_max
        )
        log_P_res = -0.5 * (dist_res / (0.1 * d_min + 1e-9)) ** 2

        # Angular kernel (Multi-scale: Peak + Wide Background)
        # Using a mixture of a narrow peak and a heavy-tailed background ensures
        # we have a strong gradient near the peak and a stable signal far away.
        dist_ang = 1.0 - top_cosines

        # log_K_peak = -dist_ang / tolerance^2
        # log_K_wide = -log(1 + dist_ang / (wide_tol^2))
        # We use LogSumExp to smoothly combine them
        log_K_peak = -dist_ang / (tolerance_rad**2 + 1e-9)

        wide_tol = jnp.deg2rad(5.0)  # Always have a 5 degree capture range
        log_K_wide = -jnp.log(1.0 + dist_ang / (wide_tol**2 + 1e-9))

        # Combine (mixing weight 0.5 implicitly via LogSumExp if we don't scale)
        log_K = jax.nn.logsumexp(jnp.stack([log_K_peak, log_K_wide]), axis=0)

        # Combine into robust log-likelihood
        log_K_robust = log_K + log_P_wl + log_P_res

        # --- TIE-BREAKER PENALTIES ---
        # If multiple HKLs have identical orientation error (cosines),
        # prefer the one that matches the expected wavelength and resolution center.
        # This breaks ties caused by the regularizer (1e-9) favoring larger vectors.
        log_P_wl_tie = -1e-4 * jnp.abs(lambda_sparse - wl_mid) / (wl_half_width + 1e-9)
        log_P_res_tie = -1e-4 * jnp.abs(d_sparse - res_mid) / (res_half_width + 1e-9)
        log_K_robust += log_P_wl_tie + log_P_res_tie

        # 5. Dustbin & Softmax
        # Dustbin represents the "null" HKL match
        # Match to dustbin if outside the wide capture range (e.g. 3 * wide_tol)
        outlier_threshold_rad = jnp.minimum(jnp.deg2rad(45.0), 3.0 * wide_tol)
        dist_outlier = 1.0 - jnp.cos(outlier_threshold_rad)
        log_K_dustbin_peak = -dist_outlier / (tolerance_rad**2 + 1e-9)
        log_K_dustbin_wide = -jnp.log(1.0 + dist_outlier / (wide_tol**2 + 1e-9))
        log_K_dustbin = jax.nn.logsumexp(
            jnp.stack([log_K_dustbin_peak, log_K_dustbin_wide])
        )
        log_K_dustbin = jnp.full((batch_size, n_obs, 1), log_K_dustbin)

        log_K_extended = jnp.concatenate([log_K_robust, log_K_dustbin], axis=2)
        log_P_softmax = log_K_extended - jax.nn.logsumexp(
            log_K_extended, axis=2, keepdims=True
        )

        # 6. Score
        # We want to maximize the probability of matching ANY valid HKL (non-outlier).
        # Optimization is MINIMIZATION, so we return the negative probability sum.
        log_P_match = log_P_softmax[:, :, :-1]
        log_prob_any = jax.nn.logsumexp(log_P_match, axis=2)
        score = -jnp.sum(weights * jnp.exp(log_prob_any), axis=1)

        # Metrics for reporting
        best_k_idx = jnp.argmax(log_P_match, axis=2)
        best_hkl_idx = jnp.take_along_axis(
            top_idxs, best_k_idx[:, :, None], axis=2
        ).squeeze(2)
        best_hkl = jnp.take(hkl_pool_padded.T, best_hkl_idx, axis=0)
        best_lamb = jnp.take_along_axis(
            lambda_sparse, best_k_idx[:, :, None], axis=2
        ).squeeze(2)

        return score, jnp.exp(log_prob_any), best_hkl, best_lamb
