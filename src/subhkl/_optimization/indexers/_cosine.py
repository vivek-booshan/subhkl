import jax
import jax.numpy as jnp

from ._utils import check_hkl_symmetry


def cosine_indexer(
    ub_mat,
    wl_min_val,
    wl_max_val,
    d_min,
    d_max,
    k_sq_init,
    num_candidates,
    weights,
    centering,
    mask_range_h,
    mask_range_k,
    mask_range_l,
    valid_hkl_mask,
    kf_ki_sample,
    *,
    k_sq_override=None,
    tolerance_rad=0.002,
):
    # Use solve for better precision than inv + matmul
    v = jnp.linalg.solve(ub_mat, kf_ki_sample)
    abs_v = jnp.abs(v)
    max_v_val = jnp.max(abs_v, axis=1)
    n_start = max_v_val / wl_max_val
    start_int = jnp.ceil(n_start)

    k_sq = k_sq_override if k_sq_override is not None else k_sq_init[None, :]

    # kappa for von Mises-Fisher-like concentration in HKL space
    # Uniform angular tolerance: sigma_h approx tolerance_rad * h.

    initial_carry = (
        jnp.full(max_v_val.shape, -1e12),
        jnp.full(max_v_val.shape, -1e12),
        jnp.zeros((v.shape[0], 3, v.shape[2]), dtype=jnp.int32),
        jnp.zeros(max_v_val.shape),
    )

    def scan_body(carry, i):
        curr_sum, curr_max, curr_best_hkl, curr_best_lamb = carry
        n = start_int + i
        n_safe = jnp.where(n == 0, 1e-9, n)
        lamda_cand = max_v_val / n_safe

        # --- DYNAMIC WAVELENGTH OPTIMIZATION ---
        # Instead of just using lamda_cand, we find the lambda that best
        # satisfies the Laue condition for the nearest integer HKL.
        hkl_int = jnp.round(v / lamda_cand[:, None, :]).astype(jnp.int32)
        # Batched matmul: (S, 3, 3) @ (S, 3, N) -> (S, 3, N)
        q_int = jnp.matmul(ub_mat, hkl_int.astype(jnp.float32))
        k_dot_q = jnp.sum(kf_ki_sample * q_int, axis=1)
        safe_dot = jnp.where(jnp.abs(k_dot_q) < 1e-9, 1e-9, k_dot_q)
        lambda_opt = jnp.clip(k_sq / safe_dot, wl_min_val, wl_max_val)

        # Recalculate HKL float at the optimal wavelength for the cosine kernel
        hkl_float = v / lambda_opt[:, None, :]

        # Robust Multi-Scale Kernel: Mixture of Narrow + Wide peaks
        # We use an isotropic tolerance based on the total HKL magnitude
        # to represent a uniform angular tolerance (sigma_h approx tolerance_rad * |h|).
        # This prevents the 'delta function' behavior for components near zero.
        hkl_mag_sq = jnp.sum(hkl_float**2, axis=1, keepdims=True)
        # Use 1.0 as floor to ensure low-order reflections don't have infinite precision
        hkl_mag_sq_safe = jnp.maximum(hkl_mag_sq, 1.0)

        kappa_scaled = 1.0 / (
            hkl_mag_sq_safe * (tolerance_rad + 1e-9) ** 2 * 4 * jnp.pi**2
        )
        # Use stable sin^2 form: cos(2pi x) - 1 = -2 sin^2(pi x)
        cos_diff_stable = -2.0 * jnp.sin(jnp.pi * hkl_float) ** 2
        log_p_narrow = jnp.sum(kappa_scaled * cos_diff_stable, axis=1)

        kappa_wide_scaled = 1.0 / (
            hkl_mag_sq_safe * jnp.deg2rad(5.0) ** 2 * 4 * jnp.pi**2
        )
        log_p_wide = jnp.sum(kappa_wide_scaled * cos_diff_stable, axis=1)

        # Combine via LogSumExp with 1% weight on wide kernel
        log_prob = jax.nn.logsumexp(
            jnp.stack([log_p_narrow, log_p_wide - 4.605]), axis=0
        )

        # --- VALIDATION LOGIC ---
        # Resolution Filter
        q_sq = jnp.sum(q_int**2, axis=1)
        d_est = 1.0 / jnp.sqrt(q_sq + 1e-9)
        valid_res = (d_est >= d_min) & (d_est <= d_max)

        # Symmetry Mask
        h, k, l = (  # noqa: E741
            hkl_int[:, 0, :],
            hkl_int[:, 1, :],
            hkl_int[:, 2, :],
        )
        is_allowed = check_hkl_symmetry(
            centering, h, k, l, mask_range_h, mask_range_k, mask_range_l, valid_hkl_mask
        )

        # Combine all masks
        final_mask = is_allowed & valid_res

        # Use LogSumExp style accumulation for robustness
        log_prob_masked = jnp.where(final_mask, log_prob, -1e12)

        # Update carry
        # curr_sum will now store the logsumexp of valid candidates
        new_sum = jax.nn.logsumexp(jnp.stack([curr_sum, log_prob_masked]), axis=0)

        update_mask = log_prob_masked > curr_max
        new_max = jnp.where(update_mask, log_prob_masked, curr_max)
        new_best_hkl = jnp.where(update_mask[:, None, :], hkl_int, curr_best_hkl)
        new_best_lamb = jnp.where(update_mask, lambda_opt, curr_best_lamb)
        return (new_sum, new_max, new_best_hkl, new_best_lamb), None

    final_carry, _ = jax.lax.scan(scan_body, initial_carry, jnp.arange(num_candidates))
    accum_probs, log_prob_max, best_hkl, best_lamb = final_carry
    # score = -jnp.sum(weights * jnp.exp(accum_probs), axis=1)
    # Use max probability per peak for the score
    score = -jnp.sum(weights * jnp.exp(log_prob_max), axis=1)
    return (
        score,
        jnp.exp(log_prob_max),
        best_hkl.transpose((0, 2, 1)),
        best_lamb,
    )
