import jax
import jax.numpy as jnp


from ._utils import check_hkl_symmetry


def soft_indexer(
    ub_mat,
    kf_ki_sample,
    wl_min_val: float,
    wl_max_val: float,
    k_sq_init,
    peak_radii,
    d_min: float,
    d_max: float,
    weights,
    num_candidates: int,
    centering: str,
    mask_range_h: int,
    mask_range_k: int,
    mask_range_l: int,
    valid_hkl_mask: jnp.array,
    k_sq_override=None,
    tolerance_rad=0.002,
):
    ub_inv = jnp.linalg.inv(ub_mat)
    # Batched matmul: (S, 3, 3) @ (S, 3, N) -> (S, 3, N)
    v = jnp.matmul(ub_inv, kf_ki_sample)
    abs_v = jnp.abs(v)
    max_v_val = jnp.max(abs_v, axis=1)
    n_start = max_v_val / wl_max_val
    start_int = jnp.ceil(n_start)
    k_sq = k_sq_override if k_sq_override is not None else k_sq_init[None, :]
    k_norm = jnp.sqrt(k_sq)

    initial_carry = (
        jnp.zeros(max_v_val.shape),
        jnp.zeros(max_v_val.shape),
        jnp.zeros((v.shape[0], 3, v.shape[2]), dtype=jnp.int32),
        jnp.zeros(max_v_val.shape),
    )

    def scan_body(carry, i):
        curr_sum, curr_max, curr_best_hkl, curr_best_lamb = carry
        n = start_int + i
        n_safe = jnp.where(n == 0, 1e-9, n)
        lamda_cand = max_v_val / n_safe
        hkl_float = v / lamda_cand[:, None, :]
        hkl_int = jnp.round(hkl_float).astype(jnp.int32)
        # Batched matmul: (S, 3, 3) @ (S, 3, N) -> (S, 3, N)
        q_int = jnp.matmul(ub_mat, hkl_int.astype(jnp.float32))
        k_dot_q = jnp.sum(kf_ki_sample * q_int, axis=1)
        safe_dot = jnp.where(jnp.abs(k_dot_q) < 1e-9, 1e-9, k_dot_q)
        lambda_opt = jnp.clip(k_sq / safe_dot, wl_min_val, wl_max_val)
        q_obs = kf_ki_sample / lambda_opt[:, None, :]
        dist_sq = jnp.sum((q_obs - q_int) ** 2, axis=1)

        effective_sigma = (tolerance_rad + peak_radii[None, :]) * (k_norm / lambda_opt)
        # Robust Multi-Scale Kernel
        # 1. Narrow (High precision)
        log_p_narrow = -dist_sq / (2 * effective_sigma**2 + 1e-9)

        # 2. Wide (Capture range: 5 degrees)
        sigma_wide = jnp.deg2rad(5.0) * (k_norm / lambda_opt)
        log_p_wide = -dist_sq / (2 * sigma_wide**2 + 1e-9)

        # Combine via LogSumExp with 1% weight on wide kernel
        log_prob = jax.nn.logsumexp(
            jnp.stack([log_p_narrow, log_p_wide - 4.605]), axis=0
        )
        prob = jnp.exp(log_prob)

        # 1. Calc |Q|^2 for predicted HKL
        q_sq_pred = jnp.sum(q_int**2, axis=1)
        # 2. Convert to d = 1/|Q| (Crystallographic units)
        d_pred = 1.0 / jnp.sqrt(q_sq_pred + 1e-9)
        valid_res = (d_pred >= d_min) & (d_pred <= d_max)

        h, k, l = (  # noqa: E741
            hkl_int[:, 0, :],
            hkl_int[:, 1, :],
            hkl_int[:, 2, :],
        )
        is_allowed = check_hkl_symmetry(
            centering, h, k, l, mask_range_h, mask_range_k, mask_range_l, valid_hkl_mask
        )

        # Combine masks
        final_mask = is_allowed & valid_res

        prob = jnp.where(final_mask, prob, 0.0)

        new_sum = curr_sum + prob
        update_mask = prob > curr_max
        new_max = jnp.where(update_mask, prob, curr_max)
        new_best_hkl = jnp.where(update_mask[:, None, :], hkl_int, curr_best_hkl)
        new_best_lamb = jnp.where(update_mask, lambda_opt, curr_best_lamb)
        return (new_sum, new_max, new_best_hkl, new_best_lamb), None

    final_carry, _ = jax.lax.scan(scan_body, initial_carry, jnp.arange(num_candidates))
    accum_probs, prob_max, best_hkl, best_lamb = final_carry
    # score = -jnp.sum(weights * accum_probs, axis=1) # Original sum
    score = -jnp.sum(weights * prob_max, axis=1)
    return score, prob_max, best_hkl.transpose((0, 2, 1)), best_lamb
