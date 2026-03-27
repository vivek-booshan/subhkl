import jax.lax as lax
import jax.numpy as jnp

from ._utils import check_hkl_symmetry


def binary_indexer(
    ub_mat,
    kf_ki_sample,
    B,
    k_sq_init,
    wl_min_val,
    wl_max_val,
    d_min,
    d_max,
    peak_radii,
    weights,
    search_window_size,
    pool_hkl_sorted,
    pool_phi_sorted,
    centering,
    mask_range_h,
    mask_range_k,
    mask_range_l,
    valid_hkl_mask,
    k_sq_override=None,
    tolerance_rad=0.002,
    window_batch_size=32,
):
    k_sq = k_sq_override if k_sq_override is not None else k_sq_init[None, :]
    k_norm = jnp.sqrt(k_sq)
    ub_inv = jnp.linalg.inv(ub_mat)
    # Batched matmul: (S, 3, 3) @ (S, 3, N) -> (S, 3, N)
    hkl_float = jnp.matmul(ub_inv, kf_ki_sample)
    # Broadcasted matmul: (3, 3) @ (S, 3, N) -> (S, 3, N)
    hkl_cart_approx = jnp.matmul(B[None, ...], hkl_float)
    phi_obs = jnp.arctan2(hkl_cart_approx[:, 1, :], hkl_cart_approx[:, 0, :])
    idx_centers = jnp.searchsorted(pool_phi_sorted, phi_obs)
    half_win = search_window_size // 2
    raw_offsets = jnp.arange(-half_win, half_win + 1)
    pad_len = (
        window_batch_size - (raw_offsets.shape[0] % window_batch_size)
    ) % window_batch_size
    offsets_padded = jnp.pad(raw_offsets, (0, pad_len), constant_values=raw_offsets[-1])
    offset_batches = offsets_padded.reshape(-1, window_batch_size)
    init_min_dist = jnp.full(idx_centers.shape, 1e9)
    init_best_hkl = jnp.zeros((*idx_centers.shape, 3))
    init_best_lamb = jnp.zeros(idx_centers.shape)
    init_carry = (init_min_dist, init_best_hkl, init_best_lamb)

    def scan_body(carry, batch_offsets):
        curr_min_dist, curr_best_hkl, curr_best_lamb = carry
        gather_idx = idx_centers[..., None] + batch_offsets[None, None, :]
        pool_T = pool_hkl_sorted.T
        hkl_cands = jnp.take(pool_T, gather_idx, axis=0, mode="wrap")
        # Broadcasted matmul: (S, 3, 3) @ (S, M, W, 3, 1) -> (S, M, W, 3, 1)
        # hkl_cands is (S, M, W, 3)
        q_pred = jnp.matmul(ub_mat[:, None, None, ...], hkl_cands[..., None]).squeeze(
            -1
        )
        k_obs = jnp.transpose(kf_ki_sample, (0, 2, 1))[:, :, None, :]
        k_dot_q = jnp.sum(k_obs * q_pred, axis=3)
        lambda_opt = k_sq[..., None] / jnp.where(jnp.abs(k_dot_q) < 1e-9, 1e-9, k_dot_q)
        valid_lamb = (lambda_opt >= wl_min_val) & (lambda_opt <= wl_max_val)
        q_sq = jnp.sum(q_pred**2, axis=3)
        d_spacings = 1.0 / jnp.sqrt(q_sq + 1e-9)  # crystallographic convention
        valid_res = (d_spacings >= d_min) & (d_spacings <= d_max)
        h, k, l = (  # noqa: E741
            hkl_cands[..., 0],
            hkl_cands[..., 1],
            hkl_cands[..., 2],
        )
        valid_sym = check_hkl_symmetry(
            centering, h, k, l, mask_range_h, mask_range_k, mask_range_l, valid_hkl_mask
        )
        valid_mask = valid_lamb & valid_res & valid_sym
        q_obs_opt = k_obs / jnp.where(lambda_opt == 0, 1.0, lambda_opt)[..., None]
        diff = q_obs_opt - q_pred
        dist_sq = jnp.sum(diff**2, axis=3)
        dist_sq_masked = jnp.where(valid_mask, dist_sq, 1e9)
        batch_min_dist = jnp.min(dist_sq_masked, axis=2)
        batch_best_local_idx = jnp.argmin(dist_sq_masked, axis=2)
        batch_best_hkl = jnp.take_along_axis(
            hkl_cands, batch_best_local_idx[..., None, None], axis=2
        ).squeeze(axis=2)
        batch_best_lamb = jnp.take_along_axis(
            lambda_opt, batch_best_local_idx[..., None], axis=2
        ).squeeze(axis=2)
        improve_mask = batch_min_dist < curr_min_dist
        new_min_dist = jnp.where(improve_mask, batch_min_dist, curr_min_dist)
        new_best_hkl = jnp.where(improve_mask[..., None], batch_best_hkl, curr_best_hkl)
        new_best_lamb = jnp.where(improve_mask, batch_best_lamb, curr_best_lamb)
        return (new_min_dist, new_best_hkl, new_best_lamb), None

    final_carry, _ = lax.scan(scan_body, init_carry, offset_batches)
    best_dist_sq, best_hkl, best_lamb = final_carry

    # Use dynamic lambda for accurate physical tolerance scaling
    effective_sigma = (tolerance_rad + peak_radii[None, :]) * (k_norm / best_lamb)
    probs = jnp.exp(-best_dist_sq / (2 * effective_sigma**2 + 1e-9))
    score = -jnp.sum(weights * probs, axis=1)
    return score, probs, best_hkl, best_lamb
