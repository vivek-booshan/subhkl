import jax.numpy as jnp


# noqa: E741
def check_hkl_symmetry(
    centering: str,
    h: int,
    k: int,
    l: int,
    mask_range_h: int,
    mask_range_k: int,
    mask_range_l: int,
    valid_hkl_mask: jnp.array,
):
    """
    Robust symmetry check in JAX. Uses pre-computed mask for speed,
    and falls back to centring parity checks for out-of-bounds HKLs.
    """
    rh, rk, rl = mask_range_h, mask_range_k, mask_range_l
    idx_h = jnp.clip(h + rh, 0, 2 * rh).astype(jnp.int32)
    idx_k = jnp.clip(k + rk, 0, 2 * rk).astype(jnp.int32)
    idx_l = jnp.clip(l + rl, 0, 2 * rl).astype(jnp.int32)

    in_bounds = (h >= -rh) & (h <= rh) & (k >= -rk) & (k <= rk) & (l >= -rl) & (l <= rl)

    # Parity checks for centring
    h_even = h % 2 == 0
    k_even = k % 2 == 0
    l_even = l % 2 == 0

    if centering == "F":
        # All odd or all even
        allowed_out = (h_even == k_even) & (k_even == l_even)
    elif centering == "I":
        # h+k+l is even
        allowed_out = (h + k + l) % 2 == 0
    elif centering == "A":
        # k+l is even
        allowed_out = (k + l) % 2 == 0
    elif centering == "B":
        # h+l is even
        allowed_out = (h + l) % 2 == 0
    elif centering == "C":
        # h+k is even
        allowed_out = (h + k) % 2 == 0
    elif centering == "R":
        # -h+k+l is divisible by 3
        allowed_out = (-h + k + l) % 3 == 0
    else:
        # P or other: Assume allowed unless we have a specific reason to reject
        allowed_out = True

    return jnp.where(in_bounds, valid_hkl_mask[idx_h, idx_k, idx_l], allowed_out)
