import jax
import jax.numpy as jnp
import numpy as np

from subhkl.core.crystallography import LatticeSOA
from subhkl.core.math import rotation_from_axis_angle, rotation_from_rodrigues

def update_add(arr, idx, val):
    return arr.at[idx].add(val)


def update_set(arr, idx, val):
    return arr.at[idx].set(val)


def _inverse_map_param(value, bound):
    if bound < 1e-12:
        return 0.5
    norm = (value + bound) / (2.0 * bound)
    return np.clip(norm, 0.0, 1.0)


def _forward_map_param(norm, bound):
    return norm * 2.0 * bound - bound


def _inverse_map_lattice(value, nominal, frac_bound):
    delta = np.abs(nominal) * frac_bound
    min_val = nominal - delta
    max_val = nominal + delta
    if (max_val - min_val) < 1e-12:
        return 0.5
    norm = (value - min_val) / (max_val - min_val)
    return np.clip(norm, 0.0, 1.0)


def _forward_map_lattice(norm, nominal, frac_bound):
    delta = np.abs(nominal) * frac_bound
    min_val = nominal - delta
    max_val = nominal + delta
    return min_val + norm * (max_val - min_val)


def reconstruct_cell_params(
    params_norm: jnp.ndarray,
    lattice_system: str,
    free_params_init: jnp.ndarray,
    lattice_bound_frac: float,
) -> jnp.ndarray:
    """Reconstructs the full 6-parameter unit cell from normalized free parameters."""
    p_free = _forward_map_lattice(params_norm, free_params_init, lattice_bound_frac)
    S = params_norm.shape[0]
    deg90 = jnp.full((S,), 90.0)
    deg120 = jnp.full((S,), 120.0)

    if lattice_system == "Cubic":
        a = p_free[:, 0]
        return jnp.stack([a, a, a, deg90, deg90, deg90], axis=1)
    if lattice_system == "Hexagonal":
        a, c = p_free[:, 0], p_free[:, 1]
        return jnp.stack([a, a, c, deg90, deg90, deg120], axis=1)
    if lattice_system == "Tetragonal":
        a, c = p_free[:, 0], p_free[:, 1]
        return jnp.stack([a, a, c, deg90, deg90, deg90], axis=1)
    if lattice_system == "Rhombohedral":
        a, alpha = p_free[:, 0], p_free[:, 1]
        return jnp.stack([a, a, a, alpha, alpha, alpha], axis=1)
    if lattice_system == "Orthorhombic":
        a, b, c = p_free[:, 0], p_free[:, 1], p_free[:, 2]
        return jnp.stack([a, b, c, deg90, deg90, deg90], axis=1)
    if lattice_system == "Monoclinic":
        a, b, c, beta = p_free[:, 0], p_free[:, 1], p_free[:, 2], p_free[:, 3]
        return jnp.stack([a, b, c, deg90, beta, deg90], axis=1)

    return p_free

def compute_goniometer_R(
    gonio_offsets_norm: jnp.ndarray,
    goniometer_bound_deg: float,
    gonio_nominal_offsets: jnp.ndarray,
    gonio_angles: jnp.ndarray,
    gonio_axes: jnp.ndarray,
    num_gonio_axes: int,
) -> jnp.ndarray:
        # NOTE: This helper uses Norm to calc Delta, then adds Nominal from self.
    """Computes the total Lab -> Sample rotation matrix R."""
    offsets_delta = _forward_map_param(gonio_offsets_norm, goniometer_bound_deg)
    total_offsets = gonio_nominal_offsets + offsets_delta

    angles_deg = total_offsets[:, :, None] + gonio_angles[None, :, :]
    S, M = total_offsets.shape[0], gonio_angles.shape[1]

    # Broadcast an identity matrix to (S, M, 3, 3) without copying memory
    R = jnp.broadcast_to(jnp.eye(3), (S, M, 3, 3))
    deg2rad = jnp.pi / 180.0

    for i in range(num_gonio_axes):
        axis_spec = gonio_axes[i]
        direction = axis_spec[0:3]
        sign = axis_spec[3]
        theta = sign * angles_deg[:, i, :] * deg2rad
        Ri = rotation_from_axis_angle(direction, theta)
        # Mantid SetGoniometer: R = R0 @ R1 @ R2
        # Each Ri should be multiplied on the RIGHT of the current accumulated matrix.
        # Batched matmul: (S, M, 3, 3) @ (S, M, 3, 3) -> (S, M, 3, 3)
        R = jnp.matmul(R, Ri)

    return R

def get_physical_params(
    x: jnp.ndarray,
    # Lattice Args
    refine_lattice: bool,
    free_params_init: jnp.ndarray,
    B: jnp.ndarray,
    # Sample Args
    refine_sample: bool,
    sample_bound: float,
    sample_nominal: jnp.ndarray,
    # Beam Args
    refine_beam: bool,
    beam_bound_deg: float,
    beam_nominal: jnp.ndarray,
    # Goniometer Args
    refine_goniometer: bool,
    num_gonio_axes: int,
    num_active_gonio: int,
    gonio_mask: jnp.ndarray,
    goniometer_bound_deg: float,
    gonio_nominal_offsets: jnp.ndarray,
    gonio_angles: jnp.ndarray,
    gonio_axes: jnp.ndarray,
):
    """Reconstruct physical parameters (Base + Delta) for a batch of solutions x."""
    idx = 0

    # 1. Orientation
    rot_params = x[:, idx : idx + 3]
    U = jax.vmap(rotation_from_rodrigues)(rot_params)
    idx += 3

    # 2. Lattice
    if refine_lattice:
        n_lat = free_params_init.size
        cell_params_norm = x[:, idx : idx + n_lat]
        _B = LatticeSOA.compute_B_batched(cell_params_norm)
        idx += n_lat
        UB = jnp.matmul(U, _B)
    else:
        _B = B
        UB = jnp.matmul(U, _B[None, ...])

    # 3. Sample
    if refine_sample:
        s_norm = x[:, idx : idx + 3]
        idx += 3
        sample_delta = _forward_map_param(s_norm, sample_bound)
        sample_total = sample_nominal + sample_delta
    else:
        # Zero-copy view instead of .repeat()
        sample_total = jnp.broadcast_to(sample_nominal, (x.shape[0], 3))

    # 4. Beam
    if refine_beam:
        bound_rad = jnp.deg2rad(beam_bound_deg)
        tx = _forward_map_param(x[:, idx], bound_rad)
        ty = _forward_map_param(x[:, idx + 1], bound_rad)
        idx += 2

        ki_vec = jnp.broadcast_to(beam_nominal, (x.shape[0], 3))
        ki_vec = update_add(ki_vec, (slice(None), 0), tx)
        ki_vec = update_add(ki_vec, (slice(None), 1), ty)
        ki_vec = ki_vec / jnp.linalg.norm(ki_vec, axis=1, keepdims=True)
    else:
        ki_vec = jnp.broadcast_to(beam_nominal, (x.shape[0], 3))

    # 5. Goniometer
    if refine_goniometer:
        gonio_norm = jnp.full((x.shape[0], num_gonio_axes), 0.5)
        if num_active_gonio > 0:
            gonio_norm = update_set(
                gonio_norm,
                (slice(None), gonio_mask),
                x[:, idx : idx + num_active_gonio],
            )
            idx += num_active_gonio

        offsets_delta = _forward_map_param(gonio_norm, goniometer_bound_deg)
        offsets_total = gonio_nominal_offsets + offsets_delta
        R = compute_goniometer_R(
            gonio_norm,
            goniometer_bound_deg,
            gonio_nominal_offsets,
            gonio_angles,
            gonio_axes,
            num_gonio_axes,
        )

    elif gonio_axes is not None:
        offsets_total = jnp.broadcast_to(
            gonio_nominal_offsets, (x.shape[0], num_gonio_axes)
        )
        gonio_norm = jnp.full((x.shape[0], num_gonio_axes), 0.5)
        R = compute_goniometer_R(
            gonio_norm,
            goniometer_bound_deg,
            gonio_nominal_offsets,
            gonio_angles,
            gonio_axes,
            num_gonio_axes,
        )
    else:
        offsets_total = None
        R = None

    return UB, _B, sample_total, ki_vec, offsets_total, R

