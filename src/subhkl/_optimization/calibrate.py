# NOTE(vivek): pondering whether to move to io/loader:ExperimentLoader
from dataclasses import replace
import h5py
import numpy as np

from subhkl.core.crystallography import Lattice
from subhkl.core.experiment import ExperimentData
from subhkl.core.models import LATTICE_CONFIG
from subhkl.instrument.goniometer import Goniometer

# NOTE(vivek): public facing allows passage of entire experiment data class
# internally handles passing only the relevant data
# allows internal func to be pure for jax optimization (i think)
def calibrate(
    experiment: ExperimentData,
    filename: str,
    refine_lattice=False,
    refine_sample=False,
    refine_beam=False,
    refine_goniometer=False,
    refine_goniometer_axes=False,
    lattice_bound_frac=0.05,
    sample_bound_meters=0.002,
    beam_bound_deg=1.0,
    goniometer_bound_deg=5.0,
) -> (np.ndarray, ExperimentData):
    x0, new_lat, new_g_offsets, new_ki, new_sample = _get_bootstrap_params(
        lattice=experiment.lattice,
        goniometer=experiment.goniometer,
        space_group=experiment.space_group,
        ki_vec=experiment.ki_vec,
        base_sample_offsets=experiment.base_sample_offset,
        bootstrap_filename=filename,
        refine_lattice=refine_lattice,
        refine_sample=refine_sample,
        refine_beam=refine_beam,
        refine_goniometer=refine_goniometer,
        refine_goniometer_axes=refine_goniometer_axes,
        lattice_bound_frac=lattice_bound_frac,
        sample_bound_meters=sample_bound_meters,
        beam_bound_deg=beam_bound_deg,
        goniometer_bound_deg=goniometer_bound_deg,
    )
    new_goniometer = replace(experiment.goniometer, base_offsets=new_g_offsets)
    bootstrapped_experiment = replace(
        experiment,
        lattice=new_lat,
        goniometer=new_goniometer,
        ki_vec=new_ki,
        base_sample_offset=new_sample,
    )
    return x0, bootstrapped_experiment

def _get_bootstrap_params(
    lattice: Lattice,
    goniometer: Goniometer,
    space_group: str,
    ki_vec: np.ndarray,
    base_sample_offsets: np.ndarray,
    bootstrap_filename: str,
    refine_lattice=False,
    refine_sample=False,
    refine_beam=False,
    refine_goniometer=False,
    refine_goniometer_axes=False,
    lattice_bound_frac=0.05,
    sample_bound_meters=0.002,
    beam_bound_deg=1.0,
    goniometer_bound_deg=5.0,
):
    print(f"Bootstrapping from physical solution: {bootstrap_filename}")

    updated_lattice = lattice
    updated_ki_vec = ki_vec
    updated_sample_offsets = base_sample_offsets
    updated_gonio_offsets = goniometer.base_offsets

    with h5py.File(bootstrap_filename, "r") as f:
        raw_x = f.get("optimization/best_params")
        new_params = [raw_x[()][:3] if raw_x is not None else np.zeros(3)]

        if refine_lattice:
            updated_lattice, _ = Lattice.infer_system(
                lattice_input=(
                    f["sample/a"][()],
                    f["sample/b"][()],
                    f["sample/c"][()],
                    f["sample/alpha"][()],
                    f["sample/beta"][()],
                    f["sample/gamma"][()],
                ),
                space_group=space_group,
                return_type="lattice",
            )
            print(
                f"  > Recentered Lattice: {updated_lattice.a:.2f}, {updated_lattice.b:.2f}..."
            )

        num_free = LATTICE_CONFIG[updated_lattice.system]["num_params"]
        new_params.append(np.full(num_free, 0.5))

        updated_sample_offsets = f.get("sample/offset", np.zeros(3))[()]
        if refine_sample:
            new_params.append(np.full(3, 0.5))

        updated_ki_vec = f.get("beam/ki_vec", np.array([0.0, 0.0, 1.0]))[()]
        if refine_beam:
            new_params.append(np.full(2, 0.5))

        h5_gonio = f.get("optimization/goniometer_offsets")
        if h5_gonio is not None:
            updated_gonio_offsets = h5_gonio[()]
            print(f"  > Setting Base Goniometer Offsets: {updated_gonio_offsets}")
        elif goniometer.axes is not None:
            updated_gonio_offsets = np.zeros(len(goniometer.axes))

        if refine_goniometer:
            ref_list = goniometer.names or range(len(goniometer.axes or []))
            if refine_goniometer_axes and goniometer.names:
                mask = [
                    any(req in name for req in refine_goniometer_axes)
                    for name in goniometer.names
                ]
            else:
                mask = [True] * len(ref_list)

            if (n_active := sum(mask)) > 0:
                new_params.append(np.full(n_active, 0.5))

    x0 = np.concatenate([np.atleast_1d(p) for p in new_params])
    return (
        x0,
        updated_lattice,
        updated_gonio_offsets,
        updated_ki_vec,
        updated_sample_offsets,
    )
