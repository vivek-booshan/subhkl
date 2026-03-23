from dataclasses import fields
import warnings

import numpy as np

from subhkl.core.crystallography import Lattice
from subhkl.core.experiment import PeaksData, ExperimentData
from subhkl.instrument.goniometer import Goniometer
from subhkl._optimization.optimization import (
    _forward_map_lattice,
    _forward_map_param, # only for legacy tests that need it
    _inverse_map_param,
)

from subhkl.utils.shim import (
    HAS_JAX,
    OPTIMIZATION_BACKEND, # for legacy tests that need it
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


# NOTE(vivek): switch to builder api
class FindUB:
    def __init__(self, data: ExperimentData):
        self.state = data
        self.lattice: Lattice
        self.goniometer: Goniometer
        self.peaks: PeaksData

        # NOTE(vivek): if slots=False then we can use dict logic
        # for key, value in vars(data).items():
        #     setattr(self, key, value)

        for field in fields(data):
            name = field.name
            value = getattr(data, name)
            setattr(self, name, value)

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
            print("JAX not available - using SciPy-based optimization")
            from ._scipy_minimize import _minimize_scipy
            (num_indiexed, hkl, lamda, U, x), refined_state = _minimize_scipy(
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
        else:
            from ._jax_minimize import _jax_minimize
            (num_indexed, hkl, lamda, U, x), refined_state = _jax_minimize(
                state=self.state,
                strategy_name=strategy_name,
                population_size=population_size,
                num_generations=num_generations,
                n_runs=n_runs,
                seed=seed,
                tolerance_deg=tolerance_deg,
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
                search_window_size=search_window_size,
                window_batch_size=window_batch_size,
                chunk_size=chunk_size,
                num_iters=num_iters,
                top_k=top_k,
                batch_size=batch_size,
                sigma_init=sigma_init,
                softness=softness,
                B_sharpen=B_sharpen,
            )

        for field in fields(refined_state):
            name = field.name
            value = getattr(refined_state, name)
            setattr(self, name, value)

        self.x = x
        self.state = refined_state
        return num_indexed, hkl, lamda, U
