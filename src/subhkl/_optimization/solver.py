from dataclasses import dataclass, fields, field, replace, asdict, astuple
# from enum import IntFlag, auto
from typing import Optional, List

import numpy as np

from subhkl.core import ExperimentData


@dataclass(frozen=True, slots=True)
class Result:
    num_indexed: int
    hkl: np.ndarray
    wavelengths: np.ndarray
    U: np.ndarray
    x: np.ndarray
    state: ExperimentData

    def __iter__(self):
        return iter(astuple(self))


# NOTE(vivek): switch to bit set?
# class RefineFlags(IntFlag):
#     NONE = 0
#     LATTICE = auto()
#     GONIOMETER = auto()
#     SAMPLE = auto()
#     BEAM = auto()
#     ALL = LATTICE | GONIOMETER | SAMPLE | BEAM


@dataclass(frozen=True, slots=True)
class RefinementConfig:
    refine_lattice: bool = False
    lattice_bound_frac: float = 0.05
    refine_goniometer: bool = False
    goniometer_bound_deg: float = 5.0
    refine_goniometer_axes: Optional[List[str]] = None
    refine_sample: bool = False
    sample_bound_meters: float = 0.002
    refine_beam: bool = False
    beam_bound_deg: float = 1.0


@dataclass(frozen=True, slots=True)
class IndexingConfig:
    d_min: Optional[float] = None
    d_max: float = 100.0
    hkl_search_range: int = 20
    tolerance_deg: float = 0.1
    loss_method: str = "gaussian"
    softness: float = 0.01
    B_sharpen: float = 50.0
    search_window_size: int = 256
    window_batch_size: int = 32
    chunk_size: int = 2048
    num_iters: int = 20
    top_k: int = 32


@dataclass(frozen=True, slots=True)
class SolverConfig:
    strategy_name: str = "cma_es"
    population_size: int = 1000
    num_generations: int = 100
    sigma_init: Optional[float] = None
    n_runs: int = 1
    seed: int = 0
    batch_size: Optional[int] = None


@dataclass(frozen=True, slots=True)
class UBSolver:
    strategy: str = "cma_es"
    refinement_cfg: RefinementConfig = field(default_factory=RefinementConfig)
    indexing_cfg: IndexingConfig = field(default_factory=IndexingConfig)
    solver_cfg: SolverConfig = field(default_factory=SolverConfig)

    _x0: np.ndarray = None
    _calibrated_state: ExperimentData = None
    """
    # Solver API for solving the UB matrix.
    The UB matrix links measured data (peaks on the detector) to physical orientation and geometry of the crystal
    sample.

    The B matrix transforms miller indices (h, k, l) (title drop!!), which
    are integer coords in reciprocal space, to cartesian coordinates (qx, qy, qz) in
    a reference frame tied to the crystal.

    The U matrix is a standard 3x3 rotation matrix, representing the orientation
    of the crystal sample relative to the instrument's lab frame. If you rotate
    the crystal on a goniometer, the B matrix stays the same (atoms in crystal
    haven't moved relative to each other) but the U matrix changes.

    Q_sample = UB(h, k, l) # vertical coord

    Q_sample is the scattering vector in the sample's coordinate system

    Strategies: cma_es, de, pos (explain)
    """

    def __post_init__(self):
        assert self.strategy.lower() in ["cma_es", "de", "pso"], (
            "strategy must be one of the following: [cma_es, de, pos]"
        )

    def resolution(self, d_min, d_max):
        """Set crystallographic resolution limits"""
        return self.set(d_min=d_min, d_max=d_max)

    def with_strategy(self, strategy: str, pop_size: int = 1000, num_gens: int = 100):
        """Configure the evolutionary strategy."""
        return self.set(
            strategy=strategy, population_size=pop_size, num_generations=num_gens
        )

    def refinement_options(
        self,
        refine_lattice: bool = None,
        lattice_bound_frac: float = None,
        refine_goniometer: bool = None,
        goniometer_bound_deg: float = None,
        refine_goniometer_axes: List[str] = None,
        refine_sample: bool = None,
        sample_bound_meters: float = None,
        refine_beam: bool = None,
        beam_bound_deg: float = None,
    ):
        """Configure physical model refinement parameters and boundaries."""
        updates = {k: v for k, v in locals().items() if v is not None and k != "self"}
        return replace(self, refinement_cfg=replace(self.refinement_cfg, **updates))

    def indexing_options(
        self,
        d_min: float = None,
        d_max: float = None,
        hkl_search_range: int = None,
        tolerance_deg: float = None,
        loss_method: str = None,
        softness: float = None,
        B_sharpen: float = None,
        search_window_size: int = None,
        window_batch_size: int = None,
        chunk_size: int = None,
        num_iters: int = None,
        top_k: int = None,
    ):
        """Configure the mathematical indexing engine and HKL pool generation."""
        updates = {k: v for k, v in locals().items() if v is not None and k != "self"}
        return replace(self, indexing_cfg=replace(self.indexing_cfg, **updates))

    def solver_options(
        self,
        strategy_name: str = None,
        population_size: int = None,
        num_generations: int = None,
        sigma_init: float = None,
        n_runs: int = None,
        seed: int = None,
        batch_size: int = None,
    ):
        """Configure the evolutionary strategy and JAX execution parameters."""
        updates = {k: v for k, v in locals().items() if v is not None and k != "self"}
        return replace(self, solver_cfg=replace(self.solver_cfg, **updates))

    def search_depth(self, range_hkl: int):
        """Sets the HKL pool size. Use larger values (>30) for large unit cells."""
        return self.set(hkl_search_range=range_hkl)

    def parallel_runs(self, count: int, seed: int = None):
        """Set how many independent JAX runs to perform in parallel via vmap."""
        updates = {"n_runs": count}
        if seed is not None:
            updates["seed"] = seed
        return self.set(**updates)

    def precise(self, tolerance_deg: float = 0.05, num_iters: int = 50):
        """Adjusts the indexer to be more strict and perform more internal iterations."""
        return self.set(tolerance_deg=tolerance_deg, num_iters=num_iters)

    def refine(
        self,
        lattice: bool = False,
        goniometer: bool = False,
        sample: bool = False,
        beam: bool = False,
    ):
        updates = {}
        if lattice is not None:
            updates["refine_lattice"] = lattice
        if sample is not None:
            updates["refine_sample"] = sample
        if goniometer is not None:
            updates["refine_goniometer"] = goniometer
        if beam is not None:
            updates["refine_beam"] = beam

        return replace(self, refinement_cfg=replace(self.refinement_cfg, **updates))

    def physical_constraints(
        self, lattice_frac=None, gonio_deg=None, sample_meters=None, beam_deg=None
    ):
        """Sets the search boundaries for refined physical parameters."""
        updates = {}
        if lattice_frac is not None:
            updates["lattice_bound_frac"] = lattice_frac
        if gonio_deg is not None:
            updates["goniometer_bound_deg"] = gonio_deg
        if sample_meters is not None:
            updates["sample_bound_meters"] = sample_meters
        if beam_deg is not None:
            updates["beam_bound_deg"] = beam_deg
        return replace(self, refinement_cfg=replace(self.refinement_cfg, **updates))

    def set(self, **kwargs):
        """Generic method to modify any arbitrary internal kwarg."""
        r_up, i_up, s_up = {}, {}, {}
        strategy = kwargs.pop("strategy", self.strategy)

        r_names = {f.name for f in fields(RefinementConfig)}
        i_names = {f.name for f in fields(IndexingConfig)}
        s_names = {f.name for f in fields(SolverConfig)}

        for k, v in kwargs.items():
            if k in r_names:
                r_up[k] = v
            elif k in i_names:
                i_up[k] = v
            elif k in s_names:
                s_up[k] = v
            else:
                raise AttributeError(
                    f"'{k}' is not a valid parameter for any config cluster."
                )

        return replace(
            self,
            strategy=strategy,
            refinement_cfg=RefinementConfig(**r_up) if r_up else self.refinement_cfg,
            indexing_cfg=IndexingConfig(**i_up) if i_up else self.indexing_cfg,
            solver_cfg=SolverConfig(**s_up) if s_up else self.solver_cfg,
        )

    def calibrate(self, state: ExperimentData, filename: str):
        import subhkl._optimization.calibrate as cal

        x0, calibrated_state = cal(state, filename, **asdict(self.refinement_cfg))
        return replace(self, _x0=x0, _calibrated_state=calibrated_state)

    # NOTE(vivek): let minimizer just directly take the args. Check if goniometer_* is just state.goniometer.*
    # NOTE(vivek): remove scipy minimize option entirely
    def solve(
        self,
        state: Optional[ExperimentData] = None,
        init_params: Optional[np.ndarray] = None,
        goniometer_axes: Optional[List[str]] = None,
        goniometer_angles: Optional[np.ndarray] = None,
        goniometer_names: Optional[List[str]] = None,
    ):
        """
        Minimize the objective using evolutionary strategies.
        """
        if state is not None:
            if self._calibrated_state is not None:
                print(
                    ">>> Notice: Overriding calibrated state with manually passed ExperimentData."
                )
            target_state = state
        else:
            assert self._calibrated_state is not None, (
                "No ExperimentData found. You must either pass 'state' to .solve() "
                "or call .calibrate(state, filename) first to initialize the solver."
            )
            print(">>> Using bootstrapped ExperimentData from calibration.")
            target_state = self._calibrated_state

        if init_params is not None:
            x0 = init_params
        elif self._x0 is not None:
            x0 = self._x0
        else:
            print(
                ">>> No initial parameters found. Starting from default heuristic (0.5)."
            )
            x0 = None

        rcfg = self.refinement_cfg
        icfg = self.indexing_cfg
        scfg = self.solver_cfg

        from ._jax_minimize import _jax_minimize

        result = _jax_minimize(
            state=target_state,
            init_params=x0,
            goniometer_axes=goniometer_axes,
            goniometer_angles=goniometer_angles,
            goniometer_names=goniometer_names,
            # refinement
            refine_lattice=rcfg.refine_lattice,
            lattice_bound_frac=rcfg.lattice_bound_frac,
            refine_goniometer=rcfg.refine_goniometer,
            refine_goniometer_axes=rcfg.refine_goniometer_axes,
            goniometer_bound_deg=rcfg.goniometer_bound_deg,
            refine_sample=rcfg.refine_sample,
            sample_bound_meters=rcfg.sample_bound_meters,
            refine_beam=rcfg.refine_beam,
            beam_bound_deg=rcfg.beam_bound_deg,
            # indexing
            d_min=icfg.d_min,
            d_max=icfg.d_max,
            hkl_search_range=icfg.hkl_search_range,
            tolerance_deg=icfg.tolerance_deg,
            loss_method=icfg.loss_method,
            search_window_size=icfg.search_window_size,
            window_batch_size=icfg.window_batch_size,
            chunk_size=icfg.chunk_size,
            num_iters=icfg.num_iters,
            top_k=icfg.top_k,
            softness=icfg.softness,
            B_sharpen=icfg.B_sharpen,
            # solver
            strategy_name=scfg.strategy_name,
            population_size=scfg.population_size,
            num_generations=scfg.num_generations,
            n_runs=scfg.n_runs,
            seed=scfg.seed,
            batch_size=scfg.batch_size,
            sigma_init=scfg.sigma_init,
        )

        return result
