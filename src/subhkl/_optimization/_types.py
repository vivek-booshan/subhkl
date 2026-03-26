from dataclasses import dataclass, astuple
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


