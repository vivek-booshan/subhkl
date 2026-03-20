from dataclasses import dataclass, field
import numpy as np
from typing import Optional

from subhkl.instrument.goniometer import Goniometer
from subhkl.core.crystallography import Lattice

@dataclass(frozen=True)
class PeaksData:
    two_theta : np.ndarray # (N,)
    azimuthal : np.ndarray # (N,)
    intensity : np.ndarray # (N,)
    sigma     : np.ndarray # (N,)
    radius    : np.ndarray # (N,)
    xyz       : np.ndarray # (N, 3)

@dataclass(frozen=True)
class ExperimentData:
    lattice: Lattice
    peaks: PeaksData
    goniometer: Goniometer

    space_group: str
    wavelength: [float, float]
    run_indices: np.ndarray
    ki_vec: np.ndarray = None # Shape: (3,) - Incident beam vector
    sample_offset: Optional[np.ndarray] = None # Refined sample position
    base_sample_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
