from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np

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
    # Physical Lattice State 
    # NOTE(vivek): Lattice shadows individual 
    lattice: Lattice
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    space_group: str

    # idk where to put these guys
    wavelength: [float, float]

    # Observed Peak Data
    # peaks: PeaksData
    two_theta: np.ndarray       # Shape: (N_peaks,)
    az_phi: np.ndarray          # Shape: (N_peaks,)
    intensity: np.ndarray       # Shape: (N_peaks,)
    sigma_intensity: np.ndarray # Shape: (N_peaks,)
    radii: np.ndarray           # Shape: (N_peaks,)
    run_indices: np.ndarray     # Shape: (N_peaks,) mapping to R or gonio_angles
    peak_xyz: Optional[np.ndarray] = None  # Shape: (N_peaks, 3) - Peak positions in Lab space
    
    ki_vec: np.ndarray = None # Shape: (3,) - Incident beam vector
    sample_offset: Optional[np.ndarray] = None # Refined sample position
    base_sample_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Goniometer Logic 
    # R is the static/pre-computed rotation stack
    R: Optional[np.ndarray] = None  # Shape: (N_runs, 3, 3) or (3, 3)
    goniometer_axes: Optional[np.ndarray] = None   # Physical axis definitions
    goniometer_angles: Optional[np.ndarray] = None # Raw motor positions: (N_axes, N_runs)
    goniometer_offsets: Optional[np.ndarray] = None # Refined motor offsets
    goniometer_names: List[str] = field(default_factory=list)
    base_gonio_offset: Optional[np.ndarray] = None
    # goniometer: Goniometer
