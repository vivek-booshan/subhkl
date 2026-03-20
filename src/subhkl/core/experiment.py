from dataclasses import dataclass, field
import numpy as np
from typing import Optional

from subhkl.instrument.goniometer import Goniometer
from subhkl.core.crystallography import Lattice

@dataclass(frozen=True, slots=True)
class PeaksData:
    two_theta : np.ndarray # (N,)
    azimuthal : np.ndarray # (N,)
    intensity : np.ndarray # (N,)
    sigma     : np.ndarray # (N,)
    radius    : np.ndarray # (N,)
    xyz       : np.ndarray # (N, 3)

    def refine_weights(self, B_sharpen: float=None) -> np.ndarray:
        """
        Calculate normalized weights for optimization based on SNR and
        optional Wilson B-factor sharpening  
        """
        snr = self.intensity / (self.sigma + 1e-12)

        weights = snr
        if B_sharpen is not None:
            # 2sin(theta)/lambda basically related to 1/d^2
            theta_rad = np.deg2rad(self.two_theta) / 2.0
            sin_sq_theta = np.sin(theta_rad)**2
            wilson_correction = np.exp(B_sharpen * sin_sq_theta)
            weights *= wilson_correction

        weights = weights / (np.mean(weights) + 1e-12)
        return np.clip(weights, 0, 10.0)

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
