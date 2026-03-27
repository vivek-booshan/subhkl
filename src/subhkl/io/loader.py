import os
import re
import h5py
import numpy as np

from subhkl.config import beamlines, reduction_settings

from subhkl.core.crystallography import Lattice
from subhkl.core.experiment import ExperimentData, PeaksData
from subhkl.integration.image_data import ImageData
from subhkl.instrument.goniometer import Goniometer


class ImageLoader:
    def from_nexus(filename: str, instrument) -> ImageData:
        detectors = beamlines[instrument]
        settings = reduction_settings[instrument]
        ims = {}
        with h5py.File(filename, "r") as f:
            keys = []
            banks = []
            for key in f["/entry/"].keys():
                match = re.match(r"bank(\d+).*", key)
                if match is not None:
                    keys.append(key)
                    banks.append(int(match.groups()[0]))
            for rel_key, bank in zip(keys, banks, strict=False):
                key = "/entry/" + rel_key + "/event_id"
                array = f[key][()]
                det = detectors.get(str(bank))
                if det is not None:
                    m, n, offset = det["m"], det["n"], det["offset"]
                    bc = np.bincount(array - offset, minlength=m * n)
                    if np.sum(bc) > 0:
                        if settings.get("YAxisIsFastVaryingIndex"):
                            ims[bank] = bc.reshape(m, n).T
                        else:
                            ims[bank] = bc.reshape(n, m)

        return ImageData(ims=ims)

    def from_h5(filename: str) -> ImageData:
        ims = {}
        bank_mapping = {}
        with h5py.File(filename, "r") as f:
            images = f["images"]
            N = images.shape[0]
            if "bank_ids" in f:
                bank_ids = f["bank_ids"][()]
            else:
                bank_ids = np.zeros(N, dtype=int)
            if "files" in f and "file_offsets" in f:
                image_files_raw = [
                    n.decode("utf-8") if isinstance(n, bytes) else str(n)
                    for n in f["files"][()]
                ]
                file_offsets = f["file_offsets"][()]
            else:
                image_files_raw = None
                file_offsets = None
            data = images[()]
            for i in range(N):
                ims[i] = data[i]
                bank_mapping[i] = int(bank_ids[i])

        return ImageData(
            ims=ims,
            raw_files=image_files_raw,
            file_offsets=file_offsets,
            bank_mapping=bank_mapping,
        )


class GoniometerLoader:
    def from_nexus(filename: str, instrument: str):
        """
        Get goniometer axes and rotation angles from Nexus file

        Parameters
        ----------
        filename : str
            Name of nexus file to load angles from

        instrument : str
            Name of instrument used to collect data

        Returns
        -------
        axes : list[length 4 numpy array]
            List of axes in format used by Mantid `SetGoniometer`
        angles : list[float]
            List of angles in degrees about the axes
        names : list[str]
            List of axis names
        """
        settings = reduction_settings[instrument]
        axes, angles, names = [], [], []
        with h5py.File(filename) as f:
            try:
                das_logs = f["entry/DASlogs"]

                # We can iterate directly over settings["Goniometer"] as of Python 3.6
                # which guarantees that `json.load` keeps the iteration order of keys
                # the same as it is in the original file.
                # So this should work fine--assuming the order is correct in
                # `reduction_settings.json`, that is!
                for axis_name, axis_spec in settings["Goniometer"].items():
                    angle_deg = float(das_logs[axis_name]["average_value"][0])
                    axis = np.array(axis_spec, dtype=float)
                    angles.append(angle_deg)
                    axes.append(axis)
                    names.append(axis_name)
            except Exception:
                pass

        rotation = Goniometer.get_rotation(axes, angles)

        return Goniometer(
            axes=axes,
            angles=np.array(angles),
            names=names,
            rotation=rotation,
        )

    def from_h5(h5_file):
        axes = h5_file["goniometer/axes"][()]
        angles = h5_file["goniometer/angles"][()]
        names = None
        if "goniometer/names" in h5_file:
            names = [
                n.decode() if isinstance(n, bytes) else str(n)
                for n in h5_file["goniometer/names"][()]
            ]

        # Handle 2D angles (stack of matrices) vs 1D angles
        if angles.ndim == 2:
            rotation = np.stack([Goniometer.get_rotation(axes, ang) for ang in angles])
        else:
            rotation = Goniometer.get_rotation(axes, angles)

        return Goniometer(axes=axes, angles=angles, names=names, rotation=rotation)


class ExperimentLoader:
    @classmethod
    def from_h5(cls, filename: str) -> ExperimentData:
        """Entry point for HDF5 files."""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Could not find peak file: {filename}")

        # Flatten HDF5 groups into a flat dictionary for easier mapping
        with h5py.File(os.path.abspath(filename), "r") as f:
            raw_data = {}

            def _visitor(name, obj):
                if isinstance(obj, h5py.Dataset):
                    raw_data[name] = obj[()]

            f.visititems(_visitor)

            return cls.from_dict(raw_data)

    @classmethod
    def from_dict(cls, data: dict) -> ExperimentData:
        """The core mapping logic. Decouples dictionary keys from class attributes."""

        sg = data.get("sample/space_group", "P1")
        if isinstance(sg, bytes):
            sg = sg.decode("utf-8")
        else:
            sg = str(sg)

        sample_cell = (
            data["sample/a"],
            data["sample/b"],
            data["sample/c"],
            data["sample/alpha"],
            data["sample/beta"],
            data["sample/gamma"],
        )

        system, _ = Lattice.infer_system(sample_cell, sg)
        lattice = Lattice(*sample_cell, system)

        run_indices = cls._resolve_run_indices(data)

        goniometer_names = data.get("goniometer/names")
        if goniometer_names is not None and isinstance(goniometer_names[0], bytes):
            goniometer_names = [n.decode("utf-8") for n in goniometer_names]

        peaks = PeaksData(
            two_theta=data["peaks/two_theta"],
            azimuthal=data["peaks/azimuthal"],
            intensity=data["peaks/intensity"],
            sigma=data["peaks/sigma"],
            radius=data.get("peaks/radius"),
            xyz=data.get("peaks/xyz"),
        )

        goniometer = Goniometer(
            axes=data.get("goniometer/axes"),
            angles=data.get("goniometer/angles"),
            names=goniometer_names,
            rotation=data.get("goniometer/R", np.eye(3)),
            offsets=data.get("optimization/goniometer_offsets"),
        )

        return ExperimentData(
            lattice=lattice,
            peaks=peaks,
            goniometer=goniometer,
            space_group=sg,
            wavelength=data["instrument/wavelength"],
            run_indices=run_indices,
            ki_vec=np.array(data.get("beam/ki_vec", [0.0, 0.0, 1.0])),
            base_sample_offset=np.array(data.get("sample/offset", [0.0, 0.0, 0.0])),
        )

    @staticmethod
    def _resolve_run_indices(data: dict) -> np.ndarray:
        r_stack = data.get("goniometer/R")
        idx_run = data.get("peaks/run_index")
        idx_img = data.get("peaks/image_index")
        idx_bank = data.get("bank", data.get("bank_ids"))

        if r_stack is not None and r_stack.ndim == 3:
            num_rot = r_stack.shape[0]

            for candidate in [idx_run, idx_img, idx_bank]:
                if candidate is not None:
                    if (int(np.max(candidate)) + 1) == num_rot:
                        return candidate

        for fallback in [idx_run, idx_img, idx_bank]:
            if fallback is not None:
                return fallback

        num_peaks = len(data["peaks/two_theta"])
        return np.zeros(num_peaks, dtype=int)
