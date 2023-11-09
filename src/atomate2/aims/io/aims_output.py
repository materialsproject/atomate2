"""A representation of FHI-aims output (based on ASE output parser)."""
# from __future__ import annotations

# from typing import TYPE_CHECKING, Any

# import numpy as np
# from monty.json import MontyDecoder, MSONable

# from atomate2.aims.io.parsers import read_aims_header_info, read_aims_output

# if TYPE_CHECKING:
#     from collections.abc import Sequence
#     from pathlib import Path

#     from emmet.core.math import Matrix3D, Vector3D

#     from atomate2.aims.utils.msonable_atoms import MSONableAtoms


# class AimsOutput(MSONable):
#     """The main output file for FHI-aims."""

#     def __init__(
#         self,
#         results: Sequence[MSONableAtoms],
#         metadata: dict[str, Any],
#         atoms_summary: dict[str, Any],
#     ):
#         """AimsOutput object constructor.

#         Parameters
#         ----------
#         results: Sequence[.MSONableAtoms]
#             A list of all images in an output file
#         metadata: Dict[str, Any]
#             The metadata of the executable used to perform the calculation
#         atoms_summary: Dict[str, Any]
#             The summary of the starting atomic structure
#         """
#         self._results = results
#         self._metadata = metadata
#         self._atoms_summary = atoms_summary

#     def as_dict(self) -> dict[str, Any]:
#         """Create a dict representation of the outputs for MSONable."""
#         d: dict[str, Any] = {
#             "@module": self.__class__.__module__,
#             "@class": self.__class__.__name__,
#         }

#         d["results"] = self._results
#         d["metadata"] = self._metadata
#         d["atoms_summary"] = self._atoms_summary
#         return d

#     @classmethod
#     def from_outfile(cls, outfile: str | Path):
#         """Construct an AimsOutput from an output file.

#         Parameters
#         ----------
#         outfile: str or Path
#             The aims.out file to parse
#         """
#         metadata, atoms_summary = read_aims_header_info(outfile)
#         results = read_aims_output(outfile, index=slice(0, None))

#         return cls(results, metadata, atoms_summary)

#     @classmethod
#     def from_dict(cls, d: dict[str, Any]):
#         """Construct an AimsOutput from a dictionary.

#         Parameters
#         ----------
#         d: Dict[str, Any]
#             The dictionary used to create AimsOutput
#         """
#         decoded = {
#             k: MontyDecoder().process_decoded(v)
#             for k, v in d.items()
#             if not k.startswith("@")
#         }
#         return cls(decoded["results"], decoded["metadata"], decoded["atoms_summary"])

#     def get_results_for_image(self, image_ind: int) -> MSONableAtoms:
#         """Get the results dictionary for a particular image or slice of images.

#         Parameters
#         ----------
#         image_ind: int
#             The index of the image to get the results for

#         Returns
#         -------
#         The results for that image
#         """
#         return self._results[image_ind]

#     @property
#     def atoms_summary(self) -> dict[str, Any]:
#         """The summary of the material/molecule that the calculations represent."""
#         return self._atoms_summary

#     @property
#     def metadata(self) -> dict[str, Any]:
#         """The system metadata."""
#         return self._metadata

#     @property
#     def n_images(self) -> int:
#         """The number of images in results."""
#         return len(self._results)

#     @property
#     def initial_structure(self) -> MSONableAtoms:
#         """The initial structure for the calculations."""
#         return self._atoms_summary["initial_atoms"]

#     @property
#     def final_structure(self) -> MSONableAtoms:
#         """The final structure for the calculation."""
#         return self._results[-1]

#     @property
#     def structures(self) -> Sequence[MSONableAtoms]:
#         """All images in the output file."""
#         return self._results

#     @property
#     def fermi_energy(self) -> float:
#         """The Fermi energy for the final structure in the calculation."""
#         return self.get_results_for_image(-1).calc.results["fermi_energy"]

#     @property
#     def vbm(self) -> float:
#         """The HOMO level for the final structure in the calculation."""
#         return self.get_results_for_image(-1).calc.results["vbm"]

#     @property
#     def cbm(self) -> float:
#         """The LUMO level for the final structure in the calculation."""
#         return self.get_results_for_image(-1).calc.results["cbm"]

#     @property
#     def band_gap(self) -> float:
#         """The band gap for the final structure in the calculation."""
#         return self.get_results_for_image(-1).calc.results["gap"]

#     @property
#     def direct_band_gap(self) -> float:
#         """The direct band gap for the final structure in the calculation."""
#         return self.get_results_for_image(-1).calc.results["direct_gap"]

#     @property
#     def final_energy(self) -> float:
#         """The total energy for the final structure in the calculation."""
#         return self.get_results_for_image(-1).calc.results["energy"]

#     @property
#     def completed(self) -> bool:
#         """Did the calculation complete."""
#         return len(self._results) > 0

#     @property
#     def aims_version(self) -> str:
#         """The version of FHI-aims used for the calculation."""
#         return self._metadata["version_number"]

#     @property
#     def forces(self) -> list[Vector3D]:
#         """The forces for the final image of the calculation."""
#         force_array = self.get_results_for_image(-1).calc.results.get("forces", None)
#         if isinstance(force_array, np.ndarray):
#             return force_array.tolist()

#         return force_array

#     @property
#     def stress(self) -> Matrix3D:
#         """The stress for the final image of the calculation."""
#         return self.get_results_for_image(-1).calc.results.get("stress", None)

#     @property
#     def stresses(self) -> list[Matrix3D]:
#         """The atomic virial stresses for the final image of the calculation."""
#         stresses_array = self.get_results_for_image(-1).calc.results.get(
#             "stresses", None
#         )
#         if isinstance(stresses_array, np.ndarray):
#             return stresses_array.tolist()
#         return stresses_array

#     @property
#     def all_forces(self) -> list[list[Vector3D]]:
#         """The forces for all images in the calculation."""
#         all_forces_arr = [
#             res.calc.results.get("forces", None) for res in self._results
#         ]
#         return [
#             af.tolist() if isinstance(af, np.ndarray) else af for af in all_forces_arr
#         ]
