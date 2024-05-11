"""Define common jobs used in EOS workflows, electronic-structure code agnostic."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from jobflow import job
from monty.json import MSONable
from pymatgen.alchemy.materials import TransformedStructure
from pymatgen.analysis.eos import EOS, EOSError
from pymatgen.transformations.standard_transformations import (
    DeformStructureTransformation,
)
from scipy.optimize import leastsq

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from jobflow import Job
    from pymatgen.core import Structure


class EOSPostProcessor(MSONable):
    """
    Fit data to an EOS.

    name : str
        Name of the class
    eos_attrs : tuple[str,...]
        Physical quantities that can enter the EOS fit
    job_types : tuple[str,...]
        Types of jobs included in the EOS data
    min_data_points : int or None
        Minimum number of data points needed to perform a fit.
    """

    name: str = "EOS postprocessor"
    eos_attrs: tuple[str, ...] = ("energy", "volume", "stress", "pressure")
    job_types: tuple[str, ...] = ("relax", "static")
    min_data_points: int | None = None

    def __init__(self) -> None:
        self.results: dict[str, dict] = {}

    def sort_by_quantity(self, quantity: str = "volume") -> None:
        """
        Sort input data by given kwarg.

        Parameters
        ----------
        quantity : str = "volume"
            kwarg to sort by
        """
        for job_type in self._use_job_types:
            sort_by_vol = np.argsort(self.results[job_type][quantity])
            for key in self.eos_attrs:
                if self.results[job_type].get(key):
                    self.results[job_type][key] = [
                        self.results[job_type][key][index] for index in sort_by_vol
                    ]

    def eval(self) -> None:
        """Fit the EOS according to a user-implemented function."""
        raise NotImplementedError

    def fit(self, eos_flow_output: dict[str, Any]) -> None:
        """
        Fit the EOS.

        Parameters
        ----------
        eos_flow_output : dict
            Volume, energy, and (optionally) stress and pressure data in dict form,
            {
                "relax" <required> and "static" <optional> : {
                    "energy": list, <required>
                    "volume": list, <required>
                    "stress": list <optional>
                },
                "initial_<key>": {"E0": float, "V0": float} <optional>,
                    for <key> in ("relax", "static")
            }
        """
        self.results.update(eos_flow_output)
        self._use_job_types = [key for key in self.job_types if self.results.get(key)]
        if self.min_data_points and any(
            len(self.results[job_type].get("volume", [])) < self.min_data_points
            for job_type in self._use_job_types
        ):
            raise ValueError(
                f"{self.__class__} requires {self.min_data_points} "
                "frames to fit an EOS."
            )

        self.sort_by_quantity()
        self.eval()

    @job
    def make(self, eos_flow_output: dict[str, Any]) -> Job:
        """Run the fit as a jobflow job.

        Parameters
        ----------
        eos_flow_output : dict
            Volume, energy, and (optionally) stress and pressure data in dict form,
            {
                "relax" <required> and "static" <optional> : {
                    "energy": list, <required>
                    "volume": list, <required>
                    "stress": list <optional>
                },
                "initial_<key>": {"E0": float, "V0": float} <optional>,
                    for <key> in ("relax", "static")
            }
        """
        self.fit(eos_flow_output)
        return self.results


class PostProcessEosEnergy(EOSPostProcessor):
    """
    Fit energy vs. volume data to an EOS.

    Parameters
    ----------
    eos_flow_output : dict
        Volume, energy, and (optionally) stress and pressure data in dict form,
        {
            "relax" <required> and "static" <optional> : {
                "energy": list, <required>
                "volume": list, <required>
                "stress": list <optional>
            },
            "initial_<key>": {"E0": float, "V0": float} <optional>,
                for <key> in ("relax", "static")
        }
    name : str
        Name of the class
    eos_attrs : tuple[str,...]
        Physical quantities that can enter the EOS fit
    job_types : tuple[str,...]
        Types of jobs included in the EOS data
    min_data_points : int or None
        Minimum number of data points needed to perform a fit.
    eos_models : tuple[str,...]
        List of names of EOSes to fit to.
    """

    name: str = "EOS energy vs volume fit"
    min_data_points: int | None = 4
    eos_models: tuple[str, ...] = (
        "murnaghan",
        "birch",
        "birch_murnaghan",
        "pourier_tarantola",
        "vinet",
    )

    def eval(self) -> None:
        """Fit the input data to each EOS in `self.eos_models."""
        for jobtype in self._use_job_types:
            self.results[jobtype]["EOS"] = {}
            for eos_name in self.eos_models:
                try:
                    eos = EOS(eos_name=eos_name).fit(
                        self.results[jobtype]["volume"], self.results[jobtype]["energy"]
                    )
                    self.results[jobtype]["EOS"][eos_name] = {
                        **eos.results,
                        "b0 GPa": float(eos.b0_GPa),
                    }
                except EOSError as exc:
                    self.results[jobtype]["EOS"][eos_name] = {"exception": str(exc)}


class PostProcessEosPressure(EOSPostProcessor):
    """
    Fit pressure vs. volume data to an EOS.

    Parameters
    ----------
    eos_flow_output : dict
        Volume, energy, and (optionally) stress and pressure data in dict form,
        {
            "relax" <required> and "static" <optional> : {
                "energy": list, <required>
                "volume": list, <required>
                "stress": list <optional>
            },
            "initial_<key>": {"E0": float, "V0": float} <optional>,
                for <key> in ("relax", "static")
        }
    name : str
        Name of the class
    eos_attrs : tuple[str,...]
        Physical quantities that can enter the EOS fit
    job_types : tuple[str,...]
        Types of jobs included in the EOS data
    min_data_points : int or None
        Minimum number of data points needed to perform a fit.

    If only stresses are specified, it is assumed that the elements of "stress"
    are 3 x 3 tensors, and the pressure is computed as
        pressure = Trace(stress tensor)/3
    The overall sign is irrelevant for a successful fit, as the overall sign
    of the pressure indicates internal/external stress.
    """

    name: str = "EOS pressure vs volume fit"
    min_data_points: int | None = 3

    @staticmethod
    def _birch_murnaghan_pressure(
        volume: float, b0: float, b1: float, v0: float
    ) -> float:
        """
        Compute pressure from Birch-Murnaghan equation of state.

        Parameters
        ----------
        volume : float
            A single volume or list of them to evaluate the pressure.
        b0 : float
            The Birch-Murnaghan (BM) bulk modulus at the equilibrium volume V = v0
        b1 : float
            The derivative of the bulk modulus wrt pressure at v0
        v0 : float
            The equilibrium volume

        Returns
        -------
        float : the BM pressure

        BM EOS for E(V) has the form
            E(V) = E0 + 9 B0 V0 / 16 * (
                (B1 - 4)*eta**6 + (14 - 3*B1)*eta**4 + (3*B1 - 16)*eta**2 + 6 - B1
            )
            eta = (V0/V)**(1/3).
        This function computes p = - dE / dV via the chain rule,
            p = d E / d eta * (- d eta / dV)
            = eta**4/(3*V0) * d E / d eta
        """
        eta = (v0 / volume) ** (1.0 / 3.0)
        return (
            3
            * b0
            * eta**5
            / 8.0
            * (3 * (b1 - 4) * eta**4 + 2 * (14.0 - 3 * b1) * eta**2 + 3 * b1 - 16.0)
        )

    def _initial_fit(self) -> dict:
        """Generate initial polynomial fit for p(V) curve.

        p(V) / V = a + b V + c V**2
        """
        init_pars = {}
        for jobtype in self._use_job_types:
            if self.results[jobtype].get("stress") and (
                not self.results[jobtype].get("pressure")
            ):
                self.results[jobtype]["pressure"] = [
                    1.0 / 3.0 * np.trace(np.array(stress_tensor))
                    for stress_tensor in self.results[jobtype]["stress"]
                ]
            poly_pars = np.polyfit(
                self.results[jobtype]["volume"],
                np.array(self.results[jobtype]["pressure"])
                / np.array(self.results[jobtype]["volume"]),
                deg=2,
            )

            radicand = poly_pars[1] ** 2 - 4.0 * poly_pars[0] * poly_pars[2]
            if radicand < 0.0:
                v0 = self.results[jobtype]["volume"][
                    np.argmin(self.results[jobtype]["energy"])
                ]
            else:
                min_abs_pressure = 1e20
                for i in range(2):
                    _v0 = (-poly_pars[1] + (-1) ** i * radicand ** (0.5)) / (
                        2.0 * poly_pars[0]
                    )
                    pressure = _v0 * np.polyval(poly_pars, _v0)
                    if _v0 > 0.0 and abs(pressure) < min_abs_pressure:
                        min_abs_pressure = abs(pressure)
                        v0 = _v0

            b0 = -(
                3 * poly_pars[0] * v0**3 + 2 * poly_pars[1] * v0**2 + poly_pars[0] * v0
            )
            b1 = (
                v0
                * (9 * poly_pars[0] * v0**2 + 4 * poly_pars[1] * v0 + poly_pars[0])
                / b0
            )

            init_pars[jobtype] = [b0, b1, v0]

        return init_pars

    def _objective(self, pars: Sequence, jobtype: str) -> float:
        return np.array(
            self.results[jobtype]["pressure"]
        ) - self._birch_murnaghan_pressure(
            np.array(self.results[jobtype]["volume"]), *pars
        )

    def eval(self) -> None:
        """Fit the input data to the Birch-Murnaghan pressure EOS."""
        initial_pars = self._initial_fit()
        for jobtype in self._use_job_types:
            eos_params, ierr = leastsq(
                self._objective, initial_pars[jobtype], args=(jobtype,)
            )

            self.results[jobtype]["EOS"] = {}
            if ierr not in (1, 2, 3, 4):
                self.results[jobtype]["EOS"]["exception"] = (
                    "Optimal EOS parameters not found."
                )
            else:
                for i, key in enumerate(["b0", "b1", "v0"]):
                    self.results[jobtype]["EOS"][key] = eos_params[i]


@job
def apply_strain_to_structure(structure: Structure, deformations: list) -> list:
    """
    Apply strain(s) to input structure and return transformation(s) as list.

    Parameters
    ----------
    structure: .Structure
        Input structure to apply strain to
    deformations: list[.Deformation]
        A list of deformations to apply **independently** to the input
        structure, in anticipation of performing an EOS fit.
        Deformations should be of the form of a 3x3 matrix, e.g.,
        [[1.2, 0., 0.], [0., 1.2, 0.], [0., 0., 1.2]]
        or
        ((1.2, 0., 0.), (0., 1.2, 0.), (0., 0., 1.2))

    Returns
    -------
    list
        A list of .TransformedStructure objects corresponding to the
        list of input deformations.
    """
    transformations = []
    for deformation in deformations:
        # deform the structure
        ts = TransformedStructure(
            structure,
            transformations=[DeformStructureTransformation(deformation=deformation)],
        )
        transformations += [ts]
    return transformations
