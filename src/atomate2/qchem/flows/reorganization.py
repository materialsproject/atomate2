"""Flows for inner-sphere reorganization energies."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from jobflow import Flow, Maker

from atomate2.qchem.flows.core import FrequencyOptFlatteningMaker
from atomate2.qchem.jobs.core import SinglePointMaker
from atomate2.qchem.jobs.reorganization import (
    STATE_KEYS,
    analyze_inner_reorganization,
    prepare_vertical_molecules,
)

if TYPE_CHECKING:
    from pymatgen.core.structure import Molecule

    from atomate2.qchem.jobs.base import BaseQCMaker


@dataclass
class InnerReorganizationMaker(Maker):
    """
    Maker for the inner-sphere reorganization energy of an electron transfer.

    For two states of a molecule (for example a neutral molecule and its radical
    anion), the flow optimizes both states (unless ``opt_maker`` is None, in which
    case the given geometries are used as they are) and runs four single points at
    the same level of theory: each state at its own optimized geometry and at the
    optimized geometry of the other state (the four-point scheme). The final job
    returns an :obj:`.InnerReorganizationDocument` with the reorganization energies
    on both potential energy surfaces, the vertical and adiabatic energies, and a
    check that the electron transfer does not change the bonding.

    With an equilibrium solvent model (or in vacuum) this gives the intramolecular
    reorganization energy only. The outer-sphere (solvent) contribution depends on
    the environment of the electron transfer and is not computed here.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    opt_maker : .Maker or None
        Maker used to optimize both states, typically a
        :obj:`.FrequencyOptFlatteningMaker` or an :obj:`.OptMaker`. None uses the
        geometries passed to :meth:`make` as optimized geometries.
    sp_maker : .BaseQCMaker
        Maker for the four single points. Use the same level of theory (functional,
        basis set and solvent model) as ``opt_maker``.
    tight_scf : bool
        Converge the single points to ``scf_convergence = 8`` unless the single-point
        maker sets ``scf_convergence`` itself. Q-Chem's default for single points
        (5) is looser than for optimizations and frequencies (8), which matters for
        energy differences of a few meV.
    """

    name: str = "inner reorganization"
    opt_maker: Maker | None = field(default_factory=FrequencyOptFlatteningMaker)
    sp_maker: BaseQCMaker = field(default_factory=SinglePointMaker)
    tight_scf: bool = True

    def make(self, molecule_1: Molecule, molecule_2: Molecule) -> Flow:
        """
        Create the flow.

        Parameters
        ----------
        molecule_1, molecule_2 : .Molecule
            The two states, with their charges and spin multiplicities, and with the
            same atoms in the same order (the geometries can be identical if they
            are optimized first).

        Returns
        -------
        Flow
            The flow, whose output is the :obj:`.InnerReorganizationDocument`.
        """
        if [str(s) for s in molecule_1.species] != [str(s) for s in molecule_2.species]:
            raise ValueError(
                "molecule_1 and molecule_2 must have the same atoms in the same order."
            )

        jobs: list = []
        opt_1 = opt_2 = None
        if self.opt_maker is not None:
            opt_1 = self.opt_maker.make(molecule_1)
            opt_1.name = "optimization state 1"
            opt_2 = self.opt_maker.make(molecule_2)
            opt_2.name = "optimization state 2"
            jobs += [opt_1, opt_2]
            sources: tuple[Any, Any] = (opt_1.output, opt_2.output)
        else:
            sources = (molecule_1, molecule_2)

        molecules = prepare_vertical_molecules(
            *sources,
            int(molecule_1.charge),
            int(molecule_1.spin_multiplicity),
            int(molecule_2.charge),
            int(molecule_2.spin_multiplicity),
        )
        molecules.name = "prepare vertical molecules"
        jobs.append(molecules)

        sp_maker = self._sp_maker()
        single_points = {}
        for key in STATE_KEYS:
            state, geometry = key.split("@")
            single_point = sp_maker.make(molecules.output[key])
            single_point.name = f"single point state {state} at geometry {geometry}"
            single_points[key] = single_point
        jobs += list(single_points.values())

        analysis = analyze_inner_reorganization(
            molecules.output,
            {key: sp.output for key, sp in single_points.items()},
            single_point_uuids={key: sp.uuid for key, sp in single_points.items()},
            opt_1=None if opt_1 is None else opt_1.output,
            opt_2=None if opt_2 is None else opt_2.output,
        )
        analysis.name = "analyze inner reorganization"
        jobs.append(analysis)

        return Flow(jobs, output=analysis.output, name=self.name)

    def _sp_maker(self) -> BaseQCMaker:
        """Copy of the single-point maker, with tight SCF convergence if requested."""
        sp_maker = deepcopy(self.sp_maker)
        if not self.tight_scf:
            return sp_maker
        generator = sp_maker.input_set_generator
        overwrite_inputs: dict[str, Any] = deepcopy(generator.overwrite_inputs) or {}
        rem = overwrite_inputs.get("rem", {})
        if "scf_convergence" not in rem:
            overwrite_inputs["rem"] = {**rem, "scf_convergence": "8"}
            generator.overwrite_inputs = overwrite_inputs
        return sp_maker
