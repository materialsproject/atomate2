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

    The unconstrained four-point energies are only meaningful if each state is the
    same electronic state at both geometries. When the added electron (or hole)
    localizes differently at the two geometries, e.g. on a metal ion at one and on a
    ligand at the other, the document's ``electronic_state_change`` flag is set and
    the diabatic mode should be used: with ``cdft_fragment``, the four single points
    are constrained DFT calculations that keep each state's charge (and spin) on
    the given fragment at both geometries, so that both energies of a state lie on
    the same diabatic surface. For example, for Li+-EC (atoms 1-10 EC, atom 11 Li)
    and its reduction to Li+ - EC radical anion::

        InnerReorganizationMaker(
            cdft_fragment=(1, 10),
            cdft_fragment_states=((0, 1), (-1, 2)),
        )

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
    cdft_fragment : tuple of int or None
        First and last atom (numbered from 1, inclusive) of the fragment whose
        charge (and spin) is constrained in the diabatic mode. It must be at the
        start or the end of the molecule, so that the remaining atoms form the other
        fragment. None (default) runs unconstrained single points.
    cdft_fragment_states : tuple or None
        (charge, spin multiplicity) of the constrained fragment in state 1 and in
        state 2, e.g. ``((0, 1), (-1, 2))`` for a neutral closed-shell fragment that
        becomes a radical anion. The other fragment gets the remaining charge and
        unpaired electrons.
    cdft_constrain_spin : bool
        Also constrain the net spin of the fragment (number of unpaired electrons
        on it), not only its charge.
    cdft_partition : str
        Q-Chem's ``cdft_pop``: "fbh" (fragment-based Hirshfeld, default, robust for
        ions such as Li+) or "becke".
    run_unconstrained : bool
        In the diabatic mode, also run the four unconstrained single points, which
        give the vertical and adiabatic energies, the gaps between the diabatic and
        the ground states, and the electronic-state check.
    """

    name: str = "inner reorganization"
    opt_maker: Maker | None = field(default_factory=FrequencyOptFlatteningMaker)
    sp_maker: BaseQCMaker = field(default_factory=SinglePointMaker)
    tight_scf: bool = True
    cdft_fragment: tuple[int, int] | None = None
    cdft_fragment_states: tuple[tuple[int, int], tuple[int, int]] | None = None
    cdft_constrain_spin: bool = True
    cdft_partition: str = "fbh"
    run_unconstrained: bool = True

    def __post_init__(self) -> None:
        """Check the diabatic-mode settings."""
        if (self.cdft_fragment is None) != (self.cdft_fragment_states is None):
            raise ValueError(
                "cdft_fragment and cdft_fragment_states must be given together."
            )

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

        diabatic = self.cdft_fragment is not None
        cdft_settings = (
            self._cdft_settings(molecule_1, molecule_2) if diabatic else None
        )

        single_points: dict[str, Any] = {}
        unconstrained: dict[str, Any] = {}
        if not diabatic or self.run_unconstrained:
            sp_maker = self._sp_maker()
            for key in STATE_KEYS:
                state, geometry = key.split("@")
                single_point = sp_maker.make(molecules.output[key])
                single_point.name = f"single point state {state} at geometry {geometry}"
                unconstrained[key] = single_point
            jobs += list(unconstrained.values())
        if diabatic:
            cdft_makers = {
                state: self._sp_maker(cdft_settings["states"][state])
                for state in ("1", "2")
            }
            for key in STATE_KEYS:
                state, geometry = key.split("@")
                single_point = cdft_makers[state].make(molecules.output[key])
                single_point.name = (
                    f"cdft single point state {state} at geometry {geometry}"
                )
                single_points[key] = single_point
            jobs += list(single_points.values())
        else:
            single_points = unconstrained
            unconstrained = {}

        analysis = analyze_inner_reorganization(
            molecules.output,
            {key: sp.output for key, sp in single_points.items()},
            single_point_uuids={key: sp.uuid for key, sp in single_points.items()},
            opt_1=None if opt_1 is None else opt_1.output,
            opt_2=None if opt_2 is None else opt_2.output,
            unconstrained_points=(
                {key: sp.output for key, sp in unconstrained.items()}
                if unconstrained
                else None
            ),
            cdft_settings=cdft_settings,
        )
        analysis.name = "analyze inner reorganization"
        jobs.append(analysis)

        return Flow(jobs, output=analysis.output, name=self.name)

    def _cdft_settings(self, molecule_1: Molecule, molecule_2: Molecule) -> dict:
        """Fragments and constraints of the two diabatic states."""
        first, last = self.cdft_fragment
        n_atoms = len(molecule_1)
        if not (1 <= first <= last <= n_atoms) or (first != 1 and last != n_atoms):
            raise ValueError(
                "cdft_fragment must be a range of atoms (numbered from 1) at the start "
                "or the end of the molecule."
            )
        if first == 1 and last == n_atoms:
            raise ValueError("cdft_fragment cannot contain all atoms.")
        rest = (last + 1, n_atoms) if first == 1 else (1, first - 1)

        states = {}
        for state, molecule, (charge, multiplicity) in zip(
            ("1", "2"), (molecule_1, molecule_2), self.cdft_fragment_states, strict=True
        ):
            rest_charge = int(molecule.charge) - charge
            rest_unpaired = (molecule.spin_multiplicity - 1) - (multiplicity - 1)
            if rest_unpaired < 0:
                raise ValueError(
                    f"The fragment multiplicity {multiplicity} of state {state} is "
                    "larger than the multiplicity of the molecule."
                )
            fragment = {
                "first_atom": first,
                "last_atom": last,
                "charge": charge,
                "spin_multiplicity": multiplicity,
            }
            other = {
                "first_atom": rest[0],
                "last_atom": rest[1],
                "charge": rest_charge,
                "spin_multiplicity": rest_unpaired + 1,
            }
            constraints = [
                {
                    "value": float(-charge),  # Q-Chem: excess electrons on the atoms
                    "coefficients": [1.0],
                    "first_atoms": [first],
                    "last_atoms": [last],
                    "types": [None],
                }
            ]
            if self.cdft_constrain_spin:
                constraints.append(
                    {
                        "value": float(multiplicity - 1),
                        "coefficients": [1.0],
                        "first_atoms": [first],
                        "last_atoms": [last],
                        "types": ["s"],
                    }
                )
            states[state] = {
                "fragments": [fragment, other] if first == 1 else [other, fragment],
                "constraints": [constraints],
            }
        return {
            "fragment": [first, last],
            "fragment_states": [list(s) for s in self.cdft_fragment_states],
            "constrain_spin": self.cdft_constrain_spin,
            "partition": self.cdft_partition,
            "states": states,
        }

    def _sp_maker(self, cdft_state: dict | None = None) -> BaseQCMaker:
        """Copy of the single-point maker, set up for tight SCF and a CDFT state."""
        sp_maker = deepcopy(self.sp_maker)
        generator = sp_maker.input_set_generator
        if cdft_state is not None:
            generator.fragments = cdft_state["fragments"]
            generator.cdft_constraints = cdft_state["constraints"]
            cdft_overwrite: dict[str, Any] = deepcopy(generator.overwrite_inputs) or {}
            cdft_overwrite["rem"] = {
                **cdft_overwrite.get("rem", {}),
                "cdft_pop": self.cdft_partition,
            }
            generator.overwrite_inputs = cdft_overwrite
        if not self.tight_scf:
            return sp_maker
        overwrite_inputs: dict[str, Any] = deepcopy(generator.overwrite_inputs) or {}
        rem = overwrite_inputs.get("rem", {})
        if "scf_convergence" not in rem:
            overwrite_inputs["rem"] = {**rem, "scf_convergence": "8"}
            generator.overwrite_inputs = overwrite_inputs
        return sp_maker
