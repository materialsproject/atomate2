"""Flow for fitting magnetic exchange parameters and estimating Tc.

This is a post-processing workflow and runs no DFT itself. Given magnetic structures and their energies (e.g. the
output of the magnetic-orderings workflow), it fits a classical Heisenberg
Hamiltonian and optionally runs Vampire Monte-Carlo for the critical temperature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Maker

from atomate2.common.jobs.exchange import (
    build_exchange_doc,
    heisenberg_mapping,
)
from atomate2.vampire.jobs.run_vampire import run_vampire

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure

    from atomate2.common.schemas.magnetism import MagneticOrderingsDocument


__all__ = ["ExchangeMaker"]


@dataclass
class ExchangeMaker(Maker):
    """Maker to fit Heisenberg exchange parameters from magnetic structures.

    Given a set of magnetic structures and their energies (per atom), this fits a
    classical Heisenberg Hamiltonian via pymatgen's ``HeisenbergMapper`` to extract
    exchange parameters ``J_ij`` and the average exchange ``<J>`` (``javg``).

    This is a post-processing workflow and runs no DFT. It is code-agnostic: the
    inputs are plain pymatgen structures and energies, typically taken from the
    magnetic-orderings workflow (see :meth:`make_from_ordering_doc`).

    Originally implemented in atomate (v1) for VASP as the ``ExchangeWF``.

    Parameters
    ----------
    name : str
        Name of the flows produced by this Maker.
    heisenberg_settings : dict
        Keyword arguments for pymatgen's HeisenbergMapper, i.e. the nearest-neighbour
        ``cutoff`` (Angstrom) and the distance-grouping ``tol``.
    run_vampire : bool
        Whether to run the Vampire Monte-Carlo step to estimate the critical
        temperature. Requires the external ``vampire-serial`` binary on PATH; the step
        raises a clear error if it is missing. Defaults to True (atomate1 parity).
    mc_settings : dict | None
        Keyword arguments for the Vampire Monte-Carlo run (e.g. ``mc_box_size``,
        ``equil_timesteps``, ``mc_timesteps``, ``avg``). Only used if ``run_vampire``.
    """

    name: str = "exchange"
    heisenberg_settings: dict = field(
        default_factory=lambda: {"cutoff": 3.0, "tol": 0.04}
    )
    run_vampire: bool = True
    mc_settings: dict | None = None

    def make(
        self,
        structures: list[Structure],
        energies: list[float],
        parent: Structure | None = None,
    ) -> Flow:
        """Make a flow to fit Heisenberg exchange parameters.

        Parameters
        ----------
        structures : list[Structure]
            Magnetic structures, each carrying a "magmom" site property.
        energies : list[float]
            Energies **per atom** (eV) corresponding to each structure.

        Returns
        -------
        Flow
            The exchange-parameter fitting workflow.
        """
        if len(structures) != len(energies):
            raise ValueError(
                f"Got {len(structures)} structures but {len(energies)} energies; "
                "these must be equal."
            )

        # sort so the ground state (lowest energy) is index 0
        order = sorted(range(len(energies)), key=lambda i: energies[i])
        structures = [structures[i] for i in order]
        energies = [energies[i] for i in order]

        # HeisenbergMapper requires a 'magmom' site property on every structure
        for idx, structure in enumerate(structures):
            if not structure.site_properties.get("magmom"):
                raise ValueError(
                    f"Structure {idx} is missing a 'magmom' site property, which is "
                    "required to fit a Heisenberg model."
                )

        hmap = heisenberg_mapping(structures, energies, parent, self.heisenberg_settings)
        jobs = [hmap]

        vampire_output = None
        if self.run_vampire:
            vmc = run_vampire(hmap.output, self.mc_settings)
            jobs.append(vmc)
            vampire_output = vmc.output

        # structures[0] is the full ground-state structure (Heisenberg strips
        # non-magnetic atoms internally, so pass the original through for provenance)
        doc = build_exchange_doc(
            hmap.output,
            parent_structure=parent or structures[0],
            vampire_output=vampire_output,
        )
        jobs.append(doc)

        formula = structures[0].composition.reduced_formula
        return Flow(
            jobs=jobs,
            output=doc.output,
            name=f"{self.name} ({formula})",
        )

    def make_from_ordering_doc(self, doc: MagneticOrderingsDocument) -> Flow:
        """Make an exchange flow from a (concrete) magnetic-orderings document.

        This is a convenience constructor for chaining onto the magnetic-orderings
        workflow: it pulls the structures and per-atom energies out of an
        already-computed ``MagneticOrderingsDocument`` and forwards them to
        :meth:`make`. Because the structures must be inspected (sorted, validated)
        when the flow is built, ``doc`` must be a resolved document, not a jobflow
        output reference.

        Parameters
        ----------
        doc : MagneticOrderingsDocument
            A computed magnetic-orderings document (e.g. from a finished
            MagneticOrderingsMaker run).

        Returns
        -------
        Flow
            The exchange-parameter fitting workflow.
        """
        structures, energies = [], []

        for output in doc.outputs:
            structure = output.structure.copy()
            # HeisenbergMapper needs magmoms; the output stores them separately
            if not structure.site_properties.get("magmom"):
                structure.add_site_property("magmom", output.magmoms)
            structures.append(structure)
            energies.append(output.energy_per_atom)

        return self.make(structures, energies, doc.parent_structure)
