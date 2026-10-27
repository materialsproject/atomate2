"""Flow for fitting magnetic exchange parameters and estimating Tc.

This is a post-processing workflow and runs no DFT itself. Given magnetic
structures and their energies (e.g. the output of the magnetic-orderings
workflow), it fits a classical Heisenberg Hamiltonian and optionally runs
Vampire Monte-Carlo for the critical temperature.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Maker

from atomate2.common.jobs.exchange import build_exchange_doc, heisenberg_mapping
from atomate2.vampire.jobs.run_vampire import run_vampire

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure

    from atomate2.common.schemas.magnetism import MagneticOrderingsDocument


__all__ = ["ExchangeMaker"]


@dataclass
class ExchangeMaker(Maker):
    """Maker to fit Heisenberg exchange parameters from magnetic structures.

    Given a set of magnetic structures and their energies (per atom), this fits a
    classical Heisenberg Hamiltonian via pymatgen's ``HeisenbergMapper``. Each
    exchange parameter ``J_ij`` is one symmetry orbit of bonds in the paramagnetic
    parent cell, fit by least squares over all supplied orderings.

    This is a post-processing workflow and runs no DFT. It is code-agnostic: the
    inputs are plain pymatgen structures and energies, typically taken from the
    magnetic-orderings workflow (see :meth:`make_from_ordering_doc`). Compute the
    orderings as static calculations in the parent geometry, i.e. run the
    ``MagneticOrderingsMaker`` with ``relax_maker=None``: relaxing each ordering on
    its own puts lattice energy into the fitted J's.

    Originally implemented in atomate (v1) for VASP as the ``ExchangeWF``.

    Parameters
    ----------
    name : str
        Name of the flows produced by this Maker.
    heisenberg_settings : dict
        Keyword arguments for pymatgen's HeisenbergMapper. The empty default keeps
        only the nearest bonds of each sublattice pair. Keys:

        - ``cutoff``: every bond shorter than this (Angstrom) couples, plus the
          nearest bonds of each sublattice pair.
        - ``symprec``: symmetry tolerance (Angstrom) for the parent's sublattices
          and interactions.
        - ``matcher``: ``StructureMatcher`` that maps each ordering onto the parent;
          loosen it if relaxation distorted the orderings.

        ``tol`` is deprecated in pymatgen and ignored.
    default_magmoms : dict | None
        The elements that can carry a moment; moments on all other elements are
        zeroed in the fit. Pass the same dict the MagneticOrderingsMaker ran with.
        If None, pymatgen's ``default_magmoms.yaml`` is used.
    run_vampire : bool
        Whether to run the Vampire Monte-Carlo step to estimate the critical
        temperature. Requires a VAMPIRE binary on PATH - ``vampire-serial`` unless
        ``ATOMATE2_VAMPIRE_CMD`` names another (e.g. ``srun -n 24 vampire-parallel``
        for the MPI build); the step raises a clear error if it is missing.
        Defaults to True (atomate1 parity).
    mc_settings : dict | None
        Keyword arguments for the Vampire Monte-Carlo run (e.g. ``mc_box_size``,
        ``equil_timesteps``, ``mc_timesteps``). Only used if ``run_vampire``.
    """

    name: str = "exchange"
    heisenberg_settings: dict = field(default_factory=dict)
    default_magmoms: dict[str, float] | None = None
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
        parent : Structure or None
            The paramagnetic parent cell the orderings were enumerated from. If None,
            pymatgen infers it from the lowest-energy ordering and warns.

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
        if any(not structure.site_properties.get("magmom") for structure in structures):
            raise ValueError(
                "Every structure needs a 'magmom' site property to fit a Heisenberg "
                "model."
            )

        hmap = heisenberg_mapping(
            structures,
            energies,
            parent,
            {**self.heisenberg_settings, "default_magmoms": self.default_magmoms},
        )
        jobs = [hmap]

        vampire_output = None
        if self.run_vampire:
            vmc = run_vampire(hmap.output, self.mc_settings)
            jobs.append(vmc)
            vampire_output = vmc.output

        doc = build_exchange_doc(
            hmap.output,
            parent_structure=parent,
            vampire_output=vampire_output,
            vampire_settings=self.mc_settings if self.run_vampire else None,
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
        :meth:`make`. Because the structures must be inspected (validated) when the
        flow is built, ``doc`` must be a resolved document, not a jobflow output
        reference.

        Parameters
        ----------
        doc : MagneticOrderingsDocument
            A computed magnetic-orderings document (e.g. from a finished
            MagneticOrderingsMaker run). It does not store the ``default_magmoms``
            the orderings ran with; set them on this maker.

        Returns
        -------
        Flow
            The exchange-parameter fitting workflow.
        """
        if any(output.relax_output for output in doc.outputs):
            warnings.warn(
                "The orderings were relaxed. The fit assumes static calculations in "
                "the parent geometry, so lattice energy ends up in the J's. Run the "
                "MagneticOrderingsMaker with relax_maker=None.",
                stacklevel=2,
            )

        structures, energies = [], []

        for output in doc.outputs:
            structure = output.structure.copy()
            # HeisenbergMapper needs magmoms; the output stores them separately
            if not structure.site_properties.get("magmom"):
                structure.add_site_property("magmom", output.magmoms)
            structures.append(structure)
            energies.append(output.energy_per_atom)

        return self.make(structures, energies, doc.parent_structure)
