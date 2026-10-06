"""Flow for CALPHAD free energy models from special quasirandom structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Maker
from pymatgen.util.due import Doi, due

from atomate2.common.jobs.calphad import (
    fit_tdb,
    get_sqs_structures,
    run_sqs_calculations,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


@due.dcite(
    Doi("10.1016/S0364-5916(02)80006-2"),
    description="ATAT: the Alloy Theoretic Automated Toolkit.",
)
@due.dcite(
    Doi("10.1016/j.calphad.2017.05.005"),
    description="sqs2tdb: CALPHAD models from special quasirandom structures.",
)
@due.dcite(
    Doi("10.1103/PhysRevLett.65.353"),
    description="Special quasirandom structures.",
)
@due.dcite(
    Doi("10.1016/0364-5916(91)90030-N"),
    description="SGTE data for pure elements.",
)
@dataclass
class BaseCalphadMaker(Maker):
    """
    Maker to fit a CALPHAD database for a binary system with ATAT sqs2tdb.

    The special quasirandom structures (SQS) of each lattice are taken from the
    ATAT database. Each solid SQS is relaxed. Each liquid SQS is melted and then
    run with MD, and the mean potential energy of that run is used. sqs2tdb fits
    the energies of each lattice and writes one TDB file.

    For lattices in the SGTE database (e.g. FCC_A1, HCP_A3 and LIQUID), sqs2tdb
    takes the free energies of the pure elements from SGTE. Only the mixing
    terms come from the calculations.

    ATAT must be installed, with its programs on the PATH.

    The jobs read the energy and is_force_converged of the relaxation outputs
    and the ionic steps of the liquid MD output, as in the force field task
    documents. So far only force field makers give these.

    This workflow is new and has not been tested widely. It might still change
    in future versions.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    lattices : list[str]
        ATAT lattice names. Each needs an entry in terms. The ordered lattices
        need the lattices of their pure element end members, e.g. NI4MO_D1A
        needs FCC_A1.
    level : int
        Level of the ATAT SQS database. Higher levels add compositions.
    terms : dict[str, list[str]]
        Lines of the sqs2tdb terms.in file for each lattice. Each line has the
        form order,level, with one pair per sublattice separated by ":". Order 1
        gives the end members and order 2 the binary interactions. Level is the
        highest Redlich-Kister order.
    relax_maker : Maker
        Maker to relax the solid SQS.
    liquid_melt_maker : Maker or None
        MD maker that melts the liquid SQS. Needed if LIQUID is in lattices.
    liquid_md_maker : Maker or None
        MD maker for the liquid energy. It must store the energy and structure
        of each frame. Needed if LIQUID is in lattices.
    n_equilibration_frames : int
        Number of stored frames at the start of the liquid MD that are left out
        of the mean energy.
    liquid_supercell : int
        Number of repeats of each liquid SQS along each lattice vector for the
        liquid MD. The energy is divided by the number of SQS cells.
    """

    name: str = "calphad"
    lattices: list[str] = field(
        default_factory=lambda: ["FCC_A1", "BCC_A2", "HCP_A3", "LIQUID"]
    )
    level: int = 2
    terms: dict[str, list[str]] = field(
        default_factory=lambda: {
            lattice: ["1,0", "2,0"]
            for lattice in ("FCC_A1", "BCC_A2", "HCP_A3", "LIQUID")
        }
    )
    relax_maker: Maker | None = None
    liquid_melt_maker: Maker | None = None
    liquid_md_maker: Maker | None = None
    n_equilibration_frames: int = 0
    liquid_supercell: int = 1

    def make(self, elements: Sequence[str]) -> Flow:
        """
        Make a flow to fit the CALPHAD database.

        Parameters
        ----------
        elements : Sequence[str]
            The two elements, e.g. ("Ni", "Re").
        """
        if "LIQUID" in self.lattices and None in (
            self.liquid_melt_maker,
            self.liquid_md_maker,
        ):
            raise ValueError("LIQUID needs liquid_melt_maker and liquid_md_maker.")

        sqs = get_sqs_structures(elements, self.lattices, self.level)
        calculations = run_sqs_calculations(
            sqs.output,
            self.relax_maker,
            self.liquid_melt_maker,
            self.liquid_md_maker,
            self.n_equilibration_frames,
            self.liquid_supercell,
        )
        fit = fit_tdb(
            elements,
            self.level,
            {lattice: self.terms[lattice] for lattice in self.lattices},
            calculations.output,
        )
        return Flow([sqs, calculations, fit], output=fit.output, name=self.name)
