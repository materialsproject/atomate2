"""Flow for the thermal expansion from third-order force constants."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Maker
from pymatgen.util.due import Doi, due

from atomate2.common.jobs.cte import compute_cte

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core.structure import Structure

    from atomate2.common.flows.elastic import BaseElasticMaker
    from atomate2.common.flows.pheasy import BasePhononMaker
    from atomate2.forcefields.jobs import ForceFieldRelaxMaker
    from atomate2.vasp.jobs.base import BaseVaspMaker


@due.dcite(
    Doi("10.1088/1361-648X/acd831"),
    description="Implementation strategies in phonopy and phono3py.",
)
@due.dcite(
    Doi("10.7566/JPSJ.92.012001"),
    description="Phonopy and phono3py.",
)
@dataclass
class BaseCTEMaker(Maker, ABC):
    """
    Maker to calculate the thermal expansion from third-order force constants.

    A tight structural relaxation is performed first. The relaxed structure is
    then passed to the pheasy phonon flow, which fits the second- and
    third-order force constants, and to the elastic flow. The two flows do not
    depend on each other, so a workflow manager can run them at the same time.
    Neither flow may relax the structure again, so that both use the same
    structure in the same frame. Finally, phono3py gives the mode
    Grueneisen tensors on a q-point mesh, and the thermal expansion tensor
    follows from the heat-capacity weighted Grueneisen tensors and the elastic
    compliance. The mode Grueneisen tensors come from the third-order force
    constants at the relaxed structure. The frequencies are not renormalized
    with temperature.

    Parameters
    ----------
    name: str
        Name of the flows produced by this maker.
    bulk_relax_maker: .ForceFieldRelaxMaker, .BaseVaspMaker, or None
        A maker to perform a tight relaxation on the bulk. Set to None to skip
        the relaxation.
    phonon_maker: .BasePhononMaker
        The pheasy phonon maker. It must have cal_anhar_fcs=True,
        use_symmetrized_structure=None and bulk_relax_maker=None. Its
        anhar_fit_methods set which force constants are used for the thermal
        expansion.
    elastic_maker: .BaseElasticMaker
        Maker for the elastic tensor. It must have bulk_relax_maker=None.
    temperatures: list[float]
        Temperatures in K.
    mesh: tuple[int, int, int] | float
        q-point mesh for the mode Grueneisen tensors, or a q-point density used
        as kppa in pymatgen's Kpoints.automatic_density for the unit cell.
    tol_imaginary_modes: float
        If a frequency on the mesh is below -tol_imaginary_modes in THz, a
        warning is raised and the thermal expansion of that fit is not computed.
    min_frequency: float
        Modes below this frequency in THz are left out of the thermal expansion.
    """

    name: str = "cte"
    bulk_relax_maker: ForceFieldRelaxMaker | BaseVaspMaker | None = None
    phonon_maker: BasePhononMaker = None
    elastic_maker: BaseElasticMaker = None
    temperatures: list[float] = field(default_factory=lambda: list(range(0, 1001, 10)))
    mesh: tuple[int, int, int] | float = (12, 12, 12)
    tol_imaginary_modes: float = 0.1
    min_frequency: float = 1e-3

    def __post_init__(self) -> None:
        """Check that the phonon and elastic makers fit this workflow."""
        if not self.phonon_maker.cal_anhar_fcs:
            raise ValueError(
                "The phonon maker needs cal_anhar_fcs=True, since the thermal "
                "expansion needs the third-order force constants."
            )
        if self.phonon_maker.use_symmetrized_structure is not None:
            raise ValueError(
                "The phonon maker needs use_symmetrized_structure=None, so that the "
                "phonon and elastic calculations use the same frame."
            )
        for label, maker in (
            ("phonon", self.phonon_maker),
            ("elastic", self.elastic_maker),
        ):
            if maker.bulk_relax_maker is not None:
                raise ValueError(
                    f"The {label} maker needs bulk_relax_maker=None. Otherwise the "
                    "structure is relaxed again, and the phonon and elastic "
                    "calculations do not use the same structure."
                )

    def make(self, structure: Structure, prev_dir: str | Path | None = None) -> Flow:
        """
        Make a flow to calculate the thermal expansion.

        Parameters
        ----------
        structure: Structure
            A pymatgen structure. Start with a structure that is nearly fully
            optimized, as the relaxation settings are strict.
        prev_dir: str or Path or None
            A previous calculation directory to use for copying outputs.
        """
        jobs = []
        equilibrium_stress = None
        if self.bulk_relax_maker is not None:
            bulk_kwargs = {}
            if self.prev_calc_dir_argname is not None:
                bulk_kwargs[self.prev_calc_dir_argname] = prev_dir
            bulk = self.bulk_relax_maker.make(structure, **bulk_kwargs)
            jobs.append(bulk)
            structure = bulk.output.structure
            prev_dir = bulk.output.dir_name
            # as in the elastic flow when it runs its own relaxation
            equilibrium_stress = bulk.output.output.stress

        phonon_flow = self.phonon_maker.make(structure, prev_dir=prev_dir)
        elastic_flow = self.elastic_maker.make(
            structure, prev_dir=prev_dir, equilibrium_stress=equilibrium_stress
        )
        cte = compute_cte(
            phonon_output=phonon_flow.output,
            elastic_tensor=elastic_flow.output.elastic_tensor.raw,
            elastic_structure=elastic_flow.output.structure,
            anhar_fit_methods=self.phonon_maker.anhar_fit_methods,
            temperatures=self.temperatures,
            mesh=self.mesh,
            tol_imaginary_modes=self.tol_imaginary_modes,
            min_frequency=self.min_frequency,
            symprec=self.phonon_maker.symprec,
        )
        jobs += [phonon_flow, elastic_flow, cte]
        return Flow(jobs, output=cte.output, name=self.name)

    @property
    @abstractmethod
    def prev_calc_dir_argname(self) -> str | None:
        """Name of the argument that passes the previous calculation directory.

        It differs between codes, so each subclass sets it.
        """
