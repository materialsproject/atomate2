"""
Module defining equation of state workflows.

Modeled on the atomate bulk_modulus workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from jobflow import Flow, Job, Maker

from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.EOS import (
    DeformationMaker,
    EosRelaxMaker,
    MPGGADeformationMaker,
    MPGGAEosRelaxMaker,
    MPGGAEosStaticMaker,
    MPLegacyDeformationMaker,
    MPLegacyEosRelaxMaker,
    MPLegacyEosStaticMaker,
    MPMetaGGADeformationMaker,
    MPMetaGGAEosPreRelaxMaker,
    MPMetaGGAEosRelaxMaker,
    MPMetaGGAEosStaticMaker,
    postprocess_EOS,
)
from atomate2.vasp.powerups import update_user_incar_settings

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core.structure import Structure

    from atomate2.vasp.jobs.base import BaseVaspMaker


@dataclass
class EosDoubleRelaxMaker(DoubleRelaxMaker):
    """
    Workflow to generate initial double relaxation for EOS fitting.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker1 : .BaseVaspMaker
        Maker to use to generate the first relaxation.
    relax_maker2 : .BaseVaspMaker
        Maker to use to generate the second relaxation.

    NB: WAVECARs are copied over to speed up calculation
    """

    name: str = "EOS double relax"
    relax_maker1: BaseVaspMaker | None = field(default_factory=EosRelaxMaker)
    relax_maker2: BaseVaspMaker = field(
        default_factory=lambda: EosRelaxMaker(
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR",)}
        )
    )


@dataclass
class EosMaker(Maker):
    """
    Workflow to generate energy vs. volume data for EOS fitting.

    First relax a structure using relax_maker.
    Then perform a series of deformations to the relaxed structure, and
    evaluate single-point energies with static_maker.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker : .BaseVaspMaker
        Maker to generate the relaxation, defaults to DoubleRelaxMaker
    deformation_maker : .BaseVaspMaker
        Maker to generate deformations + single-points, defaults to TransmuterMaker
    static_maker : .BaseVaspMaker
        Optional Maker to generate statics from transmutation.
        Original atomate workflow did not include statics, including it here
    strain : tuple[float]
        Percentage linear strain to apply as a deformation, default = -5% to 5%
    number_of_frames : int
        Number of strain calculations to do for EOS fit, default = 6
    """

    name: str = "EOS Maker"
    relax_maker: Maker | None = field(default_factory=EosDoubleRelaxMaker)
    deformation_maker: Maker = field(
        default_factory=lambda: DeformationMaker(
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR",)}
        )
    )
    static_maker: Maker | None = None
    linear_strain: tuple[float, float] = (-0.05, 0.05)
    number_of_frames: int = 6

    def make(self, structure: Structure, prev_vasp_dir: str | Path | None = None):
        """
        Run an EOS VASP job.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        prev_vasp_dir : str or Path or None
            A previous VASP calculation directory to copy output files from.
        """
        relax_jobs: list[Job] = []
        static_jobs: list[Job] = []

        relax_flow = self.relax_maker.make(
            structure=structure, prev_vasp_dir=prev_vasp_dir
        )
        relax_flow.name = "EOS equilibrium relaxation"

        if self.static_maker:
            equil_static = self.static_maker.make(
                structure=relax_flow.output.structure,
                prev_vasp_dir=relax_flow.output.dir_name,
            )
            equil_static.name = "EOS equilibrium static"

        strain_l = np.linspace(
            self.linear_strain[0], self.linear_strain[1], self.number_of_frames
        )
        insert_equil_index = np.searchsorted(strain_l, 0, side="left")
        deformation_l = [(np.identity(3) * (1 + eps)).tolist() for eps in strain_l]

        for ideformation, deformation in enumerate(deformation_l):
            # Doubly ensure that relaxations are done at fixed volume --> ISIF = 2
            deform_relax_maker = update_user_incar_settings(
                flow=self.deformation_maker, incar_updates={"ISIF": 2}
            )
            deform_relax_job = deform_relax_maker.make(
                structure=relax_flow.output.structure,
                deformation_matrix=deformation,
                prev_vasp_dir=relax_flow.output.dir_name,
            )
            deform_relax_job.name = f"EOS Deformation Relax {ideformation}"

            if ideformation == insert_equil_index:
                relax_jobs += [relax_flow]
                if self.static_maker:
                    static_jobs += [equil_static]

            relax_jobs += [deform_relax_job]
            if self.static_maker:
                deform_static_job = self.static_maker.make(
                    structure=deform_relax_job.output.structure,
                    prev_vasp_dir=deform_relax_job.output.dir_name,
                )
                deform_static_job.name = f"EOS Static {ideformation}"
                static_jobs += [deform_static_job]

        flow_output: dict[str, list] = {"relax": []}
        for iframe in range(self.number_of_frames + 1):
            flow_output["relax"].append(
                [
                    relax_jobs[iframe].output.volume,
                    relax_jobs[iframe].output.calcs_reversed[0].output.energy,
                ]
            )

        if self.static_maker:
            flow_output["static"] = []
            for iframe in range(self.number_of_frames + 1):
                flow_output["static"].append(
                    [
                        static_jobs[iframe].output.volume,
                        static_jobs[iframe].output.calcs_reversed[0].output.energy,
                    ]
                )

        postprocess = postprocess_EOS(flow_output)
        postprocess.name = self.name + "_" + postprocess.name

        return Flow(
            jobs=relax_jobs + static_jobs + [postprocess],
            output=postprocess.output,
            name=self.name,
        )


"""
MPLegacy = MP legacy, atomate1-PBE-GGA compatible WFs
"""


@dataclass
class MPLegacyEosDoubleRelaxMaker(DoubleRelaxMaker):
    """
    Workflow to generate initial MP legacy PBE-GGA double relaxation.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker1 : .BaseVaspMaker
        Maker to use to generate the first relaxation.
    relax_maker2 : .BaseVaspMaker
        Maker to use to generate the second relaxation.

    NB: WAVECARs are copied over to speed up calculation
    """

    name: str = "MP Legacy EOS double relax"
    relax_maker1: BaseVaspMaker | None = field(default_factory=MPLegacyEosRelaxMaker)
    relax_maker2: BaseVaspMaker = field(
        default_factory=lambda: MPLegacyEosRelaxMaker(
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR",)}
        )
    )


@dataclass
class MPLegacyEosMaker(EosMaker):
    """
    Workflow to generate atomate1, MP PBE-GGA compatible EOS flow.

    First relax a structure using relax_maker.
    Then perform a series of deformations to the relaxed structure, and
    evaluate single-point energies with static_maker.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker : .BaseVaspMaker
        Maker to generate the relaxation, defaults to DoubleRelaxMaker
    deformation_maker : .BaseVaspMaker
        Maker to generate deformations + single-points, defaults to TransmuterMaker
    static_maker : .BaseVaspMaker
        Optional Maker to generate statics from transmutation.
        Original atomate workflow did not include statics, including it here
    strain : tuple[float]
        Percentage linear strain to apply as a deformation, default = -5% to 5%
    number_of_frames : int
        Number of strain calculations to do for EOS fit, default = 6
    """

    name: str = "MP Legacy GGA EOS Maker"
    relax_maker: Maker | None = field(default_factory=MPLegacyEosDoubleRelaxMaker)
    deformation_maker: Maker = field(
        default_factory=lambda: MPLegacyDeformationMaker(
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR",)}
        )
    )
    static_maker: Maker | None = field(
        default_factory=lambda: MPLegacyEosStaticMaker(
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR",)}
        )
    )
    linear_strain: tuple[float, float] = (-0.05, 0.05)
    number_of_frames: int = 6


"""
MPGGA = MP PBE-GGA compatible WFs
"""


@dataclass
class MPGGAEosDoubleRelaxMaker(DoubleRelaxMaker):
    """
    Workflow to generate initial MP PBE-GGA double relaxation for EOS fitting.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker1 : .BaseVaspMaker
        Maker to use to generate the first relaxation.
    relax_maker2 : .BaseVaspMaker
        Maker to use to generate the second relaxation.

    NB: WAVECARs are copied over to speed up calculation
    """

    name: str = "MP GGA EOS double relax"
    relax_maker1: BaseVaspMaker | None = field(default_factory=MPGGAEosRelaxMaker)
    relax_maker2: BaseVaspMaker = field(
        default_factory=lambda: MPGGAEosRelaxMaker(
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR",)}
        )
    )


@dataclass
class MPGGAEosMaker(EosMaker):
    """
    Workflow to generate MP PBE-GGA compatible EOS flow.

    First relax a structure using relax_maker.
    Then perform a series of deformations to the relaxed structure, and
    evaluate single-point energies with static_maker.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker : .BaseVaspMaker
        Maker to generate the relaxation, defaults to DoubleRelaxMaker
    deformation_maker : .BaseVaspMaker
        Maker to generate deformations + single-points, defaults to TransmuterMaker
    static_maker : .BaseVaspMaker
        Optional Maker to generate statics from transmutation.
        Original atomate workflow did not include statics, including it here
    strain : tuple[float]
        Percentage linear strain to apply as a deformation, default = -5% to 5%
    number_of_frames : int
        Number of strain calculations to do for EOS fit, default = 6
    """

    name: str = "MP GGA EOS Maker"
    relax_maker: Maker | None = field(default_factory=MPGGAEosDoubleRelaxMaker)
    deformation_maker: Maker = field(
        default_factory=lambda: MPGGADeformationMaker(
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR",)}
        )
    )
    static_maker: Maker | None = field(
        default_factory=lambda: MPGGAEosStaticMaker(
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR",)}
        )
    )
    linear_strain: tuple[float, float] = (-0.05, 0.05)
    number_of_frames: int = 6


"""
MPMetaGGA prefix = MP r2SCAN Meta-GGA compatible
"""


@dataclass
class MPMetaGGAEosDoubleRelaxMaker(DoubleRelaxMaker):
    """
    Workflow to generate initial MP r2SCAN Meta-GGA double relaxation for EOS fitting.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker1 : .BaseVaspMaker
        Maker to use to generate the first relaxation.
    relax_maker2 : .BaseVaspMaker
        Maker to use to generate the second relaxation.

    NB: WAVECARs are copied over to speed up calculation
    """

    name: str = "MP Meta-GGA EOS double relax"
    relax_maker1: BaseVaspMaker | None = field(
        default_factory=MPMetaGGAEosPreRelaxMaker
    )
    relax_maker2: BaseVaspMaker = field(
        default_factory=lambda: MPMetaGGAEosRelaxMaker(
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR",)}
        )
    )


@dataclass
class MPMetaGGAEosMaker(EosMaker):
    """
    Workflow to generate atomate1, MP r2SCAN Meta-GGA EOS flow.

    First relax a structure using relax_maker.
    Then perform a series of deformations to the relaxed structure, and
    evaluate single-point energies with static_maker.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker : .BaseVaspMaker
        Maker to generate the relaxation, defaults to DoubleRelaxMaker
    deformation_maker : .BaseVaspMaker
        Maker to generate deformations + single-points, defaults to TransmuterMaker
    static_maker : .BaseVaspMaker
        Optional Maker to generate statics from transmutation.
        Original atomate workflow did not include statics, including it here
    strain : tuple[float]
        Percentage linear strain to apply as a deformation, default = -5% to 5%
    number_of_frames : int
        Number of strain calculations to do for EOS fit, default = 6
    """

    name: str = "MP Meta-GGA EOS Maker"
    relax_maker: Maker | None = field(default_factory=MPMetaGGAEosDoubleRelaxMaker)
    deformation_maker: Maker = field(
        default_factory=lambda: MPMetaGGADeformationMaker(
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR",)}
        )
    )
    static_maker: Maker | None = field(
        default_factory=lambda: MPMetaGGAEosStaticMaker(
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR",)}
        )
    )
    linear_strain: tuple[float, float] = (-0.05, 0.05)
    number_of_frames: int = 6
