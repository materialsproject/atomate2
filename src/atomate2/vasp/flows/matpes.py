"""
Module defining MatPES flows.

In case of questions, consult @janosh or @esoteric-ephemera. Makes PBE + r2SCAN
cheaper than running both separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Maker

from atomate2.vasp.jobs.matpes import MatPesGGAStaticMaker, MatPesMetaGGAStaticMaker
from atomate2.vasp.sets.matpes import MatPesGGAStaticSetGenerator

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core import Structure


@dataclass
class MatPesStaticFlowMaker(Maker):
    """MatPES flow doing a GGA static followed by meta-GGA static.

    Uses the GGA WAVECAR to speed up electronic convergence on the meta-GGA static.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    static1 : .BaseVaspMaker
        Maker to generate the first VASP static.
    static2 : .BaseVaspMaker
        Maker to generate the second VASP static.
    static3 : .BaseVaspMaker or None
        Maker to generate the optional third VASP static. Defaults to GGA static with
        +U corrections if structure contains elements with +U corrections, else to None.
    """

    name: str = "MatPES GGA plus meta-GGA static"
    static1: Maker | None = field(
        default_factory=lambda: MatPesGGAStaticMaker(
            # write WAVECAR so we can pass as pre-conditioned starting point to static2
            input_set_generator=MatPesGGAStaticSetGenerator(
                user_incar_settings={"LWAVE": True}
            ),
        )
    )
    static2: Maker = field(
        default_factory=lambda: MatPesMetaGGAStaticMaker(
            # could copy CHGCAR from GGA to meta-GGA directory too but is redundant
            # since VASP can reconstruct it from WAVECAR
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR",)}
        )
    )
    # optional 3rd PBE+U static in case structure contains elements with +U corrections
    static3: Maker | None = field(
        default_factory=lambda: MatPesGGAStaticMaker(
            input_set_generator=MatPesGGAStaticSetGenerator(
                user_incar_settings={"LDAU:": True},  # enable +U corrections
            ),
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR",)},
        )
    )

    def make(self, structure: Structure, prev_dir: str | Path | None = None):
        """
        Create a flow with MatPES statics.

        By default, a PBE static is followed by an r2SCAN static and optionally a PBE+U
        static if the structure contains elements with +U corrections. The PBE static is
        run with LWAVE=True so its WAVECAR can be passed as a pre-conditioned starting
        point to both the r2SCAN static and the PBE+U static.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous VASP calculation directory to copy output files from.

        Returns
        -------
        Flow
            A flow containing 2 or 3 statics.
        """
        static1 = self.static1.make(structure, prev_dir=prev_dir)
        static2 = self.static2.make(structure, prev_dir=static1.output.dir_name)
        output = {"static1": static1.output, "static2": static2.output}
        jobs = [static1, static2]

        # add third static if structure contains elements with +U corrections
        if self.static3 and {*structure.elements} & {
            *self.static3.input_set_generator.config_dict.get("INCAR", {}).get(
                "LDAUU", []
            )
        }:
            static3 = self.static3.make(structure, prev_dir=static1.output.dir_name)
            output["static3"] = static3.output
            jobs += [static3]

        return Flow(jobs=jobs, output=output, name=self.name)
