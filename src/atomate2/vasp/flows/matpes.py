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
class MatPesGGAPlusMetaGGAStaticMaker(Maker):
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

    def make(self, structure: Structure, prev_dir: str | Path | None = None):
        """
        Create a flow with two chained statics.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous VASP calculation directory to copy output files from.

        Returns
        -------
        Flow
            A flow containing two statics.
        """
        static1 = self.static1.make(structure, prev_dir=prev_dir)
        static2 = self.static2.make(structure, prev_dir=static1.output.dir_name)
        output = {"static1": static1.output, "static2": static2.output}
        return Flow([static1, static2], output=output, name=self.name)
