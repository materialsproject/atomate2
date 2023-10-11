"""
Module defining MatPES flows.

In case of questions, consult @janosh or @esoteric-ephemera. Makes PBE + r2SCAN
cheaper than running both separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jobflow import Maker

from atomate2.vasp.jobs.matpes import MatPesGGAStaticMaker, MatPesMetaGGAStaticMaker


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
    static1: Maker | None = field(default_factory=MatPesGGAStaticMaker)
    static2: Maker = field(
        default_factory=lambda: MatPesMetaGGAStaticMaker(
            # could copy CHGCAR from GGA to meta-GGA directory too but is redundant
            # since VASP can reconstruct it from WAVECAR
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR",)}
        )
    )
