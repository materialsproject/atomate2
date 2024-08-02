"""
Module defining MatPES flows.

In case of questions, consult @janosh or @esoteric-ephemera. Makes PBE + r2SCAN
cheaper than running both separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Maker
from pymatgen.io.vasp.sets import MatPESStaticSet

from atomate2.vasp.jobs.matpes import MatPesGGAStaticMaker, MatPesMetaGGAStaticMaker

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

    name: str = "MatPES static flow"
    static1: Maker | None = field(
        default_factory=lambda: MatPesGGAStaticMaker(
            input_set_generator=MatPESStaticSet(
                # write WAVECAR so we can use as pre-conditioned starting point for
                # static2/3
                user_incar_settings={"LWAVE": True}
            ),
        )
    )
    static2: Maker = field(
        default_factory=lambda: MatPesMetaGGAStaticMaker(
            # start from pre-conditioned WAVECAR from static1 to speed up convergence
            # could copy CHGCAR too but is redundant since VASP can reconstruct it from
            # WAVECAR
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR",)}
        )
    )
    # optional 3rd PBE+U static in case structure contains elements with +U corrections
    static3: Maker | None = field(
        default_factory=lambda: MatPesGGAStaticMaker(
            name="MatPES GGA+U static",
            input_set_generator=MatPESStaticSet(
                user_incar_settings={"LDAU:": True},  # enable +U corrections
            ),
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR",)},
        )
    )

    def make(self, structure: Structure, prev_dir: str | Path | None = None) -> Flow:
        """Create a flow with MatPES statics.

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

        # only run 3rd static if set generator not None and structure contains at least
        # one element with Hubbard +U corrections
        if self.static3 is not None:
            static3_config = self.static3.input_set_generator.config_dict
            u_corrections = static3_config.get("INCAR", {}).get("LDAUU", {})
            elems = set(map(str, structure.elements))
            if self.static3 and any(
                anion in elems and elems & {*cations}
                for anion, cations in u_corrections.items()
            ):
                static3 = self.static3.make(structure, prev_dir=static1.output.dir_name)
                output["static3"] = static3.output
                jobs += [static3]

        return Flow(jobs=jobs, output=output, name=self.name)
