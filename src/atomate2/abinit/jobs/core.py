"""Core jobs for running ABINIT calculations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from abipy.flowtk import events
from abipy.flowtk.utils import irdvars_for_ext

from atomate2.abinit.inputs.factories import NScfInputGenerator, ScfInputGenerator
from atomate2.abinit.jobs.base import BaseAbinitMaker
from atomate2.abinit.utils.common import RestartError

logger = logging.getLogger(__name__)

__all__ = ["ScfMaker", "NonScfMaker"]


@dataclass
class ScfMaker(BaseAbinitMaker):
    """Maker to create ABINIT scf jobs.

    Parameters
    ----------
    name : str
        The job name.
    """

    calc_type: str = "scf"
    name: str = "Scf calculation"
    input_generator: ScfInputGenerator = ScfInputGenerator()
    CRITICAL_EVENTS = [
        events.ScfConvergenceWarning,
    ]

    def resolve_restart_deps(self):
        """Resolve dpendencies to restart Scf calculations.

        Scf calculations can be restarted from either the WFK file or the DEN file.
        """
        # Prefer WFK over DEN files since we can reuse the wavefunctions.
        if self.restart_info.reset:
            # remove non reset keys that may have been added in a previous restart
            self.remove_restart_vars(["WFK", "DEN"])
        else:
            for ext in ("WFK", "DEN"):
                restart_file = self.restart_info.prev_outdir.has_abiext(ext)
                irdvars = irdvars_for_ext(ext)
                if restart_file:
                    break
            else:
                msg = "Cannot find WFK or DEN file to restart from."
                logger.error(msg)
                raise RestartError(msg)

            # Move out --> in.
            self.out_to_in(restart_file)

            # Add the appropriate variable for restarting.
            self.abinit_input.set_vars(irdvars)


class NonScfDeps(dict):
    def __init__(self):
        super().__init__({"scf": ["DEN"]})


@dataclass
class NonScfMaker(BaseAbinitMaker):
    """Maker to create non SCF calculations."""

    calc_type: str = "nscf"
    name: str = "non-Scf calculation"

    input_generator: NScfInputGenerator = NScfInputGenerator()
    CRITICAL_EVENTS = [
        events.NscfConvergenceWarning,
    ]

    # Here I don't know how to set a default that is mutable (dict) so I use
    # the default_factory of the dataclass field.
    dependencies: Optional[dict] = field(default_factory=NonScfDeps)

    def resolve_restart_deps(self):
        """Resolve dpendencies to restart Non-Scf calculations.

        Non-Scf calculations can only be restarted from the WFK file .
        """
        if self.restart_info.reset:
            # remove non reset keys that may have been added in a previous restart
            self.remove_restart_vars(["WFK"])
        else:
            ext = "WFK"
            restart_file = self.restart_info.prev_outdir.has_abiext(ext)
            if not restart_file:
                msg = "Cannot find the WFK file to restart from."
                logger.error(msg)
                raise RestartError(msg)

            # Move out --> in.
            self.out_to_in(restart_file)

            # Add the appropriate variable for restarting.
            irdvars = irdvars_for_ext(ext)
            self.abinit_input.set_vars(irdvars)
