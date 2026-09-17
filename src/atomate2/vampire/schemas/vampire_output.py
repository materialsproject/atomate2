"""Schema for the output of a Vampire Monte-Carlo run.

Vendored from pymatgen's ``pymatgen.command_line.vampire_caller`` (removed there
in 2026.3.23); see :mod:`atomate2.vampire.vampire_caller` for provenance details.
"""

from __future__ import annotations

from monty.json import MSONable


class VampireOutput(MSONable):
    """Process results from a Vampire Monte Carlo simulation.

    Parses the critical temperature from the simulation output.
    """

    def __init__(
        self,
        parsed_out: str | None = None,
        nmats: int | None = None,
        critical_temp: float | None = None,
    ) -> None:
        """Initialize the output.

        Args:
            parsed_out (str): JSON rep of parsed stdout DataFrame.
            nmats (int): Number of distinct materials (1 for each specie and
                up/down spin).
            critical_temp (float): Monte Carlo Tc result.
        """
        self.parsed_out = parsed_out
        self.nmats = nmats
        self.critical_temp = critical_temp
