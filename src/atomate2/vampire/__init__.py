"""Interface to the external VAMPIRE atomistic spin-dynamics code.

This subpackage vendors pymatgen's ``VampireCaller``/``VampireOutput`` (removed
from pymatgen in 2026.3.23) so atomate2's magnetic-exchange workflow can still
estimate critical temperatures via Vampire Monte-Carlo. See
:mod:`atomate2.vampire.vampire_caller` for provenance details.
"""

from __future__ import annotations

from atomate2.vampire.vampire_caller import VampireCaller, VampireOutput

__all__ = ["VampireCaller", "VampireOutput"]
