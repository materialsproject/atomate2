"""Equation of State (EOS) workflows for SIESTA."""

from atomate2.siesta.flows.eos.core import (
    EOSFullBasisConvergenceFlowMaker,
    SiestaEosFlowMaker,
    collect_eos_parameter_data,
    plot_eos_parameter_fits_from_data,
    plot_eos_parameter_timing,
    write_eos_parameter_summary,
)

# Backward compatibility aliases
EOSFlowMaker = SiestaEosFlowMaker
EOSMaker = SiestaEosFlowMaker

__all__ = [
    "SiestaEosFlowMaker",
    "EOSFullBasisConvergenceFlowMaker",
    "EOSFlowMaker",  # Alias for backward compatibility
    "EOSMaker",  # Alias for backward compatibility
    "collect_eos_parameter_data",
    "plot_eos_parameter_fits_from_data",
    "plot_eos_parameter_timing",
    "write_eos_parameter_summary",
]
