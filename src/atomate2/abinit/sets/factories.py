# coding: utf-8
"""Factory functions for Abinit input files """
from __future__ import annotations

import numpy as np
import pymatgen.io.abinit.abiobjects as aobj
from abipy.abio.inputs import AbinitInput

def bse_with_mdf_from_inputs(nscf_input, sigma_input, bs_loband, bs_nband, 
                        mdf_epsinf, mbpt_sciss, exc_type="TDA", bs_algo="haydock", accuracy="normal", spin_mode="polarized",
                        smearing="fermi_dirac:0.1 eV") -> AbinitInput:
    """Return a sigma input."""

    bse_input = nscf_input.deepcopy()
    bse_input.pop_irdvars()

    exc_ham = aobj.ExcHamiltonian(bs_loband, bs_nband, mbpt_sciss, coulomb_mode="model_df", ecuteps=sigma_input["ecuteps"],
                                  spin_mode=spin_mode, mdf_epsinf=mdf_epsinf, exc_type=exc_type, algo=bs_algo,
                                  bs_freq_mesh=None, with_lf=True, zcut=None)

    bse_input.set_vars(exc_ham.to_abivars())
    # TODO: Cannot use istwfk != 1.
    bse_input.set_vars(istwfk="*1")
    bse_input.set_vars(ecutwfn=nscf_input["ecut"])
    return bse_input
