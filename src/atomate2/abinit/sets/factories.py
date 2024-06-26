# coding: utf-8
"""Factory functions for Abinit input files """
from __future__ import annotations

import numpy as np
import pymatgen.io.abinit.abiobjects as aobj
from abipy.abio.inputs import AbinitInput
import abipy.core.abinit_units as abu

def bse_with_mdf_from_inputs(nscf_input, bs_loband, nband, 
                        mbpt_sciss=0.0, mdf_epsinf=0.0, scr_input=None, exc_type="TDA", bs_algo="haydock", accuracy="normal", spin_mode="polarized",
                        zcut="0.1 eV", ecuteps=3.0, coulomb_mode="model_df", bs_freq_mesh=[0.0, 10, 0.01]) -> AbinitInput:
    """Return a sigma input."""

    bse_input = nscf_input.deepcopy()
    bse_input.pop_irdvars()
    exc_ham = aobj.ExcHamiltonian(bs_loband=bs_loband, nband=nband, mbpt_sciss=mbpt_sciss*abu.eV_Ha, coulomb_mode=coulomb_mode, ecuteps=ecuteps,
                                  spin_mode=spin_mode, mdf_epsinf=mdf_epsinf, exc_type=exc_type, algo=bs_algo,
                                  bs_freq_mesh=np.array(bs_freq_mesh)*abu.eV_Ha, with_lf=True, zcut=zcut)

    bse_input.set_vars(exc_ham.to_abivars())
    # TODO: Cannot use istwfk != 1.
    if scr_input:
        bse_input.set_vars(ecuteps=scr_input["ecuteps"])
        bse_input.set_vars(bs_coulomb_term=11)
        bse_input.pop_vars(["mdf_epsinf"])
    bse_input.set_vars(istwfk="*1")
    bse_input.set_vars(ecutwfn=nscf_input["ecut"])
    bse_input.set_vars(bs_haydock_niter=200)
    bse_input.set_vars(nband=nband)
    return bse_input
