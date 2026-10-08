import pytest
from jobflow import Flow
from pymatgen.core.structure import Structure

from atomate2.vasp.flows.debye_waller import DebyeWallerMaker
from atomate2.vasp.flows.phonons import PhononMaker


def test_debye_waller_maker_vasp_flow(si_structure: Structure):
    """The phonon flow output goes to compute_debye_waller."""
    maker = DebyeWallerMaker(temperatures=[100, 300], mesh=(10, 10, 10))
    assert maker.phonon_maker.store_force_constants
    flow = maker.make(si_structure)
    phonon_flow, debye_waller = flow.jobs
    assert isinstance(phonon_flow, Flow)
    assert debye_waller.name == "compute_debye_waller"
    assert flow.output.uuid == debye_waller.uuid

    kwargs = debye_waller.function_kwargs
    assert kwargs["phonon_output"].uuid == phonon_flow.output.uuid
    assert kwargs["temperatures"] == [100, 300]
    assert kwargs["mesh"] == (10, 10, 10)
    assert kwargs["symprec"] == maker.phonon_maker.symprec


def test_debye_waller_maker_checks_phonon_maker():
    with pytest.raises(ValueError, match="store_force_constants=True"):
        DebyeWallerMaker(phonon_maker=PhononMaker())
