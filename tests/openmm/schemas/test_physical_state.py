import pytest

def test_phyical_state(alchemy_input_set):
    from atomate2.openmm.schemas.physical_state import PhysicalState

    physical_state = PhysicalState.from_input_set(alchemy_input_set)

    assert physical_state.temperature > 0
    assert physical_state.step_size > 0
    assert physical_state.friction_coefficient > 0
