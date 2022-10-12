import pytest

def test_input_generators(si_structure):
    from atomate2.cp2k.sets.base import Cp2kInputGenerator, multiple_input_updators
    from atomate2.cp2k.sets.core import (
        StaticSetGenerator, RelaxSetGenerator, CellOptSetGenerator, 
        HybridSetGenerator, HybridStaticSetGenerator, HybridRelaxSetGenerator, 
        HybridCellOptSetGenerator
    )

    for gen in [StaticSetGenerator(), HybridStaticSetGenerator()]:
        input_set = gen.get_input_set(si_structure)
        assert input_set.cp2k_input["GLOBAL"]["RUN_TYPE"].values[0] == "ENERGY_FORCE"

    for gen in [RelaxSetGenerator(), HybridRelaxSetGenerator()]:
        input_set = gen.get_input_set(si_structure)
        assert input_set.cp2k_input["GLOBAL"]["RUN_TYPE"].values[0] == "GEO_OPT" 
        assert input_set.cp2k_input.get("MOTION")
        assert input_set.cp2k_input['MOTION']['GEO_OPT']['BFGS']['TRUST_RADIUS'].values[0] == pytest.approx(0.1)

    for gen in [CellOptSetGenerator(), HybridCellOptSetGenerator()]:
        input_set = gen.get_input_set(si_structure)
        assert input_set.cp2k_input["GLOBAL"]["RUN_TYPE"].values[0] == "CELL_OPT" 

    gen = HybridSetGenerator()
    input_set = gen.get_input_set(si_structure)
    assert input_set.cp2k_input["GLOBAL"]["RUN_TYPE"].values[0] == "ENERGY_FORCE"
    assert input_set.cp2k_input.check("FORCE_EVAL/DFT/XC/HF") 
    assert input_set.cp2k_input.check("FORCE_EVAL/DFT/AUXILIARY_DENSITY_MATRIX_METHOD")