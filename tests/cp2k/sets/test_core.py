import pytest


def test_input_generators(si_structure):
    from atomate2.cp2k.sets.core import (
        StaticSetGenerator, RelaxSetGenerator, CellOptSetGenerator,
        HybridSetGenerator, HybridStaticSetGenerator, HybridRelaxSetGenerator,
        HybridCellOptSetGenerator, NonSCFSetGenerator, MDSetGenerator
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
    assert input_set.cp2k_input.check("FORCE_EVAL/DFT/KPOINTS")

    gen = NonSCFSetGenerator()
    input_set = gen.get_input_set(si_structure)
    assert input_set.cp2k_input.check("FORCE_EVAL/DFT/PRINT/BAND_STRUCTURE")
    assert "KPOINT_SET" in input_set.cp2k_input["FORCE_EVAL"]['DFT']['PRINT']['BAND_STRUCTURE'].subsections
    assert input_set.cp2k_input["FORCE_EVAL"]['DFT']['PRINT']['BAND_STRUCTURE']['KPOINT_SET'][0]['NPOINTS'].values[0] == gen.line_density

    gen = MDSetGenerator()
    input_set = gen.get_input_set(si_structure)
    assert input_set.cp2k_input["GLOBAL"]["RUN_TYPE"].values[0] == "MD"
    assert input_set.cp2k_input.check("MOTION/MD")