import pytest

def test_input_generators(si_structure):
    from atomate2.cp2k.sets.defect import (
        DefectSetGenerator, DefectStaticSetGenerator, DefectRelaxSetGenerator, DefectCellOptSetGenerator,
        DefectHybridStaticSetGenerator, DefectHybridRelaxSetGenerator, DefectHybridCellOptSetGenerator
    )

    # check that all generators give the correct printing
    for gen in [
        DefectSetGenerator(), DefectStaticSetGenerator(), DefectRelaxSetGenerator(),
        DefectCellOptSetGenerator(), DefectHybridStaticSetGenerator(),
        DefectHybridRelaxSetGenerator(), DefectHybridCellOptSetGenerator()
        ]:
        input_set = gen.get_input_set(si_structure)
        assert input_set.cp2k_input.check("FORCE_EVAL/DFT/PRINT/PDOS") or input_set.cp2k_input.check("FORCE_EVAL/DFT/PRINT/DOS")
        assert input_set.cp2k_input.check("FORCE_EVAL/DFT/PRINT/V_HARTREE_CUBE")
