import pytest


def test_input_generators(si_structure, basis_and_potential):
    from atomate2.cp2k.sets.core import (
        CellOptSetGenerator,
        HybridCellOptSetGenerator,
        HybridRelaxSetGenerator,
        HybridStaticSetGenerator,
        MDSetGenerator,
        NonSCFSetGenerator,
        RelaxSetGenerator,
        StaticSetGenerator,
    )

    for gen in (
        StaticSetGenerator(user_input_settings=basis_and_potential),
        HybridStaticSetGenerator(user_input_settings=basis_and_potential),
    ):
        input_set = gen.get_input_set(si_structure)
        assert input_set.cp2k_input["GLOBAL"]["RUN_TYPE"].values[0] == "ENERGY_FORCE"
        assert input_set.cp2k_input.check("FORCE_EVAL/DFT/KPOINTS")

    for gen in (
        RelaxSetGenerator(user_input_settings=basis_and_potential),
        HybridRelaxSetGenerator(user_input_settings=basis_and_potential),
    ):
        input_set = gen.get_input_set(si_structure)
        assert input_set.cp2k_input["GLOBAL"]["RUN_TYPE"].values[0] == "GEO_OPT"
        assert input_set.cp2k_input.get("MOTION")
        assert input_set.cp2k_input["MOTION"]["GEO_OPT"]["BFGS"]["TRUST_RADIUS"].values[
            0
        ] == pytest.approx(0.1)

    for gen in (
        CellOptSetGenerator(user_input_settings=basis_and_potential),
        HybridCellOptSetGenerator(user_input_settings=basis_and_potential),
    ):
        input_set = gen.get_input_set(si_structure)
        assert input_set.cp2k_input["GLOBAL"]["RUN_TYPE"].values[0] == "CELL_OPT"

    gen = HybridStaticSetGenerator(user_input_settings=basis_and_potential)
    gen.user_input_settings = {
        "activate_hybrid": {
            "hybrid_functional": "HSE06",
            "eps_schwarz": 5e-3,
            "eps_schwarz_forces": 1e-2,
        }
    }
    input_set = gen.get_input_set(si_structure)
    assert input_set.cp2k_input.check("FORCE_EVAL/DFT/XC/HF")
    assert (
        input_set.cp2k_input["FORCE_EVAL"]["DFT"]["XC"]["HF"]["INTERACTION_POTENTIAL"][
            "POTENTIAL_TYPE"
        ].values[0]
        == "SHORTRANGE"
    )
    assert input_set.cp2k_input["FORCE_EVAL"]["DFT"]["XC"]["HF"]["SCREENING"][
        "EPS_SCHWARZ"
    ].values[0] == pytest.approx(5e-3)
    assert input_set.cp2k_input["FORCE_EVAL"]["DFT"]["XC"]["HF"]["SCREENING"][
        "EPS_SCHWARZ_FORCES"
    ].values[0] == pytest.approx(1e-2)
    assert input_set.cp2k_input.check("FORCE_EVAL/DFT/AUXILIARY_DENSITY_MATRIX_METHOD")

    gen = NonSCFSetGenerator(user_input_settings=basis_and_potential)
    input_set = gen.get_input_set(si_structure)
    assert input_set.cp2k_input.check("FORCE_EVAL/DFT/PRINT/BAND_STRUCTURE")
    assert (
        "KPOINT_SET"
        in input_set.cp2k_input["FORCE_EVAL"]["DFT"]["PRINT"][
            "BAND_STRUCTURE"
        ].subsections
    )
    assert (
        input_set.cp2k_input["FORCE_EVAL"]["DFT"]["PRINT"]["BAND_STRUCTURE"][
            "KPOINT_SET"
        ][0]["NPOINTS"].values[0]
        == gen.line_density
    )

    gen = MDSetGenerator(user_input_settings=basis_and_potential)
    input_set = gen.get_input_set(si_structure)
    assert input_set.cp2k_input["GLOBAL"]["RUN_TYPE"].values[0] == "MD"
    assert input_set.cp2k_input.check("MOTION/MD")
