import pytest
from pymatgen.core.structure import Structure

from atomate2.vasp.sets.core import MLMDSetGenerator


def test_mlmd_generator(test_dir) -> None:
    struct_gan = Structure.from_file(test_dir / "structures" / "GaN.cif")

    gen = MLMDSetGenerator()
    incar_updates = gen.get_incar_updates(structure=struct_gan)
    assert "ML_LMLFF" in incar_updates
    assert incar_updates["ML_LMLFF"] is True
    assert "ML_MODE" in incar_updates
    assert incar_updates["ML_MODE"] == "train"

    gen = MLMDSetGenerator(ml_mode="run")
    incar_updates = gen.get_incar_updates(structure=struct_gan)
    assert "ML_MODE" in incar_updates
    assert incar_updates["ML_MODE"] == "run"

    with pytest.raises(
        ValueError, match=r"Supported values for ml_mode are: train, select, refit, run"
    ):
        MLMDSetGenerator(ml_mode="zztop")
