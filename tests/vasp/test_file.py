import pytest


@pytest.mark.parametrize(
    "copy_kwargs,files",
    [
        ({}, ("POSCAR", "INCAR", "KPOINTS", "POTCAR", "OUTCAR", "vasprun.xml")),
        ({"contcar_to_poscar": False}, ("CONTCAR", "INCAR", "KPOINTS")),
        ({"additional_vasp_files": ("PROCAR",)}, ("POSCAR", "INCAR", "PROCAR")),
    ],
)
def test_copy_vasp_outputs_static(vasp_test_dir, tmp_dir, copy_kwargs, files):
    from pathlib import Path

    from atomate2.vasp.file import copy_vasp_outputs

    path = vasp_test_dir / "Si_static" / "outputs"
    copy_vasp_outputs(src_dir=path, **copy_kwargs)

    for file in files:
        assert Path(file).exists()


@pytest.mark.parametrize(
    "copy_kwargs,files",
    [
        ({}, ("POSCAR", "INCAR", "KPOINTS", "POTCAR", "OUTCAR", "vasprun.xml")),
        ({"contcar_to_poscar": False}, ("CONTCAR", "INCAR", "KPOINTS")),
        ({"additional_vasp_files": ("PROCAR",)}, ("POSCAR", "INCAR", "PROCAR")),
    ],
)
def test_copy_vasp_outputs_double(vasp_test_dir, tmp_dir, copy_kwargs, files):
    from pathlib import Path

    from atomate2.vasp.file import copy_vasp_outputs

    path = vasp_test_dir / "Si_structure_optimization_double" / "outputs"
    copy_vasp_outputs(src_dir=path, **copy_kwargs)

    for file in files:
        assert Path(file).exists()


def test_get_largest_relax_extension(vasp_test_dir):
    from atomate2.vasp.file import get_largest_relax_extension

    path = vasp_test_dir / "Si_structure_optimization_double" / "outputs"
    extension = get_largest_relax_extension(directory=path)
    assert extension == ".relax2"

    path = vasp_test_dir / "Si_static" / "outputs"
    extension = get_largest_relax_extension(directory=path)
    assert extension == ""
