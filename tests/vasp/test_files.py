from pathlib import Path

import pytest

from atomate2.vasp.files import copy_vasp_outputs, get_largest_relax_extension


@pytest.mark.parametrize(
    "copy_kwargs,files",
    [
        ({}, ("POSCAR", "INCAR", "OUTCAR", "vasprun.xml")),
        ({"contcar_to_poscar": False}, ("CONTCAR", "INCAR")),
        ({"additional_vasp_files": ("CHGCAR",)}, ("POSCAR", "INCAR", "CHGCAR")),
    ],
)
def test_copy_vasp_outputs_static(vasp_test_dir, tmp_dir, copy_kwargs, files):
    path = vasp_test_dir / "Si_band_structure" / "static" / "outputs"
    copy_vasp_outputs(src_dir=path, **copy_kwargs)

    for file in files:
        assert Path(file).exists()


@pytest.mark.parametrize(
    "copy_kwargs,files",
    [
        ({}, ("POSCAR", "INCAR", "KPOINTS", "OUTCAR", "vasprun.xml")),
        ({"contcar_to_poscar": False}, ("CONTCAR", "INCAR", "KPOINTS")),
        ({"additional_vasp_files": ("vasp.out",)}, ("POSCAR", "INCAR", "vasp.out")),
    ],
)
def test_copy_vasp_outputs_double(vasp_test_dir, tmp_dir, copy_kwargs, files):
    path = vasp_test_dir / "Si_old_double_relax" / "outputs"
    copy_vasp_outputs(src_dir=path, **copy_kwargs)

    for file in files:
        assert Path(file).exists()


def test_get_largest_relax_extension(vasp_test_dir):
    path = vasp_test_dir / "Si_old_double_relax" / "outputs"
    extension = get_largest_relax_extension(directory=path)
    assert extension == ".relax2"

    path = vasp_test_dir / "Si_band_structure" / "static" / "outputs"
    extension = get_largest_relax_extension(directory=path)
    assert extension == ""
