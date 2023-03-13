import pytest


@pytest.mark.parametrize(
    "copy_kwargs,files",
    [
        (
            {"additional_cp2k_files": ("electron_density",)},
            ("CP2K-ELECTRON_DENSITY-1_0.cube", "cp2k.out", "cp2k.inp"),
        ),
    ],
)
def test_copy_cp2k_outputs_static(cp2k_test_dir, tmp_dir, copy_kwargs, files):
    from pathlib import Path

    from atomate2.cp2k.files import copy_cp2k_outputs

    path = cp2k_test_dir / "Si_band_structure" / "static" / "outputs"
    copy_cp2k_outputs(src_dir=path, **copy_kwargs)

    for file in files:
        assert Path(file).exists()


def test_get_largest_relax_extension(cp2k_test_dir):
    from atomate2.cp2k.files import get_largest_relax_extension

    path = cp2k_test_dir / "Si_band_structure" / "static" / "outputs"
    extension = get_largest_relax_extension(directory=path)
    assert extension == ""
