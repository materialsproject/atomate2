"""Tests for atomate2siesta-maker CLI."""

from pathlib import Path

import pytest
from click.testing import CliRunner
from pymatgen.core import Lattice, Structure

from atomate2.siesta.cli.maker import cli
from atomate2.siesta.cli.maker.templates import TEMPLATES, RelaxTemplate


@pytest.fixture
def si_structure_file(tmp_path):
    """Create a test Silicon structure file."""
    lattice = Lattice.cubic(5.43)
    structure = Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])

    structure_file = tmp_path / "Si.cif"
    structure.to(filename=str(structure_file), fmt="cif")

    return structure_file


def test_relax_template_class():
    """Test RelaxTemplate class methods."""
    template = RelaxTemplate()

    assert template.name == "relax"
    assert "relaxation" in template.description.lower()
    assert len(template.output_files) > 0

    # Test imports generation
    imports = template.generate_imports({})
    assert "RelaxMaker" in imports
    assert "Structure" in imports

    # Test maker generation (fixed cell)
    maker_code = template.generate_maker({"cell_type": "fixed"})
    assert "fixed_cell_relaxation" in maker_code

    # Test maker generation (variable cell)
    maker_code = template.generate_maker({"cell_type": "variable"})
    assert "variable_cell_relaxation" in maker_code

    # Test with preset
    maker_code = template.generate_maker({"preset": "relax_standard"})
    assert "apply_tier_preset" in maker_code
    assert "relax_standard" in maker_code


def test_template_registry():
    """Test that template registry is populated."""
    assert "relax" in TEMPLATES
    assert "static" in TEMPLATES
    assert "bands" in TEMPLATES
    assert "dos" in TEMPLATES

    # Verify all templates are instances of WorkflowTemplate
    for name, template in TEMPLATES.items():
        assert hasattr(template, "generate")
        assert hasattr(template, "name")
        assert hasattr(template, "description")


def test_relax_template_generation(si_structure_file):
    """Test complete script generation for relax template."""
    template = RelaxTemplate()

    command = "atomate2siesta-maker make relax Si.cif"
    options = {"cell_type": "fixed", "dry_run": False}

    script = template.generate(str(si_structure_file), command, options)

    # Check header
    assert "#!/usr/bin/env python" in script
    assert "atomate2siesta-maker" in script
    assert command in script

    # Check imports
    assert "from jobflow import run_locally" in script
    assert "from pymatgen.core import Structure" in script
    assert "from atomate2.siesta.jobs.core import RelaxMaker" in script

    # Check config check
    assert "SIESTA_PP_PATH" in script
    assert "SETTINGS" in script

    # Check structure loading
    assert f'Structure.from_file("{si_structure_file}")' in script

    # Check maker creation
    assert "RelaxMaker.fixed_cell_relaxation()" in script

    # Check execution
    assert "run_locally" in script
    assert "create_folders=True" in script


def test_cli_list_command():
    """Test CLI list command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["list"])

    assert result.exit_code == 0
    assert "relax" in result.output
    assert "static" in result.output
    assert "bands" in result.output


def test_cli_make_relax(si_structure_file, tmp_path):
    """Test CLI relax command."""
    runner = CliRunner()

    output_file = tmp_path / "test_relax.py"

    result = runner.invoke(
        cli,
        [
            "relax",
            str(si_structure_file),
            "-o",
            str(output_file),
        ],
    )

    assert result.exit_code == 0
    assert output_file.exists()

    # Check generated script content
    script_content = output_file.read_text()
    assert "RelaxMaker" in script_content
    assert "fixed_cell_relaxation" in script_content
    assert str(si_structure_file) in script_content


def test_cli_make_with_preset(si_structure_file, tmp_path):
    """Test CLI relax command with preset."""
    runner = CliRunner()

    output_file = tmp_path / "test_relax_preset.py"

    result = runner.invoke(
        cli,
        [
            "relax",
            str(si_structure_file),
            "-o",
            str(output_file),
            "--preset",
            "relax_standard",
        ],
    )

    assert result.exit_code == 0
    assert output_file.exists()

    script_content = output_file.read_text()
    assert "apply_tier_preset" in script_content
    assert "relax_standard" in script_content


def test_cli_make_variable_cell(si_structure_file, tmp_path):
    """Test CLI relax command with variable cell."""
    runner = CliRunner()

    output_file = tmp_path / "test_relax_variable.py"

    result = runner.invoke(
        cli,
        [
            "relax",
            str(si_structure_file),
            "-o",
            str(output_file),
            "--cell-type",
            "variable",
        ],
    )

    assert result.exit_code == 0
    assert output_file.exists()

    script_content = output_file.read_text()
    assert "variable_cell_relaxation" in script_content


def test_cli_make_bands(si_structure_file, tmp_path):
    """Test CLI bands command."""
    runner = CliRunner()

    output_file = tmp_path / "test_bands.py"

    result = runner.invoke(
        cli,
        [
            "bands",
            str(si_structure_file),
            "-o",
            str(output_file),
            "--kpath-density",
            "30",
        ],
    )

    assert result.exit_code == 0
    assert output_file.exists()

    script_content = output_file.read_text()
    assert "BandStructureMaker" in script_content
    assert "kpath_density=30" in script_content


def test_cli_make_invalid_workflow(si_structure_file):
    """Test CLI with invalid workflow type."""
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "invalid_workflow",
            str(si_structure_file),
        ],
    )

    assert result.exit_code != 0
    assert "No such command" in result.output


def test_cli_make_missing_structure():
    """Test CLI with missing structure file."""
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "relax",
            "nonexistent.cif",
        ],
    )

    assert result.exit_code != 0


def test_cli_make_default_output_name(si_structure_file, tmp_path):
    """Test that default output name is generated correctly."""
    runner = CliRunner()

    # Change to tmp_path to avoid polluting current directory
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(
            cli,
            [
                "relax",
                str(si_structure_file),
            ],
        )

        assert result.exit_code == 0

        # Check that relax_Si.py was created
        expected_file = Path("relax_Si.py")
        assert expected_file.exists()

        script_content = expected_file.read_text()
        assert "RelaxMaker" in script_content


def test_script_is_executable(si_structure_file, tmp_path):
    """Test that generated script is executable."""
    runner = CliRunner()

    output_file = tmp_path / "test_relax.py"

    result = runner.invoke(
        cli,
        [
            "relax",
            str(si_structure_file),
            "-o",
            str(output_file),
        ],
    )

    assert result.exit_code == 0

    # Check that file is executable
    assert output_file.stat().st_mode & 0o111  # Check execute permission


def test_cli_make_phonon_with_supercell(si_structure_file, tmp_path):
    """Test CLI phonon command with space-separated supercell."""
    runner = CliRunner()

    output_file = tmp_path / "test_phonon_supercell.py"

    result = runner.invoke(
        cli,
        [
            "phonon",
            str(si_structure_file),
            "-o",
            str(output_file),
            "--supercell",
            "2",
            "2",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert output_file.exists()

    script_content = output_file.read_text()
    assert "SiestaPhononMaker" in script_content
    assert "supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]]" in script_content


def test_cli_make_neb(si_structure_file, tmp_path):
    """Test CLI NEB command with two structures."""
    runner = CliRunner()

    # Create a copy as final structure
    final_file = tmp_path / "Si_final.cif"
    final_file.write_text(si_structure_file.read_text())

    output_file = tmp_path / "test_neb.py"

    result = runner.invoke(
        cli,
        [
            "neb",
            str(si_structure_file),
            str(final_file),
            "-o",
            str(output_file),
            "--number-of-images",
            "7",
        ],
    )

    assert result.exit_code == 0
    assert output_file.exists()

    script_content = output_file.read_text()
    assert "NebDirectFlowMaker" in script_content
    assert "number_of_images=7" in script_content
    assert "initial_structure=initial" in script_content
    assert "final_structure=final" in script_content


def test_cli_make_surface(si_structure_file, tmp_path):
    """Test CLI surface command."""
    runner = CliRunner()

    output_file = tmp_path / "test_surface.py"

    result = runner.invoke(
        cli,
        [
            "surface",
            str(si_structure_file),
            "-o",
            str(output_file),
            "--miller-indices",
            "1,1,1",
        ],
    )

    assert result.exit_code == 0
    assert output_file.exists()

    script_content = output_file.read_text()
    assert "SurfaceEnergyFlowMaker" in script_content
    assert "miller_indices=(1, 1, 1)" in script_content


def test_cli_make_adsorption(si_structure_file, tmp_path):
    """Test CLI adsorption command."""
    runner = CliRunner()

    output_file = tmp_path / "test_adsorption.py"

    result = runner.invoke(
        cli,
        [
            "adsorption",
            str(si_structure_file),
            "-o",
            str(output_file),
            "--grid-size",
            "4",
            "4",
            "--height",
            "2.5",
        ],
    )

    assert result.exit_code == 0
    assert output_file.exists()

    script_content = output_file.read_text()
    assert "AdsorptionScanFlowMaker" in script_content
    assert "grid_size=(4, 4)" in script_content
    assert "height=2.5" in script_content
