"""Tests the band structure input set generator"""

from atomate2.aims.sets.bs import BandStructureSetGenerator
from tests.aims import compare_files


def comp_system(atoms, user_params, test_name, work_path, ref_path):
    k_point_density = user_params.pop("k_point_density", 20)
    generator = BandStructureSetGenerator(
        user_parameters=user_params, k_point_density=k_point_density
    )
    input_set = generator.get_input_set(atoms)
    input_set.write_input(work_path / test_name)
    compare_files(test_name, work_path, ref_path)


def test_si_bs(Si, species_dir, tmp_path, ref_path):
    parameters = {"species_dir": str(species_dir), "k_grid": [8, 8, 8]}
    comp_system(Si, parameters, "static-si-bs", tmp_path, ref_path)


def test_si_bs_output(Si, species_dir, tmp_path, ref_path):
    parameters = {
        "species_dir": str(species_dir),
        "k_grid": [8, 8, 8],
        "output": [
            "json_log",
        ],
    }
    comp_system(Si, parameters, "static-si-bs-output", tmp_path, ref_path)


def test_si_bs_density(Si, species_dir, tmp_path, ref_path):
    parameters = {
        "species_dir": str(species_dir),
        "k_grid": [8, 8, 8],
        "k_point_density": 40,
    }
    comp_system(Si, parameters, "static-si-bs-density", tmp_path, ref_path)
