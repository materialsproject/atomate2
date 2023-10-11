"""Tests the GW input set generator"""
from pathlib import Path

from atomate2.aims.sets.bs import GWSetGenerator
from tests import compare_files


def comp_system(atoms, user_params, test_name, work_path, ref_path):
    k_point_density = user_params.pop("k_point_density", 20)
    generator = GWSetGenerator(
        user_parameters=user_params, k_point_density=k_point_density
    )
    input_set = generator.get_input_set(atoms)
    input_set.write_input(work_path / test_name)
    compare_files(test_name, work_path, ref_path)


def test_si_gw(Si, species_dir, tmp_path, ref_path):
    parameters = {
        "species_dir": str(species_dir),
        "k_grid": [2, 2, 2],
        "k_point_density": 10,
    }
    comp_system(Si, parameters, "static-si-gw", tmp_path, ref_path)
