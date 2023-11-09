"""Tests the GW input set generator"""
import gzip
import json
from glob import glob
from pathlib import Path

from atomate2.aims.sets.bs import GWSetGenerator


def compare_files(test_name, work_dir, ref_dir):
    for file in glob(f"{work_dir / test_name}/*in"):
        with open(file) as test_file:
            test_lines = [
                line.strip()
                for line in test_file.readlines()[4:]
                if len(line.strip()) > 0
            ]

        with gzip.open(f"{ref_dir / test_name / Path(file).name}.gz", "rt") as ref_file:
            ref_lines = [
                line.strip()
                for line in ref_file.readlines()[4:]
                if len(line.strip()) > 0
            ]

        assert test_lines == ref_lines

    with open(f"{ref_dir / test_name}/parameters.json") as ref_file:
        ref = json.load(ref_file)
    ref.pop("species_dir", None)

    with open(f"{work_dir / test_name}/parameters.json") as check_file:
        check = json.load(check_file)
    check.pop("species_dir", None)

    assert ref == check


def comp_system(atoms, user_params, test_name, work_path, ref_path):
    k_point_density = user_params.pop("k_point_density", 20)
    generator = GWSetGenerator(
        user_parameters=user_params, k_point_density=k_point_density
    )
    input_set = generator.get_input_set(atoms)
    input_set.write_input(work_path / test_name)
    # input_set.write_input(ref_path / test_name)
    compare_files(test_name, work_path, ref_path)


def test_si_gw(Si, species_dir, tmp_path, ref_path):
    parameters = {
        "species_dir": str(species_dir / "light"),
        "k_grid": [2, 2, 2],
        "k_point_density": 10,
    }
    comp_system(Si, parameters, "static-si-gw", tmp_path, ref_path)
