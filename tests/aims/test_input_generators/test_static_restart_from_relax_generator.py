"""The test of input sets generating from restart information"""

import json
import os
import shutil
from pathlib import Path


from atomate2.aims.sets.core import StaticSetGenerator
from tests.aims import compare_files


def comp_system(atoms, prev_dir, test_name, work_path, ref_path, species_dir):
    generator = StaticSetGenerator(user_parameters={})
    # adjust species dir in the prev_dir
    params_file = Path(prev_dir) / "parameters.json"
    shutil.copy(params_file, Path(prev_dir) / "~parameters.json")
    with open(params_file) as f:
        params = json.load(f)
    params["species_dir"] = species_dir.as_posix()
    with open(params_file, "w") as f:
        json.dump(params, f)

    input_set = generator.get_input_set(
        atoms, prev_dir, properties=["energy", "forces", "stress"]
    )
    input_set.write_input(work_path / test_name)
    compare_files(test_name, work_path, ref_path)
    shutil.move(Path(prev_dir) / "~parameters.json", params_file)


def test_static_from_relax_si(Si, species_dir, tmp_path, ref_path):
    comp_system(
        Si,
        f"{ref_path}/relax-si/outputs",
        "static-from-prev-si",
        tmp_path,
        ref_path,
        species_dir,
    )


def test_static_from_relax_si_no_kgrid(Si, species_dir, tmp_path, ref_path):
    comp_system(
        Si,
        f"{ref_path}/relax-no-kgrid-si/",
        "static-from-prev-no-kgrid-si",
        tmp_path,
        ref_path,
        species_dir,
    )


def test_static_from_relax_default_species_dir(Si, species_dir, tmp_path, ref_path):
    sd_def = os.getenv("AIMS_SPECIES_DIR", None)
    os.environ["AIMS_SPECIES_DIR"] = str(species_dir)

    comp_system(
        Si,
        f"{ref_path}/relax-default-sd-si/",
        "static-from-prev-default-sd-si",
        tmp_path,
        ref_path,
        species_dir,
    )

    if sd_def:
        os.environ["AIMS_SPECIES_DIR"] = sd_def
    else:
        os.unsetenv("AIMS_SPECIES_DIR")


def test_static_from_relax_o2(O2, species_dir, tmp_path, ref_path):
    comp_system(
        O2,
        f"{ref_path}/relax-o2/",
        "static-from-prev-o2",
        tmp_path,
        ref_path,
        species_dir,
    )


def test_static_from_relax_default_species_dir_o2(O2, species_dir, tmp_path, ref_path):
    sd_def = os.getenv("AIMS_SPECIES_DIR", None)
    os.environ["AIMS_SPECIES_DIR"] = str(species_dir)

    comp_system(
        O2,
        f"{ref_path}/relax-default-sd-o2/",
        "static-from-prev-default-sd-o2",
        tmp_path,
        ref_path,
        species_dir,
    )

    if sd_def:
        os.environ["AIMS_SPECIES_DIR"] = sd_def
    else:
        os.unsetenv("AIMS_SPECIES_DIR")
