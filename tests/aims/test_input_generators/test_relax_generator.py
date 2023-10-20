import os

from atomate2.aims.sets.core import RelaxSetGenerator
from tests.aims import compare_files


def comp_system(atoms, user_params, test_name, work_path, ref_path):
    generator = RelaxSetGenerator(user_parameters=user_params)
    input_set = generator.get_input_set(atoms)
    input_set.write_input(work_path / test_name)
    compare_files(test_name, work_path, ref_path)


def test_relax_si(Si, species_dir, tmp_path, ref_path):
    parameters = {"species_dir": str(species_dir / "light"), "k_grid": [2, 2, 2]}
    comp_system(Si, parameters, "relax-si/inputs", tmp_path, ref_path)


def test_relax_si_no_kgrid(Si, species_dir, tmp_path, ref_path):
    parameters = {"species_dir": str(species_dir / "light")}
    comp_system(Si, parameters, "relax-no-kgrid-si", tmp_path, ref_path)


def test_relax_default_species_dir(Si, species_dir, tmp_path, ref_path):
    sd_def = os.getenv("AIMS_SPECIES_DIR", None)
    os.environ["AIMS_SPECIES_DIR"] = str(species_dir / "light")
    parameters = {"k_grid": [2, 2, 2]}

    comp_system(Si, parameters, "relax-default-sd-si", tmp_path, ref_path)

    if sd_def:
        os.environ["AIMS_SPECIES_DIR"] = sd_def
    else:
        os.unsetenv("AIMS_SPECIES_DIR")


def test_relax_o2(O2, species_dir, tmp_path, ref_path):
    parameters = {"species_dir": str(species_dir / "light")}
    comp_system(O2, parameters, "relax-o2", tmp_path, ref_path)


def test_relax_default_species_dir_o2(O2, species_dir, tmp_path, ref_path):
    sd_def = os.getenv("AIMS_SPECIES_DIR", None)
    os.environ["AIMS_SPECIES_DIR"] = str(species_dir / "light")
    parameters = {"k_grid": [2, 2, 2]}

    comp_system(O2, parameters, "relax-default-sd-o2", tmp_path, ref_path)

    if sd_def:
        os.environ["AIMS_SPECIES_DIR"] = sd_def
    else:
        os.unsetenv("AIMS_SPECIES_DIR")
