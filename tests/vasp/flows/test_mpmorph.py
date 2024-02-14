"""Test MPMorph VASP flows."""
import pytest

from atomate2.common.flows.amorphous import EquilibriumVolumeMaker
from atomate2.vasp.jobs.md import MDMaker
from jobflow import run_locally

from pymatgen.core import Structure

def test_equilibrium_volume_maker(mock_vasp, clean_dir, vasp_test_dir):

    ref_paths = {
        "Equilibrium Volume Maker molecular dynamics 1": "Li_mp_morph/Li_0.8",
        "Equilibrium Volume Maker molecular dynamics 2": "Li_mp_morph/Li_1.0",
        "Equilibrium Volume Maker molecular dynamics 3": "Li_mp_morph/Li_1.2"
    }

    mock_vasp(ref_paths)

    intial_structure = Structure.from_file(
        f"{vasp_test_dir}/Li_mp_morph/Li_1.0/inputs/POSCAR.gz"
    )

    flow = EquilibriumVolumeMaker(
        md_maker = MDMaker(),
    ).make(
        structure = intial_structure
    )

    responses = run_locally(flow,create_folders=True, ensure_success=True)
    head_uuid = next(iter(responses))
    print([responses[uuid][1].output for uuid in responses])
    # asserting False so that stdout is printed by pytest
    assert False

    