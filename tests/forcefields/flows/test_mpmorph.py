"""Test MPMorph forcefield flows."""

import pytest
from pymatgen.analysis.structure_matcher import StructureMatcher
from jobflow import run_locally
from atomate2.forcefields.flows.mpmorph import (
    MPMorphLJMDMaker,
    MPMorphCHGNetMDMaker,
    MPMorphSlowQuenchLJMDMaker,
    MPMorphSlowQuenchCHGNetMDMaker,
    MPMorphFastQuenchLJMDMaker,
    MPMorphFastQuenchCHGNetMDMaker,
)

name_to_maker = {
    "LJ": MPMorphLJMDMaker,
    "LJ Slow Quench": MPMorphSlowQuenchLJMDMaker,
    "LJ Fast Quench": MPMorphFastQuenchLJMDMaker,
    "CHGNet": MPMorphCHGNetMDMaker,
    "CHGNet Slow Quench": MPMorphSlowQuenchCHGNetMDMaker,
    "CHGNet Fast Quench": MPMorphFastQuenchCHGNetMDMaker,
}


@pytest.mark.parametrize(
    "ff_name",
    [
        "LJ",
        "LJ Slow Quench",
        "LJ Fast Quench",
        "CHGNet",
        "CHGNet Slow Quench",
        "CHGNet Fast Quench",
    ],
)
def test_mpmorph_mlff_maker(ff_name, sr_ti_o3_structure, test_dir, clean_dir):
    temp = 500
    n_steps_convergence = 500
    n_steps_production = 1000

    ref_energies_per_atom = {
        "CHGNet": -5.280157089233398,
        "MACE": -5.311369895935059,
        "LJ": 0,
    }

    # ASE can slightly change tolerances on structure positions
    matcher = StructureMatcher()

    unit_cell_structure = sr_ti_o3_structure.copy()

    structure = unit_cell_structure.to_conventional() * (2, 2, 2)

    print(name_to_maker[ff_name])
    flow = name_to_maker[ff_name](
        temperature=temp,
        steps_convergence=n_steps_convergence,
        steps_total_production=n_steps_production,
    ).make(structure)

    response = run_locally(flow, ensure_success=True)
    task_doc = response[next(iter(response))][1].output
