import json
from pathlib import Path

import pytest
from emmet.core.openmm import OpenMMInterchange, OpenMMTaskDocument
from jobflow import Flow, JobStore, run_locally
from maggma.stores import MemoryStore
from monty.json import MontyDecoder
from pymatgen.core import Composition, Structure

from atomate2.common.jobs.mpmorph import get_random_packed_structure
from atomate2.forcefields.utils import revert_default_dtype
from atomate2.openmm.jobs.core import NVTMaker


@pytest.fixture
def run_job(tmp_path):
    def run_job(job):
        response_dict = run_locally(job, ensure_success=True, root_dir=tmp_path)
        return list(response_dict.values())[-1][1].output

    return run_job


@pytest.fixture(scope="package")
def openmm_data(test_dir):
    return test_dir / "openmm"


@pytest.fixture(scope="package")
def interchange(openmm_data):
    # we use openff to generate the interchange object that we test on
    # but we don't want to create a logical dependency on openff, in
    # case the user has another way of generating the interchange object
    regenerate_test_data = False
    if regenerate_test_data:
        import openff.toolkit as tk
        from openff.interchange import Interchange
        from openff.interchange.components._packmol import pack_box
        from openff.toolkit import ForceField
        from openff.units import unit

        from atomate2.openff.utils import (
            create_mol_spec,
            merge_specs_by_name_and_smiles,
        )
        from atomate2.openmm.utils import openff_to_openmm_interchange

        o = create_mol_spec("O", 300, charge_method="mmff94")
        cco = create_mol_spec("CCO", 10, charge_method="mmff94")
        cco2 = create_mol_spec("CCO", 20, name="cco2", charge_method="mmff94")
        mol_specs = [o, cco, cco2]
        mol_specs.sort(
            key=lambda x: tk.Molecule.from_json(x.openff_mol).to_smiles() + x.name
        )

        topology = pack_box(
            molecules=[tk.Molecule.from_json(spec.openff_mol) for spec in mol_specs],
            number_of_copies=[spec.count for spec in mol_specs],
            mass_density=0.8 * unit.grams / unit.milliliter,
        )

        mol_specs = merge_specs_by_name_and_smiles(mol_specs)

        openff_interchange = Interchange.from_smirnoff(
            force_field=ForceField("openff_unconstrained-2.1.1.offxml"),
            topology=topology,
            charge_from_molecules=[
                tk.Molecule.from_json(spec.openff_mol) for spec in mol_specs
            ],
            allow_nonintegral_charges=True,
        )

        openmm_interchange = openff_to_openmm_interchange(openff_interchange)

        with open(openmm_data / "interchange.json", "w") as file:
            file.write(openmm_interchange.model_dump_json())

    with open(openmm_data / "interchange.json") as file:
        return OpenMMInterchange.model_validate_json(file.read())


@pytest.fixture
def output_dir(test_dir):
    return test_dir / "classical_md" / "output_dir"


@pytest.fixture(scope="session")
def random_structure(test_dir) -> Structure:
    test_files = test_dir / "test_files"
    test_files.mkdir(parents=True, exist_ok=True)
    struct_file = test_files / "random_structure.json"

    # disable this flag to speed up local testing
    regenerate_test_data = True
    if regenerate_test_data:
        struct_file.unlink(missing_ok=True)
        composition = Composition("Fe100")

        n_atoms = 60
        struct = get_random_packed_structure(
            composition=composition,
            target_atoms=n_atoms,
            packmol_seed=1,
        )
        struct.to_file(str(struct_file))
    return Structure.from_file(struct_file)


@pytest.fixture(scope="session")
def task_doc(random_structure: Structure, test_dir: Path) -> OpenMMInterchange:
    from atomate2.openmm.jobs.mace import generate_mace_interchange

    output_dir = test_dir / "test_files" / "output_dir"
    output_dir.mkdir(parents=True, exist_ok=True)

    # disable this flag to speed up local testing
    regenerate_test_data = True
    if regenerate_test_data:
        (output_dir / "taskdoc.json").unlink(missing_ok=True)
        generate_job = generate_mace_interchange(
            random_structure,
        )
        nvt_job = NVTMaker(
            n_steps=2, traj_interval=1, state_interval=1, save_structure=True
        ).make(generate_job.output.interchange, prev_dir=generate_job.output.dir_name)

        job_store = JobStore(MemoryStore(), additional_stores={"data": MemoryStore()})

        with revert_default_dtype():
            run_locally(
                Flow([generate_job, nvt_job]),
                store=job_store,
                ensure_success=True,
                root_dir=output_dir,
            )

    task_doc_dict = json.load((output_dir / "taskdoc.json").open(), cls=MontyDecoder)

    return OpenMMTaskDocument.model_validate(task_doc_dict)


@pytest.fixture(scope="session")
def mace_interchange(task_doc: OpenMMTaskDocument) -> OpenMMInterchange:
    return OpenMMInterchange.model_validate_json(task_doc.interchange)
