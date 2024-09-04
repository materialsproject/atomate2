from pathlib import Path

from jobflow import run_locally
from pymatgen.core.structure import Structure
from pymatgen.phonon.gruneisen import (
    GruneisenParameter,
    GruneisenPhononBandStructureSymmLine,
)

from atomate2.common.schemas.gruneisen import (
    GruneisenDerivedProperties,
    GruneisenInputDirs,
    GruneisenParameterDocument,
    PhononRunsImaginaryModes,
)
from atomate2.vasp.flows.gruneisen import GruneisenMaker
from atomate2.vasp.flows.phonons import PhononMaker
from atomate2.vasp.powerups import update_user_incar_settings


def test_gruneisen_wf_vasp(clean_dir, mock_vasp, si_diamond: Structure, tmp_path: Path):
    # mapping from job name to directory containing test files
    ref_paths = {
        "tight relax 1": "Si_gruneisen_1/tight_relax_1",
        "tight relax 2": "Si_gruneisen_1/tight_relax_2",
        "tight relax 1 plus": "Si_gruneisen_1/tight_relax_plus",
        "tight relax 2 plus": "Si_gruneisen_1/tight_relax_plus_2",
        "tight relax 1 minus": "Si_gruneisen_1/tight_relax_minus",
        "tight relax 2 minus": "Si_gruneisen_1/tight_relax_minus_2",
        "phonon static 1/1 ground": "Si_gruneisen_1/phonon_ground",
        "phonon static 1/1 plus": "Si_gruneisen_1/phonon_plus",
        "phonon static 1/1 minus": "Si_gruneisen_1/phonon_minus",
    }
    mock_vasp(ref_paths)

    flow = GruneisenMaker(
        symprec=1e-4,
        phonon_maker=PhononMaker(
            create_thermal_displacements=False,
            store_force_constants=False,
            prefer_90_degrees=False,
            min_length=10,
            born_maker=None,
            bulk_relax_maker=None,
            static_energy_maker=None,
        ),
        compute_gruneisen_param_kwargs={
            "gruneisen_mesh": f"{tmp_path}/gruneisen_mesh.pdf",
            "gruneisen_bs": f"{tmp_path}/gruneisen_band.pdf",
        },
    ).make(structure=si_diamond)

    flow = update_user_incar_settings(flow, {"NPAR": 4, "ISMEAR": 0})

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    # validate the outputs types
    gp_doc = responses[flow.output.uuid][1].output
    assert isinstance(gp_doc, GruneisenParameterDocument)
    assert isinstance(gp_doc.gruneisen_parameter, GruneisenParameter)
    assert isinstance(
        gp_doc.gruneisen_band_structure, GruneisenPhononBandStructureSymmLine
    )
    assert isinstance(gp_doc.derived_properties, GruneisenDerivedProperties)
    assert isinstance(gp_doc.gruneisen_parameter_inputs, GruneisenInputDirs)
    assert isinstance(gp_doc.phonon_runs_has_imaginary_modes, PhononRunsImaginaryModes)

    # check if plots are generated in specified directory / thus testing kwargs as well
    assert Path(f"{tmp_path}/gruneisen_mesh.pdf").exists()
    assert Path(f"{tmp_path}/gruneisen_band.pdf").exists()
