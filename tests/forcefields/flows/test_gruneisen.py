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
from atomate2.forcefields.flows.gruneisen import GruneisenMaker
from atomate2.forcefields.flows.phonons import PhononMaker


def test_gruneisen_wf_ff(clean_dir, si_structure: Structure, tmp_path: Path):
    flow = GruneisenMaker(
        symprec=1e-2,
        compute_gruneisen_param_kwargs={
            "gruneisen_mesh": f"{tmp_path}/gruneisen_mesh.pdf",
            "gruneisen_bs": f"{tmp_path}/gruneisen_band.pdf",
        },
        phonon_maker=PhononMaker(
            use_symmetrized_structure="conventional",
            create_thermal_displacements=False,
            store_force_constants=False,
            prefer_90_degrees=False,
        ),
    ).make(structure=si_structure)

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
