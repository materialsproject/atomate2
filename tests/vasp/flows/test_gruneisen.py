# from pathlib import Path
#
# import torch
# from jobflow import run_locally
# from pymatgen.core.structure import Structure
# from pymatgen.phonon.gruneisen import (
#     GruneisenParameter,
#     GruneisenPhononBandStructureSymmLine,
# )
#
# from atomate2.common.schemas.gruneisen import (
#     GruneisenDerivedProperties,
#     GruneisenInputDirs,
#     GruneisenParameterDocument,
#     PhononRunsImaginaryModes,
# )
# from atomate2.vasp.flows.gruneisen import GruneisenMaker
# from atomate2.vasp.flows.phonons import PhononMaker
#
#
# def test_gruneisen_wf_vasp(clean_dir, mock_vasp, si_diamond: Structure, tmp_path: Path):
#
#     # mapping from job name to directory containing test files
#     ref_paths = {
#         "tight relax 1": "Si_gruneisen/tight_relax_1",
#         "tight relax 2": "Si_gruneisen/tight_relax_2",
#         "tight relax 1 ground": "Si_gruneisen/tight_relax_ground",
#         "tight relax 1 plus": "Si_gruneisen/tight_relax_plus",
#         "tight relax 1 minus": "Si_gruneisen/tight_relax_minus",
#         "phonon": "Si_gruneisen/phonon",
#         "phonon 1": "Si_gruneisen/phonon_1",
#         "phonon 2": "Si_gruneisen/phonon_2",
#
#     }
#
#     # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
#     fake_run_vasp_kwargs = {
#         "tight relax 1": {"incar_settings": ["NSW", "ISMEAR", "ISIF"]},
#         "tight relax 2": {"incar_settings": ["NSW", "ISMEAR", "ISIF"]},
#         "tight relax ground": {"incar_settings": ["NSW", "ISMEAR", "ISIF"]},
#         "tight relax plus": {"incar_settings": ["NSW", "ISMEAR", "ISIF"]},
#         "tight relax minus": {"incar_settings": ["NSW", "ISMEAR", "ISIF"]},
#         "static": {"incar_settings": ["NSW", "ISMEAR", "ISIF"]},
#     }
#
#     # automatically use fake VASP and write POTCAR.spec during the test
#     mock_vasp(ref_paths, fake_run_vasp_kwargs)
#
#
#     flow = GruneisenMaker(
#         phonon_maker=PhononMaker(
#             create_thermal_displacements=False,
#             store_force_constants=False,
#             prefer_90_degrees=False,
#         ),
#         compute_gruneisen_param_kwargs={
#             "gruneisen_mesh": f"{tmp_path}/gruneisen_mesh.pdf",
#             "gruneisen_bs": f"{tmp_path}/gruneisen_band.pdf",
#         },
#     ).make(structure=si_diamond)
#
#     # run the flow or job and ensure that it finished running successfully
#     responses = run_locally(flow, create_folders=True, ensure_success=True)
#
#     # validate the outputs types
#     gp_doc = responses[flow.output.uuid][1].output
#     assert isinstance(gp_doc, GruneisenParameterDocument)
#     assert isinstance(gp_doc.gruneisen_parameter, GruneisenParameter)
#     assert isinstance(
#         gp_doc.gruneisen_band_structure, GruneisenPhononBandStructureSymmLine
#     )
#     assert isinstance(gp_doc.derived_properties, GruneisenDerivedProperties)
#     assert isinstance(gp_doc.gruneisen_parameter_inputs, GruneisenInputDirs)
#     assert isinstance(gp_doc.phonon_runs_has_imaginary_modes, PhononRunsImaginaryModes)
#
#     # check if plots are generated in specified directory / thus testing kwargs as well
#     assert Path(f"{tmp_path}/gruneisen_mesh.pdf").exists()
#     assert Path(f"{tmp_path}/gruneisen_band.pdf").exists()
