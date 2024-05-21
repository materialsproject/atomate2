import json

def test_anharmonic_quantification():   
    import numpy as np
    import json
    from jobflow import run_locally
    from pymatgen.io.aims.sets.core import StaticSetGenerator

    from atomate2.aims.flows.phonons import PhononMaker
    from atomate2.aims.jobs.core import RelaxMaker, StaticMaker
    from atomate2.aims.jobs.phonons import PhononDisplacementMaker
    from atomate2.common.flows.anharmonicity import BaseAnharmonicityMaker

    from pathlib import Path
    from pymatgen.core.structure import Structure, Lattice

    from atomate2.common.jobs.anharmonicity import get_force_constants, get_emode_efreq, calc_sigma_A_oneshot

    from pymatgen.io.phonopy import get_phonopy_structure
    from phonopy import Phonopy
    from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
    from atomate2.common.schemas.phonons import get_factor

    species_dir = Path(
        "/Users/kevinbeck/Desktop/Research/atomate2/tests/aims/species_dir"
    ).resolve().parent / "species_dir"
    si = Structure(
        lattice=Lattice(
            [[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]]
        ),
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )


    # mapping from job name to directory containing test files
    ref_paths = {
        "Relaxation calculation": "phonon-relax-si",
        "phonon static aims 1/1": "phonon-disp-si",
        "SCF Calculation": "phonon-energy-si",
    }

    # settings passed to fake_run_aims; adjust these to check for certain input settings
    fake_run_aims_kwargs = {}

    # automatically use fake FHI-aims

    parameters = {
        "k_grid": [2, 2, 2],
        "species_dir": (species_dir / "light").as_posix(),
    }
    # generate job

    parameters_phonon_disp = dict(compute_forces=True, **parameters)
    """
    maker = PhononMaker(
        bulk_relax_maker=RelaxMaker.full_relaxation(user_params=parameters),
        static_energy_maker=StaticMaker(
            input_set_generator=StaticSetGenerator(user_params=parameters)
        ),
        use_symmetrized_structure="primitive",
        phonon_displacement_maker=PhononDisplacementMaker(
            input_set_generator=StaticSetGenerator(
                user_params=parameters_phonon_disp,
                user_kpoints_settings={"density": 5.0, "even": True},
            )
        ), 
    )
        """
    maker = BaseAnharmonicityMaker(
        code="aims",
        bulk_relax_maker=RelaxMaker.full_relaxation(user_params=parameters),
        static_energy_maker=StaticMaker(
            input_set_generator=StaticSetGenerator(user_params=parameters)
        ),
        use_symmetrized_structure="primitive",
        phonon_displacement_maker=PhononDisplacementMaker(
            input_set_generator=StaticSetGenerator(
                user_params=parameters_phonon_disp,
                user_kpoints_settings={"density": 5.0, "even": True},
            )
        ),
    ) 
    maker.name = "anharmonicity"
    flow = maker.make(
        si,
        supercell_matrix=np.array([-1, 1, 1, 1, -1, 1, 1, 1, -1]).reshape((3, 3)),
        socket = True
    )

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)
    return responses

test_anharmonic_quantification()
"""
# validation the outputs of the job
output = responses[flow.job_uuids[-1]][1].output

phonopy_settings_schema = {
    "description": "Collection to store computational settings for the "
    "phonon computation.",
    "properties": {
        "npoints_band": {
            "default": "number of points for band structure computation",
            "title": "Npoints Band",
            "type": "integer",
        },
        "kpath_scheme": {
            "default": "indicates the kpath scheme",
            "title": "Kpath Scheme",
            "type": "string",
        },
        "kpoint_density_dos": {
            "default": "number of points for computation of free energies and"
            " densities of states",
            "title": "Kpoint Density Dos",
            "type": "integer",
        },
    },
    "title": "PhononComputationalSettings",
    "type": "object",
}
assert output.code == "aims"
assert output.born is None
assert not output.has_imaginary_modes

assert output.temperatures == list(range(0, 500, 10))
assert output.heat_capacities[0] == 0.0
assert np.round(output.heat_capacities[-1], 2) == 23.06
assert output.phonopy_settings.schema_json() == json.dumps(phonopy_settings_schema)
assert np.round(output.phonon_bandstructure.bands[-1, 0], 2) == 14.41

# TODO: Delete this later
kpath_dict, kpath_concrete = output.get_kpath(
            structure = si,
            kpath_scheme = "seekpath",
            symprec = 1e-4,
        )
qpoints, connections = get_band_qpoints_and_path_connections(
            kpath_concrete,  
        )
fc = output.force_constants
cell = get_phonopy_structure(si)
ph = Phonopy(cell, supercell_matrix=np.array([-1, 1, 1, 1, -1, 1, 1, 1, -1]).reshape((3, 3)),primitive_matrix=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            ], factor = get_factor("aims"),symprec=1e-4, is_symmetry=True)
ph.force_constants = fc.force_constants
dm = ph.dynamical_matrix
# Convert qpoints into array of 3-element arrays
q_vectors = []
for i in range(len(qpoints)):
    for j in range(len(qpoints[i])):
        q_vectors.append(qpoints[i][j])
q_vectors = np.array(q_vectors)
# To get dynamical matrix
ph.run_qpoints(q_vectors)
dyn_mat = ph.dynamical_matrix.dynamical_matrix
"""