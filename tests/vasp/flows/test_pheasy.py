from jobflow import run_locally
from numpy.testing import assert_allclose
from pymatgen.core.structure import Structure
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos

from atomate2.common.flows.pheasy import BasePhononMaker
from atomate2.common.powerups import add_metadata_to_flow
from atomate2.common.schemas.pheasy import (
    Forceconstants,
    PhononBSDOSDoc,
    PhononComputationalSettings,
    PhononJobDirs,
    PhononUUIDs,
)
from atomate2.vasp.flows.pheasy import PhononMaker
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.powerups import update_user_incar_settings


def test_pheasy_wf_vasp(mock_vasp, clean_dir, si_structure: Structure, test_dir):
    # mapping from job name to directory containing test files
    ref_paths = {
        "tight relax 1": "Si_pheasy/tight_relax_1",
        "tight relax 2": "Si_pheasy/tight_relax_2",
        "phonon static 1/2": "Si_pheasy/phonon_static_1_2",
        "phonon static 2/2": "Si_pheasy/phonon_static_2_2",
        "static": "Si_pheasy/static",
        "dielectric": "Si_pheasy/dielectric",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "tight relax 1": {"incar_settings": ["NSW", "ISMEAR", "KSPACING"]},
        "tight relax 2": {"incar_settings": ["NSW", "ISMEAR", "KSPACING"]},
        "phonon static 1/2": {"incar_settings": ["NSW", "ISMEAR"]},
        "phonon static 2/2": {"incar_settings": ["NSW", "ISMEAR"]},
        "static": {"incar_settings": ["NSW", "ISMEAR"]},
        "dielectric": {"incar_settings": ["NSW", "ISMEAR"]},
    }

    # automatically use fake VASP and write POTCAR.spec dulsring the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    si_struct = Structure.from_file(
        test_dir / "vasp/Si_pheasy/tight_relax_1/inputs/POSCAR.gz"
    )

    job = PhononMaker(
        force_diagonal=True,
        min_length=12,
        mp_id="mp-149",
        cal_anhar_fcs=False,
        # use_symmetrized_structure="primitive"
    ).make(structure=si_struct)

    job = update_user_incar_settings(
        job,
        {
            "ENCUT": 600,
            "ISMEAR": 0,
            "SIGMA": 0.05,
            "KSPACING": 0.15,
            "ISPIN": 1,
            "EDIFFG": -1e-04,
            "EDIFF": 1e-07,
        },
    )
    job = add_metadata_to_flow(
        flow=job,
        additional_fields={"mp_id": "mp-149", "unit_testing": "yes"},
        class_filter=(BaseVaspMaker, BasePhononMaker, PhononMaker),
    )

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # validate the outputs
    assert isinstance(responses[job.jobs[-1].uuid][1].output, PhononBSDOSDoc)
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.free_energies,
        [
            5792.458116272716,
            5792.451271742757,
            5792.308574619996,
            5791.3893705576875,
            5788.26230090264,
            5781.251118319793,
            5768.997781886127,
            5750.565423632229,
            5725.332982293328,
            5692.876443331414,
            5652.890004107427,
            5605.143827748774,
            5549.4640389105325,
            5485.723569667365,
            5413.837265156259,
            5333.758132354205,
            5245.473570542215,
            5149.001346193473,
            5044.385428634828,
            4931.691884236595,
            4811.004999276292,
            4682.423743675511,
            4546.058632404462,
            4402.028999079057,
            4250.460668093644,
            4091.483995048405,
            3925.2322370663665,
            3751.840212042715,
            3571.4432067784546,
            3384.176096803778,
            3190.172644484756,
            2989.5649460944087,
            2782.483002538277,
            2569.054392149452,
            2349.4040273117657,
            2123.6539796025945,
            1891.9233606773441,
            1654.3282482754973,
            1410.9816485521276,
            1161.9934874706462,
            907.4706252726158,
            647.5168891063854,
            382.2331197810191,
            111.71722934513599,
            -163.9357332035992,
            -444.6335102719554,
            -730.2865598681723,
            -1020.8079770828365,
            -1316.1134140135116,
            -1616.1209990625455,
        ],
        rtol=1e-5,  # relaxed relative tolerance
        atol=1e-5,  # add a small absolute tolerance
    )

    assert isinstance(
        responses[job.jobs[-1].uuid][1].output.phonon_bandstructure,
        PhononBandStructureSymmLine,
    )
    assert isinstance(responses[job.jobs[-1].uuid][1].output.phonon_dos, PhononDos)
    # assert isinstance(
    #     responses[job.jobs[-1].uuid][1].output.thermal_displacement_data,
    #     ThermalDisplacementData,
    # )
    assert isinstance(responses[job.jobs[-1].uuid][1].output.structure, Structure)
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.temperatures,
        [
            0,
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90,
            100,
            110,
            120,
            130,
            140,
            150,
            160,
            170,
            180,
            190,
            200,
            210,
            220,
            230,
            240,
            250,
            260,
            270,
            280,
            290,
            300,
            310,
            320,
            330,
            340,
            350,
            360,
            370,
            380,
            390,
            400,
            410,
            420,
            430,
            440,
            450,
            460,
            470,
            480,
            490,
        ],
    )
    assert responses[job.jobs[-1].uuid][1].output.has_imaginary_modes is False
    assert isinstance(
        responses[job.jobs[-1].uuid][1].output.force_constants, Forceconstants
    )
    assert isinstance(responses[job.jobs[-1].uuid][1].output.jobdirs, PhononJobDirs)
    assert isinstance(responses[job.jobs[-1].uuid][1].output.uuids, PhononUUIDs)
    assert_allclose(responses[job.jobs[-1].uuid][1].output.total_dft_energy, -5.7466748)
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.born,
        [
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ],
    )
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.epsilon_static,
        (
            (13.31020238, 0.0, -0.000000000000000000000000000000041086505480261033),
            (0.000000000000000000000000000000032869204384208823, 13.31020238, 0.0),
            (
                0.00000000000000000000000000000003697785493223493,
                -0.00000000000000000000000000000005310360021821649,
                13.31020238,
            ),
        ),
        atol=1e-8,
    )
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.supercell_matrix,
        [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
    )
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.primitive_matrix,
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        rtol=1e-5,
        atol=1e-10,
    )
    assert responses[job.jobs[-1].uuid][1].output.code == "vasp"
    assert isinstance(
        responses[job.jobs[-1].uuid][1].output.phonopy_settings,
        PhononComputationalSettings,
    )
    assert responses[job.jobs[-1].uuid][1].output.phonopy_settings.npoints_band == 101
    assert (
        responses[job.jobs[-1].uuid][1].output.phonopy_settings.kpath_scheme
        == "seekpath"
    )
    assert (
        responses[job.jobs[-1].uuid][1].output.phonopy_settings.kpoint_density_dos
        == 7000
    )
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.entropies,
        [
            0.0,
            0.0029053715285381693,
            0.03534406735813481,
            0.17371749757534496,
            0.4806355132850958,
            0.9443138235461701,
            1.5217487290037728,
            2.1748672785891556,
            2.8785588902287014,
            3.61777903114171,
            4.383308818024208,
            5.168810618964096,
            5.969257468971105,
            6.780249611796599,
            7.597788260092625,
            8.418241566262875,
            9.238367696860214,
            10.055337466350297,
            10.866738450845892,
            11.670559750796489,
            12.465162508699496,
            13.249242154996521,
            14.02178733336692,
            14.78203900188199,
            15.529451895610356,
            16.26365953641141,
            16.984443284189616,
            17.691705481526853,
            18.38544648297612,
            19.065745223657316,
            19.732742925263075,
            20.38662953003233,
            21.027632473643276,
            21.6560074426566,
            22.27203080258059,
            22.87599342375645,
            23.468195671240995,
            24.04894336028136,
            24.618544510282547,
            25.177306757323354,
            25.725535308514058,
            26.26353134118291,
            26.791590766448167,
            27.31000329059948,
            27.81905171927258,
            28.31901145900843,
            28.810150178756654,
            29.29272760048077,
            29.76699539348012,
            30.23319715155372,
        ],
        atol=1e-6,
    )
    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.heat_capacities,
        [
            0.0,
            0.009726834044170778,
            0.13866101670569114,
            0.6571570545060572,
            1.5667423561930287,
            2.639591924628341,
            3.7231364670350806,
            4.771184569421581,
            5.786825421391361,
            6.782392815371432,
            7.764289299349057,
            8.731219108699392,
            9.676996516283827,
            10.593722262345167,
            11.473976773888314,
            12.311941879701372,
            13.103752616984227,
            13.847404360575291,
            14.542454296999962,
            15.189663439692012,
            15.790656197118794,
            16.347630811439014,
            16.86312946277534,
            17.33986465971092,
            17.780593548325292,
            18.188030602925156,
            18.564789807070603,
            18.913348785224382,
            19.23602883664659,
            19.53498619439316,
            19.812210987727507,
            20.069531311525665,
            20.30862052298697,
            20.531006428284563,
            20.738081424927493,
            20.931112960999076,
            21.111253886165763,
            21.279552422144512,
            21.436961588185195,
            21.584347992205277,
            21.722499949574136,
            21.852134925891033,
            21.97390632235039,
            22.088409636021186,
            22.19618803517433,
            22.29773739353601,
            22.393510828356888,
            22.483922786412425,
            22.56935272015279,
            22.65014839366568,
        ],
        rtol=1e-5,  # relaxed relative tolerance
        atol=1e-5,  # add a small absolute tolerance
    )

    assert_allclose(
        responses[job.jobs[-1].uuid][1].output.internal_energies,
        [
            5792.458116272716,
            5792.480324532992,
            5793.015455042023,
            5796.60089455924,
            5807.487720506598,
            5828.466808566305,
            5860.302704690473,
            5902.806132190803,
            5955.617692560522,
            6018.476555173027,
            6091.22088493709,
            6173.712994848893,
            6265.774934186432,
            6367.156018184099,
            6477.527620534776,
            6596.4943662401865,
            6723.612400966099,
            6858.408714377748,
            7000.398348669136,
            7149.098235746229,
            7304.03749984975,
            7464.76459503267,
            7630.851844526549,
            7801.897968265965,
            7977.529121766132,
            8157.3988778485345,
            8341.187489623593,
            8528.600690692962,
            8719.3682206193,
            8913.242210240971,
            9109.995520608829,
            9309.420098917732,
            9511.32539258518,
            9715.536846674577,
            9921.894498604659,
            10130.251676299567,
            10340.473800672748,
            10552.437289894393,
            10766.028560740182,
            10981.143121073083,
            11197.684746889987,
            11415.564737168335,
            11634.701239831205,
            11855.018642409685,
            12076.447021347745,
            12298.92164431772,
            12522.382520360086,
            12746.773993107487,
            12972.044372785333,
            13198.145603091058,
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    assert responses[job.jobs[-1].uuid][1].output.chemsys == "Si"
