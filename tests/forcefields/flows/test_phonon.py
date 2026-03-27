import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from ase.calculators.calculator import Calculator
from emmet.core.phonon import (
    CalcMeta,
    PhononBS,
    PhononBSDOSDoc,
    PhononComputationalSettings,
    PhononDOS,
)
from jobflow import Flow, JobStore, run_locally
from numpy.testing import assert_allclose
from pymatgen.core.structure import Structure

from atomate2.forcefields.flows.phonons import PhononMaker
from atomate2.forcefields.jobs import ForceFieldRelaxMaker, ForceFieldStaticMaker
from atomate2.forcefields.utils import MLFF

from ..conftest import mlff_is_installed  # noqa: TID252

# TODO fix GAP, currently fails with RuntimeError, see
# https://github.com/materialsproject/atomate2/pull/918#issuecomment-2253659694

# skip m3gnet and matpes models due to matcalc requiring
# DGL which is PyTorch 2.4 incompatible, raises
# "FileNotFoundError: Cannot find DGL C++ libgraphbolt_pytorch_2.4.1.so"
skip_mlff = set(
    map(
        MLFF,
        [
            "Forcefield",
            "GAP",
            "M3GNet",
            "MATPES_R2SCAN",
            "MATPES_PBE",
            "Allegro",
            "FAIRChem",
        ],
    )
)


@pytest.mark.parametrize(
    "mlff",
    [mlff for mlff in MLFF if mlff not in skip_mlff and mlff_is_installed(mlff)],
)
def test_phonon_maker_initialization_with_all_mlff(
    mlff,
    si_structure: Structure,
    test_dir: Path,
    get_deepmd_pretrained_model_path: Path,
):
    """Test PhononMaker can be initialized with all MLFF static and relax makers."""

    chk_pt_dir = test_dir / "forcefields"

    calc_kwargs = {
        MLFF.Nequip: {
            "compile_path": f"{chk_pt_dir}/nequip/nequip_ff_sr_ti_o3.nequip.pth"
        },
        MLFF.NEP: {"model_filename": f"{test_dir}/forcefields/nep/nep.txt"},
        MLFF.DeepMD: {"model": get_deepmd_pretrained_model_path},
        MLFF.UPET: {"model": "pet-mad-xs"},
    }.get(mlff, {})
    static_maker = ForceFieldStaticMaker(
        name=f"{mlff} static",
        force_field_name=str(mlff),
        calculator_kwargs=calc_kwargs,
    )
    relax_maker = ForceFieldRelaxMaker(
        name=f"{mlff} relax",
        force_field_name=str(mlff),
        relax_kwargs={"fmax": 0.00001},
        calculator_kwargs=calc_kwargs,
    )

    try:
        phonon_maker = PhononMaker(
            bulk_relax_maker=relax_maker,
            static_energy_maker=static_maker,
            phonon_displacement_maker=static_maker,
            use_symmetrized_structure="conventional",
            create_thermal_displacements=False,
            store_force_constants=False,
        )

        flow = phonon_maker.make(si_structure)
        assert isinstance(flow, Flow)
        assert len(flow) == 7, f"{len(flow)=}"
        assert flow[1].name == f"{mlff} relax", f"{flow[1].name=}"
        assert flow[3].name == f"{mlff} static", f"{flow[3].name=}"
        assert flow[4].name == "generate_phonon_displacements", f"{flow[4].name=}"
        assert flow[5].name == "run_phonon_displacements", f"{flow[5].name=}"

        # expected_calc = ase_calculator(mlff)
        relax_calc = phonon_maker.bulk_relax_maker.calculator
        if mlff == MLFF.Forcefield:
            assert relax_calc is None, f"{relax_calc=}"
        else:
            assert isinstance(relax_calc, Calculator), f"{type(relax_calc)=}"
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize PhononMaker with {mlff=} makers"
        ) from exc


@pytest.mark.skipif(not mlff_is_installed("CHGNet"), reason="matgl is not installed")
@pytest.mark.parametrize("from_name", [False, True])
def test_phonon_wf_force_field(
    clean_dir, si_structure: Structure, tmp_path: Path, from_name: bool
):
    # TODO brittle due to inability to adjust dtypes in CHGNetRelaxMaker

    # See issue: https://github.com/materialsproject/atomate2/issues/1395
    # Testing JSON store explicitly to avoid regression for this flow
    json_dir = TemporaryDirectory()
    json_store = JobStore.from_dict_spec(
        {
            "docs_store": {
                "type": "JSONStore",
                "paths": str(Path(json_dir.name) / "output.json"),
                "read_only": False,
            },
            "additional_stores": {
                "data": {
                    "type": "JSONStore",
                    "paths": str(Path(json_dir.name) / "blob_output.json"),
                    "read_only": False,
                }
            },
        }
    )

    phonon_kwargs = dict(
        use_symmetrized_structure="conventional",
        create_thermal_displacements=False,
        store_force_constants=False,
        prefer_90_degrees=False,
        generate_frequencies_eigenvectors_kwargs={
            "tstep": 100,
            "filename_bs": (filename_bs := f"{tmp_path}/phonon_bs_test.png"),
            "filename_dos": (filename_dos := f"{tmp_path}/phonon_dos_test.pdf"),
        },
    )

    if from_name:
        phonon_maker = PhononMaker.from_force_field_name("CHGNet", **phonon_kwargs)
        if phonon_kwargs.get("relax_initial_structure", True):
            assert isinstance(phonon_maker.bulk_relax_maker, ForceFieldRelaxMaker)
            assert "CHGNet" in phonon_maker.bulk_relax_maker.force_field_name

        for attr in ("static_energy_maker", "phonon_displacement_maker"):
            assert "CHGNet" in getattr(phonon_maker, attr).force_field_name

        assert (
            PhononMaker.from_force_field_name(
                "CHGNet", relax_initial_structure=False
            ).bulk_relax_maker
            is None
        )
    else:
        phonon_maker = PhononMaker(**phonon_kwargs)

    flow = phonon_maker.make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(
        flow, create_folders=True, ensure_success=True, store=json_store
    )

    # close temp dir for JSON store
    json_dir.cleanup()

    # validate the outputs
    ph_bs_dos_doc = responses[flow[-1].uuid][1].output
    assert isinstance(ph_bs_dos_doc, PhononBSDOSDoc)

    ph_band_struct = ph_bs_dos_doc.phonon_bandstructure
    assert isinstance(ph_band_struct, PhononBS)

    ph_dos = ph_bs_dos_doc.phonon_dos
    assert isinstance(ph_dos, PhononDOS)
    assert ph_bs_dos_doc.thermal_displacement_data is None
    assert isinstance(ph_bs_dos_doc.structure, Structure)
    assert ph_bs_dos_doc.force_constants is None
    assert all(isinstance(cm, CalcMeta) for cm in ph_bs_dos_doc.calc_meta)
    assert_allclose(ph_bs_dos_doc.total_dft_energy, -5.37245798, 4)
    assert ph_bs_dos_doc.born is None
    assert ph_bs_dos_doc.epsilon_static is None
    assert_allclose(
        ph_bs_dos_doc.supercell_matrix,
        [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
    )
    assert_allclose(
        ph_bs_dos_doc.primitive_matrix,
        ((0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)),
        atol=1e-8,
    )
    assert ph_bs_dos_doc.code == "forcefields"
    assert isinstance(ph_bs_dos_doc.post_process_settings, PhononComputationalSettings)
    assert ph_bs_dos_doc.post_process_settings.npoints_band == 101
    assert ph_bs_dos_doc.post_process_settings.kpath_scheme == "seekpath"
    assert ph_bs_dos_doc.post_process_settings.kpoint_density_dos == 7_000

    ref_vals = {
        "entropy": [0.0, 14.6055132, 34.95943091, 51.27884372, 64.05205132],
        "heat_capacity": [0.0, 21.622918, 37.313026, 43.348308, 45.921721],
        "internal_energy": [
            10510.17946131,
            11038.76862405,
            13676.21828021,
            17534.72238986,
            21889.29538244,
        ],
        "free_energy": [
            8883.796678,
            8342.505732,
            5796.489655,
            1386.324568,
            -4481.767156
        ],
    }
    thermo_props = ph_bs_dos_doc.compute_thermo_quantities(
        [0, 100, 200, 300, 400], normalization=None
    )

    for key, vals in ref_vals.items():
        assert_allclose(thermo_props[key], vals, rtol=0.2, atol=1e-8)

    # check phonon plots exist
    assert os.path.isfile(filename_bs)
    assert os.path.isfile(filename_dos)


@pytest.mark.skipif(not mlff_is_installed("MACE"), reason="mace_torch is not installed")
def test_ext_load_phonon_initialization():
    calculator_meta = {
        "@module": "mace.calculators",
        "@callable": "mace_mp",
    }
    maker = PhononMaker.from_force_field_name(
        force_field_name=calculator_meta,
        relax_initial_structure=True,
    )
    assert maker.bulk_relax_maker.ase_calculator_name == "mace_mp"
    assert maker.static_energy_maker.ase_calculator_name == "mace_mp"
    assert maker.phonon_displacement_maker.ase_calculator_name == "mace_mp"
