from pathlib import Path

from ase.calculators.calculator import Calculator
from jobflow import Flow, run_locally
from numpy.testing import assert_allclose
from pymatgen.core import Structure

from atomate2.common.jobs.phonons import get_supercell_size
from atomate2.forcefields.flows.phonons import PhononMaker
from atomate2.forcefields.jobs import ForceFieldRelaxMaker, ForceFieldStaticMaker
from atomate2.forcefields.utils import MLFF


def test_phonon_get_supercell_size(clean_dir, si_structure: Structure):
    job = get_supercell_size(si_structure, min_length=18, prefer_90_degrees=True)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    assert_allclose(responses[job.uuid][1].output, [[6, -2, 0], [0, 6, 0], [-3, -2, 5]])


def test_phonon_maker_initialization_with_all_mlff(
    si_structure: Structure, test_dir: Path
):
    """Test PhononMaker can be initialized with all MLFF static and relax makers."""

    chk_pt_dir = test_dir / "forcefields"
    for mlff in MLFF:
        if mlff == MLFF.GAP:
            continue  # TODO fix GAP, currently fails with RuntimeError, see
            # https://github.com/materialsproject/atomate2/pull/918#issuecomment-2253659694
        calc_kwargs = {
            MLFF.Nequip: {"model_path": f"{chk_pt_dir}/nequip/nequip_ff_sr_ti_o3.pth"},
            MLFF.NEP: {"model_filename": f"{test_dir}/forcefields/nep/nep.txt"},
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
