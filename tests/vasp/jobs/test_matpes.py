import pytest
from emmet.core.tasks import TaskDoc
from jobflow import run_locally
from pymatgen.core import Structure
from pymatgen.io.vasp.sets import MatPESStaticSet

from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.matpes import MatPesGGAStaticMaker, MatPesMetaGGAStaticMaker

expected_incar = {
    "ALGO": "Normal",
    "EDIFF": 1e-05,
    "ENAUG": 1360,
    "ENCUT": 680,
    "GGA": "PE",
    "ISMEAR": 0,
    "ISPIN": 2,
    "KSPACING": 0.22,
    "LAECHG": True,
    "LASPH": True,
    "LCHARG": True,
    "LMIXTAU": True,
    "LORBIT": 11,
    "LREAL": False,
    "LWAVE": False,
    "MAGMOM": [0.6, 0.6],
    "NELM": 200,
    "NSW": 0,
    "PREC": "Accurate",
    "SIGMA": 0.05,
    "LMAXMIX": 6,
    "LDAU": False,
    "LDAUJ": {
        "F": {"Co": 0, "Cr": 0, "Fe": 0, "Mn": 0, "Mo": 0, "Ni": 0, "V": 0, "W": 0},
        "O": {"Co": 0, "Cr": 0, "Fe": 0, "Mn": 0, "Mo": 0, "Ni": 0, "V": 0, "W": 0},
    },
    "LDAUL": {
        "F": {"Co": 2, "Cr": 2, "Fe": 2, "Mn": 2, "Mo": 2, "Ni": 2, "V": 2, "W": 2},
        "O": {"Co": 2, "Cr": 2, "Fe": 2, "Mn": 2, "Mo": 2, "Ni": 2, "V": 2, "W": 2},
    },
    "LDAUTYPE": 2,
    "LDAUU": {
        "F": {
            "Co": 3.32,
            "Cr": 3.7,
            "Fe": 5.3,
            "Mn": 3.9,
            "Mo": 4.38,
            "Ni": 6.2,
            "V": 3.25,
            "W": 6.2,
        },
        "O": {
            "Co": 3.32,
            "Cr": 3.7,
            "Fe": 5.3,
            "Mn": 3.9,
            "Mo": 4.38,
            "Ni": 6.2,
            "V": 3.25,
            "W": 6.2,
        },
    },
}


@pytest.mark.parametrize("maker_cls", [MatPesGGAStaticMaker, MatPesMetaGGAStaticMaker])
def test_matpes_static_maker_default_values(maker_cls: BaseVaspMaker):
    maker = maker_cls()
    is_meta = "Meta" in maker_cls.__name__
    assert maker.name == f"MatPES {'meta-' if is_meta else ''}GGA static"
    assert isinstance(maker.input_set_generator, MatPESStaticSet)
    if is_meta:
        assert (
            maker.input_set_generator._config_dict[  # noqa: SLF001
                "INCAR"
            ].get("METAGGA")
            == "R2SCAN"
        )
    config = maker.input_set_generator.config_dict
    assert {*config} == {"INCAR", "POTCAR", "PARENT", "POTCAR_FUNCTIONAL"}
    assert all(
        v == expected_incar[k] for k, v in config["INCAR"].items() if k != "MAGMOM"
    )


def test_matpes_gga_static_maker(mock_vasp, clean_dir, vasp_test_dir):
    # map from job name to directory containing reference input/output files
    gga_job_name = "MatPES GGA static"
    ref_paths = {gga_job_name: "matpes_static_flow/pbe_static"}
    si_struct = Structure.from_file(
        f"{vasp_test_dir}/matpes_static_flow/pbe_static/inputs/POSCAR.gz"
    )

    # exclude LWAVE from INCAR checking since it defaults to False in MatPesGGAStatic
    # but is overridden to True in the MatPES static flow to speed up SCF convergence
    # of 2nd static, but we use the same reference input files in both tests
    mock_vasp(ref_paths, {gga_job_name: {"incar_exclude": ["LWAVE"]}})

    job = MatPesGGAStaticMaker().make(si_struct)

    # ensure flow runs successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # validate output
    task_doc = responses[job.uuid][1].output
    assert isinstance(task_doc, TaskDoc)
    assert task_doc.output.energy == pytest.approx(-10.84940729)


def test_matpes_meta_gga_static_maker(mock_vasp, clean_dir, vasp_test_dir):
    # map from job name to directory containing reference input/output files
    ref_paths = {"MatPES meta-GGA static": "matpes_static_flow/r2scan_static"}
    si_struct = Structure.from_file(
        f"{vasp_test_dir}/matpes_static_flow/r2scan_static/inputs/POSCAR.gz"
    )

    mock_vasp(ref_paths)

    job = MatPesMetaGGAStaticMaker().make(si_struct)

    # ensure flow runs successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # validate output
    task_doc = responses[job.uuid][1].output
    assert isinstance(task_doc, TaskDoc)
    assert task_doc.output.energy == pytest.approx(-17.53895667)
