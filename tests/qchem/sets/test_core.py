import pytest
from pymatgen.core import Molecule
from pymatgen.io.core import InputGenerator
from pymatgen.io.qchem import sets as pmg_sets
from pymatgen.io.qchem.sets import (
    ForceSetGenerator,
    FreqSetGenerator,
    OptSetGenerator,
    PESScanSetGenerator,
    QCInputGenerator,
    QCInputSet,
    SinglePointSetGenerator,
    TransitionStateSetGenerator,
)

import atomate2.qchem.sets.base
import atomate2.qchem.sets.core

# generator, job type, default basis set
GENERATORS = [
    (SinglePointSetGenerator, "sp", "def2-tzvpd"),
    (OptSetGenerator, "opt", "def2-svpd"),
    (TransitionStateSetGenerator, "ts", "def2-svpd"),
    (ForceSetGenerator, "force", "def2-tzvpd"),
    (FreqSetGenerator, "freq", "def2-svpd"),
    (PESScanSetGenerator, "pes_scan", "def2-svpd"),
]
SCAN_VARIABLES = {"stre": ["1 2 0.95 1.35 0.1"]}


@pytest.fixture(scope="module")
def water_mol() -> Molecule:
    """Dummy molecular structure for water as a test molecule."""
    return Molecule(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.121], [-0.783, 0.0, -0.485], [0.783, 0.0, -0.485]],
    )


def _get_input_set(generator_cls: type[QCInputGenerator], molecule, **kwargs):
    if generator_cls is PESScanSetGenerator:
        # a PES scan cannot be set up without something to scan over
        kwargs.setdefault("scan_variables", SCAN_VARIABLES)
    return generator_cls(**kwargs).get_input_set(molecule)


@pytest.mark.parametrize(
    "name",
    [
        "ForceSetGenerator",
        "FreqSetGenerator",
        "OptSetGenerator",
        "PESScanSetGenerator",
        "SinglePointSetGenerator",
        "TransitionStateSetGenerator",
    ],
)
def test_generators_are_pymatgen_generators(name: str) -> None:
    assert getattr(atomate2.qchem.sets.core, name) is getattr(pmg_sets, name)


def test_base_classes_are_pymatgen_classes() -> None:
    assert atomate2.qchem.sets.base.QCInputGenerator is QCInputGenerator
    assert atomate2.qchem.sets.base.QCInputSet is QCInputSet


@pytest.mark.parametrize(
    "generator_cls, expected_job_type, expected_basis_set", GENERATORS
)
def test_qc_sets(
    generator_cls: type[QCInputGenerator],
    expected_job_type: str,
    expected_basis_set: str,
    water_mol: Molecule,
) -> None:
    generator = generator_cls()
    assert isinstance(generator, QCInputGenerator)
    assert isinstance(generator, InputGenerator)
    assert {*generator.__dict__} >= {
        "job_type",
        "basis_set",
        "scf_algorithm",
        "qchem_version",
        "dft_rung",
        "pcm_dielectric",
        "isosvp_dielectric",
        "smd_solvent",
        "cmirs_solvent",
        "custom_smd",
        "opt_variables",
        "scan_variables",
        "max_scf_cycles",
        "geom_opt_max_cycles",
        "plot_cubes",
        "nbo_params",
        "geom_opt",
        "overwrite_inputs",
        "vdw_mode",
    }
    assert generator.scf_algorithm == "diis"
    assert generator.job_type == expected_job_type
    assert generator.basis_set == expected_basis_set

    input_set = _get_input_set(generator_cls, water_mol)
    assert isinstance(input_set, QCInputSet)
    qcinput = input_set.qcinput
    assert qcinput.molecule == water_mol
    assert qcinput.rem["job_type"] == expected_job_type
    assert qcinput.rem["basis"] == expected_basis_set
    assert qcinput.rem["scf_algorithm"] == "diis"
    assert qcinput.rem["method"] == "wb97mv"
    assert qcinput.rem["max_scf_cycles"] == "100"
    assert ("geom_opt_max_cycles" in qcinput.rem) == (
        expected_job_type in ("opt", "ts", "pes_scan")
    )
    assert "solvent_method" not in qcinput.rem


@pytest.mark.parametrize(
    "generator_cls, expected_job_type, expected_basis_set", GENERATORS
)
def test_extra_params_pcm(
    generator_cls: type[QCInputGenerator],
    expected_job_type: str,
    expected_basis_set: str,
    water_mol: Molecule,
) -> None:
    qcinput = _get_input_set(
        generator_cls, water_mol, dft_rung=2, pcm_dielectric=78.39
    ).qcinput
    assert qcinput.rem["method"] == "b97-d3"
    assert qcinput.rem["dft_d"] == "d3_bj"
    assert qcinput.rem["solvent_method"] == "pcm"
    assert qcinput.pcm == {
        "heavypoints": "194",
        "hpoints": "194",
        "radii": "uff",
        "theory": "cpcm",
        "vdwscale": "1.1",
    }
    assert qcinput.solvent == {"dielectric": "78.39"}
    assert qcinput.rem["job_type"] == expected_job_type
    assert qcinput.rem["basis"] == expected_basis_set


@pytest.mark.parametrize(
    "generator_cls, expected_job_type, expected_basis_set", GENERATORS
)
def test_extra_params_smd(
    generator_cls: type[QCInputGenerator],
    expected_job_type: str,
    expected_basis_set: str,
    water_mol: Molecule,
) -> None:
    qcinput = _get_input_set(generator_cls, water_mol, smd_solvent="water").qcinput
    assert qcinput.rem["solvent_method"] == "smd"
    assert qcinput.rem["ideriv"] == "1"
    assert qcinput.smx == {"solvent": "water"}
    assert qcinput.rem["job_type"] == expected_job_type
    assert qcinput.rem["basis"] == expected_basis_set


@pytest.mark.parametrize(
    "generator_cls, expected_job_type, expected_basis_set", GENERATORS
)
def test_extra_params_plots(
    generator_cls: type[QCInputGenerator],
    expected_job_type: str,
    expected_basis_set: str,
    water_mol: Molecule,
) -> None:
    qcinput = _get_input_set(generator_cls, water_mol, plot_cubes=True).qcinput
    assert qcinput.plots == {"grid_spacing": "0.05", "total_density": "0"}
    assert qcinput.rem["plots"] == "true"
    assert qcinput.rem["make_cube_files"] == "true"
    assert qcinput.rem["job_type"] == expected_job_type
    assert qcinput.rem["basis"] == expected_basis_set


@pytest.mark.parametrize(
    "generator_cls, expected_job_type, expected_basis_set", GENERATORS
)
def test_extra_params_nbo(
    generator_cls: type[QCInputGenerator],
    expected_job_type: str,
    expected_basis_set: str,
    water_mol: Molecule,
) -> None:
    nbo_params = {"version": 7, "plots": "PLOT"}
    qcinput = _get_input_set(generator_cls, water_mol, nbo_params=nbo_params).qcinput
    assert qcinput.rem["nbo"] == "true"
    assert qcinput.rem["nbo_external"] == "true"
    assert qcinput.nbo == {"plots": "plot"}
    # the settings supplied by the user are left untouched
    assert nbo_params == {"version": 7, "plots": "PLOT"}
    assert qcinput.rem["job_type"] == expected_job_type
    assert qcinput.rem["basis"] == expected_basis_set


def test_pes_scan_requires_scan_variables(water_mol: Molecule) -> None:
    # the generator can be created without scan variables (e.g. by PESScanMaker) ...
    generator = PESScanSetGenerator()
    # ... but they have to be set by the time the input set is generated
    with pytest.raises(ValueError, match="without some variable to scan over"):
        generator.get_input_set(water_mol)

    generator.scan_variables = SCAN_VARIABLES
    assert generator.get_input_set(water_mol).qcinput.scan == SCAN_VARIABLES


def test_generator_is_not_modified(water_mol: Molecule) -> None:
    generator = OptSetGenerator(
        pcm_dielectric=10.0, overwrite_inputs={"rem": {"symmetry": "true"}}
    )
    as_dict = generator.as_dict()
    first = str(generator.get_input_set(water_mol).qcinput)
    second = str(generator.get_input_set(water_mol).qcinput)
    assert first == second
    assert generator.as_dict() == as_dict
