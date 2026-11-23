import numpy as np
import pytest
from pymatgen.core import Molecule
from pymatgen.core.units import Ha_to_eV

from atomate2.qchem.flows.reorganization import InnerReorganizationMaker
from atomate2.qchem.jobs.core import SinglePointMaker
from atomate2.qchem.jobs.reorganization import prepare_vertical_molecules
from atomate2.qchem.schemas.reorganization import (
    InnerReorganizationDocument,
    kabsch_rmsd,
)
from atomate2.qchem.sets.core import SinglePointSetGenerator


@pytest.fixture
def nh3():
    return Molecule(
        ["N", "H", "H", "H"],
        [
            [0.0, 0.0, 0.0],
            [0.9306, 0.0, -0.375],
            [-0.4653, 0.8059, -0.375],
            [-0.4653, -0.8059, -0.375],
        ],
    )


@pytest.fixture
def planar_nh3_cation():
    return Molecule(
        ["N", "H", "H", "H"],
        [
            [0.0, 0.0, 0.0],
            [1.02, 0.0, 0.0],
            [-0.51, 0.8833, 0.0],
            [-0.51, -0.8833, 0.0],
        ],
        charge=1,
        spin_multiplicity=2,
    )


def test_inner_reorganization_document(nh3, planar_nh3_cation):
    energies = {"1@1": -56.5, "1@2": -56.49, "2@2": -56.1, "2@1": -56.08}
    doc = InnerReorganizationDocument.from_energies(
        nh3, planar_nh3_cation, energies, free_energy_1=-1.0, free_energy_2=9.5
    )
    assert (doc.charge_1, doc.spin_multiplicity_1) == (0, 1)
    assert (doc.charge_2, doc.spin_multiplicity_2) == (1, 2)
    assert doc.lambda_1 == pytest.approx(0.01 * Ha_to_eV)
    assert doc.lambda_2 == pytest.approx(0.02 * Ha_to_eV)
    assert doc.lambda_mean == pytest.approx(0.015 * Ha_to_eV)
    assert doc.vertical_energy_1_to_2 == pytest.approx(0.42 * Ha_to_eV)
    assert doc.vertical_energy_2_to_1 == pytest.approx(-0.39 * Ha_to_eV)
    assert doc.adiabatic_energy == pytest.approx(0.4 * Ha_to_eV)
    assert doc.adiabatic_free_energy == pytest.approx(10.5)
    assert doc.geometry_rmsd > 0
    # the round trip through a dict keeps the molecules
    assert InnerReorganizationDocument(**doc.model_dump()).molecule_2 == doc.molecule_2


def test_inner_reorganization_document_connectivity(nh3):
    pytest.importorskip("openbabel")
    stretched = nh3.copy()
    stretched.translate_sites([1], [3.0, 0.0, 0.0])
    energies = dict.fromkeys(("1@1", "1@2", "2@2", "2@1"), -1.0)
    assert (
        InnerReorganizationDocument.from_energies(
            nh3, nh3, energies
        ).connectivity_change
        == "no_change"
    )
    doc = InnerReorganizationDocument.from_energies(nh3, stretched, energies)
    assert doc.connectivity_change != "no_change"


def test_kabsch_rmsd(nh3):
    rotated = nh3.copy()
    rotated.rotate_sites(theta=1.0, axis=[1.0, 2.0, 3.0])
    rotated.translate_sites(range(4), [1.0, -2.0, 0.5])
    assert kabsch_rmsd(nh3, rotated) == pytest.approx(0.0, abs=1e-8)
    with pytest.raises(ValueError, match="same atoms"):
        kabsch_rmsd(nh3, Molecule(["O", "H", "H", "H"], nh3.cart_coords))


def test_prepare_vertical_molecules(nh3, planar_nh3_cation):
    molecules = prepare_vertical_molecules.original(nh3, planar_nh3_cation, 0, 1, 1, 2)
    expected = {
        "1@1": (nh3, 0, 1),
        "1@2": (planar_nh3_cation, 0, 1),
        "2@2": (planar_nh3_cation, 1, 2),
        "2@1": (nh3, 1, 2),
    }
    for key, (geometry, charge, spin) in expected.items():
        molecule = molecules[key]
        assert molecule.cart_coords == pytest.approx(geometry.cart_coords)
        assert (molecule.charge, molecule.spin_multiplicity) == (charge, spin)
    assert molecules["1"] is molecules["1@1"]
    assert molecules["2"] is molecules["2@2"]

    # optimized geometries can also come from task documents
    opt_output = {"output": {"optimized_molecule": planar_nh3_cation.as_dict()}}
    freq_output = {
        "output": {"optimized_molecule": None, "initial_molecule": nh3.as_dict()}
    }
    molecules = prepare_vertical_molecules.original(freq_output, opt_output, 0, 1, 1, 2)
    assert molecules["1@2"].cart_coords == pytest.approx(planar_nh3_cation.cart_coords)
    assert molecules["2@1"].cart_coords == pytest.approx(nh3.cart_coords)


def test_inner_reorganization_maker(nh3, planar_nh3_cation):
    flow = InnerReorganizationMaker().make(nh3, planar_nh3_cation)
    assert [job.name for job in flow.jobs] == [
        "optimization state 1",
        "optimization state 2",
        "prepare vertical molecules",
        "single point state 1 at geometry 1",
        "single point state 1 at geometry 2",
        "single point state 2 at geometry 2",
        "single point state 2 at geometry 1",
        "analyze inner reorganization",
    ]

    flow = InnerReorganizationMaker(opt_maker=None).make(nh3, planar_nh3_cation)
    assert flow.jobs[0].name == "prepare vertical molecules"
    assert len(flow.jobs) == 6

    with pytest.raises(ValueError, match="same atoms in the same order"):
        InnerReorganizationMaker().make(
            nh3, Molecule(["H", "N", "H", "H"], nh3.cart_coords)
        )


def test_inner_reorganization_maker_tight_scf():
    maker = InnerReorganizationMaker()
    rem = maker._sp_maker().input_set_generator.overwrite_inputs["rem"]  # noqa: SLF001
    assert rem["scf_convergence"] == "8"
    assert maker.sp_maker.input_set_generator.overwrite_inputs == {}  # not modified

    # an explicit setting of the single-point maker is kept
    sp_maker = SinglePointMaker(
        input_set_generator=SinglePointSetGenerator(
            overwrite_inputs={"rem": {"scf_convergence": "6"}, "smx": {"solvent": "w"}}
        )
    )
    maker = InnerReorganizationMaker(sp_maker=sp_maker)
    overwrite = maker._sp_maker().input_set_generator.overwrite_inputs  # noqa: SLF001
    assert overwrite == {"rem": {"scf_convergence": "6"}, "smx": {"solvent": "w"}}

    maker = InnerReorganizationMaker(tight_scf=False)
    overwrite = maker._sp_maker().input_set_generator.overwrite_inputs  # noqa: SLF001
    assert overwrite == {}


def test_negative_reorganization_warns(nh3, planar_nh3_cation, caplog):
    energies = {"1@1": -56.5, "1@2": -56.6, "2@2": -56.1, "2@1": -56.08}
    doc = InnerReorganizationDocument.from_energies(nh3, planar_nh3_cation, energies)
    assert doc.lambda_1 < 0
    assert "Negative reorganization energy" in caplog.text
    assert np.isfinite(doc.lambda_mean)


def test_inner_reorganization_maker_run(mock_qchem, clean_dir, qchem_test_dir):
    from pathlib import Path

    from jobflow import JobStore, run_locally
    from maggma.stores import MemoryStore
    from pymatgen.io.qchem.inputs import QCInput

    # NH3 -> NH3+ (wB97M-V/def2-SVPD, vacuum) from optimized geometries: pyramidal
    # NH3 and planar NH3+ (from a FrequencyOptFlatteningMaker run of each state)
    ref_dir = Path(qchem_test_dir) / "inner_reorganization_nh3"
    names = [
        f"single point state {state} at geometry {geometry}"
        for state, geometry in (("1", "1"), ("1", "2"), ("2", "2"), ("2", "1"))
    ]
    mock_qchem({name: ref_dir / name.replace(" ", "_") for name in names})

    def ref_molecule(name):
        path = ref_dir / name.replace(" ", "_") / "inputs" / "mol.qin.gz"
        return QCInput.from_file(path).molecule

    molecule_1 = ref_molecule(names[0])
    molecule_2 = ref_molecule(names[2])
    assert (molecule_2.charge, molecule_2.spin_multiplicity) == (1, 2)

    maker = InnerReorganizationMaker(
        opt_maker=None,
        sp_maker=SinglePointMaker(
            input_set_generator=SinglePointSetGenerator(basis_set="def2-svpd")
        ),
    )
    flow = maker.make(molecule_1, molecule_2)
    store = JobStore(MemoryStore(), additional_stores={"data": MemoryStore()})
    run_locally(flow, store=store, create_folders=True, ensure_success=True)
    doc = flow.output.resolve(store)

    assert doc.energy_1_at_1 == pytest.approx(-56.48627336, abs=1e-8)
    assert doc.energy_1_at_2 == pytest.approx(-56.47801546, abs=1e-8)
    assert doc.energy_2_at_2 == pytest.approx(-56.11243227, abs=1e-8)
    assert doc.energy_2_at_1 == pytest.approx(-56.08586262, abs=1e-8)
    # the neutral surface is soft (close to the NH3 inversion barrier), the cation
    # surface much stiffer
    assert doc.lambda_1 == pytest.approx(0.2247, abs=1e-4)
    assert doc.lambda_2 == pytest.approx(0.7230, abs=1e-4)
    assert doc.lambda_mean == pytest.approx(0.4739, abs=1e-4)
    # vertical and adiabatic ionization energies (expt. ~10.8 and ~10.07 eV)
    assert doc.vertical_energy_1_to_2 == pytest.approx(10.8957, abs=1e-4)
    assert doc.adiabatic_energy == pytest.approx(10.1727, abs=1e-4)
    assert doc.adiabatic_free_energy is None  # no thermochemistry without opt_maker
    assert doc.geometry_rmsd == pytest.approx(0.1778, abs=1e-4)
    assert doc.solvation_lot_info == "wB97M-V/def2-SVPD/VACUUM(NONE)"
    assert set(doc.single_point_dirs) == {"1@1", "1@2", "2@2", "2@1"}


def test_free_energy_from_task_document():
    from emmet.core.molecules.thermo import get_free_energy

    from atomate2.qchem.jobs.reorganization import _free_energy

    output = {"output": {"final_energy": -56.5, "enthalpy": 25.0, "entropy": 45.0}}
    assert _free_energy(output) == pytest.approx(get_free_energy(-56.5, 25.0, 45.0))
    assert _free_energy({"output": {"final_energy": -56.5}}) is None
    assert _free_energy(None) is None


def _emmet_parses_fragments() -> bool:
    from emmet.core.qchem.calculation import CalculationInput
    from pymatgen.io.qchem.inputs import QCInput

    fragments = [
        Molecule(["H"], [[0, 0, 0]], charge=0, spin_multiplicity=2),
        Molecule(["Li"], [[0, 0, 2]], charge=1, spin_multiplicity=1),
    ]
    qcinput = QCInput(
        molecule=fragments, rem={"job_type": "sp", "basis": "sto-3g", "method": "hf"}
    )
    try:
        CalculationInput.from_qcinput(qcinput)
    except ValueError:
        return False
    return True


def test_split_into_fragments():
    from atomate2.qchem.sets.base import split_into_fragments

    molecule = Molecule(
        ["O", "H", "H", "Li"],
        [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0], [0, 0, 2.0]],
        charge=1,
        spin_multiplicity=1,
    )
    water = {"first_atom": 1, "last_atom": 3, "charge": 0, "spin_multiplicity": 1}
    lithium = {"first_atom": 4, "last_atom": 4, "charge": 1, "spin_multiplicity": 1}
    fragments = split_into_fragments(molecule, [water, lithium])
    assert [len(f) for f in fragments] == [3, 1]
    assert [(f.charge, f.spin_multiplicity) for f in fragments] == [(0, 1), (1, 1)]
    assert fragments[1].cart_coords[0] == pytest.approx([0, 0, 2.0])

    with pytest.raises(ValueError, match="cover all atoms"):
        split_into_fragments(molecule, [water])
    with pytest.raises(ValueError, match="charges must add up"):
        split_into_fragments(molecule, [water, {**lithium, "charge": 0}])
    with pytest.raises(ValueError, match="multiplicities must combine"):
        split_into_fragments(molecule, [water, {**lithium, "spin_multiplicity": 3}])


def test_cdft_input_set():
    molecule = Molecule(
        ["O", "H", "H", "Li"],
        [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0], [0, 0, 2.0]],
        charge=1,
        spin_multiplicity=1,
    )
    constraints = [
        [
            {
                "value": 0.0,
                "coefficients": [1.0],
                "first_atoms": [1],
                "last_atoms": [3],
                "types": [None],
            }
        ]
    ]
    generator = SinglePointSetGenerator(
        fragments=[
            {"first_atom": 1, "last_atom": 3, "charge": 0, "spin_multiplicity": 1},
            {"first_atom": 4, "last_atom": 4, "charge": 1, "spin_multiplicity": 1},
        ],
        cdft_constraints=constraints,
    )
    qcinput = generator.get_input_set(molecule).qcinput
    # Q-Chem 6.2 ignores $cdft with GEN_SCFMAN, so it must be switched off
    assert qcinput.rem["gen_scfman"] == "false"
    assert qcinput.rem["cdft"] == "true"
    assert qcinput.rem["cdft_pop"] == "fbh"
    assert qcinput.cdft == constraints
    assert [len(f) for f in qcinput.molecule] == [3, 1]

    # without constraints nothing changes
    plain = SinglePointSetGenerator().get_input_set(molecule).qcinput
    assert plain.rem["gen_scfman"] == "true"
    assert "cdft" not in plain.rem
    assert isinstance(plain.molecule, Molecule)


def test_inner_reorganization_maker_cdft_settings(nh3, planar_nh3_cation):
    # NH3 split as N (atom 1) and the three hydrogens, purely to test the bookkeeping
    maker = InnerReorganizationMaker(
        cdft_fragment=(1, 1), cdft_fragment_states=((0, 1), (1, 2))
    )
    settings = maker._cdft_settings(nh3, planar_nh3_cation)  # noqa: SLF001
    state_1, state_2 = settings["states"]["1"], settings["states"]["2"]
    assert state_1["fragments"] == [
        {"first_atom": 1, "last_atom": 1, "charge": 0, "spin_multiplicity": 1},
        {"first_atom": 2, "last_atom": 4, "charge": 0, "spin_multiplicity": 1},
    ]
    assert state_2["fragments"][1] == {
        "first_atom": 2,
        "last_atom": 4,
        "charge": 0,
        "spin_multiplicity": 1,
    }
    # Q-Chem constraint values are excess electrons (-charge) and unpaired spins
    assert [c["value"] for c in state_2["constraints"][0]] == [-1.0, 1.0]
    assert [c["types"] for c in state_2["constraints"][0]] == [[None], ["s"]]

    # a fragment at the end of the molecule, charge constraint only
    maker = InnerReorganizationMaker(
        cdft_fragment=(2, 4),
        cdft_fragment_states=((0, 1), (1, 2)),
        cdft_constrain_spin=False,
    )
    settings = maker._cdft_settings(nh3, planar_nh3_cation)  # noqa: SLF001
    assert settings["states"]["2"]["fragments"][0]["last_atom"] == 1
    assert len(settings["states"]["2"]["constraints"][0]) == 1

    for fragment in ((2, 3), (1, 4)):
        maker = InnerReorganizationMaker(
            cdft_fragment=fragment, cdft_fragment_states=((0, 1), (1, 2))
        )
        with pytest.raises(ValueError, match="cdft_fragment"):
            maker._cdft_settings(nh3, planar_nh3_cation)  # noqa: SLF001
    maker = InnerReorganizationMaker(
        cdft_fragment=(1, 1), cdft_fragment_states=((0, 3), (1, 2))
    )
    with pytest.raises(ValueError, match="larger than the multiplicity"):
        maker._cdft_settings(nh3, planar_nh3_cation)  # noqa: SLF001
    with pytest.raises(ValueError, match="given together"):
        InnerReorganizationMaker(cdft_fragment=(1, 1))


def test_inner_reorganization_maker_diabatic_flow(nh3, planar_nh3_cation):
    maker = InnerReorganizationMaker(
        opt_maker=None, cdft_fragment=(1, 1), cdft_fragment_states=((0, 1), (1, 2))
    )
    flow = maker.make(nh3, planar_nh3_cation)
    names = [job.name for job in flow.jobs]
    assert names[1:5] == [
        f"single point state {s} at geometry {g}"
        for s, g in (("1", "1"), ("1", "2"), ("2", "2"), ("2", "1"))
    ]
    assert names[5:9] == [f"cdft {name}" for name in names[1:5]]
    cdft_generator = flow.jobs[5].function.__self__.input_set_generator
    assert cdft_generator.cdft_constraints is not None
    assert cdft_generator.overwrite_inputs["rem"]["cdft_pop"] == "fbh"
    assert cdft_generator.overwrite_inputs["rem"]["scf_convergence"] == "8"
    assert flow.jobs[1].function.__self__.input_set_generator.cdft_constraints is None

    maker = InnerReorganizationMaker(
        opt_maker=None,
        cdft_fragment=(1, 1),
        cdft_fragment_states=((0, 1), (1, 2)),
        run_unconstrained=False,
    )
    assert len(maker.make(nh3, planar_nh3_cation).jobs) == 6


def test_inner_reorganization_document_diabatic(nh3, planar_nh3_cation):
    energies = {"1@1": -56.49, "1@2": -56.40, "2@2": -56.10, "2@1": -56.00}
    unconstrained = {"1@1": -56.50, "1@2": -56.41, "2@2": -56.11, "2@1": -56.09}
    doc = InnerReorganizationDocument.from_energies(
        nh3,
        planar_nh3_cation,
        energies,
        unconstrained_energies=unconstrained,
        max_spin_change={"1": 0.0, "2": 0.85},
        diabatic=True,
    )
    assert doc.diabatic
    # reorganization energies from the constrained (diabatic) energies
    assert doc.lambda_1 == pytest.approx(0.09 * Ha_to_eV)
    assert doc.lambda_2 == pytest.approx(0.10 * Ha_to_eV)
    # vertical and adiabatic energies from the unconstrained ones
    assert doc.vertical_energy_1_to_2 == pytest.approx(0.41 * Ha_to_eV)
    assert doc.adiabatic_energy == pytest.approx(0.39 * Ha_to_eV)
    assert doc.diabatic_gaps["2@1"] == pytest.approx(0.09 * Ha_to_eV)
    assert doc.electronic_state_change is True


def test_max_spin_change():
    from types import SimpleNamespace

    from atomate2.qchem.jobs.reorganization import _max_spin_change

    def task(mulliken):
        return SimpleNamespace(output=SimpleNamespace(mulliken=mulliken))

    closed = task([np.array([0.1, -0.1])])
    points = {
        "1@1": closed,
        "1@2": closed,
        "2@2": task([np.array([[0.0, 1.0], [0.0, 0.0]])]),
        "2@1": task([np.array([[0.0, 0.4], [0.0, 0.6]])]),
    }
    assert _max_spin_change(points) == pytest.approx({"1": 0.0, "2": 0.6})
    assert _max_spin_change({**points, "1@1": task(None)}) is None


def test_inner_reorganization_maker_diabatic_run(mock_qchem, clean_dir, qchem_test_dir):
    if not _emmet_parses_fragments():
        pytest.skip("this emmet version cannot parse Q-Chem inputs with fragments")
    from pathlib import Path

    from jobflow import JobStore, run_locally
    from maggma.stores import MemoryStore
    from pymatgen.io.qchem.inputs import QCInput

    # Li+-EC (state 1) and its reduction to Li+ chelated by the EC radical anion
    # (state 2): wB97M-V/def2-SVPD, SMD (eps 18.5), optimized geometries. The
    # unconstrained reduced molecule at the Li+-EC geometry puts the electron on Li
    # instead of EC; the CDFT points keep it on EC (atoms 1-10).
    ref_dir = Path(qchem_test_dir) / "inner_reorganization_liec_cdft"
    names = [
        f"{prefix}single point state {s} at geometry {g}"
        for prefix in ("", "cdft ")
        for s, g in (("1", "1"), ("1", "2"), ("2", "2"), ("2", "1"))
    ]
    mock_qchem({name: ref_dir / name.replace(" ", "_") for name in names})

    def ref_molecule(name, charge, spin_multiplicity):
        molecule = QCInput.from_file(
            ref_dir / name.replace(" ", "_") / "inputs" / "mol.qin.gz"
        ).molecule
        molecule.set_charge_and_spin(charge, spin_multiplicity)
        return molecule

    molecule_1 = ref_molecule(names[0], 1, 1)
    molecule_2 = ref_molecule(names[2], 0, 2)
    smd = {
        "smx": {
            "solvent": "other",
            "epsilon": "18.5",
            "soln": "1.415",
            "sola": "0.00",
            "solb": "0.735",
            "solg": "20.2",
            "solc": "0.00",
            "solh": "0.00",
        }
    }
    maker = InnerReorganizationMaker(
        opt_maker=None,
        sp_maker=SinglePointMaker(
            input_set_generator=SinglePointSetGenerator(
                basis_set="def2-svpd",
                smd_solvent="other",
                custom_smd="18.5,1.415,0.00,0.735,20.2,0.00,0.00",
                overwrite_inputs=smd,
            )
        ),
        cdft_fragment=(1, 10),
        cdft_fragment_states=((0, 1), (-1, 2)),
    )
    flow = maker.make(molecule_1, molecule_2)
    store = JobStore(MemoryStore(), additional_stores={"data": MemoryStore()})
    run_locally(flow, store=store, create_folders=True, ensure_success=True)
    doc = flow.output.resolve(store)

    assert doc.diabatic
    assert doc.lambda_1 == pytest.approx(2.1647, abs=1e-4)
    assert doc.lambda_2 == pytest.approx(1.2000, abs=1e-4)
    assert doc.adiabatic_energy == pytest.approx(-1.9977, abs=1e-4)
    # the diabatic states are close to the ground states at their own minima...
    assert doc.diabatic_gaps["1@1"] == pytest.approx(0.0459, abs=1e-4)
    assert doc.diabatic_gaps["2@2"] == pytest.approx(0.0426, abs=1e-4)
    # ...but not for the reduced molecule at the Li+-EC geometry, where the
    # unconstrained electron sits on Li
    assert doc.diabatic_gaps["2@1"] == pytest.approx(0.9145, abs=1e-4)
    assert doc.max_spin_change["2"] > 0.8
    assert doc.electronic_state_change is True
    assert doc.unconstrained_energies["2@1"] == pytest.approx(-349.56285443, abs=1e-8)
