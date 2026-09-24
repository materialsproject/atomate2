"""Tests for the thermal expansion workflow.

The end-to-end test runs the force field CTEMaker with ASE's EMT potential, so
it needs no DFT reference data. It lives here and not under tests/forcefields,
because pheasy and ALM are only installed in the test-non-ase CI job.
"""

from pathlib import Path

import numpy as np
import phono3py.phonon3.gruneisen as gruneisen_module
import phonopy
import pytest
from ase.build import bulk
from jobflow import run_locally
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from pymatgen.analysis.elasticity import ElasticTensor
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.constants import Boltzmann, Planck
from scipy.spatial.transform import Rotation

from atomate2.common.jobs.cte import compute_cte
from atomate2.common.schemas.cte import CTEDocument, _expand_born_to_unitcell, get_cte
from atomate2.forcefields.flows.cte import CTEMaker

EMT = {"@module": "ase.calculators.emt", "@callable": "EMT"}


def _heat_capacity(frequency: float, temperature: float) -> float:
    x = Planck * frequency * 1e12 / (Boltzmann * temperature)
    return Boltzmann * x**2 * np.exp(x) / (np.exp(x) - 1) ** 2


def test_get_cte_cubic():
    """For a cubic crystal with gamma = g * I, alpha = g * Cv / (3 * B * V)."""
    c11, c12, c44 = 170.0, 120.0, 75.0
    elastic = np.zeros((6, 6))
    elastic[:3, :3] = c12
    np.fill_diagonal(elastic[:3, :3], c11)
    elastic[3:, 3:] = np.eye(3) * c44
    bulk_modulus = (c11 + 2 * c12) / 3 * 1e9  # Pa

    frequencies = np.array([[2.0, 5.0, 7.0], [3.0, 4.0, 6.0]])
    weights = np.array([1, 3])
    gruneisen = 1.7 * np.broadcast_to(np.eye(3), (2, 3, 3, 3))
    volume, temperature = 45.0, 300.0

    alpha, mean_gruneisen = get_cte(
        frequencies, gruneisen, weights, elastic, volume, [temperature]
    )

    heat_capacity = (
        sum(
            w * _heat_capacity(f, temperature)
            for w, row in zip(weights, frequencies, strict=True)
            for f in row
        )
        / weights.sum()
    )
    expected = 1.7 * heat_capacity / (3 * bulk_modulus * volume * 1e-30)
    assert alpha[0] == pytest.approx(expected * np.eye(3), rel=1e-10, abs=1e-20)
    assert mean_gruneisen[0] == pytest.approx(1.7 * np.eye(3))


def test_get_cte_rotation():
    """Rotating the inputs must rotate alpha, which checks the shear terms.

    The mode Grueneisen tensors from phono3py are not symmetric, so the input
    tensors here are not symmetric either.
    """
    # hexagonal elastic tensor, with C66 = (C11 - C12) / 2
    c11, c12, c13, c33, c44 = 350.0, 120.0, 90.0, 400.0, 110.0
    elastic = np.array(
        [
            [c11, c12, c13, 0, 0, 0],
            [c12, c11, c13, 0, 0, 0],
            [c13, c13, c33, 0, 0, 0],
            [0, 0, 0, c44, 0, 0],
            [0, 0, 0, 0, c44, 0],
            [0, 0, 0, 0, 0, (c11 - c12) / 2],
        ]
    )
    rng = np.random.default_rng(7)
    frequencies = rng.uniform(1.0, 10.0, size=(4, 6))
    gruneisen = rng.normal(1.0, 0.5, size=(4, 6, 3, 3))
    weights = np.ones(4)
    temperatures = [100.0, 500.0]

    alpha, _ = get_cte(frequencies, gruneisen, weights, elastic, 30.0, temperatures)
    rotation = Rotation.from_euler("zxz", [30, 50, 70], degrees=True).as_matrix()
    rotated_elastic = ElasticTensor.from_voigt(elastic).rotate(rotation).voigt
    rotated_gruneisen = np.einsum("ik,qbkl,jl->qbij", rotation, gruneisen, rotation)
    rotated_alpha, _ = get_cte(
        frequencies, rotated_gruneisen, weights, rotated_elastic, 30.0, temperatures
    )

    expected = np.einsum("ik,tkl,jl->tij", rotation, alpha, rotation)
    assert np.abs(expected[:, 0, 1]).max() > 1e-7  # the shear terms are tested
    assert rotated_alpha == pytest.approx(expected, rel=1e-8, abs=1e-15)


def test_get_cte_left_out_modes():
    """Modes below min_frequency are left out, and T = 0 gives zero."""
    elastic = np.diag([200.0, 200.0, 200.0, 80.0, 80.0, 80.0])
    frequencies = np.array([[0.0, 0.0, 0.0, 4.0], [-0.5, 3.0, 5.0, 6.0]])
    gruneisen = np.ones((2, 4, 3, 3))
    gruneisen[0, :3] = np.nan  # NaN checks that the left-out modes are ignored

    alpha, mean_gruneisen = get_cte(
        frequencies, gruneisen, np.ones(2), elastic, 40.0, [0.0, 300.0]
    )
    assert np.all(alpha[0] == 0.0)
    assert mean_gruneisen[0] is None
    assert mean_gruneisen[1] == pytest.approx(np.ones((3, 3)))

    # only the modes at 4, 3, 5 and 6 THz count, over two q-points
    heat_capacity = sum(_heat_capacity(f, 300.0) for f in (4.0, 3.0, 5.0, 6.0)) / 2
    thermal_stress = heat_capacity / (40.0 * 1e-30)
    expected = np.full((3, 3), thermal_stress / 80e9 / 2)
    np.fill_diagonal(expected, thermal_stress / 200e9)
    assert alpha[1] == pytest.approx(expected, rel=1e-10)


def test_expand_born_to_unitcell():
    """Born charges of the primitive cell are mapped onto the conventional cell."""
    atoms = bulk("MgO", "rocksalt", a=4.2, cubic=True)
    unitcell = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.cell[:],
        scaled_positions=atoms.get_scaled_positions(),
    )
    phonon = Phonopy(unitcell, supercell_matrix=np.eye(3), primitive_matrix="auto")
    assert len(phonon.primitive) == 2
    born_primitive = {"Mg": 1.9, "O": -1.9}
    phonon.nac_params = {
        "born": [np.eye(3) * born_primitive[s] for s in phonon.primitive.symbols],
        "dielectric": np.eye(3) * 3.0,
    }

    born = _expand_born_to_unitcell(phonon)
    assert born.shape == (8, 3, 3)
    for symbol, charges in zip(unitcell.symbols, born, strict=True):
        assert charges == pytest.approx(np.eye(3) * born_primitive[symbol])


def test_cte_maker_emt(clean_dir, monkeypatch):
    """Run the whole force field workflow with EMT forces on fcc Cu."""
    structure = AseAtomsAdaptor.get_structure(bulk("Cu", "fcc", a=3.61, cubic=True))
    maker = CTEMaker.from_force_field_name(
        EMT, temperatures=[0, 100, 300], mesh=(8, 8, 8)
    )
    # a 2x2x2 supercell of the cubic cell, 32 atoms, and an fc3 cutoff of 6 Bohr
    # (3.2 A), which covers the nearest neighbours at 2.55 A
    maker.phonon_maker.min_length = 7.0
    maker.phonon_maker.fcs_cutoff_radius = [-1, 6, 6]
    maker.phonon_maker.anhar_fit_methods = ["cocktail", "one-shot"]

    flow = maker.make(structure)
    # the uuid of the phonon flow output that compute_cte reads
    phonon_uuid = flow.jobs[-1].function_kwargs["phonon_output"].uuid
    responses = run_locally(flow, create_folders=True, ensure_success=True)
    doc = responses[flow.output.uuid][1].output
    assert isinstance(doc, CTEDocument)
    assert doc.mesh == (8, 8, 8)
    assert [result.fit_method for result in doc.results] == ["cocktail", "one-shot"]

    # EMT elastic constants of Cu in GPa
    assert doc.elastic_tensor[0][0] == pytest.approx(172.6, rel=0.01)
    assert doc.elastic_tensor[0][1] == pytest.approx(115.4, rel=0.01)
    assert doc.elastic_tensor[3][3] == pytest.approx(89.9, rel=0.01)

    for result in doc.results:
        assert not result.has_imaginary_modes
        alpha = np.array(result.thermal_expansion)
        assert np.all(alpha[0] == 0.0)
        # cubic, so alpha is isotropic in every frame
        assert alpha[2] == pytest.approx(alpha[2, 0, 0] * np.eye(3), abs=1e-12)
        assert result.volumetric_thermal_expansion[2] == pytest.approx(
            3 * alpha[2, 0, 0]
        )
    # linear thermal expansion at 300 K in 1/K, and the mean Grueneisen parameter.
    # The one-shot fit has few supercells in this small cell, so its value is only
    # compared with the cocktail value.
    cocktail, one_shot = doc.results
    assert cocktail.thermal_expansion[2][0][0] == pytest.approx(1.859e-5, rel=0.02)
    assert np.trace(cocktail.average_gruneisen[2]) / 3 == pytest.approx(2.237, rel=0.02)
    assert one_shot.thermal_expansion[2][0][0] == pytest.approx(
        cocktail.thermal_expansion[2][0][0], rel=0.2
    )
    assert Path(doc.phonon_job_dir, "one_shot", "fc3.hdf5").exists()

    # the same force constants, with a q-point density instead of a mesh. The
    # negative tolerance flags every frequency, to test the imaginary mode check.
    phonon_output = responses[phonon_uuid][1].output
    job = compute_cte(
        phonon_output=phonon_output,
        elastic_tensor=doc.elastic_tensor,
        elastic_structure=doc.structure,
        anhar_fit_methods=["one-shot"],
        temperatures=[300],
        mesh=100.0,
        tol_imaginary_modes=-10.0,
    )
    with pytest.warns(UserWarning, match="thermal expansion is not"):
        responses = run_locally(job, create_folders=True, ensure_success=True)
    flagged_doc = responses[job.uuid][1].output
    assert flagged_doc.mesh == (2, 2, 2)
    (flagged,) = flagged_doc.results
    assert flagged.has_imaginary_modes
    assert flagged.thermal_expansion is None

    # the non-analytical term correction with zero Born charges leaves alpha
    # unchanged. pheasy only stores Born charges for VASP, so they are added here.
    # The phonopy primitive cell has one atom and the unit cell four, so the
    # charges must be expanded to four atoms before they reach phono3py.
    phonon_job_dir = Path(doc.phonon_job_dir)
    phonon = phonopy.load(phonon_job_dir / "phonopy.yaml", produce_fc=False)
    assert len(phonon.primitive) == 1
    phonon.nac_params = {
        "born": np.zeros((1, 3, 3)),
        "dielectric": np.eye(3) * 10.0,
        "factor": 14.399652,
    }
    phonon.save("phonopy_nac.yaml")

    nac_params_used = []
    original_gruneisen = gruneisen_module.Gruneisen

    def _gruneisen(*args, **kwargs):
        nac_params_used.append(kwargs["nac_params"])
        return original_gruneisen(*args, **kwargs)

    monkeypatch.setattr(gruneisen_module, "Gruneisen", _gruneisen)
    cocktail_files = {
        "cocktail": (phonon_job_dir / "FORCE_CONSTANTS", phonon_job_dir / "fc3.hdf5")
    }
    settings = {
        "force_constant_files": cocktail_files,
        "elastic_tensor": doc.elastic_tensor,
        "temperatures": [300],
        "mesh": (8, 8, 8),
        "tol_imaginary_modes": 0.1,
        "min_frequency": 1e-3,
        "symprec": 1e-5,
    }
    nac_doc = CTEDocument.from_force_constants(
        phonopy_yaml="phonopy_nac.yaml", structure=doc.structure, **settings
    )
    (nac_params,) = nac_params_used
    assert nac_params["born"].shape == (4, 3, 3)
    assert nac_params["dielectric"] == pytest.approx(np.eye(3) * 10.0)
    assert np.array(nac_doc.results[0].thermal_expansion[0]) == pytest.approx(
        np.array(cocktail.thermal_expansion[2]), rel=1e-6, abs=1e-15
    )

    # a structure with another lattice or other atoms is refused
    strained = doc.structure.copy()
    strained.apply_strain(0.01)
    with pytest.raises(ValueError, match="not in the same frame"):
        CTEDocument.from_force_constants(
            phonopy_yaml=phonon_job_dir / "phonopy.yaml", structure=strained, **settings
        )
    other_atoms = doc.structure.copy()
    other_atoms.replace_species({"Cu": "Au"})
    with pytest.raises(ValueError, match="atoms of the elastic calculation"):
        CTEDocument.from_force_constants(
            phonopy_yaml=phonon_job_dir / "phonopy.yaml",
            structure=other_atoms,
            **settings,
        )
    with pytest.raises(ValueError, match="must not be negative"):
        CTEDocument.from_force_constants(
            phonopy_yaml=phonon_job_dir / "phonopy.yaml",
            structure=doc.structure,
            **{**settings, "temperatures": [-10, 300]},
        )


def test_cte_maker_force_field_defaults():
    """The default phonon maker uses min_length=12.0 for the supercells."""
    maker = CTEMaker()
    assert maker.phonon_maker.min_length == 12.0
    assert maker.phonon_maker.cal_anhar_fcs
    assert maker.phonon_maker.displacement_anhar == 0.03
