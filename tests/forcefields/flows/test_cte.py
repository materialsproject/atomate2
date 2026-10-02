"""Tests for the force field thermal expansion workflow.

The tests run the force field CTEMaker with ASE's EMT potential, so they need no
DFT reference data and no machine-learned force field. pheasy, ALM and phono3py
are only installed in the numpy-limited forcefield CI job, so the tests are
skipped in the other forcefield jobs.
"""
# ruff: noqa: E402

from pathlib import Path

import pytest

pytest.importorskip("pheasy")
gruneisen_module = pytest.importorskip("phono3py.phonon3.gruneisen")

import numpy as np
import phonopy
from ase.build import bulk
from jobflow import run_locally
from pymatgen.io.ase import AseAtomsAdaptor

from atomate2.common.jobs.cte import compute_cte
from atomate2.common.schemas.cte import CTEDocument
from atomate2.forcefields.flows.cte import CTEMaker

EMT = {"@module": "ase.calculators.emt", "@callable": "EMT"}


def test_cte_maker_emt(clean_dir, monkeypatch):
    """Run the whole force field workflow with EMT forces on fcc Cu."""
    structure = AseAtomsAdaptor.get_structure(bulk("Cu", "fcc", a=3.61, cubic=True))
    # the conventional cell keeps the 4-atom cubic unit cell used below
    maker = CTEMaker.from_force_field_name(
        EMT,
        use_symmetrized_structure="conventional",
        temperatures=[0, 100, 300],
        mesh=(8, 8, 8),
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
    assert np.array(doc.supercell_matrix) == pytest.approx(2 * np.eye(3))
    assert [result.fit_method for result in doc.results] == ["cocktail", "one-shot"]

    # EMT elastic constants of Cu in GPa
    assert doc.elastic_tensor[0][0] == pytest.approx(172.6, rel=0.01)
    assert doc.elastic_tensor[0][1] == pytest.approx(115.4, rel=0.01)
    assert doc.elastic_tensor[3][3] == pytest.approx(89.9, rel=0.01)

    for result in doc.results:
        assert not result.has_imaginary_modes
        alpha = np.array(result.thermal_expansion_tensor)
        assert np.all(alpha[0] == 0.0)
        # cubic, so alpha is isotropic in every frame
        assert alpha[2] == pytest.approx(alpha[2, 0, 0] * np.eye(3), abs=1e-12)
        assert result.thermal_expansion[2] == pytest.approx(3 * alpha[2, 0, 0])
    # linear thermal expansion at 300 K in 1/K, and the mean Grueneisen parameter.
    # The one-shot fit has few supercells in this small cell, so its value is only
    # compared with the cocktail value.
    cocktail, one_shot = doc.results
    assert cocktail.thermal_expansion_tensor[2][0][0] == pytest.approx(
        1.859e-5, rel=0.02
    )
    assert np.trace(cocktail.average_gruneisen[2]) / 3 == pytest.approx(2.237, rel=0.02)
    assert one_shot.thermal_expansion_tensor[2][0][0] == pytest.approx(
        cocktail.thermal_expansion_tensor[2][0][0], rel=0.2
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
    assert flagged.thermal_expansion_tensor is None

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
    assert np.array(nac_doc.results[0].thermal_expansion_tensor[0]) == pytest.approx(
        np.array(cocktail.thermal_expansion_tensor[2]), rel=1e-6, abs=1e-15
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
    assert maker.use_symmetrized_structure == "primitive"
    assert maker.phonon_maker.min_length == 12.0
    assert maker.phonon_maker.cal_anhar_fcs
    assert maker.phonon_maker.displacement_anhar == 0.03
