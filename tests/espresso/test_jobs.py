import numpy as np
import pytest
from jobflow import run_locally
from pymatgen.core import Lattice, Structure

from atomate2.espresso.jobs import (
    EspressoDielectricMaker,
    EspressoRelaxMaker,
    EspressoStaticMaker,
    rotate_to_standard_orientation,
)
from atomate2.espresso.schemas import EspressoTaskDoc


def run(maker, structure, kspacing=0.5):
    # the reference runs used the test pseudopotentials and coarse k-point grids
    maker.input_set_generator.pseudo_family = "test_family"
    maker.input_set_generator.kspacing = kspacing
    job = maker.make(structure)
    responses = run_locally(job, create_folders=True, ensure_success=True)
    return responses[job.uuid][1].output


def test_static_maker(mock_espresso, clean_dir, si_primitive):
    mock_espresso("static")
    displaced = si_primitive.copy()
    displaced.translate_sites([0], [0.01, 0, 0], frac_coords=False)
    doc = run(EspressoStaticMaker(), displaced)

    assert isinstance(doc, EspressoTaskDoc)
    assert doc.task_label == "espresso static"
    assert doc.input.parameters["control"]["calculation"] == "scf"
    assert doc.output.energy == pytest.approx(-213.98870255)
    assert doc.output.energy_per_atom == pytest.approx(doc.output.energy / 2)
    assert doc.output.forces[0][0] == pytest.approx(-0.13357498)
    assert sum(force[0] for force in doc.output.forces) == pytest.approx(0, abs=1e-6)
    # pw.out prints a total stress of 4.55 kbar (xx) and 1.93 kbar (xy)
    assert doc.output.stress[0][0] == pytest.approx(4.5514, abs=1e-4)
    assert doc.output.stress[0][1] == pytest.approx(1.9300, abs=1e-4)
    assert doc.output.epsilon_static is None
    assert doc.structure == displaced


def test_relax_maker(mock_espresso, clean_dir, si_primitive):
    mock_espresso("relax")
    strained = si_primitive.copy()
    strained.scale_lattice(si_primitive.volume * 1.03)
    doc = run(EspressoRelaxMaker(), strained)

    assert doc.input.parameters["control"]["calculation"] == "vc-relax"
    assert doc.input.parameters["control"]["forc_conv_thr"] == 1e-4
    assert doc.input.structure.volume == pytest.approx(strained.volume)
    # the final self-consistent calculation on the relaxed cell
    assert doc.structure.volume == pytest.approx(41.2279, abs=1e-4)
    assert doc.output.energy == pytest.approx(-213.99066721)
    assert doc.output.stress[0][0] == pytest.approx(-0.1265, abs=1e-4)


def test_relax_maker_damped_dynamics(mock_espresso, clean_dir, si_primitive):
    mock_espresso("relax_damped")
    structure = si_primitive.copy()
    structure.scale_lattice(si_primitive.volume * 1.03)
    structure.translate_sites([0], [0.05, 0, 0], frac_coords=False)
    maker = EspressoRelaxMaker()
    maker.input_set_generator.calculation = "relax"
    maker.input_set_generator.user_input_data = {"ions": {"ion_dynamics": "damp"}}
    doc = run(maker, structure)

    # pw.out prints "Damped Dynamics: convergence achieved" instead of
    # "bfgs converged", and "relax" keeps the cell
    assert doc.input.parameters["control"]["calculation"] == "relax"
    assert doc.structure.volume == pytest.approx(structure.volume)
    assert doc.output.forces[0][0] == pytest.approx(-0.000723, abs=1e-6)


@pytest.mark.parametrize(
    ("name", "maker", "user_input_data", "message"),
    [
        (
            "static_unconverged",
            EspressoStaticMaker(),
            {"electrons": {"electron_maxstep": 2}},
            "did not converge",
        ),
        (
            "relax_unconverged",
            EspressoRelaxMaker(),
            {"control": {"nstep": 1}},
            "did not relax",
        ),
    ],
)
def test_maker_not_converged(
    mock_espresso, clean_dir, si_primitive, name, maker, user_input_data, message
):
    # pw.x stops with exit code 0 in both cases, so the job has to catch it
    mock_espresso(name)
    structure = si_primitive.copy()
    if name == "static_unconverged":
        structure.translate_sites([0], [0.01, 0, 0], frac_coords=False)
    else:
        structure.scale_lattice(si_primitive.volume * 1.03)
        structure.translate_sites([0], [0.05, 0, 0], frac_coords=False)
    maker.input_set_generator.pseudo_family = "test_family"
    maker.input_set_generator.kspacing = 0.5
    maker.input_set_generator.user_input_data = user_input_data

    with pytest.raises(RuntimeError, match=message):
        maker.make.original(maker, structure)


def test_dielectric_maker(mock_espresso, clean_dir):
    mock_espresso("dielectric")
    sic = Structure(
        Lattice.from_parameters(3.095, 3.095, 3.095, 60, 60, 60),
        ["Si", "C"],
        [[0, 0, 0], [0.25, 0.25, 0.25]],
    )
    doc = run(EspressoDielectricMaker(), sic, kspacing=0.3)

    assert doc.input.parameters["system"]["occupations"] == "fixed"
    assert doc.input.parameters["system"]["ecutwfc"] == 30
    assert doc.input.ph_parameters["inputph"]["epsil"] is True
    assert doc.output.epsilon_static[0][0] == pytest.approx(6.983653)
    # ph.x applies the acoustic sum rule, so the charges of Si and C cancel
    assert doc.output.born[0][0][0] == pytest.approx(2.69267)
    assert doc.output.born[1][0][0] == pytest.approx(-2.69267)


def test_rotate_to_standard_orientation(si_primitive):
    si_primitive.add_site_property("magmom", [1, -1])
    rotated = rotate_to_standard_orientation.original(si_primitive, symprec=1e-3)

    # the standard primitive fcc cell has the cubic axes along x, y, and z
    half = 3.867 / 2**0.5
    expected = np.array([[half, 0, half], [half, half, 0], [0, half, half]])
    assert rotated.lattice.matrix == pytest.approx(expected, abs=1e-6)
    assert rotated.frac_coords == pytest.approx(si_primitive.frac_coords)
    assert rotated.site_properties["magmom"] == [1, -1]
