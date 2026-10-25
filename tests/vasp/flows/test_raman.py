from dataclasses import dataclass

import numpy as np
import pytest
from jobflow import Maker, job, run_locally
from phonopy import Phonopy
from pymatgen.core import Lattice, Structure
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure

from atomate2.utils import raman as rm
from atomate2.vasp.flows.raman import RamanMaker
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.schemas.raman import RamanDoc


def rutile():
    return Structure.from_spacegroup(
        "P4_2/mnm",
        Lattice.tetragonal(4.59, 2.96),
        ["Ti", "O"],
        [[0, 0, 0], [0.3048, 0.3048, 0]],
    )


def diamond():
    return Structure.from_spacegroup(
        "Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]]
    ).get_primitive_structure()


def hcp():
    return Structure.from_spacegroup(
        "P6_3/mmc", Lattice.hexagonal(2.56, 4.18), ["Cu"], [[1 / 3, 2 / 3, 0.25]]
    )


def toy_force_constants(structure):
    from ase.calculators.lj import LennardJones
    from pymatgen.io.ase import AseAtomsAdaptor

    dmin = structure.distance_matrix[np.triu_indices(len(structure), 1)].min()
    ph = Phonopy(get_phonopy_structure(structure), supercell_matrix=np.diag([2, 2, 2]))
    ph.generate_displacements(distance=0.01)
    forces = []
    for sc in ph.supercells_with_displacements:
        atoms = AseAtomsAdaptor.get_atoms(get_pmg_structure(sc))
        atoms.calc = LennardJones(sigma=0.9 * dmin / 2 ** (1 / 6), rc=6.0, smooth=True)
        forces.append(atoms.get_forces())
    ph.produce_force_constants(forces=forces)
    return ph.force_constants, np.diag([2, 2, 2]).tolist(), ph.primitive_matrix.tolist()


def toy_epsilon(structure, cutoff=3.0):
    eps = np.eye(3) * 5.0
    for site in structure:
        for nb in structure.get_neighbors(site, cutoff):
            d = nb.coords - site.coords
            r = np.linalg.norm(d)
            eps += np.exp(-(r - 2.0)) * (0.3 * np.outer(d, d) / r**2 + 0.1 * np.eye(3))
    return eps / len(structure)


@pytest.mark.parametrize(("make", "n_active"), [(rutile, 5), (diamond, 3), (hcp, 2)])
def test_selection_rules(make, n_active):
    s = make()
    fc, scm, pm = toy_force_constants(s)
    ph = rm.build_phonopy(s, {"force_constants": fc.tolist()}, scm, pm)
    nm = rm.gamma_normal_modes(ph)
    nmodes = len(nm["frequencies_thz"])
    assert np.allclose(
        np.sort(nm["frequencies_thz"]),
        np.sort(ph.get_frequencies([0, 0, 0])),
        atol=1e-6,
    )
    rules = rm.gamma_selection_rules(ph, nmodes)
    acoustic = rm.acoustic_mode_indices(nm["frequencies_thz"])
    active = [
        k for k in range(nmodes) if rules["raman_active"][k] and k not in acoustic
    ]
    assert len(active) == n_active

    prim = nm["structure"]
    for k in set(range(nmodes)) - set(acoustic) - set(active):
        eps = [
            toy_epsilon(
                rm.displace_along_mode(prim, nm["eigenvecs"][k], nm["masses"], d)
            )
            for d in (0.005, -0.005)
        ]
        norm = rm.mode_displacement_pattern(nm["eigenvecs"][k], nm["masses"])[1]
        tensor = rm.raman_tensor_from_epsilons([0.005, -0.005], eps, prim.volume, norm)
        assert np.abs(tensor).max() < 1e-6


def test_intensity_definitions():
    a = np.random.default_rng(0).normal(size=(3, 3))
    a = a + a.T
    assert np.isclose(rm.crd_intensity(a), rm.raman_activity(a) / 3)


def test_encut_from_potcar():
    class P:
        def __init__(self, enmax):
            self.enmax = enmax

    class Gen:
        def get_input_set(self, structure):
            return type("VIS", (), {"potcar": [P(400.0), P(258.689)]})()

    assert rm.get_encut_from_potcar(rutile(), Gen(), 1.3) == 520.0


@job
def fake_phonons(structure):
    fc, scm, pm = toy_force_constants(structure)
    return {
        "structure": structure,
        "force_constants": {"force_constants": fc.tolist()},
        "supercell_matrix": scm,
        "primitive_matrix": pm,
    }


@dataclass
class FakePhononMaker(Maker):
    name: str = "fake phonon"
    symprec: float = 1e-4
    store_force_constants: bool = True

    def make(self, structure, **kwargs):
        return fake_phonons(structure)


@job
def fake_eps(structure):
    eps = toy_epsilon(structure).tolist()
    return {
        "calcs_reversed": [{"output": {"epsilon_static": eps}}],
        "uuid": "x",
        "dir_name": "fake_dir",
    }


@dataclass
class FakeEpsMaker(Maker):
    name: str = "fake eps"
    write_additional_data: dict = None

    def make(self, structure, prev_dir=None):
        return fake_eps(structure)


def test_raman_flow_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    maker = RamanMaker(
        phonon_maker=FakePhononMaker(),
        raman_maker=FakeEpsMaker(),
        encut_enmax_factor=None,
    )
    flow = maker.make(rutile(), mp_id="mp-2657")
    out = run_locally(flow, create_folders=True, ensure_success=True)
    doc = out[flow.output.uuid][1].output
    doc = RamanDoc(**doc) if isinstance(doc, dict) else doc
    assert doc.computed_modes == sorted(doc.computed_modes)
    assert {m.irrep for m in doc.modes} == {"A1g", "B1g", "B2g", "Eg"}
    assert len(doc.computed_modes) == 5
    assert all(len(m.epsilons) == 2 for m in doc.modes)


def test_raman_flow_encut(monkeypatch):
    monkeypatch.setattr(
        "atomate2.vasp.flows.raman.get_encut_from_potcar", lambda *_: 336.2957
    )
    flow = RamanMaker().make(rutile())
    makers = []
    for j, _ in flow.iterflow():
        if j.maker is not None and isinstance(j.maker, BaseVaspMaker):
            makers.append(j.maker)
        makers += [
            a for a in j.function_kwargs.values() if isinstance(a, BaseVaspMaker)
        ]
    assert makers
    assert all(
        m.input_set_generator.user_incar_settings.get("ENCUT") == 336.2957
        for m in makers
    )
    names = {m.name for m in makers}
    assert {"tight relax 1", "static", "dielectric", "raman dielectric"} <= names


@pytest.mark.parametrize(
    ("gap", "rk"),
    [(None, 40), (0.0, 40), (0.99, 40), (1.0, 30), (1.5, 30), (2.0, 30), (2.01, 20)],
)
def test_kpoints_length_from_bandgap(gap, rk):
    assert rm.kpoints_length_from_bandgap(gap) == rk


def test_bandgap_reference():
    from atomate2.vasp.flows.phonons import PhononMaker

    flow = RamanMaker().make(rutile(), encut=520)
    jobs = {j.uuid: j.name for j, _ in flow.iterflow()}
    disp = next(j for j, _ in flow.iterflow() if j.name == "raman displacements")
    assert jobs[disp.function_kwargs["bandgap"].uuid] == "static"

    maker = RamanMaker(
        phonon_maker=PhononMaker(static_energy_maker=None, born_maker=None)
    )
    flow = maker.make(rutile(), encut=520)
    jobs = {j.uuid: j.name for j, _ in flow.iterflow()}
    disp = next(j for j, _ in flow.iterflow() if j.name == "raman displacements")
    assert jobs[disp.function_kwargs["bandgap"].uuid] == "tight relax 2"


@pytest.mark.parametrize(("gap", "rk"), [(0.5, 40), (1.5, 30), (3.0, 20)])
def test_rk_applied_to_dielectric_jobs(gap, rk):
    from atomate2.vasp.jobs.raman import run_raman_displacements

    modes_data = {
        "step_size": 0.005,
        "displaced": [
            {"mode": 3, "displacement": d, "structure": rutile()}
            for d in (-0.005, 0.005)
        ],
    }
    response = run_raman_displacements.original(modes_data, bandgap=gap)
    jobs = response.replace.jobs
    assert all(j.maker.input_set_generator.kpoints_length == rk for j in jobs)
    assert response.replace.output["kpoints_length"] == rk
    vis = jobs[0].maker.input_set_generator.get_input_set(rutile(), potcar_spec=True)
    assert vis.kpoints.kpts[0][0] == rk

    from pymatgen.io.vasp import Kpoints

    assert Kpoints.from_str(str(vis.kpoints)).kpts[0][0] == rk
