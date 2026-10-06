"""Tests for the force field pheasy workflow.

pheasy and ALM are only installed in the numpy-limited forcefield CI job, so the
tests are skipped in the other forcefield jobs.
"""

import pytest

pytest.importorskip("pheasy")

import numpy as np
from ase.build import bulk
from emmet.core.phonon import (
    PhononBS,
    PhononBSDOSDoc,
    PhononDOS,
    ThermalDisplacementData,
)
from jobflow import run_locally
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor

import atomate2.common.jobs.pheasy as pheasy_jobs
from atomate2.forcefields.flows.pheasy import PhononMaker
from atomate2.forcefields.jobs import (
    ForceFieldDielectricMaker,
    ForceFieldRelaxMaker,
    ForceFieldStaticMaker,
)

EMT = {"@module": "ase.calculators.emt", "@callable": "EMT"}


def _cu_structure() -> Structure:
    return AseAtomsAdaptor.get_structure(bulk("Cu", "fcc", a=3.61, cubic=True))


def test_get_supercell_size_kwargs(monkeypatch):
    received = {}
    transformation = pheasy_jobs.CubicSupercellTransformation

    def record_kwargs(**kwargs):
        received.update(kwargs)
        return transformation(**kwargs)

    monkeypatch.setattr(pheasy_jobs, "CubicSupercellTransformation", record_kwargs)

    # the maker passes get_supercell_size_kwargs on to the job
    maker = PhononMaker(get_supercell_size_kwargs={"angle_tolerance": 0.1})
    job = maker.get_supercell_matrix(_cu_structure())
    assert job.function_kwargs == {"angle_tolerance": 0.1}

    # the job passes them to CubicSupercellTransformation. The other default
    # is kept.
    job.function(*job.function_args, **job.function_kwargs)
    assert received["angle_tolerance"] == 0.1
    assert received["allow_orthorhombic"] is False


def test_pheasy_wf_force_field(clean_dir):
    """Run the harmonic force field pheasy workflow with EMT forces on fcc Cu."""
    # a 2x2x2 supercell of the 4-atom cubic cell
    maker = PhononMaker(
        min_length=7.0,
        use_symmetrized_structure="conventional",
        create_thermal_displacements=True,
        bulk_relax_maker=ForceFieldRelaxMaker(
            force_field_name=EMT, relax_kwargs={"fmax": 0.00001}
        ),
        static_energy_maker=ForceFieldStaticMaker(force_field_name=EMT),
        phonon_displacement_maker=ForceFieldStaticMaker(force_field_name=EMT),
    )
    flow = maker.make(_cu_structure())
    responses = run_locally(flow, create_folders=True, ensure_success=True)
    ph_doc = responses[flow.jobs[-1].uuid][1].output

    assert isinstance(ph_doc, PhononBSDOSDoc)
    assert isinstance(ph_doc.phonon_bandstructure, PhononBS)
    assert isinstance(ph_doc.phonon_dos, PhononDOS)
    assert isinstance(ph_doc.thermal_displacement_data, ThermalDisplacementData)
    assert isinstance(ph_doc.structure, Structure)
    assert ph_doc.has_imaginary_modes is False
    assert isinstance(ph_doc.force_constants, list)


def test_pheasy_wf_force_field_born_charges(clean_dir, fake_dielectric_calculator):
    structure = Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(4.17), ["Ni", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    ).get_primitive_structure()
    maker = PhononMaker(
        bulk_relax_maker=None,
        static_energy_maker=None,
        phonon_displacement_maker=ForceFieldStaticMaker(force_field_name=EMT),
        born_maker=ForceFieldDielectricMaker(),
        min_length=8,
        create_thermal_displacements=False,
        store_force_constants=False,
    )
    flow = maker.make(structure)
    responses = run_locally(flow, create_folders=True, ensure_success=True)
    ph_doc = responses[flow.jobs[-1].uuid][1].output

    assert np.array(ph_doc.born) == pytest.approx(
        np.array([2, -2])[:, None, None] * np.eye(3)
    )
    # the LO-TO splitting raises the highest frequency
    assert np.max(ph_doc.phonon_bandstructure.frequencies) == pytest.approx(
        17.974, abs=1e-2
    )
