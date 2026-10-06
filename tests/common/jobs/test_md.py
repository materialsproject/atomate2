"""Tests of the shared MD jobs and flows."""

from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from ase.io import write as ase_write
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from atomate2.common.flows.md import ChainedMDMaker
from atomate2.common.jobs.md import get_md_restart_structure
from atomate2.forcefields.md import ForceFieldMDMaker

TEST_DIR = Path(__file__).resolve().parents[2] / "test_data"


def test_get_md_restart_structure():
    cu_supercell = AseAtomsAdaptor.get_structure(
        bulk("Cu", "fcc", a=3.61, cubic=True) * (2, 2, 2)
    )
    rng = np.random.default_rng(4)
    velocities = rng.normal(0, 0.01, size=(len(cu_supercell), 3))
    reference = cu_supercell.copy(site_properties={"magmom": [1.0] * 32})

    # a CONTCAR of a VASP MD, with the predictor-corrector block after the
    # velocities
    md_dir = TEST_DIR / "vasp/Si_multi_md/molecular_dynamics_1/outputs"
    si_reference = Structure(
        np.eye(3) * 3.87, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]]
    )
    restart = get_md_restart_structure.original(str(md_dir), si_reference)
    assert restart.site_properties["velocities"][0] == pytest.approx(
        [0.40252568e-03, -0.31480439e-02, -0.16045943e-02]
    )
    assert "magmom" not in restart.site_properties

    Path("vasp").mkdir()
    structure = cu_supercell.copy(site_properties={"velocities": velocities})
    structure.to(filename="vasp/CONTCAR", fmt="poscar")
    restart = get_md_restart_structure.original(str(Path("vasp").resolve()), reference)
    assert np.allclose(restart.site_properties["velocities"], velocities, atol=1e-8)
    assert restart.site_properties["magmom"] == [1.0] * 32

    Path("ase").mkdir()
    atoms = AseAtomsAdaptor.get_atoms(cu_supercell)
    atoms.set_velocities(velocities)
    ase_write(Path("ase") / "md.traj", [atoms.copy(), atoms])
    restart = get_md_restart_structure.original(
        str(Path("ase").resolve()), reference, "md.traj"
    )
    assert np.allclose(restart.site_properties["velocities"], velocities)
    assert np.allclose(restart.cart_coords, cu_supercell.cart_coords)
    assert restart.site_properties["magmom"] == [1.0] * 32


def test_chained_md_maker_needs_ase_trajectory():
    structure = Structure(np.eye(3) * 3.61, ["Cu"], [[0, 0, 0]])
    maker = ChainedMDMaker(md_makers=[ForceFieldMDMaker(), ForceFieldMDMaker()])
    with pytest.raises(ValueError, match="trajectory in the ASE format"):
        maker.make(structure)
