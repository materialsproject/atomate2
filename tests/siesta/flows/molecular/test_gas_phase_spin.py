"""Regression tests for open-shell molecule spin initialization (TASKS #2a).

O2 is a ground-state triplet. The DM.InitSpin auto-generator defaults to
antiferromagnetic ordering, which flips signs by atom index and turns the
molecular ferromagnetic intent (+1/+1, net S=2) into +1/-1 (net S=0, a
singlet 0.44 eV too high). GasPhaseMoleculeMaker must request "custom"
ordering so the applied moments are preserved.
"""

import pytest
from pymatgen.core import Lattice, Molecule, Structure

from atomate2.siesta.dataclass.spin_settings import SpinSettings


def _o2_box_with_magmoms():
    """O2 in a box carrying ferromagnetic magmoms (+1/+1), as the maker applies."""
    struct = Structure(
        Lattice.cubic(20.0),
        ["O", "O"],
        [[0.5, 0.5, 0.45], [0.5, 0.5, 0.55]],
    )
    struct.add_site_property("magmom", [1.0, 1.0])
    return struct


def _net_moment(block: list[str] | None) -> float:
    if not block:
        return 0.0
    return sum(float(line.split()[1]) for line in block)


@pytest.mark.parametrize("ordering", ["custom", "ferromagnetic"])
def test_dm_initspin_preserves_triplet(ordering):
    """custom/ferromagnetic ordering keeps O2 at net S=2 (triplet)."""
    inst = SpinSettings.setup_spin_settings(
        user_params={"Spin": "polarized"},
        structure=_o2_box_with_magmoms(),
        magnetic_ordering=ordering,
    )
    assert inst.dm_init_spin_block, "DM.InitSpin block must not be empty"
    assert _net_moment(inst.dm_init_spin_block) == pytest.approx(2.0)


def test_dm_initspin_afm_breaks_triplet():
    """Document the bug: antiferromagnetic ordering collapses O2 to net S=0."""
    inst = SpinSettings.setup_spin_settings(
        user_params={"Spin": "polarized"},
        structure=_o2_box_with_magmoms(),
        magnetic_ordering="antiferromagnetic",
    )
    assert _net_moment(inst.dm_init_spin_block) == pytest.approx(0.0)


def test_gas_phase_maker_requests_custom_ordering():
    """GasPhaseMoleculeMaker injects custom ordering for an open-shell molecule."""
    from atomate2.siesta.flows.molecular.gas_phase import GasPhaseMoleculeMaker

    o2 = Molecule(["O", "O"], [[0, 0, 0], [0, 0, 1.21]])
    maker = GasPhaseMoleculeMaker(box_size=20.0)
    maker.relax_maker.dry_run = True
    flow = maker.make(o2)

    # The relax job's generator must carry the custom-ordering request so the
    # applied +1/+1 moments survive into DM.InitSpin (the powerup applies the
    # molecular params to the job's maker, not the outer maker).
    orderings = [
        gen.user_params.get("a2s_magnetic_ordering")
        for job in flow.jobs
        if (mk := getattr(job, "maker", None)) is not None
        and (gen := getattr(mk, "input_set_generator", None)) is not None
        and gen.user_params
    ]
    assert "custom" in orderings
