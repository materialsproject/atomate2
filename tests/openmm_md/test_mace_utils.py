from pathlib import Path

import numpy as np
import pytest
import torch
from mace.calculators.foundations_models import download_mace_mp_checkpoint
from pymatgen.core import Structure

from atomate2.openmm.mace_utils import MacePotential, nnpops_nl, wrapping_nl
from atomate2.openmm.utils import structure_to_topology


@pytest.mark.openmm_mace
def test_mace_potential(random_structure: Structure):
    ff_path = Path(download_mace_mp_checkpoint())

    potential = MacePotential(model_path=ff_path)

    topology = structure_to_topology(random_structure)
    topology.setPeriodicBoxVectors(random_structure.lattice.matrix / 10)
    system = potential.create_system(topology)

    assert system.getNumParticles() == len(random_structure)
    assert len(system.getForces()) == 1


@pytest.fixture(scope="module")
def large_box() -> tuple[torch.Tensor, torch.Tensor, float, bool]:
    """Fixture for a large orthorhombic box and random positions."""
    num_atoms = 50
    cell_lengths = torch.tensor([10.0, 10.0, 10.0])
    cell = torch.diag(cell_lengths)
    positions = torch.rand((num_atoms, 3)) * cell_lengths
    cutoff = 4.0  # Less than half of smallest box length (5.0)
    pbc = True
    return positions, cell, cutoff, pbc


@pytest.fixture(scope="module")
def small_box() -> tuple[torch.Tensor, torch.Tensor, float, bool]:
    """Fixture for a small orthorhombic box and random positions."""
    num_atoms = 50
    cell_lengths = torch.tensor([5.0, 5.0, 5.0])
    cell = torch.diag(cell_lengths)
    positions = torch.rand((num_atoms, 3)) * cell_lengths
    cutoff = 4.9  # Greater than half of smallest box length (2.5)
    pbc = True
    return positions, cell, cutoff, pbc


@pytest.fixture(scope="module")
def triclinic_cell() -> tuple[torch.Tensor, torch.Tensor, float, bool]:
    """Fixture for a triclinic cell and random positions."""
    num_atoms = 50
    a = 5.0
    b = 5.0
    c = 5.0
    alpha = 90
    beta = 90
    gamma = 90  # Non-orthogonal angle

    # Convert cell parameters to cell vectors
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)

    cell = torch.zeros((3, 3))
    cell[0, 0] = a
    cell[1, 0] = b * np.cos(gamma_rad)
    cell[1, 1] = b * np.sin(gamma_rad)
    cell[2, 0] = c * np.cos(beta_rad)
    cell[2, 1] = (
        c
        * (np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad))
        / np.sin(gamma_rad)
    )
    cell[2, 2] = c * np.sqrt(
        1
        - np.cos(beta_rad) ** 2
        - (
            (np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad))
            / np.sin(gamma_rad)
        )
        ** 2
    )

    positions = torch.rand((num_atoms, 3)) @ cell
    cutoff = 3.0
    pbc = True
    return positions, cell, cutoff, pbc


@pytest.mark.openmm_mace
def test_nl_agreement_in_large_box(
    large_box: tuple[torch.Tensor, torch.Tensor, float, bool],
) -> None:
    """Test that nnpops_nl and wrapped_nl produce the same results in a large box."""
    positions, cell, cutoff, pbc = large_box

    # Run both functions
    neighbors_simple, shifts_simple = nnpops_nl(positions, cell, pbc, cutoff)
    neighbors_wrapped, shifts_wrapped = wrapping_nl(positions, cell, pbc, cutoff)

    # convert neighbor lists so they can be easily compared
    neighbors_simple_set = {tuple(pair) for pair in neighbors_simple.t().tolist()}
    neighbors_wrapped_self_set = {
        tuple(pair) for pair in neighbors_wrapped.t().tolist()
    }
    assert neighbors_simple_set == neighbors_wrapped_self_set

    # convert shift lists so they can be easily compared
    shifts_simple_set = {tuple(pair) for pair in shifts_simple.tolist()}
    shifts_wrapped_self_set = {tuple(pair) for pair in shifts_wrapped.tolist()}
    assert shifts_simple_set == shifts_wrapped_self_set


@pytest.mark.openmm_mace
def test_nl_approximately_correct_in_small_box(
    small_box: tuple[torch.Tensor, torch.Tensor, float, bool],
) -> None:
    """Test that wrapped_nl works in a small box with large cutoff."""
    positions, cell, cutoff, pbc = small_box

    neighbors_wrapped, shifts_wrapped = wrapping_nl(positions, cell, pbc, cutoff)

    # Check that the function runs and returns expected types
    assert neighbors_wrapped.shape[0] == 2, "Neighbors should have shape [2, N]"
    assert shifts_wrapped.shape[0] == neighbors_wrapped.shape[1], (
        "Shifts should match number of neighbor pairs"
    )
    assert 500 < neighbors_wrapped.shape[1] < 50_000, (
        "Shifts should be a reasonable size"
    )


@pytest.mark.openmm_mace
def test_nl_approximately_correct_in_triclinic_cell(
    triclinic_cell: tuple[torch.Tensor, torch.Tensor, float, bool],
) -> None:
    """Test that wrapped_nl works with a triclinic cell."""
    positions, cell, cutoff, pbc = triclinic_cell

    neighbors_wrapped, shifts_wrapped = wrapping_nl(positions, cell, pbc, cutoff)

    # Check that the function runs and returns expected types
    assert neighbors_wrapped.shape[0] == 2, "Neighbors should have shape [2, N]"
    assert shifts_wrapped.shape[0] == neighbors_wrapped.shape[1], (
        "Shifts should match number of neighbor pairs"
    )
    assert 200 < neighbors_wrapped.shape[1] < 20_000, (
        "Shifts should be a reasonable size"
    )


@pytest.mark.openmm_mace
def test_exact_pairs_between_four_atoms_on_line() -> None:
    """Test wrapped_nl with deterministically placed atoms and known cutoff."""

    # Define cell parameters
    cell_length = 10.0
    cell = torch.diag(torch.tensor([cell_length, cell_length, cell_length]))
    pbc = True
    cutoff = 2.5

    # Define atom positions
    positions = torch.tensor(
        [
            [1.0, 1.0, 1.0],  # Atom 0
            [1.0, 1.0, 3.0],  # Atom 1
            [1.0, 1.0, 7.0],  # Atom 2
            [1.0, 1.0, 9.0],  # Atom 3
        ]
    )

    # Expected neighbor pairs (after symmetrization)
    expected_pairs = {
        (0, 1),
        (1, 0),
        (0, 3),
        (3, 0),
        (2, 3),
        (3, 2),
    }

    # Run wrapped_nl
    neighbors_wrapped, shifts_wrapped = wrapping_nl(positions, cell, pbc, cutoff)

    # Extract neighbor pairs
    neighbor_pairs = (
        neighbors_wrapped.t().tolist()
    )  # Transpose and convert to list of pairs

    # Convert neighbor pairs to set of tuples for comparison
    neighbor_pairs_set = {tuple(pair) for pair in neighbor_pairs}

    # Assert that the neighbor pairs match the expected pairs
    assert neighbor_pairs_set == expected_pairs

    # Assert that the number of neighbor pairs is as expected
    expected_num_pairs = len(expected_pairs)
    actual_num_pairs = neighbors_wrapped.shape[1]
    assert actual_num_pairs == expected_num_pairs, (
        f"Expected {expected_num_pairs} neighbor pairs, got {actual_num_pairs}."
    )


@pytest.mark.openmm_mace
@pytest.mark.parametrize(
    "cutoff, n_pairs",
    [
        (1.1, 2),
        (1.6, 6),
        (2.6, 10),
        (3.1, 12),
    ],
)
def test_n_neighbors_between_three_atoms_on_line(cutoff: float, n_pairs: int) -> None:
    """Test wrapped_nl with deterministically placed atoms and known cutoff."""
    # Define cell parameters
    cell_length = 4.0
    cell = torch.diag(torch.tensor([100, 100, cell_length]))
    pbc = True

    # Define atom positions
    positions = torch.tensor(
        [
            [1.0, 1.0, 0.5],
            [1.0, 1.0, 2.0],
            [1.0, 1.0, 3.5],
        ]  # Atom 0  # Atom 1  # Atom 2
    )

    # Run wrapped_nl
    neighbors_wrapped, shifts_wrapped = wrapping_nl(positions, cell, pbc, cutoff)

    # Assert that the number of neighbor pairs is as expected
    expected_num_pairs = n_pairs
    actual_num_pairs = neighbors_wrapped.shape[1]
    assert actual_num_pairs == expected_num_pairs, (
        f"Expected {expected_num_pairs} neighbor pairs, got {actual_num_pairs}."
    )


@pytest.mark.openmm_mace
@pytest.mark.parametrize(
    "cutoff, n_pairs",
    [
        (1.1, 2),
        (2.9, 4),
        (3.9, 8),
        (4.1, 10),
    ],
)
def test_n_neighbors_between_two_atoms_on_line(cutoff: float, n_pairs: int) -> None:
    """Test wrapped_nl with deterministically placed atoms and known cutoff."""
    # Define cell parameters
    cell_length = 3.0
    cell = torch.diag(torch.tensor([100, 100, cell_length]))
    pbc = True

    # Define atom positions
    positions = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 2.0]])  # Atom 0  # Atom 1

    # Run wrapped_nl
    neighbors_wrapped, shifts_wrapped = wrapping_nl(positions, cell, pbc, cutoff)

    # Assert that the number of neighbor pairs is as expected
    expected_num_pairs = n_pairs
    actual_num_pairs = neighbors_wrapped.shape[1]
    assert actual_num_pairs == expected_num_pairs, (
        f"Expected {expected_num_pairs} neighbor pairs, got {actual_num_pairs}."
    )


@pytest.mark.openmm_mace
@pytest.mark.parametrize(
    "cutoff, n_pairs",
    [
        (1.1, 2),
        (2.1, 4),
        (3.1, 6),
    ],
)
def test_n_neighbors_with_one_atom_on_line(cutoff: float, n_pairs: int) -> None:
    """Test wrapped_nl with deterministically placed atoms and known cutoff."""
    # Define cell parameters
    cell_length = 1.0
    cell = torch.diag(torch.tensor([100, 100, cell_length]))
    pbc = True

    # Define atom positions
    positions = torch.tensor([[1.0, 1.0, 0.5]])  # Atom 0

    # Run wrapped_nl
    neighbors_wrapped, shifts_wrapped = wrapping_nl(positions, cell, pbc, cutoff)

    # Assert that the number of neighbor pairs is as expected
    expected_num_pairs = n_pairs
    actual_num_pairs = neighbors_wrapped.shape[1]
    assert actual_num_pairs == expected_num_pairs, (
        f"Expected {expected_num_pairs} neighbor pairs, got {actual_num_pairs}."
    )


@pytest.mark.openmm_mace
@pytest.mark.parametrize(
    "cutoff, n_pairs",
    [
        (1.1, 4),
        (1.42, 8),
        (2.01, 12),
        (2.4, 20),
    ],
)
def test_n_neighbors_with_one_atom_on_grid(cutoff: float, n_pairs: int) -> None:
    """Test wrapped_nl with deterministically placed atoms and known cutoff."""
    # Define cell parameters
    cell_length = 1.0
    cell = torch.diag(torch.tensor([100, cell_length, cell_length]))
    pbc = True

    # Define atom positions
    positions = torch.tensor([[1.0, 1.0, 0.5]])  # Atom 0

    # Run wrapped_nl
    neighbors_wrapped, shifts_wrapped = wrapping_nl(positions, cell, pbc, cutoff)

    # Assert that the number of neighbor pairs is as expected
    expected_num_pairs = n_pairs
    actual_num_pairs = neighbors_wrapped.shape[1]
    assert actual_num_pairs == expected_num_pairs, (
        f"Expected {expected_num_pairs} neighbor pairs, got {actual_num_pairs}."
    )
