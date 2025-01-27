"""Neighbor list utilities."""

try:
    import torch
    from NNPOps.neighbors import getNeighborPairs
except ImportError as err:
    raise ImportError(
        "NNPOps is not installed. Please install it from conda-forge."
    ) from err


def nnpops_nl(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: bool,
    cutoff: float,
    sorti: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a neighbor list computation using NNPOps.

    It outputs neighbors and shifts in the same format as ASE
    https://wiki.fysik.dtu.dk/ase/ase/neighborlist.html#ase.neighborlist.primitive_neighbor_list

    neighbors, shifts = nnpops_nl(..)
    is equivalent to

    [i, j], S = primitive_neighbor_list( quantities="ijS", ...)

    Parameters
    ----------
    positions : torch.Tensor
        Atom positions, shape (num_atoms, 3)
    cell : torch.Tensor
        Unit cell, shape (3, 3)
    pbc : bool
        Whether to use periodic boundary conditions
    cutoff : float
        Cutoff distance for neighbors
    sorti : bool, optional
        Whether to sort the neighbor list by the first index.
        Defaults to False.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple containing:
        - neighbors (torch.Tensor): Neighbor list, shape (2, num_neighbors)
        - shifts (torch.Tensor): Shift vectors, shape (num_neighbors, 3)
    """
    device = positions.device
    neighbors, deltas, _, _ = getNeighborPairs(
        positions,
        cutoff=cutoff,
        max_num_pairs=-1,
        box_vectors=cell if pbc else None,
        check_errors=False,
    )

    neighbors = neighbors.to(dtype=torch.long)

    # remove empty neighbors
    mask = neighbors[0] > -1
    neighbors = neighbors[:, mask]
    deltas = deltas[mask, :]

    # compute shifts TODO: pass deltas and distance directly to model
    # From ASE docs:
    # wrapped_delta = pos[i] - pos[j] - shift.cell
    # => shift = ((pos[i]-pos[j]) - wrapped_delta).cell^-1
    if pbc:
        shifts = torch.mm(
            (positions[neighbors[0]] - positions[neighbors[1]]) - deltas,
            torch.linalg.inv(cell),
        )
    else:
        shifts = torch.zeros(deltas.shape, device=device)

    # we have i<j, get also i>j
    neighbors = torch.hstack((neighbors, torch.stack((neighbors[1], neighbors[0]))))
    shifts = torch.vstack((shifts, -shifts))

    if sorti:
        idx = torch.argsort(neighbors[0])
        neighbors = neighbors[:, idx]
        shifts = shifts[idx, :]

    return neighbors, shifts


@torch.jit.script
def wrapping_nl(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: bool,
    cutoff: float,
    sorti: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Neighbor list including self-interactions across periodic boundaries.

    Parameters
    ----------
    positions : torch.Tensor
        Atom positions, shape (num_atoms, 3)
    cell : torch.Tensor
        Unit cell, shape (3, 3)
    pbc : bool
        Whether to use periodic boundary conditions
    cutoff : float
        Cutoff distance for neighbors
    sorti : bool, optional
        Whether to sort the neighbor list by the first index.
        Defaults to False.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple containing:
        - neighbors (torch.Tensor): Neighbor list, shape (2, num_neighbors)
        - shifts (torch.Tensor): Shift vectors, shape (num_neighbors, 3)
    """
    num_atoms = positions.shape[0]
    device = positions.device
    dtype = positions.dtype

    # Get all unique pairs including self-pairs (i <= j)
    uij = torch.triu_indices(num_atoms, num_atoms, offset=0, device=device)
    i_indices = uij[0]
    j_indices = uij[1]

    if pbc:
        # Compute displacement vectors between atom pairs
        deltas = positions[i_indices] - positions[j_indices]

        # Compute inverse cell matrix
        inv_cell = torch.linalg.inv(cell)

        # Compute fractional coordinates of displacement vectors
        frac_deltas = torch.matmul(deltas, inv_cell)

        # Determine the maximum number of shifts needed along each axis
        cell_lengths = torch.linalg.norm(cell, dim=0)
        n_max = torch.ceil(cutoff / cell_lengths).to(torch.int32)

        # Extract scalar values from n_max
        n_max0 = int(n_max[0])
        n_max1 = int(n_max[1])
        n_max2 = int(n_max[2])

        # Generate shift ranges
        shift_range_x = torch.arange(-n_max0, n_max0 + 1, device=device, dtype=dtype)
        shift_range_y = torch.arange(-n_max1, n_max1 + 1, device=device, dtype=dtype)
        shift_range_z = torch.arange(-n_max2, n_max2 + 1, device=device, dtype=dtype)

        # Generate all combinations of shifts within the range [-n_max, n_max]
        shift_x, shift_y, shift_z = torch.meshgrid(
            shift_range_x, shift_range_y, shift_range_z, indexing="ij"
        )

        shifts_list = torch.stack(
            (shift_x.reshape(-1), shift_y.reshape(-1), shift_z.reshape(-1)), dim=1
        )

        # Total number of shifts
        num_shifts = shifts_list.shape[0]

        # Expand atom pairs and shifts
        num_pairs = i_indices.shape[0]
        i_indices_expanded = i_indices.repeat_interleave(num_shifts)
        j_indices_expanded = j_indices.repeat_interleave(num_shifts)
        shifts_expanded = shifts_list.repeat(num_pairs, 1)

        # Expand fractional displacements
        frac_deltas_expanded = frac_deltas.repeat_interleave(num_shifts, dim=0)

        # Apply shifts to fractional displacements
        shifted_frac_deltas = frac_deltas_expanded - shifts_expanded

        # Convert back to Cartesian coordinates
        shifted_deltas = torch.matmul(shifted_frac_deltas, cell)

        # Compute distances
        distances = torch.linalg.norm(shifted_deltas, dim=1)

        # Apply cutoff filter
        within_cutoff = distances <= cutoff

        # Exclude self-pairs where shift is zero (no periodic boundary crossing)
        shift_zero = (shifts_expanded == 0).all(dim=1)
        i_eq_j = i_indices_expanded == j_indices_expanded
        exclude_self_zero_shift = i_eq_j & shift_zero
        within_cutoff = within_cutoff & (~exclude_self_zero_shift)

        num_within_cutoff = int(within_cutoff.sum())

        i_indices_final = i_indices_expanded[within_cutoff]
        j_indices_final = j_indices_expanded[within_cutoff]
        shifts_final = shifts_expanded[within_cutoff]

        # Generate neighbor pairs and shifts
        neighbors = torch.stack((i_indices_final, j_indices_final), dim=0)
        shifts = shifts_final

        # Add symmetric pairs (j, i) and negate shifts,
        # but avoid duplicates for self-pairs
        i_neq_j = i_indices_final != j_indices_final
        neighbors_sym = torch.stack(
            (j_indices_final[i_neq_j], i_indices_final[i_neq_j]), dim=0
        )
        shifts_sym = -shifts_final[i_neq_j]

        neighbors = torch.cat((neighbors, neighbors_sym), dim=1)
        shifts = torch.cat((shifts, shifts_sym), dim=0)

        if sorti:
            idx = torch.argsort(neighbors[0])
            neighbors = neighbors[:, idx]
            shifts = shifts[idx, :]

        return neighbors, shifts

    # Non-periodic case
    deltas = positions[i_indices] - positions[j_indices]
    distances = torch.linalg.norm(deltas, dim=1)

    # Apply cutoff filter
    within_cutoff = distances <= cutoff

    # Exclude self-pairs where distance is zero
    i_eq_j = i_indices == j_indices
    exclude_self_zero_distance = i_eq_j & (distances == 0)
    within_cutoff = within_cutoff & (~exclude_self_zero_distance)

    num_within_cutoff = int(within_cutoff.sum())

    i_indices_final = i_indices[within_cutoff]
    j_indices_final = j_indices[within_cutoff]

    shifts_final = torch.zeros((num_within_cutoff, 3), device=device, dtype=dtype)

    # Generate neighbor pairs and shifts
    neighbors = torch.stack((i_indices_final, j_indices_final), dim=0)
    shifts = shifts_final

    # Add symmetric pairs (j, i) and shifts (only if i != j)
    i_neq_j = i_indices_final != j_indices_final
    neighbors_sym = torch.stack(
        (j_indices_final[i_neq_j], i_indices_final[i_neq_j]), dim=0
    )
    shifts_sym = shifts_final[i_neq_j]  # shifts are zero

    neighbors = torch.cat((neighbors, neighbors_sym), dim=1)
    shifts = torch.cat((shifts, shifts_sym), dim=0)

    if sorti:
        idx = torch.argsort(neighbors[0])
        neighbors = neighbors[:, idx]
        shifts = shifts[idx, :]

    return neighbors, shifts
