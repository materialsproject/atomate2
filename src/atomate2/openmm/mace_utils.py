"""Supports easy instantiation of OpenMM Systems with the Mace force field.

This code is based off of the openmm-ml package. In particular,
it borrows from the MLPotential class written by Peter Eastman and the MACEForce
class written by Harry Moore. The nnpops_nl function
is also from openmm-ml and was written by Harry Moore.

The original code is licensed as below

Portions copyright (c) 2021 Stanford University and the Authors.
Authors: Peter Eastman

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import openmm
import openmm.app
import openmmtorch
import torch
from e3nn.util import jit
from mace.tools import atomic_numbers_to_indices, to_one_hot, utils

try:
    from NNPOps.neighbors import getNeighborPairs
except ImportError as err:
    raise ImportError(
        "NNPOps is not installed. Please install it from conda-forge."
    ) from err


class MaceForce(torch.nn.Module):
    """Computes the energy of a system using a MACE model.

    Attributes
    ----------
        model (torch.nn.Module): The MACE model.
        device (str): The device (CPU or GPU) on which computations are performed.
        nl (Callable): The neighbor list function used for atom interactions.
        periodic (bool): Whether to use periodic boundary conditions.
        default_dtype (torch.dtype): The default data type for tensor operations.
        r_max (float): The maximum cutoff radius for atomic interactions.
        z_table (utils.AtomicNumberTable): Table for converting between
            atomic numbers and indices.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        atomic_numbers: list[int],
        device: torch.device | None,
        nl: Callable,
        *,
        periodic: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize the MaceForce object.

        Args:
            model (torch.nn.Module): The MACE neural network model.
            atomic_numbers (list[int]): List of atomic numbers for the system.
            device (str | None): The device to run computations on ('cuda', 'cpu',
                or None for auto-detection).
            nl (Callable): The neighbor list function to use.
            periodic (bool, optional): Whether to use periodic boundary conditions.
                Defaults to True.
            dtype (torch.dtype, optional): The data type for tensor operations.
                Defaults to torch.float32.
        """
        super().__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.nl = nl
        self.periodic = periodic
        self.default_dtype = dtype

        torch.set_default_dtype(self.default_dtype)

        logging.info(
            f"Running MACEForce on device: {self.device} "
            f"with dtype: {self.default_dtype} "
            f"and neighbour list: {nl}"
        )
        # conversion constants
        self.nm_to_A = 10.0
        self.eV_to_kj = 96.48533288

        self.model = model.to(dtype=self.default_dtype, device=self.device)
        self.model.eval()

        # set model properties
        self.r_max = self.model.r_max
        self.z_table = utils.AtomicNumberTable(
            [int(z) for z in self.model.atomic_numbers]
        )
        self.model.atomic_numbers = torch.tensor(
            self.model.atomic_numbers.clone(), device=self.device
        )

        # compile model
        self.model = jit.compile(self.model)

        # setup system
        n_atoms = len(atomic_numbers)
        self.ptr = torch.tensor([0, n_atoms], dtype=torch.long, device=self.device)
        self.batch = torch.zeros(n_atoms, dtype=torch.long, device=self.device)

        # one hot encoding of atomic number
        self.node_attrs = to_one_hot(
            torch.tensor(
                atomic_numbers_to_indices(atomic_numbers, z_table=self.z_table),
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(-1),
            num_classes=len(self.z_table),
        )

        if periodic:
            self.pbc = torch.tensor([True, True, True], device=self.device)
        else:
            self.pbc = torch.tensor([False, False, False], device=self.device)

    def forward(
        self, positions: torch.Tensor, boxvectors: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute the energy of the system given atomic positions and box vectors.

        This method calculates the neighbor list, prepares the input for the MACE
        model, and returns the computed energy.

        Args:
            positions (torch.Tensor): Atomic positions in nanometers.
            boxvectors (torch.Tensor | None, optional): Box vectors for
                periodic systems. Defaults to None.

        Returns
        -------
            torch.Tensor: The computed energy of the system in kJ/mol.
        """
        positions = positions.to(device=self.device, dtype=self.default_dtype)
        positions = positions * self.nm_to_A

        if boxvectors is not None:
            cell = (
                boxvectors.to(device=self.device, dtype=self.default_dtype)
                * self.nm_to_A
            )
        else:
            # TODO: it's not clear what the best fallback should be
            # cell = torch.eye(3, device=self.device)
            cell = torch.zeros((3, 3), device=self.device)

        # calculate neighbor list
        mapping, shifts_idx = self.nl(positions, cell, self.periodic, self.r_max)
        edge_index = torch.stack((mapping[0], mapping[1]))
        shifts = torch.mm(shifts_idx, cell)

        # get model output
        out = self.model(
            dict(
                ptr=self.ptr,
                node_attrs=self.node_attrs,
                batch=self.batch,
                pbc=self.pbc,
                cell=cell,
                positions=positions,
                edge_index=edge_index,
                unit_shifts=shifts_idx,
                shifts=shifts,
            ),
            compute_force=False,
        )

        energy = out["interaction_energy"]
        if energy is None:
            energy = torch.tensor(0.0, device=self.device)

        # return energy tensor
        return energy * self.eV_to_kj


class MacePotential:
    """A potential function class for molecular simulations using MACE models.

    Attributes
    ----------
        model (torch.nn.Module | None): The MACE model, if provided directly.
        model_path (str | Path | None): Path to the MACE model file, if the
            model is to be loaded from disk.
    """

    def __init__(
        self, model: torch.nn.Module | None = None, model_path: str | Path | None = None
    ) -> None:
        """Initialize a MacePotential object.

        Exactly one of 'model' or 'model_path' must be provided.

        Args:
            model (torch.nn.Module | None, optional): The MACE model. Defaults to None.
            model_path (str | Path | None, optional): Path to the MACE model file.
                Defaults to None.

        Raises
        ------
            ValueError: If neither model nor model_path is provided,
                or if both are provided.

        """
        if (model_path is None) == (model is None):
            raise ValueError(
                "Exactly one of 'model_paths' or 'models' must be provided"
            )
        self.model = model
        self.model_path = model_path

    def create_system(self, topology: openmm.app.Topology, **kwargs) -> openmm.System:
        """Create a System for running a simulation with this potential function.

        Parameters
        ----------
        topology : openmm.app.Topology
            The Topology for which to create a System
        **kwargs : dict
            Additional keyword arguments for customizing the potential function

        Returns
        -------
        openmm.System
            A newly created System object that uses this potential function to model
            the Topology
        """
        system = openmm.System()
        if topology.getPeriodicBoxVectors() is not None:
            system.setDefaultPeriodicBoxVectors(*topology.getPeriodicBoxVectors())
        for atom in topology.atoms():
            if atom.element is None:
                system.addParticle(0)
            else:
                system.addParticle(atom.element.mass)
        self.add_forces(topology, system, **kwargs)
        return system

    def add_forces(
        self,
        topology: openmm.app.Topology,
        system: openmm.System,
        nl: Callable | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Add MACE forces to an existing OpenMM System.

        This method creates and adds a TorchForce to the provided System, which computes
        interactions using the MACE potential.

        Args:
            topology (openmm.app.Topology): The system topology.
            system (openmm.System): The OpenMM System to which forces will be added.
            nl (Callable | None, optional): The neighbor list method to use.
                If None, an appropriate method will be chosen based on system size.
                Defaults to None.
            device (str | None, optional): The device to use for computations
                ('cuda', 'cpu', or None for auto-detection). Defaults to None.
            dtype (str, optional): The data type to use for computations.
                Defaults to "float32".
        """
        periodic = (
            topology.getPeriodicBoxVectors() is not None
        ) or system.usesPeriodicBoundaryConditions()

        atoms = list(topology.atoms())
        atomic_numbers = [atom.element.atomic_number for atom in atoms]

        # get length of shortest box vector
        box_vectors = topology.getPeriodicBoxVectors()
        min_length = np.min(np.linalg.norm(box_vectors, axis=1))

        # nnpops is both faster and O(n) but can't be used on small systems
        if nl is None:
            mace_cutoff = 5
            nl = nnpops_nl if min_length > (2 * mace_cutoff) else wrapping_nl

        # serialize the MACEForce into a module
        maceforce = MaceForce(
            self.model or torch.load(self.model_path),
            atomic_numbers,
            device=device,
            nl=nl,
            periodic=periodic,
            dtype=dtype,
        )
        module = torch.jit.script(maceforce)

        # Create the TorchForce and add it to the System.
        force = openmmtorch.TorchForce(module)
        force.setForceGroup(0)
        force.setUsesPeriodicBoundaryConditions(periodic)
        system.addForce(force)


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
