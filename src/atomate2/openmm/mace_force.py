"""Supports easy instantiation of OpenMM Systems with the Mace force field.

This code is based off of the openmm-ml package. In particular,
it borrows from the MLPotential class written by Peter Eastman and the MACEForce
class written by Harry Moore. The nnpops_nl function in the neighbors file
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

from atomate2.openmm.neighbors import nnpops_nl, wrapping_nl


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
