"""Utilities for molecule manipulation and orientation."""

from __future__ import annotations

import numpy as np
from pymatgen.core import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor


def rotate_molecule(
    mol: Molecule,
    normal_vector: np.ndarray,
    target_vector: np.ndarray | None = None,
    extra_rotation: float = 0.0,
    rotation_axis: np.ndarray | None = None,
) -> Molecule:
    """
    Rotate molecule so that its normal_vector aligns with target_vector.

    Optionally apply an additional rotation around rotation_axis.

    Parameters
    ----------
    mol : Molecule
        Molecule to rotate.
    normal_vector : np.ndarray
        Current normal vector of the molecule (e.g., [0, 0, 1] for flat molecule).
    target_vector : np.ndarray, optional
        Target direction (default: [0, 0, 1]).
    extra_rotation : float
        Additional rotation in degrees (default: 0.0).
    rotation_axis : np.ndarray, optional
        Axis for additional rotation (default: [0, 0, 1]).

    Returns
    -------
    Molecule
        Rotated molecule.

    Examples
    --------
    >>> from pymatgen.core import Molecule
    >>> mol = Molecule(["C", "O"], [[0, 0, 0], [0, 0, 1.2]])
    >>> # Rotate CO to point in x-direction
    >>> rotated = rotate_molecule(mol, [0, 0, 1], [1, 0, 0])
    """
    if target_vector is None:
        target_vector = np.array([0, 0, 1])
    if rotation_axis is None:
        rotation_axis = np.array([0, 0, 1])

    # Normalize vectors
    normal_vector = np.array(normal_vector) / np.linalg.norm(normal_vector)
    target_vector = np.array(target_vector) / np.linalg.norm(target_vector)
    rotation_axis = np.array(rotation_axis) / np.linalg.norm(rotation_axis)

    # Compute rotation to align normal with target vector
    rotation_axis_normal = np.cross(normal_vector, target_vector)

    if np.linalg.norm(rotation_axis_normal) > 1e-6:
        rotation_axis_normal = rotation_axis_normal / np.linalg.norm(
            rotation_axis_normal
        )
        cos_theta = np.dot(normal_vector, target_vector)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        # Rodrigues' rotation formula
        K = np.array(
            [
                [0, -rotation_axis_normal[2], rotation_axis_normal[1]],
                [rotation_axis_normal[2], 0, -rotation_axis_normal[0]],
                [-rotation_axis_normal[1], rotation_axis_normal[0], 0],
            ]
        )
        identity_matrix = np.eye(3)
        R = identity_matrix + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
    else:
        R = np.eye(3)

    # Apply additional rotation if specified
    if extra_rotation != 0.0:
        theta_extra = np.radians(extra_rotation)
        K_extra = np.array(
            [
                [0, -rotation_axis[2], rotation_axis[1]],
                [rotation_axis[2], 0, -rotation_axis[0]],
                [-rotation_axis[1], rotation_axis[0], 0],
            ]
        )
        R_extra = (
            np.eye(3)
            + np.sin(theta_extra) * K_extra
            + (1 - np.cos(theta_extra)) * K_extra @ K_extra
        )
        R = R_extra @ R

    # Apply rotation to molecule
    coords = np.array([site.coords for site in mol.sites])
    com = coords.mean(axis=0)
    coords_centered = coords - com
    rotated_coords = coords_centered @ R.T
    rotated_coords += com

    # Create new molecule with rotated coordinates
    species = [str(site.specie) for site in mol.sites]
    return Molecule(species, rotated_coords)


def prepare_molecule_with_orientation(
    molecule: Molecule | str,
    custom_file: str | None = None,
    plane_atoms: list[int] | None = None,
    target_vector: list[float] | None = None,
    extra_rotation: float = 0.0,
    rotation_axis: list[float] | None = None,
) -> Molecule:
    """
    Prepare molecule with custom file loading and orientation.

    Parameters
    ----------
    molecule : Molecule | str
        Molecule object or chemical formula string.
    custom_file : str, optional
        Path to custom molecule file (XYZ, CIF, etc.).
    plane_atoms : list[int], optional
        List of 3 atom indices defining plane normal for rotation.
    target_vector : list[float], optional
        Target direction for molecule orientation [x, y, z].
    extra_rotation : float
        Additional rotation in degrees.
    rotation_axis : list[float], optional
        Axis for additional rotation [x, y, z].

    Returns
    -------
    Molecule
        Prepared molecule with proper orientation.

    Examples
    --------
    >>> # Load benzene and orient it flat on surface
    >>> mol = prepare_molecule_with_orientation(
    ...     "C6H6", plane_atoms=[0, 1, 2], target_vector=[0, 0, 1]
    ... )

    >>> # Load custom molecule and rotate 45 degrees
    >>> mol = prepare_molecule_with_orientation(
    ...     custom_file="my_molecule.xyz", target_vector=[0, 0, 1], extra_rotation=45.0
    ... )
    """
    # Load molecule
    if custom_file:
        # Use ASE to read custom file, then convert to pymatgen
        from ase.io import read

        adaptor = AseAtomsAdaptor()
        ase_mol = read(custom_file)
        mol = adaptor.get_molecule(ase_mol)
    elif isinstance(molecule, str):
        # Try to create from ASE molecule database
        from ase.build import molecule as ase_molecule

        adaptor = AseAtomsAdaptor()
        try:
            ase_mol = ase_molecule(molecule)
            mol = adaptor.get_molecule(ase_mol)
        except Exception:
            # Fallback: single atom
            mol = Molecule([molecule], [[0, 0, 0]])
    else:
        mol = molecule

    # Apply orientation if specified
    if target_vector is not None:
        # Calculate normal vector
        coords = np.array([site.coords for site in mol.sites])

        if len(mol) == 2:
            # For diatomic molecules, use bond vector
            normal_vector = coords[1] - coords[0]
        elif plane_atoms is not None and len(plane_atoms) == 3:
            # For molecules with 3+ atoms, use plane normal
            if max(plane_atoms) >= len(mol):
                raise ValueError("Plane atom indices exceed number of atoms")
            p1, p2, p3 = (
                coords[plane_atoms[0]],
                coords[plane_atoms[1]],
                coords[plane_atoms[2]],
            )
            v1 = p2 - p1
            v2 = p3 - p1
            normal_vector = np.cross(v1, v2)
        else:
            # Default: use principal axis or z-direction
            normal_vector = np.array([0, 0, 1])

        # Normalize
        if np.linalg.norm(normal_vector) > 1e-6:
            mol = rotate_molecule(
                mol,
                normal_vector,
                np.array(target_vector),
                extra_rotation,
                np.array(rotation_axis) if rotation_axis else np.array([0, 0, 1]),
            )

    return mol


def molecule_to_structure_in_box(
    molecule: Molecule, box_size: float = 20.0
) -> Structure:
    """
    Convert Molecule to Structure by placing it in a cubic box.

    This is necessary for SIESTA calculations which require periodic boundary conditions.

    Parameters
    ----------
    molecule : Molecule
        Molecule to convert.
    box_size : float
        Size of cubic box in Angstroms (default: 20.0).

    Returns
    -------
    Structure
        Structure with molecule centered in a cubic box.

    Examples
    --------
    >>> from pymatgen.core import Molecule
    >>> mol = Molecule(["C", "O"], [[0, 0, 0], [0, 0, 1.2]])
    >>> struct = molecule_to_structure_in_box(mol, box_size=15.0)
    >>> struct.lattice.abc
    (15.0, 15.0, 15.0)
    """
    from pymatgen.core import Lattice

    # Create cubic lattice
    lattice = Lattice.cubic(box_size)

    # Get molecule center of mass
    coords = np.array([site.coords for site in molecule.sites])
    com = coords.mean(axis=0)

    # Center molecule in box
    box_center = np.array([box_size / 2, box_size / 2, box_size / 2])
    centered_coords = coords - com + box_center

    # Create structure
    species = [str(site.specie) for site in molecule.sites]
    return Structure(lattice, species, centered_coords, coords_are_cartesian=True)
