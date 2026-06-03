# fmt: off

"""SiestaInput"""
import warnings
import logging
import numpy as np

from ase import Atoms
from ase.constraints import FixAtoms, FixCartesian, FixedLine, FixedPlane
logger = logging.getLogger(__name__)


class SiestaInput:
    """SiestaInput"""
    @classmethod
    def is_along_cartesian(cls, norm_dir: np.ndarray) -> bool:
        """Return whether `norm_dir` is along a Cartesian coordinate."""
        directions = [
            [+1, 0, 0], [-1, 0, 0],
            [0, +1, 0], [0, -1, 0],
            [0, 0, +1], [0, 0, -1],
        ]
        for direction in directions:
            if np.allclose(norm_dir, direction, rtol=0.0, atol=1e-6):
                return True
        return False

    @classmethod
    def generate_kpts(cls, kpts):
        """Write kpts."""
        yield '\n'
        yield '#KPoint grid\n'
        yield '%block kgrid_Monkhorst_Pack\n'
        for i in range(3):
            s = ''
            if i < len(kpts):
                number = kpts[i]
                displace = 0.0
            else:
                number = 1
                displace = 0
            for j in range(3):
                write_this = number if j == i else 0
                s += f' {write_this:d} '
            s += f'{displace:1.1f}\n'
            yield s
        yield '%endblock kgrid_Monkhorst_Pack\n'
        yield '\n'

    @classmethod
    def get_species(cls, atoms: Atoms, species: list, basis_set: str):
        """
        Determine species from atoms object and species input, using atoms.info if available.

        Args:
            atoms (Atoms): ASE Atoms object, potentially with info['species_dict'] and info['species_Z_dict'].
            species (list): List of species dictionaries with keys 'symbol', 'tag', 'basis_set', 'pseudopotential', 'ghost'.
            basis_set (str): Default basis set for species not specified in the species list.

        Returns:
            tuple: (all_species, species_numbers)
                - all_species: List of Species objects.
                - species_numbers: Array of species indices for each atom (1-based for FDF).
        """
        from atomate2.siesta.sets.ase.parameters import Species
        logger.debug("SiestaInput.get_species()")

        species_numbers = np.zeros(len(atoms), dtype=int)
        all_species = []
        tags = atoms.get_tags()

        # Check if atoms.info contains species_dict and species_Z_dict
        if 'species_dict' in atoms.info and 'species_Z_dict' in atoms.info:
            logger.debug("Using species information from atoms.info")
            species_dict = atoms.info['species_dict']
            species_Z_dict = atoms.info['species_Z_dict']
            species_labels = atoms.info.get('species_labels', atoms.get_chemical_symbols())

            # Create a map of labels to species indices
            unique_labels = sorted(set(species_dict.values()))
            species_map = {label: idx for idx, label in enumerate(unique_labels, 1)}  # 1-based indexing

            # Match species list entries to species_dict
            for idx, label in species_dict.items():
                atomic_number = species_Z_dict.get(idx, 0)
                is_ghost = atomic_number < 0
                base_symbol = label.split('_')[0] if '_' in label else label

                # Find matching species in the provided species list
                matching_species = [s for s in species if s['symbol'] == base_symbol and s['tag'] == label]
                if matching_species:
                    # Use the basis_set from the provided species list
                    spec = Species(
                        symbol=base_symbol,
                        basis_set=matching_species[0]['basis_set'],
                        tag=label if label != base_symbol else None,
                        pseudopotential=matching_species[0]['pseudopotential'],
                        ghost=is_ghost
                    )
                else:
                    # Fall back to default basis_set if no match found
                    spec = Species(
                        symbol=base_symbol,
                        basis_set=basis_set,
                        tag=label if label != base_symbol else None,
                        pseudopotential=None,
                        ghost=is_ghost
                    )
                all_species.append(spec)

            # Assign species numbers based on species_labels
            for i, label in enumerate(species_labels):
                species_numbers[i] = species_map.get(label, 0)
        else:
            logger.debug("No species_dict in atoms.info, using default species logic")
            # Original logic for default species
            default_species = [
                s for s in species
                if (s['tag'] is None) and s['symbol'] in atoms.symbols
            ]
            default_symbols = [s['symbol'] for s in default_species]
            for symbol in atoms.symbols:
                if symbol not in default_symbols:
                    spec = Species(
                        symbol=symbol,
                        basis_set=basis_set,
                        tag=None
                    )
                    default_species.append(spec)
                    default_symbols.append(symbol)
            assert len(default_species) == len(set(atoms.symbols))

            # Assign default species numbers
            i = 1
            for spec in default_species:
                mask = atoms.symbols == spec['symbol']
                species_numbers[mask] = i
                i += 1

            # Handle non-default species
            non_default_species = [s for s in species if s['tag'] is not None]
            for spec in non_default_species:
                mask1 = tags == spec['tag']
                mask2 = atoms.symbols == spec['symbol']
                mask = np.logical_and(mask1, mask2)
                if sum(mask) > 0:
                    species_numbers[mask] = i
                    i += 1

            all_species = default_species + non_default_species

        logger.debug(f"Generated species: {all_species}")
        logger.debug(f"Species numbers: {species_numbers.tolist()}")
        return all_species, species_numbers

    @classmethod
    def make_xyz_constraints(cls, atoms: Atoms):
        """Create coordinate-resolved list of constraints [natoms, 0:3]."""
        moved = np.ones((len(atoms), 3), dtype=int)
        for const in atoms.constraints:
            if isinstance(const, FixAtoms):
                moved[const.get_indices()] = 0
            elif isinstance(const, FixedLine):
                norm_dir = const.dir / np.linalg.norm(const.dir)
                if not cls.is_along_cartesian(norm_dir):
                    raise RuntimeError(f'norm_dir {norm_dir} is not one of the Cartesian axes')
                norm_dir = norm_dir.round().astype(int)
                moved[const.get_indices()] = norm_dir
            elif isinstance(const, FixedPlane):
                norm_dir = const.dir / np.linalg.norm(const.dir)
                if not cls.is_along_cartesian(norm_dir):
                    raise RuntimeError(f'norm_dir {norm_dir} is not one of the Cartesian axes')
                norm_dir = norm_dir.round().astype(int)
                moved[const.get_indices()] = abs(1 - norm_dir)
            elif isinstance(const, FixCartesian):
                moved[const.get_indices()] = 1 - const.mask.astype(int)
            else:
                warnings.warn(f'Constraint {const!s} is ignored')
        return moved
