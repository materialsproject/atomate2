"""Utility jobs to apply transformations as a job."""

from __future__ import annotations

import os
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from jobflow import Maker, job
from monty.serialization import dumpfn
from pymatgen.transformations.advanced_transformations import SQSTransformation

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymatgen.core import Structure
    from pymatgen.transformations.transformation_abc import AbstractTransformation


@dataclass
class Transformer(Maker):
    """Apply a pymatgen transformation, as a job."""

    transformation: AbstractTransformation

    def _make(self, structure: Structure, **kwargs) -> Structure:
        """Define methods to apply transformations to a structure."""
        return self.transformation.apply_transformation(structure, **kwargs)

    @job
    def make(self, structure: Structure, **kwargs) -> Structure:
        """Run the transformation, should not be modified in subclasses."""
        return self._make(structure, **kwargs)


@dataclass
class SQS(Transformer):
    """Generate special quasi-random structures (SQSs)."""

    transformation: SQSTransformation = field(
        default_factory=SQSTransformation(
            search_time=60,
            directory=Path(".") / "sqs_runs",
            remove_duplicate_structures=True,
            best_only=True,
        )
    )

    @staticmethod
    def check_structure(structure: Structure, scaling: Sequence[int]) -> Structure:
        """Ensure that a disordered structure and scaling factor(s) are sensible."""
        struct = structure.copy()
        struct.remove_oxidation_states()
        if struct.is_ordered:
            raise ValueError("Your structure is likely ordered!")

        if isinstance(scaling, int):
            nsites = scaling * len(struct)
        elif hasattr(scaling, "__len__") and len(scaling) == 3:
            nsites = len(struct * scaling)
        else:
            raise ValueError(
                "`scaling` must be an int or sequence of three int, "
                f"found {type(scaling)}."
            )

        num_sites: dict[str, int | float] = {
            str(element): count * nsites
            for element, count in struct.composition.items()
        }

        if not all(
            abs(num_sites[element] - round(num_sites[element])) < 1e-3
            for element in num_sites
        ):
            raise ValueError(
                f"Incompatible supercell number of sites {nsites} "
                f"for composition {struct.composition}"
            )
        return struct

    def _make(  # type: ignore[override]
        self,
        structure: Structure,
        return_ranked_list: bool | int = False,
        output_filename: str | Path | None = None,
        archive_instances: bool = False,
    ) -> dict:
        """Perform a parallelized SQS search.

        For Monte Carlo methods, mcsqs and icet-monte_carlo, this
        executes parallel SQS searches from the same starting structure.

        For the icet-enumeration method, this divides the labor of
        searching through a list of structures.

        Parameters
        ----------
        structure : Structure
            Disordered structure to order.
        return_ranked_list: bool | int = False
            Whether to return a list of SQS structures ranked by objective function
            (bool), or how many to return (int). False returns only the best.
        output_filename : str | Path | None = None
            If a str, the name of the file to log SQS output.
            If None, no file is written.

        Returns
        -------
        dict
            A dict of the best SQS structure, its objective (if saved), and
            the ranked SQS structures (if saved).
        """
        original_directory = os.getcwd()

        structure = self.check_structure(structure, self.transformation.scaling)
        if return_ranked_list and self.transformation.instances == 1:
            raise ValueError(
                "`return_ranked_list` should only be used for parallel MCSQS runs."
            )

        sqs_structs = self.transformation.apply_transformation(
            structure, return_ranked_list=return_ranked_list
        )

        if return_ranked_list:
            best_sqs = sqs_structs[0]["structure"]
            best_objective = sqs_structs[0]["objective_function"]
        else:
            best_sqs = sqs_structs
            best_objective = None

            if (
                self.transformation.sqs_method == "mcsqs"
                and (mcsqs_corr_file := Path("bestcorr.out")).exists()
            ):
                best_objective = float(
                    mcsqs_corr_file.read_text().split("Objective_function=")[-1].strip()
                )

        output = {
            "input_structure": structure,
            "sqs_structures": sqs_structs,
            "best_sqs_structure": best_sqs,
            "best_objective": best_objective,
        }

        # MCSQS caller changes the directory
        os.chdir(original_directory)
        if output_filename:
            dumpfn(output, output_filename)

        if archive_instances and self.transformation.sqs_method == "mcsqs":
            # MCSQS is the only SQS maker which requires a working directory
            mcsqs_dir = Path(self.transformation.directory)
            archive_name = str(self.transformation.directory)
            if archive_name[-1] == os.path.sep:
                archive_name = archive_name[:-1]
            archive_name += ".tar.gz"

            # add files to tarball
            with tarfile.open(archive_name, "w:gz") as tarball:
                files: list[Path] = []
                for file in os.scandir(mcsqs_dir):
                    if (filename := mcsqs_dir / file).is_file():
                        files.append(filename)
                        tarball.add(filename)

            # cleanup
            _ = [file.unlink() for file in files]  # type: ignore[func-returns-value]

            if len(list(os.scandir(mcsqs_dir))) == 0:
                mcsqs_dir.unlink()

        return output
