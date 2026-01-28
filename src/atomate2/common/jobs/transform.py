"""Utility jobs to apply transformations as a job."""

from __future__ import annotations

import os
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from jobflow import Maker, job
from pymatgen.transformations.advanced_transformations import SQSTransformation

from atomate2.common.schemas.transform import SQSTask, TransformTask

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymatgen.core import Structure
    from pymatgen.transformations.transformation_abc import AbstractTransformation


@dataclass
class Transformer(Maker):
    """Apply a pymatgen transformation, as a job.

    For many of the standard and advanced transformations,
    this should "just work" by supplying the transformation.
    """

    transformation: AbstractTransformation
    name: str = "pymatgen transformation maker"

    @job
    def make(
        self, structure: Structure, **kwargs
    ) -> TransformTask | list[TransformTask]:
        """Evaluate the transformation.

        Parameters
        ----------
        structure : Structure to transform
        **kwargs : to pass to the `apply_transformation` method

        Returns
        -------
        list of TransformTask, if `self.transformation.is_one_to_many`
        (many structures are produced from a single transformation)

        TransformTask, otherwise
        """
        transformed_structure = self.transformation.apply_transformation(
            structure, **kwargs
        )
        if self.transformation.is_one_to_many:
            return [
                TransformTask(
                    input_structure=structure,
                    final_structure=dct["structure"],
                    transformation=dct.get("transformation") or self.transformation,
                )
                for dct in transformed_structure
            ]
        return TransformTask(
            input_structure=structure,
            final_structure=transformed_structure,
            transformation=self.transformation,
        )


@dataclass
class SQS(Transformer):
    """Generate special quasi-random structures (SQSs)."""

    name: str = "SQS"

    transformation: SQSTransformation = field(
        default_factory=SQSTransformation(
            scaling=1,
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
        elif (
            hasattr(scaling, "__len__")
            and all(isinstance(sf, int) for sf in scaling)
            and len(scaling) == 3
        ):
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

    @job
    def make(  # type: ignore[override]
        self,
        structure: Structure,
        return_ranked_list: bool | int = False,
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

        Returns
        -------
        dict
            A dict of the best SQS structure, its objective (if saved), and
            the ranked SQS structures (if saved).
        """
        original_directory = os.getcwd()

        valid_struct = self.check_structure(structure, self.transformation.scaling)
        if return_ranked_list and self.transformation.instances == 1:
            raise ValueError(
                "`return_ranked_list` should only be used for parallel MCSQS runs."
            )

        sqs_structs = self.transformation.apply_transformation(
            valid_struct, return_ranked_list=return_ranked_list
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

        # MCSQS caller changes the directory
        os.chdir(original_directory)

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

        # For MCSQS, check whether the `perfect_match` was found
        # otherwise, SQSTask will throw a validation error
        found_perfect_match = False
        if (
            isinstance(best_objective, str)
            and best_objective.lower() == "perfect_match"
        ):
            best_objective = None
            found_perfect_match = True

        sqs_structures = None
        sqs_scores = None
        if isinstance(sqs_structs, list) and len(sqs_structs) > 1:
            sqs_structures = [entry["structure"] for entry in sqs_structs[1:]]
            sqs_scores = [entry["objective_function"] for entry in sqs_structs[1:]]
            for i, score in enumerate(sqs_scores):
                if isinstance(score, str) and score.lower() == "perfect_match":
                    sqs_scores[i] = None
                    found_perfect_match = True

        return SQSTask(
            transformation=self.transformation,
            input_structure=structure,
            final_structure=best_sqs,
            final_objective=best_objective,
            sqs_structures=sqs_structures,
            sqs_scores=sqs_scores,
            sqs_method=self.transformation.sqs_method,
            found_perfect_match=found_perfect_match,
        )
