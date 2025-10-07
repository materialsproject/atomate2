"""Deprecated stub to use in place of modern emmet-core dependencies."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from emmet.core.phonon import (
    PhononBSDOSDoc,
    PhononComputationalSettings,
    ThermalDisplacementData,
)
from monty.json import MSONable
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from emmet.core.math import Matrix3D

warnings.warn(
    "atomate2.common.schemas.phonons is deprecated "
    "and will be removed on 1 January, 2026. "
    "Please migrate your code to use emmet.core.phonon",
    stacklevel=2,
    category=DeprecationWarning,
)

__all__ = [
    "ForceConstants",
    "PhononBSDOSDoc",
    "PhononComputationalSettings",
    "PhononJobDirs",
    "PhononUUIDs",
    "ThermalDisplacementData",
]


class ForceConstants(MSONable):
    """DEPRECATED: A force constants class."""

    def __init__(self, force_constants: list[list[Matrix3D]]) -> None:
        self.force_constants = force_constants


class PhononUUIDs(BaseModel):
    """DEPRECATED: Collection to save all uuids connected to the phonon run."""

    optimization_run_uuid: str | None = Field(None, description="optimization run uuid")
    displacements_uuids: list[str] | None = Field(
        None, description="The uuids of the displacement jobs."
    )
    static_run_uuid: str | None = Field(None, description="static run uuid")
    born_run_uuid: str | None = Field(None, description="born run uuid")


class PhononJobDirs(BaseModel):
    """DEPRECATED: Save all job directories relevant for the phonon run."""

    displacements_job_dirs: list[str | None] | None = Field(
        None, description="The directories where the displacement jobs were run."
    )
    static_run_job_dir: str | None = Field(
        None, description="Directory where static run was performed."
    )
    born_run_job_dir: str | None = Field(
        None, description="Directory where born run was performed."
    )
    optimization_run_job_dir: str | None = Field(
        None, description="Directory where optimization run was performed."
    )
    taskdoc_run_job_dir: str | None = Field(
        None, description="Directory where task doc was generated."
    )
