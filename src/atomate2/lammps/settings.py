from pydantic import BaseSettings, Field


class LammpsSettings(BaseSettings):
    """Settings that control the invocation of LAMMPS."""

    LAMMPS_CMD: str = Field("lmp", description="The command to run LAMMPS.")
    LAMMPS_SUFFIX: list[str] | str | None = Field(
        None,
        description=(
            "The LAMMPS style suffix(es) to use."
            "See https://docs.lammps.org/Run_options.html#suffix for more information."
        ),
        examples=["gpu", "kk", "intel", "omp", "opt", [["gpu", "kk"]]],
    )
    LAMMPS_PACKAGES: list[str] | str | None = Field(
        None,
        description=(
            "Options to pass to the package command-line flag that controls subpackage "
            "styles and parameters, e.g., `'gpu 1'` will call `lmp -pk gpu 1` will tell "
            "LAMMPS to use 1 GPU for this calculation. "
            "List values are passed with separate '-pk' invocations, e.g., `lmp -pk gpu 1 -pk omp 4`."
            "See https://docs.lammps.org/Run_options.html#package for more information."
        ),
        examples=["gpu 0", "gpu 1 split 0.75", "gpu 2 split -1.0", "gpu 1 omp 4"],
    )
    MPI_CMD: str = Field("mpirun", description="The command to invoke MPI.")
    MPI_NUM_PROCESSES_FLAG: str = Field(
        "-n",
        description="The flag with which to provide the number of processes to use in the MPI execution.",
    )


LAMMPS_SETTINGS = LammpsSettings()

__all__ = ("LAMMPS_SETTINGS", "LammpsSettings")