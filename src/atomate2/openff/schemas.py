"""Solvent and solvation schemas for OpenFF which are not yet production ready."""

from __future__ import annotations

from io import StringIO
from typing import TYPE_CHECKING, Annotated

import pandas as pd
from MDAnalysis import Universe
from MDAnalysis.analysis.dielectric import DielectricConstant
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    PlainValidator,
    WithJsonSchema,
)
from solvation_analysis.solute import Solute
from transport_analysis.viscosity import ViscosityHelfand

if TYPE_CHECKING:
    from typing import Any


def data_frame_validater(o: Any) -> pd.DataFrame:
    """Define custom validator for pandas DataFrame.

    Parameters
    ----------
    o : Any

    Returns
    -------
    pandas DataFrame
    """
    if isinstance(o, pd.DataFrame):
        return o
    if isinstance(o, str):
        return pd.read_csv(StringIO(o))
    raise ValueError(f"Invalid DataFrame: {o}")


def data_frame_serializer(df: pd.DataFrame) -> str:
    """Serialize pandas DataFrame as CSV."""
    return df.to_csv()


DataFrame = Annotated[
    pd.DataFrame,
    PlainValidator(data_frame_validater),
    PlainSerializer(data_frame_serializer),
    WithJsonSchema({"type": "string"}),
]


class SolventBenchmarkingDoc(BaseModel):
    """Define document for benchmarking solvent properties."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    density: float | None = Field(None, description="Density of the solvent")

    viscosity_function_values: list[float] | None = Field(
        None, description="Viscosity function over time"
    )

    viscosity: float | None = Field(None, description="Viscosity of the solvent")

    dielectric: float | None = Field(
        None, description="Dielectric constant of the solvent"
    )

    job_uuid: str | None = Field(
        None, description="The UUID of the flow that generated this data."
    )

    flow_uuid: str | None = Field(
        None, description="The UUID of the top level host from that job."
    )

    dielectric_run_kwargs: dict | None = Field(
        None, description="kwargs passed to the DielectricConstant.run method"
    )

    viscosity_run_kwargs: dict | None = Field(
        None, description="kwargs passed to the ViscosityHelfand.run method"
    )

    tags: list[str] | None = Field(
        [], title="tag", description="Metadata tagged to the parent job."
    )

    @classmethod
    def from_universe(
        cls,
        u: Universe,
        temperature: float | None = None,
        density: float | None = None,
        job_uuid: str | None = None,
        flow_uuid: str | None = None,
        dielectric_run_kwargs: dict | None = None,
        viscosity_run_kwargs: dict | None = None,
        tags: list[str] | None = None,
    ) -> SolventBenchmarkingDoc:
        """Create document from openmm Universe."""
        if temperature is not None:
            dielectric = DielectricConstant(
                u.atoms, temperature=temperature, make_whole=False
            )
            dielectric_run_kwargs = dielectric_run_kwargs or {}
            dielectric.run(**dielectric_run_kwargs)
            eps = dielectric.results.eps_mean
        else:
            eps = None

        if u.atoms.ts.has_velocities:
            start, stop = int(0.2 * len(u.trajectory)), int(0.8 * len(u.trajectory))
            viscosity_helfand = ViscosityHelfand(
                u.atoms,
                temp_avg=temperature,
                linear_fit_window=(start, stop),
            )
            viscosity_run_kwargs = viscosity_run_kwargs or {}
            viscosity_helfand.run(**viscosity_run_kwargs)
            viscosity_function_values = viscosity_helfand.results.timeseries.tolist()
            viscosity = viscosity_helfand.results.viscosity

        else:
            viscosity_function_values = None
            viscosity = None

        return cls(
            density=density,
            viscosity_function_values=viscosity_function_values,
            viscosity=viscosity,
            dielectric=eps,
            job_uuid=job_uuid,
            flow_uuid=flow_uuid,
            dielectric_run_kwargs=dielectric_run_kwargs,
            viscosity_run_kwargs=viscosity_run_kwargs,
            tags=tags,
        )


# class SolvationDoc(ClassicalMDDoc, arbitrary_types_allowed=True):
class SolvationDoc(BaseModel):
    """Schematize solvation calculation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    solute_name: str | None = Field(None, description="Name of the solute")

    solvent_names: list[str] | None = Field(None, description="Names of the solvents")

    is_electrolyte: bool | None = Field(
        None, description="Whether system is an electrolyte"
    )

    # Solute.coordination

    coordination_numbers: dict[str, float] | None = Field(
        None,
        description="A dictionary where keys are residue names and values are "
        "the mean coordination number of that residue.",
    )

    # coordination_numbers_by_frame: DataFrame | None= Field(
    #     None,
    #     description="Coordination number in each frame of the trajectory.",
    # )

    coordinating_atoms: DataFrame | None = Field(
        None,
        description="Fraction of each atom_type participating in solvation, "
        "calculated for each solvent.",
    )

    coordination_vs_random: dict[str, float] | None = Field(
        None,
        description="Coordination number relative to random coordination.",
    )

    # Solute.networking

    # TODO: In the worst case, this could be extremely large.
    #       Need to consider what else we might want from this object.
    # network_df: DataFrame | None= Field(
    #     None,
    #     description="All solute-solvent networks in the system, "
    # .    "indexed by the `frame` and a 'network_ix'. "
    #     "Columns are the species name and res_ix.",
    # )

    network_sizes: DataFrame | None = Field(
        None,
        description="Sizes of all networks, indexed by frame. Column headers are "
        "network sizes, e.g. the integer number of solutes + solvents in the network."
        "The values in each column are the number of networks with that size in each "
        "frame.",
    )

    solute_status: dict[str, float] | None = Field(
        None,
        description="A dictionary where the keys are the “status” of the "
        "solute and the values are the fraction of solute with that "
        "status, averaged over all frames. “isolated” means that the solute not "
        "coordinated with any of the networking solvents, network size is 1. "
        "“paired” means the solute and is coordinated with a single networking "
        "solvent and that solvent is not coordinated to any other solutes, "
        "network size is 2. “networked” means that the solute is coordinated to "
        "more than one solvent or its solvent is coordinated to more than one "
        "solute, network size >= 3.",
    )

    # solute_status_by_frame: DataFrame | None= Field(
    #     None, description="Solute status in each frame of the trajectory."
    # )

    # Solute.pairing

    solvent_pairing: dict[str, float] | None = Field(
        None, description="Fraction of each solvent coordinated to the solute."
    )

    # pairing_by_frame: DataFrame | None= Field(
    #     None, description="Solvent pairing in each frame."
    # )

    fraction_free_solvents: dict[str, float] | None = Field(
        None, description="Fraction of each solvent not coordinated to solute."
    )

    diluent_composition: dict[str, float] | None = Field(
        None, description="Fraction of diluent constituted by each solvent."
    )

    # diluent_composition_by_frame: DataFrame | None= Field(
    #     None, description="Diluent composition in each frame."
    # )

    diluent_counts: DataFrame | None = Field(
        None, description="Solvent counts in each frame."
    )

    # Solute.residence

    residence_times: dict[str, float] | None = Field(
        None,
        description="Average residence time of each solvent."
        "Calculated by 1/e cutoff on autocovariance function.",
    )

    residence_times_fit: dict[str, float] | None = Field(
        None,
        description="Average residence time of each solvent."
        "Calculated by fitting the autocovariance function to an exponential decay.",
    )

    # Solute.speciation

    speciation_fraction: DataFrame | None = Field(
        None, description="Fraction of shells of each type."
    )

    solvent_co_occurrence: DataFrame | None = Field(
        None,
        description="The actual co-occurrence of solvents divided by "
        "the expected co-occurrence in randomly distributed solvation shells."
        "i.e. given a molecule of solvent i in the shell, the probability of "
        "solvent j's presence relative to choosing a solvent at random "
        "from the pool of all coordinated solvents. ",
    )

    job_uuid: str | None = Field(
        None, description="The UUID of the flow that generated this data."
    )

    flow_uuid: str | None = Field(
        None, description="The UUID of the top level host from that job."
    )

    @classmethod
    def from_solute(
        cls,
        solute: Solute,
        job_uuid: str | None = None,
        flow_uuid: str | None = None,
    ) -> SolvationDoc:
        """Create a SolvationDoc from openmm Solute."""
        # as a dict
        props = {
            "solute_name": solute.solute_name,
            "solvent_names": list(solute.solvents.keys()),
            "is_electrolyte": True,
            "job_uuid": job_uuid,
            "flow_uuid": flow_uuid,
        }
        if hasattr(solute, "coordination"):
            for k in (
                "coordination_numbers",
                "coordinating_atoms",
                "coordination_vs_random",
            ):
                props[k] = getattr(solute.coordination, k, None)
        if hasattr(solute, "pairing"):
            for k in (
                "solvent_pairing",
                "fraction_free_solvents",
                "diluent_composition",
                "diluent_counts",
            ):
                props[k] = getattr(solute.pairing, k, None)
        if hasattr(solute, "speciation"):
            for k in ("speciation_fraction", "solvent_co_occurrence"):
                props[k] = getattr(solute.speciation, k, None)
        if hasattr(solute, "networking"):
            for k in ("network_sizes", "solute_status"):
                props[k] = getattr(solute.networking, k, None)
        if hasattr(solute, "residence"):
            for k, v in {
                "residence_times_cutoff": "residence_times",
                "residence_times_fit": "residence_times_fit",
            }.items():
                props[v] = getattr(solute.residence, k, None)

        return SolvationDoc(**props)
