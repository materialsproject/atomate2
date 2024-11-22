"""Core definitions of a JDFTx calculation document."""

# mypy: ignore-errors

import logging
from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, Field
from pymatgen.core.structure import Structure
from pymatgen.io.jdftx.inputs import JDFTXInfile
from pymatgen.io.jdftx.outputs import JDFTXOutfile
from pymatgen.io.jdftx.joutstructure import JOutStructure
from pymatgen.core.trajectory import Trajectory

from atomate2.jdftx.schemas.enums import (
    TaskType, 
    CalcType, 
    JDFTxStatus, 
    SolvationType
)

__author__ = "Cooper Tezak <cote3804@colorado.edu>"
logger = logging.getLogger(__name__)

class Convergence(BaseModel):
    """Schema for calculation convergence"""
    converged: bool = Field(
        True,
        description="Whether the JDFTx calculation converged"
    )
    geom_converged: Optional[bool] = Field(
        True,
        description="Whether the ionic/lattice optimization converged"
    )
    elec_converged: Optional[bool] = Field(
        True,
        description="Whether the last electronic optimization converged"
    )
    geom_converged_reason: Optional[str] = Field(
        None,
        description="Reason ionic/lattice convergence was reached"
    )
    elec_converged_reason: Optional[str] = Field(
        None,
        description="Reason electronic convergence was reached"
    )

    @classmethod
    def from_jdftxoutput(cls, jdftxoutput: JDFTXOutfile):
        converged = jdftxoutput.is_converged
        jstrucs = jdftxoutput.jstrucs
        geom_converged = jstrucs.geom_converged
        geom_converged_reason = jstrucs.geom_converged_reason
        elec_converged = jstrucs.elec_converged
        elec_converged_reason = jstrucs.elec_converged_reason
        return cls(
            converged=converged,
            geom_converged=geom_converged,
            geom_converged_reason=geom_converged_reason,
            elec_converged=elec_converged,
            elec_converged_reason=elec_converged_reason
        )

class RunStatistics(BaseModel):
    """JDFTx run statistics."""
    total_time: Optional[float] = Field(
        0, 
        description="Total wall time for this calculation"
    )

    @classmethod
    def from_jdftxoutput(cls, jdftxoutput:JDFTXOutfile):
        """Initialize RunStatistics from JDFTXOutfile"""
        if hasattr(jdftxoutput, "t_s"):
            t_s = jdftxoutput.t_s
        else:
            t_s = None

        return cls(
            total_time=t_s
        )

class CalculationInput(BaseModel):
    """Document defining JDFTx calculation inputs."""

    structure: Structure = Field(
        None, description="input structure to JDFTx calculation"
    )
    jdftxinfile: dict = Field(
        None, 
        description="input tags in JDFTx in file"
    )

    @classmethod
    def from_jdftxinput(cls, jdftxinput: JDFTXInfile) -> "CalculationInput":
        """
        Create a JDFTx InputDoc schema from a JDFTXInfile object.

        Parameters
        ----------
        jdftxinput
            A JDFTXInfile object.

        Returns
        -------
        CalculationInput
            The input document.
        """
        return cls(
            structure=jdftxinput.structure,
            jdftxinfile=jdftxinput.as_dict(),
        )


class CalculationOutput(BaseModel):
    """Document defining JDFTx calculation outputs."""

    structure: Optional[Structure] = Field(
        None,
        description="optimized geometry of the structure after calculation",
    )
    parameters: Optional[dict] = Field(
        None,
        description="JDFTXOutfile dictionary from last JDFTx run",
    )
    forces: Optional[list] = Field(
        None, 
        description="forces from last ionic step"
    )
    energy: float = Field(
        None, 
        description="Final energy"
    )
    energy_type: str = Field(
        "F",
        description="Type of energy returned by JDFTx (e.g., F, G)"
    )
    mu: float = Field(
        None, 
        description="Fermi level of last electronic step"
    )
    lowdin_charges: Optional[list] = Field(
        None, 
        description="Lowdin charges from last electronic optimizaiton"
    )
    total_charge: float = Field(
        None,
        description="Total system charge from last electronic step in number of electrons"
    )
    stress: Optional[list[list]] = Field(
        None, 
        description="Stress from last lattice optimization step"
    )
    cbm: Optional[float] = Field(
        None, 
        description="Conduction band minimum / LUMO from last electronic optimization"
    )
    vbm: Optional[float] = Field(
        None,
        description="Valence band maximum /HOMO from last electonic optimization"
    )
    trajectory: Trajectory = Field(
        None,
        description="Ionic trajectory from last JDFTx run"
    ),
    @classmethod
    def from_jdftxoutput(cls, jdftxoutput: JDFTXOutfile, **kwargs) -> "CalculationOutput":
        """
        Create a JDFTx output document from a JDFTXOutfile object.

        Parameters
        ----------
        jdftxoutput
            A JDFTXOutfile object.

        Returns
        -------
        CalculationOutput
            The output document.
        """
        optimized_structure: JOutStructure = jdftxoutput.structure
        if hasattr(optimized_structure, "forces"):
            forces = optimized_structure.forces.tolist()
        else:
            forces = None
        if hasattr(optimized_structure, "stress"):
            if optimized_structure.stress == None:
                stress = None
            else:
                stress = optimized_structure.stress.tolist()
        else:
            stress = None
        energy = jdftxoutput.e
        energy_type = jdftxoutput.eopt_type
        mu = jdftxoutput.mu
        lowdin_charges = optimized_structure.charges
        # total charge in number of electrons (negative of oxidation state)
        total_charge = jdftxoutput.total_electrons_uncharged - jdftxoutput.total_electrons
        cbm = jdftxoutput.lumo
        vbm = jdftxoutput.homo
        if kwargs.get("store_trajectory", True) == True:
            trajectory = jdftxoutput.trajectory
        else:
            trajectory = None


        return cls(
            structure=optimized_structure, 
            forces=forces,
            energy=energy,
            energy_type=energy_type,
            mu=mu,
            lowdin_charges=lowdin_charges,
            total_charge=total_charge,
            stress=stress,
            cbm=cbm,
            vbm=vbm,
            trajectory=trajectory,
            parameters=jdftxoutput.to_dict(),
        )

class Calculation(BaseModel):
    """Full JDFTx calculation inputs and outputs."""

    dir_name: str = Field(
        None, description="The directory for this JDFTx calculation"
    )
    input: CalculationInput = Field(
        None, description="JDFTx input settings for the calculation"
    )
    output: CalculationOutput = Field(
        None, description="The JDFTx calculation output document"
    )
    converged: Convergence = Field(
        None, description="JDFTx job conversion information"
    )
    run_stats: RunStatistics = Field(
        0,
        description="Statistics for the JDFTx run"
    )
    calc_type: CalcType = Field(
        None,
        description="Calculation type (e.g. PBE)"
    )
    task_type: TaskType = Field(
        None,
        description="Task type (e.g. Lattice Optimization)"
    )
    solvation_type: SolvationType = Field(
        None, 
        description="Type of solvation model used (e.g. LinearPCM CANDLE)"
    )

    @classmethod
    def from_files(
        cls,
        dir_name: Union[Path, str],
        jdftxinput_file: Union[Path, str],
        jdftxoutput_file: Union[Path, str],
        jdftxinput_kwargs: Optional[dict] = None,
        jdftxoutput_kwargs: Optional[dict] = None,
        # task_name  # do we need task names? These are created by Custodian
    ) -> "Calculation":
        """
        Create a JDFTx calculation document from a directory and file paths.

        Parameters
        ----------
        dir_name
            The directory containing the JDFTx calculation outputs.
        jdftxinput_file
            Path to the JDFTx in file relative to dir_name.
        jdftxoutput_file
            Path to the JDFTx out file relative to dir_name.
        jdftxinput_kwargs
            Additional keyword arguments that will be passed to the
            :obj:`.JDFTXInFile.from_file` method
        jdftxoutput_kwargs
            Additional keyword arguments that will be passed to the
            :obj:`.JDFTXOutFile.from_file` method

        Returns
        -------
        Calculation
            A JDFTx calculation document.
        """
        jdftxinput_file = dir_name / jdftxinput_file
        jdftxoutput_file = dir_name / jdftxoutput_file

        jdftxinput_kwargs = jdftxinput_kwargs if jdftxinput_kwargs else {}
        jdftxinput = JDFTXInfile.from_file(jdftxinput_file)

        jdftxoutput_kwargs = jdftxoutput_kwargs if jdftxoutput_kwargs else {}
        jdftxoutput = JDFTXOutfile.from_file(jdftxoutput_file)

        input_doc = CalculationInput.from_jdftxinput(jdftxinput,  **jdftxinput_kwargs)
        output_doc = CalculationOutput.from_jdftxoutput(jdftxoutput, **jdftxoutput_kwargs)
        logging.log(logging.DEBUG, f"{output_doc}")
        converged = Convergence.from_jdftxoutput(jdftxoutput)
        run_stats = RunStatistics.from_jdftxoutput(jdftxoutput)

        calc_type = _calc_type(output_doc)
        task_type = _task_type(output_doc)
        solvation_type = _solvation_type(input_doc)

        return cls(
            dir_name=str(dir_name),
            input=input_doc,
            output=output_doc,
            converged=converged,
            run_stats=run_stats,
            calc_type=calc_type,
            task_type=task_type,
            solvation_type=solvation_type,
        )


def _task_type(
    outputdoc: CalculationOutput,
) -> TaskType:
    """Return TaskType for JDFTx calculation."""
    jdftxoutput: dict = outputdoc.parameters
    if jdftxoutput.get("geom_opt") == False:
        return TaskType("Single Point")
    else:
        if jdftxoutput.get("geom_opt_type") == "lattice":
            return TaskType("Lattice Optimization")
        elif jdftxoutput.get("geom_opt_type") == "ionic":
            return TaskType("Ionic Optimization")
    #TODO implement MD and frequency task types. Waiting on output parsers

    return TaskType("Unknown")

def _calc_type(
        outputdoc: CalculationOutput ,
) -> CalcType:
    jdftxoutput = outputdoc.parameters
    xc = jdftxoutput.get("xc_func", None)
    return CalcType(xc)

def _solvation_type(
        inputdoc: CalculationInput
) -> SolvationType:
    jdftxinput: JDFTXInfile = inputdoc.jdftxinfile
    fluid = jdftxinput.get("fluid", None)    
    if fluid == None:
        return SolvationType("None")
    else:
        fluid_solvent = jdftxinput.get("pcm-variant")
        fluid_type = fluid.get("type")
        solvation_type = f"{fluid_type} {fluid_solvent}"
        return SolvationType(solvation_type)
