"""Flows adapted from MPMorph *link to origin github repo*"""  # TODO: Add link to origin github repo

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import job, Flow, Maker, Response

# from atomate2.common.flows.eos import CommonEosMaker
from atomate2.common.jobs.eos import (
    _apply_strain_to_structure,
    MPMorphPVPostProcess,
)
from atomate2.vasp.jobs.md import MDMaker
from atomate2.vasp.sets.core import MDSetGenerator
from atomate2.forcefields.md import ForceFieldMDMaker

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from atomate2.common.jobs.eos import EOSPostProcessor
    from jobflow import Job

    from pymatgen.core import Structure


@dataclass
class EquilibriumVolumeMaker(Maker):
    """
    Equilibrate structure using NVT + EOS fitting.

    Parameters
    -----------
    name : str = "Equilibrium Volume Maker"
        Name of the flow
    md_maker : Maker
        Maker to perform NVT MD runs
    postprocessor : atomate2.common.jobs.eos.EOSPostProcessor
        Postprocessing step to fit the EOS
    min_strain : float, default = 0.5
        Minimum absolute percentage linear strain to apply to the structure
    max_attempts : int | None = 20
        Number of times to continue attempting to equilibrate the structure.
        If None, the workflow will not terminate if an equilibrated structure
        cannot be determined.
    """

    name: str = "Equilibrium Volume Maker"
    md_maker: Maker | None = None
    postprocessor: EOSPostProcessor = field(default_factory=MPMorphPVPostProcess)
    min_strain: float = 0.5
    max_attempts: int | None = 20

    @job
    def make(
        self,
        structure: Structure,
        prev_dir: str | Path | None = None,
        working_outputs: dict[str, dict] | None = None,
    ) -> Flow:
        """
        Run an NVT+EOS equilibration flow.

        Parameters
        -----------
        structure : Structure
            structure to equilibrate
        prev_dir : str | Path | None (default)
            path to copy files from
        working_outputs : dict or None
            contains the outputs of the flow as it recursively updates

        Returns
        -------
        .Flow, an MPMorph flow
        """

        if working_outputs is None:
            linear_strain = np.linspace(-0.2, 0.2, self.postprocessor.min_data_points)
            working_outputs: dict[str, dict] = {
                "relax": {key: [] for key in ("energy", "volume", "stress", "pressure")}
            }

        else:
            self.postprocessor.fit(working_outputs)
            # print("____EOS FIT PARAMS_____")
            # print(self.postprocessor.results)
            # print("_______________________")
            working_outputs = dict(self.postprocessor.results)
            working_outputs["relax"].pop(
                "pressure", None
            )  # remove pressure from working_outputs

            if (
                working_outputs["V0"] <= working_outputs["Vmax"]
                and working_outputs["V0"] >= working_outputs["Vmin"]
            ) or (
                self.max_attempts
                and (
                    len(working_outputs["relax"]["volume"])
                    - self.postprocessor.min_data_points
                )
                >= self.max_attempts
            ):
                final_structure = structure.copy()
                final_structure.scale_lattice(working_outputs["V0"])
                return final_structure

            elif working_outputs["V0"] > working_outputs["Vmax"]:
                v_ref = working_outputs["Vmax"]

            elif working_outputs["V0"] < working_outputs["Vmin"]:
                v_ref = working_outputs["Vmin"]

            eps_0 = (working_outputs["V0"] / v_ref) ** (1.0 / 3.0) - 1.0
            linear_strain = [np.sign(eps_0) * (abs(eps_0) + self.min_strain)]

        deformation_matrices = [np.eye(3) * (1.0 + eps) for eps in linear_strain]
        deformed_structures = _apply_strain_to_structure(
            structure, deformation_matrices
        )

        eos_jobs = []
        for index in range(len(deformation_matrices)):
            md_job = self.md_maker.make(
                structure=deformed_structures[index].final_structure,
                prev_dir=None,
            )
            md_job.name = (
                f"{self.name} {md_job.name} {len(working_outputs['relax']['energy'])+1}"
            )

            working_outputs["relax"]["energy"].append(md_job.output.output.energy)
            working_outputs["relax"]["volume"].append(md_job.output.structure.volume)
            working_outputs["relax"]["stress"].append(md_job.output.output.stress)
            eos_jobs.append(md_job)

        recursive = self.make(
            structure=structure,
            prev_dir=None,
            working_outputs=working_outputs,
        )

        new_eos_flow = Flow([*eos_jobs, recursive], output=recursive.output)

        return Response(replace=new_eos_flow, output=recursive.output)


@dataclass
class MPMorphMDMaker(Maker):
    """Base MPMorph flow for volume equilibration, quench, and production runs via molecular dynamics

    Calculates the equilibrium volume of a structure at a given temperature. A convergence fitting
    (optional) for the volume followed by quench (optional) from high temperature to low temperature
    and finally a production run(s) at a given temperature. Production run is broken up into multiple
    smaller steps to ensure simulation does not hit wall time limits.

    Check atomate2.vasp.flows.mpmorph for MPMorphVaspMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    convergence_md_maker : EquilibrateVolumeMaker
        MDMaker to generate the equilibrium volumer searcher
    quench_maker :  SlowQuenchMaker or FastQuenchMaker or None
        SlowQuenchMaker - MDMaker that quenchs structure from high temperature to low temperature
        FastQuenchMaker - DoubleRelaxMaker + Static that "quenchs" structure at 0K
    production_md_maker : Maker
        MDMaker to generate the production run(s)
    """

    name: str = "MP Morph md"
    convergence_md_maker: EquilibriumVolumeMaker = None  # check logic on this line
    # May need to fix next two into ForceFieldMDMakers later..)
    production_md_maker: Maker | None = None
    quench_maker: FastQuenchMaker | SlowQuenchMaker | None = None

    def make(
        self,
        structure: Structure,
        prev_dir: str | Path | None = None,
    ):
        """
        Create a flow with MPMorph molecular dynamics (and relax+static).

        By default, production run is broken up into multiple smaller steps. Converegence and
        quench are optional and may be used to equilibrate the unit cell volume (useful for
        high temperature production runs of structures extracted from Materials Project) and
        to quench the structure from high to low temperature (e.g. amorphous structures).

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous VASP calculation directory to copy output files from.

        Returns
        -------
        Flow
            A flow containing series of molecular dynamics run (and relax+static).
        """
        flow_jobs = []

        if self.convergence_md_maker is not None:
            convergence_flow = self.convergence_md_maker.make(
                structure, prev_dir=prev_dir
            )
            flow_jobs.append(convergence_flow)

            # convergence_flow only outputs a structure
            structure = convergence_flow.output

        self.production_md_maker.name = self.name + " production run"
        production_run = self.production_md_maker.make(
            structure=structure, prev_dir=prev_dir
        )
        flow_jobs.append(production_run)

        if self.quench_maker:
            quench_flow = self.quench_maker.make(
                structure=production_run.output.structure,
                prev_dir=production_run.output.dir_name,
            )
            flow_jobs += [quench_flow]

        return Flow(
            flow_jobs,
            output=production_run.output,
            name=self.name,
        )


@dataclass
class FastQuenchMaker(Maker):
    """Fast quench flow for quenching high temperature structures to 0K

    Quench's a provided structure with a single (or double) relaxation and a static calculation at 0K.
    Adapted from MPMorph Workflow

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker :  Maker
        Relax Maker
    relax_maker2 :  Maker or None
        Relax Maker for a second relaxation; useful for tighter convergence
    static_maker : Maker
        Static Maker
    """

    name: str = "fast quench"
    relax_maker: Maker = Maker
    relax_maker2: Maker | None = None
    static_maker: Maker = Maker

    def make(
        self,
        structure: Structure,
        prev_dir: str | Path | None = None,
    ) -> Flow:
        """
        Create a fast quench flow with relax and static makers.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous VASP calculation directory to copy output files from.

        Returns
        -------
        Flow
            A flow containing series of relax and static runs.
        """
        jobs: list[Job] = []

        relax1 = self.relax_maker.make(structure, prev_dir=prev_dir)
        jobs += [relax1]
        structure = relax1.output.structure
        prev_dir = relax1.output.dir_name

        if self.relax_maker2 is not None:
            relax2 = self.relax_maker2.make(structure, prev_dir=prev_dir)
            jobs += [relax2]
            structure = relax2.output.structure
            prev_dir = relax2.output.dir_name

        static = self.static_maker.make(structure, prev_dir=prev_dir)
        jobs += [static]
        return Flow(
            jobs,
            output=static.output,
            name=self.name,
        )


@dataclass
class SlowQuenchMaker(Maker):  # Works only for VASP and MLFFs
    """Slow quench flow for quenching high temperature structures to low temperature

    Quench's a provided structure with a molecular dyanmics run from a desired high temperature to
    a desired low temperature. Flow creates a series of MD runs that holds at a certain temperature
    and initiates the following MD run at a lower temperature (step-wise temperature MD runs).
    Adapted from MPMorph Workflow.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    md_maker :  Maker | None = None
        Can only be an MDMaker or ForceFieldMDMaker. Defaults to None. If None, will not work. #WORK IN PROGRESS.
    quench_start_temperature : int = 3000
        Starting temperature for quench; default 3000K
    quench_end_temperature : int = 500
        Ending temperature for quench; default 500K
    quench_temperature_step : int = 500
        Temperature step for quench; default 500K drop
    quench_nsteps : int = 1000
        Number of steps for quench; default 1000 steps
    """

    name: str = "slow quench"
    md_maker: Maker | None = None
    quench_start_temperature: int = 3000
    quench_end_temperature: int = 500
    quench_temperature_step: int = 500
    quench_nsteps: int = 1000

    def make(
        self, structure: Structure, prev_dir: str | Path | None = None
    ) -> (
        Flow
    ):  # TODO : main objective: modified to work with other MD codes; Only works for VASP and MLFF_MD now.
        """
        Create a slow quench flow with md maker.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous VASP calculation directory to copy output files from.

        Returns
        -------
        Flow
            A flow containing series of relax and static runs.
        """

        md_jobs = []
        for temp in np.arange(
            self.quench_start_temperature,
            self.quench_end_temperature,
            -self.quench_temperature_step,
        ):

            prev_dir = (
                None
                if temp == self.quench_start_temperature
                else md_jobs[-1].output.dir_name
            )

            md_job = self.call_md_maker(
                structure, prev_dir, temp=temp, nsteps=self.quench_nsteps
            )
            """ # Logic of call_md_maker is to substitute the following code:
            if isinstance(self.md_maker, MDMaker):
                md_job = MDMaker(
                    input_set_generator=MDSetGenerator(
                        start_temp=temp,
                        end_temp=temp,
                        nsteps=self.quench_tempature_setup["nsteps"],
                    )
                ).make(structure, prev_dir)

            elif isinstance(self.md_maker, ForceFieldMDMaker):
                self.md_maker = self.md_maker.update_kwargs(
                    update={
                        "temperature": temp,
                        "nsteps": self.quench_tempature_setup["nsteps"],
                    }
                )
                md_job = self.md_maker.make(
                    structure,
                    prev_dir,
                )
            else:
                raise ValueError(
                    "***WORK IN PROGRESS*** md_maker must be an MDMaker or ForceFieldMDMaker."
                )"""

            md_jobs.append(md_job)

            structure = md_job.output.structure

        return Flow(
            md_jobs,
            output=md_jobs[-1].output,
            name=self.name,
        )

    def call_md_maker(self, structure, prev_dir, temp, nsteps):
        if not (
            isinstance(self.md_maker, MDMaker)
            or isinstance(self.md_maker, ForceFieldMDMaker)
        ):
            raise ValueError(
                "***WORK IN PROGRESS*** md_maker must be an MDMaker or ForceFieldMDMaker."
            )
        return self.md_maker.make(structure, prev_dir)


@dataclass
class AmorphousLimitMaker(
    Maker
):  # TODO: 1st interation only, needs to be updated to be consistent with format of other Makers
    """Flow to create an amorphous structure from a desired stiochiometry, then perform
    MPMorph molecular dynamics runs on top of it.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    mpmorph_maker :  MPMorphMDMaker
        MDMaker to generate the molecular dynamics jobs specifically for MPMorph workflow
    """

    name: str = "Amorphous Limit Maker"
    mpmorph_maker: MPMorphMDMaker = MPMorphMDMaker

    def make(
        self,
        composition: Union[str, Composition] = None,
        prev_dir: str | Path | None = None,
    ) -> Flow:
        """
        Create a flow to generate an amorphous structure from a desired stiochiometry,
        then perform MPMorph molecular dynamics workflow runs on top of it.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous VASP calculation directory to copy output files from.

        Returns
        -------
        Flow
            A flow containing series of rescaled volume molecular dynamics runs, EOS fitted,
            then production run at the equilibirum volume.
        """

        original_amorphous_structure = get_random_packed(composition)
        mpmorph_flow = self.mpmorph_maker.make(
            structure=original_amorphous_structure, prev_dir=prev_dir
        )
        return Flow(
            [mpmorph_flow],
            name=self.name,
        )


### AmorphousMaker class from original MPMorph github repo. Plans to be integrated into atomate2
from __future__ import division

import os
import shutil
from collections import OrderedDict
from typing import List, Optional, Union

import numpy as np
from pymatgen.core import Composition, Structure
from pymatgen.ext.matproj import MPRester
from pymatgen.io.vasp.inputs import Poscar


class AmorphousMaker(object):
    def __init__(
        self,
        el_num_dict: dict,
        box_scale: Union[float, List[float]],
        tol: float = 2.0,
        packmol_path: str = "packmol",
        clean: bool = True,
        xyz_paths: List = None,
        time_seed: bool = True,
    ):
        """
        Class for generating initial constrained-random packed structures for the
        simulation of amorphous or liquid structures. This is a wrapper for "packmol" package.
        Only works for cubic boxes for now.
        Args:
            el_num_dict (dict): dictionary of number of atoms of each species. If
                number of molecules is specified, an xyz file with the same name
                needs to be provided as xyz_paths.
                e.g. {"V":22, "Li":10, "O":75, "B":10}
                e.g. {"H2O": 20}
            box_scale (float) or (numpy array): all lattice vectors are multiplied with this.
                e.g. if one scalar value is given, it is the edge length of a cubic
                simulation box (e.g. if np.array([1.2, 0.9, 1.0]) is given, the unit
                lattice vectors will be multiplied with this.
            tol (float): tolerance factor for how close the atoms can get (angstroms).
                e.g. tol = 2.0 angstroms
            packmol_path (str): path to the packmol executable
            clean (bool): whether the intermedite files generated are deleted.
            xyz_paths (list): list of paths (str) to xyz files correpsonding to
                molecules, if given so in el_num_dict. File names must match the
                molecule formula.
            time_seed (bool): whether to generate a random seed based on system time
        """
        self.el_num_dict = el_num_dict
        self.box_scale = box_scale
        self.tol = tol
        self._lattice = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        self._structure = None
        self._el_dict = None
        self.packmol_path = packmol_path
        self.clean = clean
        self.xyz_paths = xyz_paths
        self.time_seed = time_seed
        if self.xyz_paths:
            assert len(self.xyz_paths) == len(self.el_num_dict.keys())
            self.clean = False

    def __repr__(self):
        return (
            "AmorphousMaker: generates constrained-random packed initial structure for"
            " MD using packmol."
        )

    @property
    def box(self):
        """
        Returns: box vectors scaled with box_scale
        """
        return (np.array(self._lattice) * self.box_scale).tolist()

    @property
    def random_packed_structure(self):
        """
        Returns: A constrained-random packed Structure object
        """
        self._el_dict = self.call_packmol()
        self._structure = self.get_structure(self._el_dict, self.box)
        return self._structure

    def call_packmol(self):
        """
        Returns:
            A dict of coordinates of atoms for each element type
            e.g. {'V': [[4.969925, 8.409291, 5.462153], [9.338829, 9.638388, 9.179811], ...]
                  'Li': [[5.244308, 8.918049, 1.014577], [2.832759, 3.605796, 2.330589], ...]}
        """

        # this ensures periodic boundaries don't cause problems
        pm_l = self.tol / 2
        pm_h = self.box_scale - self.tol / 2
        try:
            len(pm_h)
        except:
            pm_h = [pm_h for i in range(3)]

        with open("packmol.input", "w") as f:
            f.write(
                "tolerance " + str(self.tol) + "\nfiletype xyz\noutput mixture.xyz\n"
            )
            for el in self.el_num_dict:
                f.write(
                    "structure "
                    + el
                    + ".xyz\n"
                    + "  number "
                    + str(self.el_num_dict[el])
                    + "\n  inside box"
                    + 3 * (" " + str(pm_l))
                    + (" " + str(pm_h[0]))
                    + (" " + str(pm_h[1]))
                    + (" " + str(pm_h[2]))
                    + "\nend structure\n\n"
                )

            if self.time_seed:
                f.write("seed -1\n")

            if self.xyz_paths:
                for path in self.xyz_paths:
                    try:
                        shutil.copy2(path, "./")
                    except:
                        pass
            else:
                for el in self.el_num_dict.keys():
                    with open(el + ".xyz", "w") as f:
                        f.write("1\ncomment\n" + el + " 0.0 0.0 0.0\n")

        try:
            os.system(self.packmol_path + " < packmol.input")
        except:
            raise OSError("packmol cannot be found!")
        if self.clean:
            for el in self.el_num_dict.keys():
                os.system("rm " + el + ".xyz")
            os.system("rm packmol.input")
        return self.xyz_to_dict("mixture.xyz")

    def xyz_to_dict(self, filename: str):
        """
        This is a generic xyz to dictionary convertor.
        Used to get the structure from packmol output.
        """
        with open(filename, "r") as f:
            lines = f.readlines()
            N = int(lines[0].rstrip("\n"))
            el_dict = {}
            for line in lines[2:]:
                l = line.rstrip("\n").split()
                if l[0] in el_dict:
                    el_dict[l[0]].append([float(i) for i in l[1:]])
                else:
                    el_dict[l[0]] = [[float(i) for i in l[1:]]]
        if N != sum([len(x) for x in el_dict.values()]):
            raise ValueError("Inconsistent number of atoms")
        self._el_dict = OrderedDict(el_dict)
        if self.clean:
            os.system("rm " + filename)
        return self._el_dict

    @staticmethod
    def get_structure(el_dict: dict, lattice: List[List]):
        """
        Args:
            el_dict (dict): coordinates of atoms for each element type
            e.g. {'V': [[4.969925, 8.409291, 5.462153], [9.338829, 9.638388, 9.179811], ...]
                  'Li': [[5.244308, 8.918049, 1.014577], [2.832759, 3.605796, 2.330589], ...]}
            lattice (list): is the lattice in the form of [[x1,x2,x3],[y1,y2,y3],[z1,z2,z3]]
        Returns: pymatgen Structure
        """
        species = []
        coords = []
        for el in el_dict.keys():
            for atom in el_dict[el]:
                species.append(el)
                coords.append(atom)
        return Structure(lattice, species, coords, coords_are_cartesian=True)

    def get_poscar(self):
        return Poscar(self.random_packed_structure)

    @staticmethod
    def xyzdict_to_poscar(el_dict: dict, lattice: List[List], filepath: str = "POSCAR"):
        """
        Generates XYZ file from element coordinate dictionary and lattice
        Args:
            el_dict (dict): coordinates of atoms for each element type
            e.g. {'V': [[4.969925, 8.409291, 5.462153], [9.338829, 9.638388, 9.179811], ...]
                  'Li': [[5.244308, 8.918049, 1.014577], [2.832759, 3.605796, 2.330589], ...]}
            lattice (list): is the lattice in the form of [[x1,x2,x3],[y1,y2,y3],[z1,z2,z3]]
            filepath (str): path to POSCAR to be generated
        Returns:
            writes a POSCAR file
        """
        with open(filepath, "w") as f:
            f.write("Parsed form XYZ file\n")
            f.write("1.0\n")
            for vec in lattice:
                f.write(" ".join([str(v) for v in vec]) + "\n")
            el_dict = OrderedDict(el_dict)
            for key in el_dict.keys():
                f.write(key + " ")
            f.write("\n")
            for key in el_dict.keys():
                f.write(str(len(el_dict[key])) + " ")
            f.write("\nCartesian\n")
            for key in el_dict.keys():
                for atom in el_dict[key]:
                    f.write(" ".join([str(i) for i in atom]) + "\n")


def get_random_packed(
    composition: Union[Composition, str],
    add_specie=None,
    target_atoms: int = 100,
    vol_per_atom: float = None,
    vol_exp: float = 1.0,
    modify_species: dict = None,
    use_time_seed: bool = True,
    mpr: Optional[MPRester] = None,
):
    """
    Helper method to use the AmorphousMaker to generate a randomly packed unit cell. If the volume (per atom) of the
    unit cell is not provided, the volume per atom will be predicted using structures from the materials project.

    :param composition: Formula or composition of the desired unit cell
    :param add_specie:
    :param target_atoms: Desired number of atoms in the unit cell. Generated unit cell will have at least this many
    atoms, although may be greater if the number of atoms in the specified formula/composition is evenly
    divide target_atoms.
    :param vol_per_atom: Volume of the unit cell in cubic Angstroms per atom (A^3/atom)
    :param vol_exp: Factor to multiply the volume by.
    :param modify_species:
    :param use_time_seed: Whether or not to use a time based random seed. Defaults to true
    :param mpr: custom MPRester object if additional specifications needed (such as API or endpoint)
    :return:
    """

    if type(composition) == str:
        composition = Composition(composition)
    if type(add_specie) == str:
        add_specie = Composition(add_specie)

    if vol_per_atom is None:
        if mpr is None:
            mpr = MPRester()

        comp_entries = mpr.get_entries(composition.reduced_formula, inc_structure=True)
        if len(comp_entries) > 0:
            vols = np.min(
                [
                    entry.structure.volume / entry.structure.num_sites
                    for entry in comp_entries
                ]
            )
        else:
            # Find all Materials project entries containing the elements in the
            # desired composition to estimate starting volume.
            _entries = mpr.get_entries_in_chemsys(
                [str(el) for el in composition.elements], inc_structure=True
            )
            entries = []
            for entry in _entries:
                if set(entry.structure.composition.elements) == set(
                    composition.elements
                ):
                    entries.append(entry)
                if len(entry.structure.composition.elements) >= 2:
                    entries.append(entry)

            vols = [
                entry.structure.volume / entry.structure.num_sites for entry in entries
            ]
        vol_per_atom = np.mean(vols)

    # Find total composition of atoms in the unit cell
    formula, factor = composition.get_integer_formula_and_factor()
    integer_composition = Composition(formula)
    full_cell_composition = integer_composition * np.ceil(
        target_atoms / integer_composition.num_atoms
    )
    if add_specie is not None:
        full_cell_composition += add_specie

    # Generate dict of elements and amounts for AmorphousMaker
    structure = {}
    for el in full_cell_composition:
        structure[str(el)] = int(full_cell_composition.element_composition.get(el))

    if modify_species is not None:
        for i, v in modify_species.items():
            structure[i] += v
    # use packmol to get a random configured structure
    packmol_path = os.environ["PACKMOL_PATH"]
    amorphous_maker_params = {
        "box_scale": (vol_per_atom * full_cell_composition.num_atoms * vol_exp)
        ** (1 / 3),
        "packmol_path": packmol_path,
        "xyz_paths": None,
        "time_seed": use_time_seed,
    }

    glass = AmorphousMaker(structure, **amorphous_maker_params)
    structure = glass.random_packed_structure
    return structure
