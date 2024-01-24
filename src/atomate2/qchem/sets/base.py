"""Module defining base QChem input set and generator."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from monty.io import zopen
from pymatgen.io.core import InputGenerator, InputSet
from pymatgen.io.qchem.inputs import QCInput
from pymatgen.io.qchem.utils import lower_and_check_unique

if TYPE_CHECKING:
    from pymatgen.core.structure import Molecule

# from pymatgen.io.qchem.sets import QChemDictSet

__author__ = "Alex Ganose, Ryan Kingsbury, Rishabh D Guha"
__copyright__ = "Copyright 2018-2022, The Materials Project"
__version__ = "0.1"

# _BASE_QCHEM_SET =
# loadfn(resource_filename("atomate2.qchem.sets", "BaseQchemSet.yaml"))

__all__ = ["QCInputSet", "QCInputGenerator"]


class QCInputSet(InputSet):
    """
    A class to represent a QChem input file as a QChem InputSet.

    Parameters
    ----------
    qcinput
        A QCInput object
    optional_files
        Any other optional input files supplied as a dict of ``{filename: object}``.
        The objects should follow standard pymatgen conventions in
        implementing an ``as_dict()`` and ``from_dict()`` method.
    """

    def __init__(
        self,
        qcinput: QCInput,
        optional_files: dict | None = None,
    ) -> None:
        self.qcinput = qcinput
        self.optional_files = {} if optional_files is None else optional_files

    def write_input(
        self,
        directory: str | Path,
        overwrite: bool = True,
    ) -> None:
        """
        Write QChem input file to directory.

        Parameters
        ----------
        directory
            Directory to write input files to.
        overwrite
            Whether to overwrite an input file if it already exists.
        """
        directory = Path(directory)
        os.makedirs(directory, exist_ok=True)

        inputs = {"Input_Dict": self.qcinput}
        inputs.update(self.optional_files)

        for k, v in inputs.items():
            if v is not None and (overwrite or not (directory / k).exists()):
                with zopen(directory / k, "wt") as f:
                    f.write(v.__str__())
            elif not overwrite and (directory / k).exists():
                raise FileExistsError(f"{directory / k} already exists.")

    @staticmethod
    def from_directory(
        directory: str | Path, optional_files: dict = None
    ) -> QCInputSet:
        """
        Load a set of QChem inputs from a directory.

        Parameters
        ----------
        directory
            Directory to read QChem inputs from.
        optional_files
            Optional files to read in as well as a dict of {filename: Object class}.
            Object class must have a static/class method from_file
        """
        directory = Path(directory)
        objs = {"Input_Dict": QCInput}

        inputs = {}
        for name, obj in objs.items():
            if (directory / name).exists():
                inputs[name.lower()] = obj.from_file(directory / name)

        optional_inputs = {}
        if optional_files is not None:
            for name, obj in optional_files.items():
                optional_inputs[name] = obj.from_file(directory / name)

        return QCInputSet(**inputs, optional_files=optional_inputs)

        # Todo
        # Implement is_valid property


@dataclass
class QCInputGenerator(InputGenerator):
    """
    A class to generate QChem input set.

    Parameters
    ----------
    job_type : str
        QChem job type to run. Valid options are "opt" for optimization,
        "sp" for single point, "freq" for frequency calculation, or "force" for
        force evaluation.

    basis_set : str
        Basis set to use. For example, "def2-tzvpd".

    scf_algorithm : str
        Algorithm to use for converging the SCF. Recommended choices are
        "DIIS", "GDM", and "DIIS_GDM". Other algorithms supported by Qchem's
        GEN_SCFMAN module will also likely perform well.
        Refer to the QChem manual for further details.

    dft_rung : int
        Select the DFT functional among 5 recommended levels of theory,
        in order of increasing accuracy/cost. 1 = B3LYP, 2=B3lYP+D3, 3=ωB97X-D,
        4=ωB97X-V, 5=ωB97M-V. (Default: 4)
        To set a functional not given by one of the above, set the overwrite_inputs
        argument to {"method":"<NAME OF FUNCTIONAL>"}
        **Note that the "rungs" in this argument do NOT correspond to rungs on "Jacob's
        Ladder of Density Functional Approximations"**

    pcm_dielectric : float
        Dielectric constant to use for PCM implicit solvation model. (Default: None)

    smd_solvent : str
        Solvent to use for SMD implicit solvation model. (Default: None)
        Examples include "water", "ethanol", "methanol", and "acetonitrile".
        Refer to the QChem manual for a complete list of solvents available.
        To define a custom solvent, set this argument to "custom" and
        populate custom_smd with the necessary parameters.

        **Note that only one of smd_solvent and pcm_dielectric may be set.**

    custom_smd : str
        List of parameters to define a custom solvent in SMD. (Default: None)
        Must be given as a string of seven comma separated values
        in the following order:
        "dielectric, refractive index, acidity, basicity,
        surface tension, aromaticity, electronegative halogenicity"
        Refer to the QChem manual for further details.

    opt_variables : dict
        A dictionary of opt sections, where each opt section is a key
        and the corresponding values are a list of strings. Strings must be formatted
        as instructed by the QChem manual.
        The different opt sections are: CONSTRAINT, FIXED, DUMMY, and CONNECT.
        Ex.
        opt =
        {"CONSTRAINT": ["tors 2 3 4 5 25.0", "tors 2 5 7 9 80.0"], "FIXED": ["2 XY"]}

    scan_variables : dict
        A dictionary of scan variables. Because two constraints of the
        same type are allowed (for instance, two torsions or two bond stretches),
        each TYPE of variable (stre, bend, tors) should be its own key in the dict,
        rather than each variable. Note that the total number of variable
        (sum of lengths of all lists) CANNOT be more than two.
        Ex. scan_variables =
        {"stre": ["3 6 1.5 1.9 0.1"], "tors": ["1 2 3 4 -180 180 15"]}

    max_scf_cycles : int
        Maximum number of SCF iterations. (Default: 100)

    geom_opt_max_cycles : int
        Maximum number of geometry optimization iterations. (Default: 200)

    plot_cubes : bool
        Whether to write CUBE files of the electron density. (Default: False)

    nbo_params : dict
        A dict containing the desired NBO params. Note that a key:value pair of
        "version":7 will trigger NBO7 analysis.
        Otherwise, NBO5 analysis will be performed,
        including if an empty dict is passed.
        Besides a key of "version", all other key:value pairs
        will be written into the $nbo section of the QChem input file. (Default: False)

    new_geom_opt : dict
        A dict containing parameters for the $geom_opt section of the QChem
        input file, which control the new geometry optimizer
        available starting in version 5.4.2.
        Further note that even passing an empty dictionary
        will trigger the new optimizer.
        (Default: False)

    overwrite_inputs : dict
        Dictionary of QChem input sections to add or overwrite variables.
        The currently available sections (keys) are rem, pcm, solvent, smx, opt,
        scan, van_der_waals, and plots. The value of each key is a dictionary
        of key value pairs relevant to that section.
        For example, to add a new variable to the rem section that sets
        symmetry to false, use
        overwrite_inputs = {"rem": {"symmetry": "false"}}
        **Note that if something like basis is added to the rem dict it will overwrite
        the default basis.**
        **Note that supplying a van_der_waals section here will automatically modify
        the PCM "radii" setting to "read".**
        **Note that all keys must be given as strings, even when they are numbers!**

    config_dict : dict
        The config dictionary to use containing the base input set settings.
    """

    def __init__(
        self,
        job_type: str,
        basis_set: str,
        scf_algorithm: str,
        dft_rung: int = 4,
        pcm_dielectric: float | None = None,
        smd_solvent: str | None = None,
        custom_smd: str | None = None,
        opt_variables: dict[str, list] | None = None,
        scan_variables: dict[str, list] | None = None,
        max_scf_cycles: int = 100,
        geom_opt_max_cycles: int = 200,
        plot_cubes: bool = False,
        nbo_params: dict | None = None,
        new_geom_opt: dict | None = None,
        overwrite_inputs: dict | None = None,
        vdw_mode: Literal["atomic", "sequential"] = "atomic",
        rem_dict: dict | None = None,
        opt_dict: dict | None = None,
        pcm_dict: dict | None = None,
        solv_dict: dict | None = None,
        smx_dict: dict | None = None,
        scan_dict: dict | None = None,
        vdw_dict: dict | None = None,
        plots_dict: dict | None = None,
        nbo_dict: dict | None = None,
        geom_opt_dict: dict | None = None,
    ) -> None:
        self.job_type = job_type
        self.basis_set = basis_set
        self.scf_algorithm = scf_algorithm
        self.dft_rung = dft_rung
        self.pcm_dielectric = pcm_dielectric
        self.smd_solvent = smd_solvent
        self.custom_smd = custom_smd
        self.opt_variables = opt_variables
        self.scan_variables = scan_variables
        self.max_scf_cycles = max_scf_cycles
        self.geom_opt_max_cycles = geom_opt_max_cycles
        self.plot_cubes = plot_cubes
        self.nbo_params = nbo_params
        self.new_geom_opt = new_geom_opt
        self.overwrite_inputs = overwrite_inputs
        self.vdw_mode = vdw_mode

        pcm_defaults = {
            "heavypoints": "194",
            "hpoints": "194",
            "radii": "uff",
            "theory": "cpcm",
            "vdwscale": "1.1",
        }

        plots_defaults = {"grid_spacing": "0.05", "total_density": "0"}

        opt_dict = {} if self.opt_variables is None else self.opt_variables

        scan_dict = {} if self.scan_variables is None else self.scan_variables

        pcm_dict = pcm_dict or {}
        solv_dict = solv_dict or {}
        smx_dict = smx_dict or {}
        vdw_dict = vdw_dict or {}
        plots_dict = plots_dict or {}
        rem_dict = {
            "job_type": self.job_type,
            "basis": self.basis_set,
            "max_scf_cycles": str(self.max_scf_cycles),
            "gen_scfman": "true",
            "xc_grid": "3",
            "thresh": "14",
            "s2thresh": "16",
            "scf_algorithm": self.scf_algorithm,
            "resp_charges": "true",
            "symmetry": "false",
            "sym_ignore": "true",
        }

        func_list = ["b3lyp", "b3lyp", "wb97xd", "wb97xv", "wb97mv"]
        qc_method = {i + 1: e for i, e in enumerate(func_list)}

        if qc_method.get(self.dft_rung) is not None:
            rem_dict["method"] = qc_method.get(self.dft_rung)
        else:
            raise ValueError("dft_rung should be between 1 and 5!")

        if self.dft_rung == 2:
            rem_dict["dft_D"] = "D3_BJ"

        if self.job_type.lower() in ["opt", "ts", "pes_scan"]:
            rem_dict["geom_opt_max_cycles"] = str(self.geom_opt_max_cycles)

        if self.pcm_dielectric is not None and self.smd_solvent is not None:
            raise ValueError("Only one of pcm or smd may be used for solvation.")

        if self.pcm_dielectric is not None:
            pcm_dict = pcm_defaults
            solv_dict["dielectric"] = self.pcm_dielectric
            rem_dict["solvent_method"] = "pcm"

        if self.smd_solvent is not None:
            if self.smd_solvent == "custom":
                smx_dict["solvent"] = "other"
            else:
                smx_dict["solvent"] = self.smd_solvent
            rem_dict["solvent_method"] = "smd"
            rem_dict["ideriv"] = "1"
            if self.smd_solvent in ("custom", "other") and self.custom_smd is None:
                raise ValueError(
                    "A user-defined SMD requires passing custom_smd,"
                    "a string of seven comma separated values in the following order: "
                    "dielectric, refractive index, acidity, basicity, surface tension,"
                    "aromaticity, electronegative halogenicity"
                )

        if self.plot_cubes:
            plots_dict = plots_defaults
            rem_dict["plots"] = "true"
            rem_dict["make_cube_files"] = "true"

        nbo_dict = self.nbo_params
        if self.nbo_params is not None:
            rem_dict["nbo"] = "true"
            if "version" in self.nbo_params:
                if self.nbo_params["version"] == 7:
                    rem_dict["nbo_external"] = "true"
                else:
                    raise RuntimeError(
                        "nbo params version should only be set to 7! Exiting..."
                    )
            nbo_dict = {}
            for key in self.nbo_params:
                if key != "version":
                    nbo_dict[key] = self.nbo_params[key]

        geom_opt_dict = self.new_geom_opt
        if self.new_geom_opt is not None:
            rem_dict["geom_opt2"] = "3"
            if "maxiter" in self.new_geom_opt:
                if self.new_geom_opt["maxiter"] != str(self.geom_opt_max_cycles):
                    raise RuntimeError(
                        "Max # of optimization cycles must be the same! Exiting..."
                    )
            else:
                self.new_geom_opt["maxiter"] = str(self.geom_opt_max_cycles)
            geom_opt_dict = {}
            for key in self.new_geom_opt:
                geom_opt_dict[key] = self.new_geom_opt[key]

        self.rem_dict = rem_dict
        self.opt_dict = opt_dict
        self.pcm_dict = pcm_dict
        self.solv_dict = solv_dict
        self.smx_dict = smx_dict
        self.scan_dict = scan_dict
        self.vdw_dict = vdw_dict
        self.plots_dict = plots_dict
        self.nbo_dict = nbo_dict
        self.geom_opt_dict = geom_opt_dict

    def get_input_set(self, molecule: Molecule = None) -> QCInputSet:
        """
        Return a QChem InputSet for a molecule.

        Parameters
        ----------
        molecule: Molecule
            Pymatgen representation of a molecule for which the QCInputSet
            will be generated

        Returns
        -------
        QchemInputSet
            A QChem input set
        """
        if self.overwrite_inputs:
            for sub, sub_dict in self.overwrite_inputs.items():
                if sub == "rem":
                    temp_rem = lower_and_check_unique(sub_dict)
                    for k, v in temp_rem.items():
                        self.rem_dict[k] = v
                if sub == "pcm":
                    temp_pcm = lower_and_check_unique(sub_dict)
                    for k, v in temp_pcm.items():
                        self.pcm_dict[k] = v
                if sub == "solvent":
                    temp_solvent = lower_and_check_unique(sub_dict)
                    for k, v in temp_solvent.items():
                        self.solv_dict[k] = v
                if sub == "smx":
                    temp_smx = lower_and_check_unique(sub_dict)
                    for k, v in temp_smx.items():
                        self.smx_dict[k] = v
                if sub == "scan":
                    temp_scan = lower_and_check_unique(sub_dict)
                    for k, v in temp_scan.items():
                        self.scan_dict[k] = v
                if sub == "van_der_waals":
                    temp_vdw = lower_and_check_unique(sub_dict)
                    for k, v in temp_vdw.items():
                        self.vdw_dict[k] = v
                    # set the PCM section to read custom radii
                    self.pcm_dict["radii"] = "read"
                if sub == "plots":
                    temp_plots = lower_and_check_unique(sub_dict)
                    for k, v in temp_plots.items():
                        self.plots_dict[k] = v
                if sub == "nbo":
                    if self.nbo_dict is None:
                        raise RuntimeError(
                            "Can't overwrite nbo params when NBO"
                            "is not being run! Exiting..."
                        )
                    temp_nbo = lower_and_check_unique(sub_dict)
                    for k, v in temp_nbo.items():
                        self.nbo_dict[k] = v
                if sub == "geom_opt":
                    if self.geom_opt_dict is None:
                        raise RuntimeError(
                            "Can't overwrite geom_opt params when"
                            "not using the new optimizer! Exiting..."
                        )
                    temp_geomopt = lower_and_check_unique(sub_dict)
                    for k, v in temp_geomopt.items():
                        self.geom_opt_dict[k] = v
                if sub == "opt":
                    temp_opts = lower_and_check_unique(sub_dict)
                    for k, v in temp_opts.items():
                        self.opt_dict[k] = v

        return QCInputSet(
            qcinput=QCInput(
                molecule,
                rem=self.rem_dict,
                opt=self.opt_dict,
                pcm=self.pcm_dict,
                solvent=self.solv_dict,
                smx=self.smx_dict,
                scan=self.scan_dict,
                van_der_waals=self.vdw_dict,
                vdw_mode=self.vdw_mode,
                plots=self.plots_dict,
                nbo=self.nbo_dict,
                geom_opt=self.geom_opt_dict,
            )
        )

    # def get_input_set_updates(self) -> dict:
    #     """
    #     Get updates to the input set for this calculation type.

    #     Parameters
    #     ----------

    #     Returns
    #     -------
    #     dict
    #         A dictionary of updates to apply.

    #     Is this something which is even necessary
    #     in context of QChem? Will have to discuss
    #     """
    #     raise NotImplementedError
