"""This module implements an interface to the VAMPIRE code for atomistic
simulations of magnetic materials.

.. note::

    This module is **vendored** from pymatgen. ``VampireCaller`` and
    ``VampireOutput`` lived in ``pymatgen.command_line.vampire_caller`` until
    they were removed in the "Major reorganization of pymatgen repo"
    (materialsproject/pymatgen#4595, merged 2026-03-02, first released in
    pymatgen 2026.3.23). atomate2's
    :class:`~atomate2.common.flows.exchange.ExchangeMaker` still needs them, so
    this started as a copy of the file's final pre-removal state (pymatgen
    commit ``8785afd0d801``, the last commit to touch it, released as
    pymatgen 2025.10.7) and has since been adapted to the reworked
    ``HeisenbergModel`` API: per-ordering ``site_labels`` and the ``igraph``
    interaction graph replace the old ``unique_site_ids`` dict and
    ``_get_j_exc`` lookup. The original author is Nathan C. Frey (``ncfrey``).
    It is excluded from ruff (see ``[tool.ruff] extend-exclude`` in
    ``pyproject.toml``).

This module depends on a compiled vampire executable available in the PATH.
Please download at https://vampire.york.ac.uk/download/ and
follow the instructions to compile the executable.

If you use this module, please cite:

"Atomistic spin model simulations of magnetic nanomaterials."
R. F. L. Evans, W. J. Fan, P. Chureemart, T. A. Ostler, M. O. A. Ellis
and R. W. Chantrell. J. Phys.: Condens. Matter 26, 103202 (2014)
"""

from __future__ import annotations

import logging
import subprocess
from shutil import which

import pandas as pd
from monty.dev import requires
from monty.json import MSONable

from atomate2.vampire.schemas.vampire_output import VampireOutput

__author__ = "ncfrey"
__version__ = "0.1"
__maintainer__ = "Nathan C. Frey"
__email__ = "ncfrey@lbl.gov"
__status__ = "Development"
__date__ = "June 2019"

logger = logging.getLogger(__name__)

VAMP_EXE = which("vampire-serial")


class VampireCaller:
    """Run Vampire on a material with magnetic ordering and exchange parameter
    information to compute the critical temperature with classical Monte Carlo.

    Attributes:
            structure (Structure): Ground state structure (magnetic ions only).
            site_labels (list[int]): Parent sublattice id of each site in the
                ground state structure.
            igraph (StructureGraph): Ground state graph with the fitted J_ij
                exchange values (meV) as edge weights.
            javg (float): <J> average exchange parameter estimate (meV).
            mat_name (str): Formula unit label for input files
            mat_id_dict (dict): Maps sites to material id # for vampire
                indexing.
    """

    @requires(
        VAMP_EXE is not None,
        "VampireCaller requires vampire-serial to be in the path."
        "Please follow the instructions at https://vampire.york.ac.uk/download/.",
    )
    def __init__(
        self,
        mc_box_size=4.0,
        equil_timesteps=2000,
        mc_timesteps=4000,
        save_inputs=False,
        hm=None,
        avg=True,
        user_input_settings=None,
    ):
        """user_input_settings is a dictionary that can contain:
        * start_t (int): Start MC sim at this temp, defaults to 0 K.
        * end_t (int): End MC sim at this temp, defaults to 1500 K.
        * temp_increment (int): Temp step size, defaults to 25 K.

        Args:
            mc_box_size (float): x=y=z dimensions (nm) of MC simulation box
            equil_timesteps (int): number of MC steps for equilibrating
            mc_timesteps (int): number of MC steps for averaging
            save_inputs (bool): if True, save scratch dir of vampire input files
            hm (HeisenbergModel): object already fit to low energy
                magnetic orderings.
            avg (bool): If True, simply use <J> exchange parameter estimate.
                If False, attempt to use NN, NNN, etc. interactions.
            user_input_settings (dict): optional commands for VAMPIRE Monte Carlo

        Todo:
            * Create input files in a temp folder that gets cleaned up after run terminates
        """
        self.mc_box_size = mc_box_size
        self.equil_timesteps = equil_timesteps
        self.mc_timesteps = mc_timesteps
        self.save_inputs = save_inputs
        self.avg = avg

        if not user_input_settings:  # set to empty dict
            self.user_input_settings = {}
        else:
            self.user_input_settings = user_input_settings

        # Attributes from HeisenbergModel
        if hm is None:
            raise ValueError("A fitted HeisenbergModel (hm=...) is required.")
        self.structure = hm.structures[0]  # ground state (magnetic ions only)
        self.site_labels = hm.site_labels[0]  # site -> parent sublattice id
        self.igraph = hm.igraph  # ground state graph, J_ij edge weights in meV
        self.javg = hm.javg

        # Full structure name before reducing to only magnetic ions
        self.mat_name = hm.formula

        # Switch to scratch dir which automatically cleans up vampire inputs files unless user specifies to save them
        # with ScratchDir(
        #     "/scratch", copy_from_current_on_enter=self.save_inputs, copy_to_current_on_exit=self.save_inputs
        # ):

        # Create input files
        self._create_mat()
        self._create_input()
        self._create_ucf()

        # Call Vampire
        with subprocess.Popen([VAMP_EXE], stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
            _stdout, stderr = process.communicate()
            stdout: str = _stdout.decode()

        if stderr:
            van_helsing = stderr.decode()
            if len(van_helsing) > 27:  # Suppress blank warning msg
                logger.warning(van_helsing)

        if process.returncode != 0:
            raise RuntimeError(f"Vampire exited with return code {process.returncode}.")

        self._stdout = stdout
        self._stderr = stderr

        # Process output
        n_mats = max(self.mat_id_dict.values())
        parsed_out, critical_temp = VampireCaller.parse_stdout("output", n_mats)
        self.output = VampireOutput(parsed_out, n_mats, critical_temp)

    def _create_mat(self):
        structure = self.structure
        mat_name = self.mat_name
        magmoms = structure.site_properties["magmom"]

        # A vampire material is a (sublattice, spin direction) group: one mat
        # per sublattice, two if it hosts both spin-up and spin-down sites.
        mat_ids = {}  # (sublattice id, spin sign) -> material id (1-indexed)
        mat_id_dict = {}  # site -> material id, for vampire inputs
        for site, (sub_id, magmom) in enumerate(zip(self.site_labels, magmoms, strict=True)):
            group = (sub_id, magmom > 0)
            mat_ids.setdefault(group, len(mat_ids) + 1)
            mat_id_dict[site] = mat_ids[group]

        n_mats = len(mat_ids)
        mat_file = [f"material:num-materials={n_mats}"]

        # One representative site per material for the element and moment
        reps = {}
        for site, mat_id in mat_id_dict.items():
            reps.setdefault(mat_id, site)

        for mat_id, site in sorted(reps.items()):
            atom = structure[site].species.reduced_formula
            spin = 1 if magmoms[site] > 0 else -1

            mat_file += [f"material[{mat_id}]:material-element={atom}"]
            mat_file += [
                f"material[{mat_id}]:damping-constant=1.0",
                f"material[{mat_id}]:uniaxial-anisotropy-constant=1.0e-24",
                # Only positive magmoms allowed
                f"material[{mat_id}]:atomic-spin-moment={abs(magmoms[site]):.2f} !muB",
                f"material[{mat_id}]:initial-spin-direction=0,0,{spin}",
            ]

        mat_file = "\n".join(mat_file)
        mat_file_name = f"{mat_name}.mat"

        self.mat_id_dict = mat_id_dict

        with open(mat_file_name, mode="w", encoding="utf-8") as file:
            file.write(mat_file)

    def _create_input(self):
        structure = self.structure
        mc_box_size = self.mc_box_size
        equil_timesteps = self.equil_timesteps
        mc_timesteps = self.mc_timesteps
        mat_name = self.mat_name

        input_script = [f"material:unit-cell-file={mat_name}.ucf"]
        input_script += [f"material:file={mat_name}.mat"]

        # Specify periodic boundary conditions
        input_script += [
            "create:periodic-boundaries-x",
            "create:periodic-boundaries-y",
            "create:periodic-boundaries-z",
        ]

        # Unit cell size in Angstrom
        abc = structure.lattice.abc
        ucx, ucy, ucz = abc[0], abc[1], abc[2]

        input_script += [f"dimensions:unit-cell-size-x = {ucx:.10f} !A"]
        input_script += [f"dimensions:unit-cell-size-y = {ucy:.10f} !A"]
        input_script += [f"dimensions:unit-cell-size-z = {ucz:.10f} !A"]

        # System size in nm
        input_script += [
            f"dimensions:system-size-x = {mc_box_size:.1f} !nm",
            f"dimensions:system-size-y = {mc_box_size:.1f} !nm",
            f"dimensions:system-size-z = {mc_box_size:.1f} !nm",
        ]

        # Critical temperature Monte Carlo calculation
        input_script += [
            "sim:integrator = monte-carlo",
            "sim:program = curie-temperature",
        ]

        # Default Monte Carlo params
        input_script += [
            f"sim:equilibration-time-steps = {equil_timesteps}",
            f"sim:loop-time-steps = {mc_timesteps}",
            "sim:time-steps-increment = 1",
        ]

        # Set temperature range and step size of simulation
        start_t = self.user_input_settings.get("start_t", 0)

        end_t = self.user_input_settings.get("end_t", 1500)

        temp_increment = self.user_input_settings.get("temp_increment", 25)

        input_script += [
            f"sim:minimum-temperature = {start_t}",
            f"sim:maximum-temperature = {end_t}",
            f"sim:temperature-increment = {temp_increment}",
        ]

        # Output to save
        input_script += [
            "output:temperature",
            "output:mean-magnetisation-length",
            "output:material-mean-magnetisation-length",
            "output:mean-susceptibility",
        ]

        input_script = "\n".join(input_script)

        with open("input", mode="w", encoding="utf-8") as file:
            file.write(input_script)

    def _create_ucf(self):
        structure = self.structure
        mat_name = self.mat_name

        abc = structure.lattice.abc
        ucx, ucy, ucz = abc[0], abc[1], abc[2]

        ucf = ["# Unit cell size:"]
        ucf += [f"{ucx:.10f} {ucy:.10f} {ucz:.10f}"]

        ucf += ["# Unit cell lattice vectors:"]
        a1 = list(structure.lattice.matrix[0])
        ucf += [f"{a1[0]:.10f} {a1[1]:.10f} {a1[2]:.10f}"]
        a2 = list(structure.lattice.matrix[1])
        ucf += [f"{a2[0]:.10f} {a2[1]:.10f} {a2[2]:.10f}"]
        a3 = list(structure.lattice.matrix[2])
        ucf += [f"{a3[0]:.10f} {a3[1]:.10f} {a3[2]:.10f}"]

        nmats = max(self.mat_id_dict.values())

        ucf += ["# Atoms num_materials; id cx cy cz mat cat hcat"]
        ucf += [f"{len(structure)} {nmats}"]

        # Fractional coordinates of atoms
        for site, r in enumerate(structure.frac_coords):
            # Back to 0 indexing for some reason...
            mat_id = self.mat_id_dict[site] - 1
            ucf += [f"{site} {r[0]:.10f} {r[1]:.10f} {r[2]:.10f} {mat_id} 0 0"]

        # J_ij exchange interaction matrix; the interaction graph carries the
        # fitted J_ij (meV) of every bond as an edge weight.
        igraph = self.igraph
        n_inter = 0
        for idx in range(len(igraph.graph.nodes)):
            n_inter += igraph.get_coordination_of_site(idx)

        ucf += ["# Interactions"]
        ucf += [f"{n_inter} isotropic"]

        iid = 0  # counts number of interaction
        for idx in range(len(igraph.graph.nodes)):
            for conn in igraph.get_connected_sites(idx):
                dx, dy, dz = conn.jimage  # relative integer coordinates of atom j
                j = conn.index  # index of neighbor

                # Just use the <J> estimate, or the fitted per-bond value
                j_exc = self.javg if self.avg is True else conn.weight

                # Convert J_ij from meV to Joules
                j_exc *= 1.6021766e-22

                j_exc = str(j_exc)  # otherwise this rounds to 0

                ucf += [f"{iid} {idx} {j} {dx} {dy} {dz} {j_exc}"]
                iid += 1

        ucf = "\n".join(ucf)
        ucf_file_name = f"{mat_name}.ucf"

        with open(ucf_file_name, mode="w", encoding="utf-8") as file:
            file.write(ucf)

    @staticmethod
    def parse_stdout(vamp_stdout, n_mats: int) -> tuple:
        """Parse stdout from Vampire.

        Args:
            vamp_stdout (txt file): Vampire 'output' file.
            n_mats (int): Number of materials in Vampire simulation.

        Returns:
            parsed_out (DataFrame): MSONable vampire output.
            critical_temp (float): Calculated critical temp.
        """
        names = [
            "T",
            "m_total",
            *[f"m_{idx + 1}" for idx in range(n_mats)],
            "X_x",
            "X_y",
            "X_z",
            "X_m",
            "nan",
        ]

        # Parsing vampire MC output
        df_stdout = pd.read_csv(vamp_stdout, sep="\t", skiprows=9, header=None, names=names).drop("nan", axis=1)

        parsed_out = df_stdout.to_json()

        # Max of susceptibility <-> critical temp
        critical_temp = df_stdout.iloc[df_stdout.X_m.idxmax()]["T"]

        return parsed_out, critical_temp
