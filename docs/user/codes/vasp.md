(codes.vasp)=

# VASP

At present, most workflows in atomate2 use the Vienna *ab initio* simulation package
(VASP) as the density functional theory code.

By default, the input sets used in atomate2 differ from the input sets used in atomate1
and are inconsistent with calculations performed in the Materials Project. The primary
differences are:

- Use of the PBEsol exchange–correlation functional instead of PBE.
- Use of up-to-date pseudopotentials (PBE_54 instead of PBE_52).

```{warning}
The different input sets used in atomate2 mean total energies cannot be compared
against energies taken from the Materials Project unless the default settings are
modified accordingly.
```

## Configuration

These workflows require VASP to be installed and on the path. Furthermore, the pymatgen
package is used to write VASP input files such as POTCARs. Accordingly, pymatgen
must be aware of where the pseudopotentials are installed. Please see the [pymatgen
POTCAR setup guide](https://pymatgen.org/installation.html#potcar-setup) for more
details.

All settings for controlling VASP execution can be set using the `~/.atomate2.yaml`
configuration file or using environment variables. For more details on configuring
atomate2, see the [Installation page](installation).

The most important settings to consider are:

- `VASP_CMD`: The command used to run the standard version of VASP. I.e., something like
  `mpi_run -n 16 vasp_std > vasp.out`.
- `VASP_GAMMA_CMD`: The command used to run the gamma-only version of VASP.
- `VASP_NCL_CMD`: The command used to run the non-collinear version of VASP.
- `VASP_INCAR_UPDATES`: Updates to apply to VASP INCAR files. This allows you to
  customise input sets on different machines, without having to change the submitted
  workflows. For example, you can set certain parallelization parameters, such as
  NCORE, KPAR etc.
- `VASP_VDW_KERNEL_DIR`: The path to the VASP Van der Waals kernel.

## FAQs

<b>How can I update the Custodian handlers used in a VASP job?</b>
- Every `Maker` which derives from `BaseVaspMaker` (see below) has a `run_vasp_kwargs` kwarg.
So, for example, to run a `StaticMaker` with only the `VaspErrorHandler` as the custodian handler, you would do this:

```py
from atomate2.vasp.jobs.core import StaticMaker
from custodian.vasp.handlers import VaspErrorHandler

maker = StaticMaker(run_vasp_kwargs={"handlers": [VaspErrorHandler]})
```

<b>How can I change the other Custodian settings used to run VASP?</b>
- These can be set through `run_vasp_kwargs.custodian_kwargs`:

```py
maker = StaticMaker(
    run_vasp_kwargs={
        "custodian_kwargs": {
            "max_errors_per_job": 5,
            "monitor_freq": 100,
        }
    }
)
```
For all possible `custodian_kwargs`, see the [`Custodian` class](https://github.com/materialsproject/custodian/blob/aa02baf5bc2a1883c5f8a8b6808340eeae324a99/src/custodian/custodian.py#L68).
NB: You cannot set the following four `Custodian` fields using `custodian_kwargs`: `handlers`, `jobs`, `validators`, `max_errors`, and `scratch_dir`.

<b>Can I change how VASP is run for each per-`Maker`/`Job`?</b>
Yes! Still using the `run_vasp_kwargs` and either the `vasp_cmd`, which represents the path to (and including) `vasp_std`, and `vasp_gamma_cmd`, which is the path to `vasp_gam`:

```py
maker = StaticMaker(run_vasp_kwargs={"vasp_cmd": "/path/to/vasp_std"})
```

<b>How can I use non-colinear VASP?</b>
The same method as before applies, you can simply set:

```py
vasp_ncl_path = "/path/to/vasp_ncl"
maker = StaticMaker(
    run_vasp_kwargs={"vasp_cmd": vasp_ncl_path, "vasp_gamma_cmd": vasp_ncl_path}
)
```

<b>How can I update the magnetic moments (MAGMOM) tag used to start a calculation?</b>
You can specify MAGMOM using a `dict` of defined values, such as:

```py
from pymatgen.io.vasp.sets import MPRelaxSet

vis = MPRelaxSet(user_incar_settings={"MAGMOM": {"In": 0.5, "Ga": 0.5, "As": -0.5}})
```
You can also specify different magnetic moments for different oxidation states, such as `{"Ga3+": 0.25}`.
However, note that `"Ga0+"`, which has been assigned zero-valued oxidation state, is distinct from `"Ga"`, which has not been assigned an oxidation state.

Alternatively, MAGMOM can be set by giving a structure assigned magnetic moments: `structure.add_site_property("magmom", list[float])`.
This will override the default MAGMOM settings of a VASP input set.

(vasp_workflows)=

## List of VASP workflows

```{eval-rst}
.. csv-table::
   :file: vasp-workflows.csv
   :widths: 40, 20, 40
   :header-rows: 1
```

### Static

A static VASP calculation (i.e., no relaxation).

### Relax

A VASP relaxation calculation. Full structural relaxation is performed.

### Tight Relax

A VASP relaxation calculation using tight convergence parameters. Full structural
relaxation is performed. This workflow is useful when small forces are required, such
as before calculating phonon properties.

### Dielectric

A VASP calculation to obtain dielectric properties. The static and high-frequency
dielectric constants are obtained using density functional perturbation theory.

### Transmuter

A generic calculation that transforms the structure (using one of the
{obj}`pymatgen.transformations`) before writing the input sets. This can be used to
perform many structure operations such as making a supercell or symmetrising the
structure.

### HSE06 Static

A static VASP calculation (i.e., no relaxation) using the HSE06 exchange correlation
functional.

### HSE06 Relax

A VASP relaxation calculation using the HSE06 functional. Full structural relaxation
is performed.

### HSE06 Tight Relax

A VASP relaxation calculation using tight convergence parameters with the HSE06
functional. Full structural relaxation is performed.

### Double Relax

Perform two back-to-back relaxations. This can often help avoid errors arising from
[Pulay stress](https://www.vasp.at/wiki/index.php/Pulay_stress).

In short: While the cell size, shape, symmetry, etc. can change during a relaxation, the *k* point grid does not change with it.
Additionally, the number of plane waves is held constant during a relaxation.
Both features lead to artificial (numerical) stress due to under-convergence of a relaxation with respect to the basis set.
To avoid this, we perform a single relaxation, and input its final structure to another relaxation calculation.
At the start of the second relaxation, the *k*-point mesh and plane waves are adjusted to reflect the new symmetry of the cell.

### Materials Project structure optimization

The Materials Project hosts a large database of, among other physical properties, optimized structures and their associated total energy, formation enthalpy, and basic electronic structure properties.
To generate this data, the Materials Project uses a simple double-relaxation followed by a final static calculation.
While in principle, if the second relaxation calculation is converged, a final static calculation would not be needed.
However, the second relaxation may have residual Pulay stress, and VASP averages some electronic structure data ([like the density of states](https://www.vasp.at/wiki/index.php/DOSCAR)) during a relaxation.
Thus we need to perform a final single-point (static) calculation, usually using the corrected tetrahedron method (`ISMEAR=-5`) to ensure accurate electronic structure properties.

The workflows used to produce PBE GGA or GGA+*U* and r<sup>2</sup>SCAN thermodynamic data are, respectively, `MPGGADoubleRelaxStaticMaker` and `MPMetaGGADoubleRelaxStaticMaker` in `atomate2.vasp.flows.mp`.
Moving forward, the Materials Project prefers r<sup>2</sup>SCAN calculations, but maintains its older set of GGA-level data which currently has wider coverage.
For documentation about the calculation parameters used, see the [Materials Project documentation.](https://docs.materialsproject.org/methodology/materials-methodology/calculation-details)

### Band Structure

Calculate the electronic band structure. This flow consists of three calculations:

1. A static calculation to generate the charge density.
2. A non-self-consistent field calculation on a dense uniform mesh.
3. A non-self-consistent field calculation on the high-symmetry k-point path to generate
   the line mode band structure.

```{note}
Band structure objects are automatically stored in the `data` store due to
limitations on MongoDB collection sizes.
```

### Uniform Band Structure

Calculate a uniform electronic band structure. This flow consists of two calculations:

1. A static calculation to generate the charge density.
2. A non-self-consistent field calculation on a dense uniform mesh.

```{note}
   Band structure objects are automatically stored in the `data` store due to
   limitations on MongoDB collection sizes.
```

### Line-Mode Band Structure

Calculate a line-mode electronic band structure. This flow consists of two calculations:

1. A static calculation to generate the charge density.
2. A non-self-consistent field calculation on a high-symmetry k-point path to generate
   the line mode band structure.

```{note}
Band structure objects are automatically stored in the `data` store due to
limitations on MongoDB collection sizes.
```

### HSE06 Band Structure

Calculate the electronic band structure using HSE06. This flow consists of three
calculations:

1. A HSE06 static calculation to generate the charge density.
2. A HSE06 calculation on a dense uniform mesh.
3. A HSE06 calculation on the high-symmetry k-point path using zero weighted k-points.

```{note}
Band structure objects are automatically stored in the `data` store due to
limitations on MongoDB collection sizes.
```

### HSE06 Uniform Band Structure

Calculate a uniform electronic band structure using HSE06. This flow consists of two
calculations:

1. A HSE06 static calculation to generate the charge density.
2. A HSE06 non-self-consistent field calculation on a dense uniform mesh.

```{note}
Band structure objects are automatically stored in the `data` store due to
limitations on MongoDB collection sizes.
```

### HSE06 Line-Mode Band Structure

Calculate a line-mode electronic band structure using HSE06. This flow consists of two
calculations:

1. A HSE06 static calculation to generate the charge density.
2. A HSE06 non-self-consistent field calculation on a high-symmetry k-point path to
   generate the line mode band structure.

```{note}
Band structure objects are automatically stored in the `data` store due to
limitations on MongoDB collection sizes.
```

### Relax and Band Structure

Perform a relaxation and then run the Band Structure workflow. By default, a
Double Relax relaxation is performed.

### Elastic Constant

Calculate the elastic constant of a material. Initially, a tight structural relaxation
is performed to obtain the structure in a state of approximately zero stress.
Subsequently, perturbations are applied to the lattice vectors and the resulting
stress tensor is calculated from DFT, while allowing for relaxation of the ionic degrees
of freedom. Finally, constitutive relations from linear elasticity, relating stress and
strain, are employed to fit the full 6x6 elastic tensor. From this, aggregate properties
such as Voigt and Reuss bounds on the bulk and shear moduli are derived.

See the Materials Project [documentation on elastic constants](
https://docs.materialsproject.org/methodology/elasticity/) for more details.

```{note}
It is strongly recommended to symmetrize the structure before running this workflow.
Otherwise, the symmetry reduction routines will not be as effective at reducing the
number of deformations needed.
```

### Optics

Calculate the frequency-dependent dielectric response of a material.

This workflow contains an initial static calculation, and then a non-self-consistent
field calculation with LOPTICS set. The purpose of the static calculation is to
determine i) if the material needs magnetism set, and ii) the total number of bands (the
non-scf calculation contains 1.3 * number of bands in the static calculation) as often
the highest bands are not properly converged in VASP.

### HSE06 Optics

Calculate the frequency-dependent dielectric response of a material using HSE06.

This workflow contains an initial static calculation, and then a uniform band structure
calculation with LOPTICS set. The purpose of the static calculation is to determine i)
if the material needs magnetism set, and ii) the total number of bands (the uniform
contains 1.3 * number of bands in the static calculation) as often the highest bands are
not properly converged in VASP.

### Phonons

Calculate the harmonic phonons of a material.

Initially, a tight structural relaxation is performed to obtain a structure without forces
on the atoms. Subsequently, supercells with one displaced atom are generated and accurate
forces are computed for these structures. With the help of phonopy, these forces are then
converted into a dynamical matrix. To correct for polarization effects, a correction of the
dynamical matrix based on BORN charges can be performed. Finally, phonon densities of states,
phonon band structures and thermodynamic properties are computed.

```{warning}
The current implementation of the workflow does not consider the initial magnetic moments
for the determination of the symmetry of the structure; therefore, they are removed from the structure.
```

```{note}
It is heavily recommended to symmetrize the structure before passing it to
this flow. Otherwise, a different space group might be detected and too
many displacement calculations will be generated.
It is recommended to check the convergence parameters here and
adjust them if necessary. The default might not be strict enough
for your specific case.
```

You can use the following code to start the default VASP version of the workflow:
```py
from atomate2.vasp.flows.phonons import PhononMaker
from pymatgen.core.structure import Structure

structure = Structure(
    lattice=[[0, 2.13, 2.13], [2.13, 0, 2.13], [2.13, 2.13, 0]],
    species=["Mg", "O"],
    coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
)

phonon_flow = PhononMaker(min_length=15.0, store_force_constants=False).make(
    structure=structure
)
```

### Grüneisen parameter workflow

Calculates mode-dependent Grüneisen parameters with the help of Phonopy.

Initially, a tight structural relaxation is performed to obtain a structure without
forces on the atoms. The optimized structure (ground state) is further expanded and
shrunk by 1 % (default) of its volume.
Subsequently, supercells with one displaced atom are generated for all the three structures
(ground state, expanded and shrunk volume) and accurate forces are computed for these structures.
With the help of phonopy, these forces are then converted into a dynamical matrix.
The dynamical matrices of three structures are then used as an input to the phonopy Grüneisen api
to compute mode-dependent Grüneisen parameters.

A Grüneisen workflow for VASP can be started as follows:
```python
from atomate2.vasp.flows.gruneisen import GruneisenMaker
from pymatgen.core.structure import Structure


structure = Structure(
    lattice=[[0, 2.13, 2.13], [2.13, 0, 2.13], [2.13, 2.13, 0]],
    species=["Mg", "O"],
    coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
)

gruneisen_flow = GruneisenMaker(
    kpath_scheme="seekpath", vol=0.01, mesh=(15, 15, 15)
).make(structure=structure)
```

### Quasi-harmonic Workflow

Uses the quasi-harmonic approximation with the help of Phonopy to compute thermodynamic properties.
First, a tight relaxation is performed. Subsequently, several optimizations at different constant
volumes are performed. At each of the volumes, an additional phonon run is performed as well.
Afterwards, equation of state fits are performed with phonopy.

The following script allows you to start the default workflow for VASP with some adjusted parameters:
```python
from atomate2.vasp.flows.qha import QhaMaker
from pymatgen.core.structure import Structure


structure = Structure(
    lattice=[[0, 2.13, 2.13], [2.13, 0, 2.13], [2.13, 2.13, 0]],
    species=["Mg", "O"],
    coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
)

qha_flow = QhaMaker(
    linear_strain=(-0.10, 0.10),
    number_of_frames=10,
).make(structure=structure)
```

### Equation of State Workflow

An equation of state (EOS) workflow has the following structure: First, a tight relaxation is performed.
Subsequently, several optimizations at different constant
volumes are performed. Additional static calculations might be performed afterwards to arrive at more
accurate energies. Then, an EOS fit is performed with pymatgen.

You can start the workflow as follows:
```python
from atomate2.vasp.flows.eos import EosMaker
from pymatgen.core.structure import Structure


structure = Structure(
    lattice=[[0, 2.13, 2.13], [2.13, 0, 2.13], [2.13, 2.13, 0]],
    species=["Mg", "O"],
    coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
)

eos_flow = EosMaker(
    linear_strain=(-0.10, 0.10),
    number_of_frames=10,
).make(structure=structure)
```

The output of the workflow is a dictionary by default containing the energy and volume data generated with DFT,
in addition to fitted equation of state parameters for all models currently available in pymatgen
(Murnaghan, Birch-Murnaghan, Poirier-Tarantola, and Vinet/UBER).

#### Materials Project-compliant workflows

If the user wishes to reproduce the EOS data currently in the Materials Project, they should use the atomate 1-compatible `MPLegacy`-prefixed flows (and jobs and input sets). For performing updated PBE-GGA EOS flows with Materials Project-compliant parameters, the user should use the `MPGGA`-prefixed classes. Lastly, the `MPMetaGGA`-prefixed classes allow the user to perform Materials Project-compliant r<sup>2</sup>SCAN EOS workflows.

**Summary:** For Materials Project-compliant equation of state (EOS) workflows, the user should use:
* `MPGGAEosMaker` for faster, lower-accuracy calculation with the PBE-GGA
* `MPMetaGGAEosMaker` for higher-accuracy but slower calculations with the r<sup>2</sup>SCAN meta-GGA
* `MPLegacyEosMaker` for consistency with the PBE-GGA data currently distributed by the Materials Project

#### Implementation details

The Materials Project-compliant EOS flows, jobs, and sets currently use three prefixes to indicate their usage.
* `MPGGA`: MP-compatible PBE-GGA (current)
* `MPMetaGGA`: MP-compatible r<sup>2</sup>SCAN meta-GGA (current)
* `MPLegacy`: a reproduction of the atomate 1 implementation, described in
  K. Latimer, S. Dwaraknath, K. Mathew, D. Winston, and K.A. Persson, npj Comput. Materials **vol. 4**, p. 40 (2018), DOI: 10.1038/s41524-018-0091-x

  For reference, the original atomate workflows can be found here:
    * [`atomate.vasp.workflows.base.wf_bulk_modulus`](https://github.com/hackingmaterials/atomate/blob/main/atomate/vasp/workflows/presets/core.py#L564)
    * [`atomate.vasp.workflows.base.bulk_modulus.get_wf_bulk_modulus`](https://github.com/hackingmaterials/atomate/blob/main/atomate/vasp/workflows/base/bulk_modulus.py#L21)

In the original atomate 1 workflow and the atomate2 `MPLegacyEosMaker`, the k-point density is **extremely** high. This is despite the convergence tests in the supplementary information
of Latimer *et al.* not showing strong sensitivity when the "number of ***k***-points per reciprocal atom" (KPPRA) is at least 3,000.

To make the `MPGGAEosMaker` and `MPMetaGGAEosMaker` more tractable for high-throughput jobs, their input sets (`MPGGAEos{Relax,Static}SetGenerator` and `MPMetaGGAEos{Relax,Static}SetGenerator` respectively) still use the highest ***k***-point density in standard Materials Project jobs, `KSPACING = 0.22` Å<sup>-1</sup>, which is comparable to KPPRA = 3,000.

This choice is justified by Fig. S12 of the supplemantary information of Latimer *et al.*, which shows that all fitted EOS parameters (equilibrium energy $E_0$, equilibrium volume $V_0$, bulk modulus $B_0$, and bulk modulus pressure derivative $B_1$) do not deviate by more than 1.5%, and typically by less than 0.1%, from well-converged values when KPPRA = 3,000.

### LOBSTER

Perform bonding analysis with [LOBSTER](http://cohp.de/) and [LobsterPy](https://github.com/jageo/lobsterpy)

Initially, a structural relaxation is performed. Within a static run, the wave function is pre-converged
with symmetry switched on. Then, another static run with the correct number of bands and without
symmetry will be performed. The wave function will then be used for LOBSTER runs with all
available basis functions in Lobster. Then, [LobsterPy](https://github.com/jageo/lobsterpy) will perform an automatic
analysis of the output files from LOBSTER.

Please add the LOBSTER command to the `atomate2.yaml` file:

```yaml
VASP_CMD: <<VASP_CMD>>
LOBSTER_CMD: <<LOBSTER_CMD>>
```

```{note}
A LOBSTER workflow with settings compatible to LOBSTER database (Naik, A.A., et al. Sci Data 10, 610 (2023). https://doi.org/10.1038/s41597-023-02477-5 , currently being integrated into Materials Project) is also available now,
which could be used by simply importing from atomate2.vasp.flows.mp > MPVaspLobsterMaker
instead of VaspLobsterMaker. Rest of the things to execute the workflow stays same as
shown below.
```

The corresponding flow could, for example, be started with the following code:

```py
from jobflow import SETTINGS
from jobflow import run_locally
from pymatgen.core.structure import Structure

from atomate2.vasp.flows.lobster import VaspLobsterMaker
from atomate2.vasp.powerups import update_user_incar_settings

structure = Structure(
    lattice=[[0, 2.13, 2.13], [2.13, 0, 2.13], [2.13, 2.13, 0]],
    species=["Mg", "O"],
    coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
)

lobster = VaspLobsterMaker().make(structure)

# update the incar
lobster = update_user_incar_settings(lobster, {"NPAR": 4})
# run the job
run_locally(lobster, create_folders=True, store=SETTINGS.JOB_STORE)
```

There are currently three different ways available to run the workflow efficiently, as VASP and LOBSTER rely on a different parallelization (MPI vs. OpenMP).
One can use a job script (with some restrictions), or [Jobflow-remote](https://matgenix.github.io/jobflow-remote/) / [Fireworks](https://github.com/materialsproject/fireworks) for high-throughput runs.


#### Running the LOBSTER workflow without database and with one job script only

It is possible to run the VASP-LOBSTER workflow efficiently with a minimal setup.
In this case, you will run the VASP calculations on the same node as the LOBSTER calculations.
In between, the different computations you will switch from MPI to OpenMP parallelization.

For example, for a node with 48 cores, you could use an adapted version of the following SLURM script:

```bash
#!/bin/bash
#SBATCH -J vasplobsterjob
#SBATCH -o ./%x.%j.out
#SBATCH -e ./%x.%j.err
#SBATCH -D ./
#SBATCH --mail-type=END
#SBATCH --mail-user=you@you.de
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#This needs to be adapted if you run with different cores
#SBATCH --ntasks=48

# ensure you load the modules to run VASP, e.g., module load vasp
module load my_vasp_module
# please activate the required conda environment
conda activate my_environment
cd my_folder
# the following script needs to contain the workflow
python xyz.py
```

The `LOBSTER_CMD` now needs an additional export of the number of threads.

```yaml
VASP_CMD: <<VASP_CMD>>
LOBSTER_CMD: OMP_NUM_THREADS=48 <<LOBSTER_CMD>>
```


#### Jobflow-remote
Please refer first to the general documentation of jobflow-remote: [https://matgenix.github.io/jobflow-remote/](https://matgenix.github.io/jobflow-remote/).

```py
from atomate2.vasp.flows.lobster import VaspLobsterMaker
from pymatgen.core.structure import Structure
from jobflow_remote import submit_flow, set_run_config
from atomate2.vasp.powerups import update_user_incar_settings

structure = Structure(
    lattice=[[0, 2.13, 2.13], [2.13, 0, 2.13], [2.13, 2.13, 0]],
    species=["Mg", "O"],
    coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
)

lobster = VaspLobsterMaker().make(structure)

resources = {"nodes": 3, "partition": "micro", "time": "00:55:00", "ntasks": 144}

resources_lobster = {"nodes": 1, "partition": "micro", "time": "02:55:00", "ntasks": 48}
lobster = set_run_config(lobster, name_filter="lobster", resources=resources_lobster)

lobster = update_user_incar_settings(lobster, {"NPAR": 4})
submit_flow(lobster, worker="my_worker", resources=resources, project="my_project")
```

The `LOBSTER_CMD` also needs an export of the threads.

```yaml
VASP_CMD: <<VASP_CMD>>
LOBSTER_CMD: OMP_NUM_THREADS=48 <<LOBSTER_CMD>>
```



#### Fireworks
Please first refer to the general documentation on running atomate2 workflows with fireworks: [https://materialsproject.github.io/atomate2/user/fireworks.html](https://materialsproject.github.io/atomate2/user/fireworks.html)

Specifically, you might want to change the `_fworker` for the LOBSTER runs and define a separate `lobster` worker within FireWorks:

```py
from fireworks import LaunchPad
from jobflow.managers.fireworks import flow_to_workflow
from pymatgen.core.structure import Structure

from atomate2.vasp.flows.lobster import VaspLobsterMaker
from atomate2.vasp.powerups import update_user_incar_settings

structure = Structure(
    lattice=[[0, 2.13, 2.13], [2.13, 0, 2.13], [2.13, 2.13, 0]],
    species=["Mg", "O"],
    coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
)

lobster = VaspLobsterMaker().make(structure)
lobster = update_user_incar_settings(lobster, {"NPAR": 4})

# update the fireworker of the Lobster jobs
for job, _ in lobster.iterflow():
    config = {"manager_config": {"_fworker": "worker"}}
    if "get_lobster" in job.name:
        config["response_manager_config"] = {"_fworker": "lobster"}
    job.update_config(config)

# convert the flow to a fireworks WorkFlow object
wf = flow_to_workflow(lobster)

# submit the workflow to the FireWorks launchpad
lpad = LaunchPad.auto_load()
lpad.add_wf(wf)
```


The `LOBSTER_CMD` can now be adapted to not include the number of threads:

```yaml
VASP_CMD: <<VASP_CMD>>
LOBSTER_CMD: <<LOBSTER_CMD>>
```

#### Analyzing outputs

Outputs from the automatic analysis with LobsterPy can easily be extracted from the database and also plotted:

```py
from jobflow import SETTINGS
from pymatgen.electronic_structure.cohp import Cohp
from pymatgen.electronic_structure.plotter import CohpPlotter

store = SETTINGS.JOB_STORE
store.connect()

result = store.query_one(
    {"name": "lobster_run_0"},
    properties=[
        "output.lobsterpy_data.cohp_plot_data",
        "output.lobsterpy_data_cation_anion.cohp_plot_data",
    ],
    load=True,
)

for number, (key, cohp) in enumerate(
    result["output"]["lobsterpy_data"]["cohp_plot_data"]["data"].items()
):
    plotter = CohpPlotter()
    cohp = Cohp.from_dict(cohp)
    plotter.add_cohp(key, cohp)
    plotter.save_plot(f"plots_all_bonds{number}.pdf")

for number, (key, cohp) in enumerate(
    result["output"]["lobsterpy_data_cation_anion"]["cohp_plot_data"]["data"].items()
):
    plotter = CohpPlotter()
    cohp = Cohp.from_dict(cohp)
    plotter.add_cohp(key, cohp)
    plotter.save_plot(f"plots_cation_anion_bonds{number}.pdf")
```


(modifying_input_sets)=
Modifying input sets
--------------------

The inputs for a calculation can be modified in several ways. Every VASP job
takes a {obj}`.VaspInputGenerator` as an argument (`input_set_generator`). One
option is to specify an alternative input set generator:

```py
from atomate2.vasp.sets.core import StaticSetGenerator
from atomate2.vasp.jobs.core import StaticMaker

# create a custom input generator set with a larger ENCUT
my_custom_set = StaticSetGenerator(user_incar_settings={"ENCUT": 800})

# initialise the static maker to use the custom input set generator
static_maker = StaticMaker(input_set_generator=my_custom_set)

# create a job using the customised maker
static_job = static_maker.make(structure)
```

The second approach is to edit the job after it has been made. All VASP jobs have a
`maker` attribute containing a *copy* of the `Maker` that made them. Updating
the `input_set_generator` attribute maker will update the input set that gets
written:

```py
static_job.maker.input_set_generator.user_incar_settings["LOPTICS"] = True
```

To update *k*-points, use the `user_kpoints_settings` keyword argument of an input set generator.
You can supply either a `pymatgen.io.vasp.inputs.Kpoints` object, or a `dict` containing certain [keys](https://github.com/materialsproject/pymatgen/blob/b54ac3e65e46b876de40402e8da59f551fb7d005/src/pymatgen/io/vasp/sets.py#L812).
We generally recommend the former approach unless the user is familiar with the specific style of *k*-point updates used by `pymatgen`.
For example, to use just the $\Gamma$ point:

```py
from pymatgen.io.vasp.inputs import Kpoints
from atomate2.vasp.sets.core import StaticSetGenerator
from atomate2.vasp.jobs.core import StaticMaker

custom_gamma_only_set = StaticSetGenerator(user_kpoints_settings=Kpoints())
gamma_only_static_maker = StaticMaker(input_set_generator=custom_gamma_only_set)
```

For those who are more familiar with manual *k*-point generation, you can use a VASP-style KPOINTS file or string to set the *k*-points as well:

```py
kpoints = Kpoints.from_str(
    """Uniform density Monkhorst-Pack mesh
0
Monkhorst-pack
5 5 5
"""
)
custom_static_set = StaticSetGenerator(user_kpoints_settings=kpoints)
```

Finally, sometimes you have a workflow containing many VASP jobs. In this case it can be
tedious to update the input sets for each job individually. Atomate2 provides helper
functions called "powerups" that can apply settings updates to all VASP jobs in a flow.
These powerups also contain filters for the name of the job and the maker used to
generate them. These functions will apply updates *only* to VASP jobs, including those
created dynamically - all other jobs in a flow will not be modified.

```py
from atomate2.vasp.powerups import update_user_incar_settings
from atomate2.vasp.flows.elastic import ElasticMaker
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.elastic import ElasticRelaxMaker

# make a flow to calculate the elastic constants
elastic_flow = ElasticMaker().make(structure)

# update the ENCUT of all VASP jobs in the flow
new_flow = update_user_incar_settings(elastic_flow, {"ENCUT": 200})

# only update VASP jobs which have "deformation" in the job name.
new_flow = update_user_incar_settings(
    elastic_flow, {"ENCUT": 200}, name_filter="deformation"
)

# only update VASP jobs which were generated by an ElasticRelaxMaker
new_flow = update_user_incar_settings(
    elastic_flow, {"ENCUT": 200}, class_filter=ElasticRelaxMaker
)

# powerups can also be applied directly to a Maker. This can be useful for makers
# that produce flows, as it allows you to update all nested makers. E.g.
relax_maker = DoubleRelaxMaker()
new_maker = update_user_incar_settings(relax_maker, {"ENCUT": 200})
flow = new_maker.make(structure)  # this flow will reflect the updated ENCUT value
```

```{note}
Powerups return a copy of the original flow or Maker and do not modify it in place.
```

In addition to the ability to change INCAR parameters on-the-fly, the
{obj}`.VaspInputGenerator`, `Maker` object, and "powerups" allow for the manual
modification of several additional VASP settings, such as the k-points
(`user_kpoints_settings`) and choice of pseudopotentials (`user_potcar_settings`).

If a greater degree of flexibility is needed, the user can define a default set of input
arguments (`config_dict`) that can be provided to the {obj}`.VaspInputGenerator`.
By default, the {obj}`.VaspInputGenerator` uses a base set of VASP input parameters (`atomate2.vasp.sets.base._BASE_VASP_SET`), which each `Maker` is built upon. If desired, the user can
define a custom `.yaml` file that contains a different base set of VASP settings to use.
An example of how this can be done is shown below for a representative static
calculation.

```
from atomate2.vasp.sets.core import StaticSetGenerator
from atomate2.vasp.jobs.core import StaticMaker
from atomate2.vasp.jobs.base import VaspInputGenerator
from monty.serialization import loadfn

# read in a custom config file
user_config_dict = loadfn("/path/to/my/CustomVaspSet.yaml")

# create a custom static set generator with user-defined defaults. Also change the
# NELMIN parameter to 6 (for demonstration purposes)
my_custom_set = StaticSetGenerator(
    user_incar_settings={"NELMIN": 6},
    config_dict=user_config_dict,
)

# initialise the static maker to use the custom input set generator
static_maker = StaticMaker(input_set_generator=my_custom_set)

# create a job using the customised maker
static_job = static_maker.make(structure)
```

(connecting_vasp_jobs)=
Chaining workflows
------------------

All VASP workflows are constructed using the `Maker.make()` function. The arguments
for this function always include:

- `structure`: A pymatgen structure.
- `prev_dir`: A previous VASP directory to copy output files from.

There are two options when chaining workflows:

1. Use only the structure from the previous calculation. This can be achieved by only
   setting the `structure` argument.
2. Use the structure and additional outputs from a previous calculation. By default,
   these outputs include INCAR settings, the band gap (used to automatically
   set KSPACING), and the magnetic moments. Some workflows will also use other outputs.
   For example, the Band Structure workflow will copy the CHGCAR file (charge
   density) from the previous calculation. This can be achieved by setting both the
   `structure` and `prev_dir` arguments.

These two examples are illustrated in the code below, where we chain a relaxation
calculation and a static calculation.

```py
from jobflow import Flow
from atomate2.vasp.jobs.core import RelaxMaker, StaticMaker
from pymatgen.core.structure import Structure

si_structure = Structure.from_file("Si.cif")

# create a relax job
relax_job = RelaxMaker().make(structure=si_structure)

# create a static job that will use only the structure from the relaxation
static_job = StaticMaker().make(structure=relax_job.output.structure)

# create a static job that will use additional outputs from the relaxation
static_job = StaticMaker().make(
    structure=relax_job.output.structure, prev_dir=relax_job.output.dir_name
)

# create a flow including the two jobs and set the output to be that of the static
my_flow = Flow([relax_job, static_job], output=static_job.output)
```
