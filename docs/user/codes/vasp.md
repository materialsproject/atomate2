(codes.vasp)=
# VASP

At present, most workflows in atomate2 use the Vienna *ab initio* simulation package
(VASP) as the density functional theory code.

By default, the input sets used in atomate2 differ from the input sets used in atomate1
and are inconsistent with calculations performed in the Materials Project. The primary
differences are:

- Use of the PBEsol exchange–correlation functional instead of PBE.
- Use of up-to-date pseudopotentials (PBE_54 instead of PBE_52).
- Use of KSPACING for most calculations.

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
Pulay stress.

### Band Structure

Calculate the electronic band structure. This flow consists of three calculations:

1. A static calculation to generate the charge density.
2. A non-self-consistent field calculation on a dense uniform mesh.
3. A non-self-consistent field calculation on the high-symmetry k-point path to generate
   the line mode band structure.

```{note}
Band structure objects are automatically stored in the `data` store due to
limitations on mongoDB collection sizes.
```

### Uniform Band Structure

Calculate a uniform electronic band structure. This flow consists of two calculations:

1. A static calculation to generate the charge density.
2. A non-self-consistent field calculation on a dense uniform mesh.

```{note}
   Band structure objects are automatically stored in the `data` store due to
   limitations on mongoDB collection sizes.
```

### Line-Mode Band Structure

Calculate a line-mode electronic band structure. This flow consists of two calculations:

1. A static calculation to generate the charge density.
2. A non-self-consistent field calculation on a high-symmetry k-point path to generate
   the line mode band structure.

```{note}
Band structure objects are automatically stored in the `data` store due to
limitations on mongoDB collection sizes.
```

### HSE06 Band Structure

Calculate the electronic band structure using HSE06. This flow consists of three
calculations:

1. A HSE06 static calculation to generate the charge density.
2. A HSE06 calculation on a dense uniform mesh.
3. A HSE06 calculation on the high-symmetry k-point path using zero weighted k-points.

```{note}
Band structure objects are automatically stored in the `data` store due to
limitations on mongoDB collection sizes.
```

### HSE06 Uniform Band Structure

Calculate a uniform electronic band structure using HSE06. This flow consists of two
calculations:

1. A HSE06 static calculation to generate the charge density.
2. A HSE06 non-self-consistent field calculation on a dense uniform mesh.

```{note}
Band structure objects are automatically stored in the `data` store due to
limitations on mongoDB collection sizes.
```

### HSE06 Line-Mode Band Structure

Calculate a line-mode electronic band structure using HSE06. This flow consists of two
calculations:

1. A HSE06 static calculation to generate the charge density.
2. A HSE06 non-self-consistent field calculation on a high-symmetry k-point path to
   generate the line mode band structure.

```{note}
Band structure objects are automatically stored in the `data` store due to
limitations on mongoDB collection sizes.
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

Calculate the frequency dependent dielectric response of a material.

This workflow contains an initial static calculation, and then a non-self-consistent
field calculation with LOPTICS set. The purpose of the static calculation is to
determine i) if the material needs magnetism set, and ii) the total number of bands (the
non-scf calculation contains 1.3 * number of bands in the static calculation) as often
the highest bands are not properly converged in VASP.

### HSE06 Optics

Calculate the frequency dependent dielectric response of a material using HSE06.

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

```{note}
It is heavily recommended to symmetrize the structure before passing it to
this flow. Otherwise, a different space group might be detected and too
many displacement calculations will be generated.
It is recommended to check the convergence parameters here and
adjust them if necessary. The default might not be strict enough
for your specific case.
```

## Lobster

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

Outputs from the automatic analysis with LobsterPy can easily be extracted from the database and also plotted:

```python
from jobflow import SETTINGS
from pymatgen.electronic_structure.cohp import Cohp
from pymatgen.electronic_structure.plotter import CohpPlotter

store = SETTINGS.JOB_STORE
store.connect()

result = store.query_one(
    {"name": "lobster_run_0"},
    properties=["output.lobsterpy_data.cohp_plot_data", "output.lobsterpy_data_cation_anion.cohp_plot_data", ],
    load=True,
)

for number, (key, cohp) in enumerate(result["output"]["lobsterpy_data"]["cohp_plot_data"].items()):
    plotter = CohpPlotter()
    cohp = Cohp.from_dict(cohp)
    plotter.add_cohp(key, cohp)
    plotter.save_plot("plots_all_bonds" + str(number) + ".pdf")

for number, (key, cohp) in enumerate(result["output"]["lobsterpy_data_cation_anion"]["cohp_plot_data"].items()):
    plotter = CohpPlotter()
    cohp = Cohp.from_dict(cohp)
    plotter.add_cohp(key, cohp)
    plotter.save_plot("plots_cation_anion_bonds" + str(number) + ".pdf")
```

(modifying_input_sets)=
Modifying input sets
--------------------

The inputs for a calculation can be modified in several ways. Every VASP job
takes a {obj}`.VaspInputGenerator` as an argument (`input_set_generator`). One
option is to specify an alternative input set generator:

```python
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

```python
static_job.maker.input_set_generator.user_incar_settings["LOPTICS"] = True
```

Finally, sometimes you have a workflow containing many VASP jobs. In this case it can be
tedious to update the input sets for each job individually. Atomate2 provides helper
functions called "powerups" that can apply settings updates to all VASP jobs in a flow.
These powerups also contain filters for the name of the job and the maker used to
generate them.

```python
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
By default, the {obj}`.VaspInputGenerator` uses a base set of VASP input parameters
from {obj}`.BaseVaspSet.yaml`, which each `Maker` is built upon. If desired, the user can
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
- `prev_vasp_dir`: A previous VASP directory to copy output files from.

There are two options when chaining workflows:

1. Use only the structure from the previous calculation. This can be achieved by only
   setting the `structure` argument.
2. Use the structure and additional outputs from a previous calculation. By default,
   these outputs include INCAR settings, the band gap (used to automatically
   set KSPACING), and the magnetic moments. Some workflows will also use other outputs.
   For example, the Band Structure workflow will copy the CHGCAR file (charge
   density) from the previous calculation. This can be achieved by setting both the
   `structure` and `prev_vasp_dir` arguments.

These two examples are illustrated in the code below, where we chain a relaxation
calculation and a static calculation.

```python
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
    structure=relax_job.output.structure, prev_vasp_dir=relax_job.output.dir_name
)

# create a flow including the two jobs and set the output to be that of the static
my_flow = Flow([relax_job, static_job], output=static_job.output)
```
