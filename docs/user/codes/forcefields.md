(codes.forcefields)=

# Machine Learning forcefields / interatomic potentials

`atomate2` includes an interface to a few common machine learning interatomic potentials (MLIPs), also known variously as machine learning forcefields (MLFFs), or foundation potentials (FPs) for universal variants.
These can be installed using `pip install 'atomate2[ase]'`.

***As of `atomate2==0.1.2`, all forcefield packages are opt-in only. You must install those forcefields which you plan to use.***

Running `pip install 'atomate2[forcefields-demo]'` will install the `chgnet` package to permit you to try the forcefield jobs/workflows.
You can then install additional forcefield libraries.

If you need a sense of which forcefields are compatible, you can use the [pyproject.toml](https://github.com/materialsproject/atomate2/blob/a8bc6505e439503a114f5346aec916aafae7f27b/pyproject.toml#L90) to see which versions are grouped together for testing.

Most of `Maker` classes using the forcefields inherit from `atomate2.forcefields.utils.ForceFieldMixin` to specify which forcefield to use.
The `ForceFieldMixin` mixin provides the following configurable parameters:

- `force_field_name`: Name of the forcefield to use.
- `calculator_kwargs`: Keyword arguments to pass to the corresponding ASE calculator.

These parameters are passed to `atomate2.forcefields.utils.ase_calculator()` to instantiate the appropriate ASE calculator.

The `force_field_name` should be either one of predefined `atomate2.forcefields.utils.MLFF` (or its string equivalent) or a dictionary decodable as a class or function for ASE calculator as follows.

## Using predefined forcefields supported via `atomate2.forcefields.utils.MLFF`

Support is provided for the following models, which can be selected using `atomate2.forcefields.utils.MLFF`, as shown in the table below (in alphabetical order):
**You need only install packages for the forcefields you wish to use.**

| Forcefield Name | `MLFF` | Reference | Description |
| ---- | ---- | ---- | ---- |
| Allegro | `Allegro` | [10.1038/s41467-023-36329-y](https://doi.org/10.1038/s41467-023-36329-y) | Requires the `nequip-allegro` package |
| CHGNet | `CHGNet` | [10.1038/s42256-023-00716-3](https://doi.org/10.1038/s42256-023-00716-3) | Available via the `chgnet` and `matgl` packages |
| DeepMD | `DeepMD` | [10.1103/PhysRevB.108.L180104](https://doi.org/10.1103/PhysRevB.108.L180104) | The Deep Potential model used for this test is `UniPero`, a universal interatomic potential for perovskite oxides. It can be downloaded [here](https://github.com/sliutheorygroup/UniPero) |
| FAIRChem | `FAIRChem` | [Meta's FAIRChem Github](https://github.com/facebookresearch/fairchem) | Proprietary, requires extra authentication. [See notes below.](#fairchem-notes) |
| Gaussian Approximation Potential (GAP) | `GAP` | [10.1103/PhysRevLett.104.136403](https://doi.org/10.1103/PhysRevLett.104.136403) |  Relies on `quippy-ase` package |
| M3GNet | `M3GNet` | [10.1038/s43588-022-00349-3](https://doi.org/10.1038/s43588-022-00349-3) | Relies on `matgl` package |
| MACE-Field | `MACE_FIELD` | [10.1103/b116-xy8k](https://doi.org/10.1103/b116-xy8k) | Born effective charges and dielectric tensor only, via `ForceFieldDielectricMaker`. Relies on the MACE-Field fork of `mace_torch` and a model file. [See notes below.](#mace-field-notes) |
| MACE-MP-0 | `MACE` or `MACE_MP_0` (recommended) | [10.1063/5.0297006](https://doi.org/10.1063/5.0297006) | Relies on `mace_torch` and optionally `torch_dftd` packages |
| MACE-MP-0b3 | `MACE_MP_0B3` | [10.1063/5.0297006](https://doi.org/10.1063/5.0297006) | Relies on `mace_torch` and optionally `torch_dftd` packages |
| MACE-MPA-0 | `MACE_MPA_0` | [10.1063/5.0297006](https://doi.org/10.1063/5.0297006) | Relies on `mace_torch` and optionally `torch_dftd` packages |
| MatPES-PBE | `MATPES_PBE` | [10.48550/arXiv.2503.04070](https://doi.org/10.48550/arXiv.2503.04070) | Relies on `matgl`. Defaults to TensorNet architecture, but can also use M3GNet or CHGNet architectures via kwargs. See `atomate2.forcefields.utils._DEFAULT_CALCULATOR_KWARGS` for more options. |
| MatPES-r<sup>2</sup>SCAN | `MATPES_R2SCAN`| [10.48550/arXiv.2503.04070](https://doi.org/10.48550/arXiv.2503.04070) | Relies on `matgl`. Defaults to TensorNet architecture, but can also use M3GNet or CHGNet architectures via kwargs. See `atomate2.forcefields.utils._DEFAULT_CALCULATOR_KWARGS` for more options. |
| MatterSim | `MatterSim` | [arXiv:2405.04967](https://arxiv.org/abs/2405.04967) | Requires the `mattersim` package |
| Neuroevolution Potential (NEP) | `NEP` | [10.1103/PhysRevB.104.104309](https://doi.org/10.1103/PhysRevB.104.104309) | Relies on `calorine` package |
| Neural Equivariant Interatomic Potentials (Nequip) | `Nequip` | [10.1038/s41467-022-29939-5](https://doi.org/10.1038/s41467-022-29939-5) | Relies on the `nequip` package |
| SevenNet | `SevenNet` | [10.1021/acs.jctc.4c00190](https://doi.org/10.1021/acs.jctc.4c00190) | Relies on the `sevenn` package |
| Universal Point Edge Transformer (UPET) | `UPET` | [10.1038/s41467-025-65662-7](https://doi.org/10.1038/s41467-025-65662-7) | Relies on the `upet` package. Defaults to the "pet-mad-s" model. |

## Using custom forcefields by dictionary

`force_field_name` also accepts an import-like string, or MSONable dictionary to specify a custom ASE calculator class or function [^calculator-meta-type-annotation].
For example, a `Job` created with the either of the following two code snippets instantiates a `chgnet.model.dynamics.CHGNetCalculator` as the ASE calculator.
```python
# simple import string
job = ForceFieldStaticMaker(
    calculator_meta="chgnet.model.dynamics.CHGNetCalculator",
).make(structure)
```

or using `force_field_name` when

```python
# monty MSONable style
job = ForceFieldStaticMaker(
    calculator_meta={
        "@module": "chgnet.model.dynamics",
        "@callable": "CHGNetCalculator",
    }
).make(structure)
```
Note that one can also specify `force_field_name = {"@module": ...,"@callable": ...}` in the second example for backwards compatibility.
However, this may not be preserved in future versions, and `calculator_meta` is preferred.

[^calculator-meta-type-annotation]: In this context, the type annotation of the decoded dict should be either `Type[Calculator]` or `Callable[..., Calculator]`, where `Calculator` is from `ase.calculators.calculator`.

## CALPHAD workflow {#calphad}

`CalphadMaker` fits a CALPHAD database for a binary system with a force field.
It uses the `sqs2tdb` tool of [ATAT](https://axelvandewalle.github.io/www-avdw/atat/) ([van de Walle et al., 2017](https://doi.org/10.1016/j.calphad.2017.05.005)).
The [CALPHAD tutorial](https://github.com/materialsproject/atomate2/blob/main/tutorials/calphad_workflow.ipynb) runs it for Ni-Re with four force fields.

```{warning}
This workflow is new and has not been tested widely.
It might still change in future versions.
```

ATAT is not a Python package, so install it yourself.
`make` and `make install` in the ATAT folder build all of ATAT, copy it to `~/bin` and write the `~/.atat.rc` file that `sqs2tdb` reads.
This workflow only needs three of the ATAT programs, which build in a few seconds:

```bash
make -C atat/src cellcvrt nntouch lsfit
echo "set atatdir=$PWD/atat" > ~/.atat.rc
export PATH=$PWD/atat/src:$PATH
```

`sqs2tdb` calls the other ATAT programs by name, so the ATAT `src` folder must be on your `PATH`.
`SQS2TDB_CMD` in the atomate2 settings only sets how `sqs2tdb` itself is called.

The special quasirandom structures (SQS) of each lattice come from the ATAT database.
The default lattices are FCC_A1, BCC_A2, HCP_A3 and LIQUID.
Each solid SQS is relaxed, including the cell, with `fmax=0.001` eV/Å.
Each liquid SQS is repeated three times along each lattice vector, which gives 864 atoms for the 32-atom SQS of the database.
With 256 atoms, the liquid mixing energy of Ni-Re with MACE-OMAT-0-medium was still off by 2 kJ/mol.
It is melted for 10 ps at `melt_temperature`.
It is then run for 20 ps at `liquid_temperature`, and the first 5 ps are left out of the mean potential energy.
Both liquid runs are isotropic NPT at zero pressure, with no net momentum.
The mean squared displacement leaves out the motion of the centre of mass.
Finally, `sqs2tdb` fits the energies of each lattice and writes one TDB file.

```py
from jobflow import run_locally

from atomate2.forcefields.flows.calphad import CalphadMaker

maker = CalphadMaker.from_force_field_name(
    "MACE-MP-0", melt_temperature=4500, liquid_temperature=2800
)
maker.lattices = ["FCC_A1", "HCP_A3", "NI3SN_D019", "NI4MO_D1A", "LIQUID"]
maker.terms["NI3SN_D019"] = ["1,0:1,0", "2,0:1,0"]
maker.terms["NI4MO_D1A"] = ["1,0:1,0", "2,0:1,0"]
flow = maker.make(["Ni", "Re"])
responses = run_locally(flow, create_folders=True)
tdb = responses[flow.output.uuid][1].output.tdb
```

The liquid temperatures depend on the system, so there are no defaults.
Choose `melt_temperature` high enough that every composition melts with the force field.
All compositions are then run at the same `liquid_temperature`.
Energies taken at a different temperature for each composition would add the different heat capacities of the liquids to the mixing energy.
The liquid mixing terms still depend on `liquid_temperature`.
For Ni-Re with GRACE-2L-OMAT, L0 of the liquid changes by about −3.7 J/mol per K.
Choose `liquid_temperature` as low as possible while every composition stays liquid during the run, which may be below the melting point of the pure elements.
Check `mean_squared_displacement` of each liquid calculation in the output.
In a liquid it grows with the length of the run.
In a crystal it stays at the size of the thermal vibrations, well below 1 Å².
A liquid that crystallizes during the run also drops in energy, which shows in the energies of the liquid MD job.
Check also `is_force_converged` and `relaxation_strain` of each solid calculation.
The ATAT `checkrelax` help calls a `relaxation_strain` above 0.1 too large for a cluster expansion.

`terms` sets the lines of the `sqs2tdb` `terms.in` file of each lattice.
Each line has the form `order,level`, with one pair per sublattice separated by `:`.
Order 1 gives the end members and order 2 the binary interactions.
Level is the highest Redlich-Kister order.
The ordered lattices need the lattices of their pure element end members.
If an ordered lattice is fitted, the stable lattice of each element must be fitted too.
The fit job stops with an error otherwise.

For lattices in the SGTE database, such as FCC_A1, HCP_A3 and LIQUID, `sqs2tdb` takes the free energies of the pure elements from SGTE.
Only the mixing terms come from the force field, so the melting points of the pure elements are those of SGTE.
The liquid mixing terms are the excess energies at `liquid_temperature`, used at all temperatures.
That they change with `liquid_temperature` shows that the liquid also has an excess entropy, which the fit leaves out.
The vibrational and short-range order options of `sqs2tdb` are not used.

The TDB file can be read with [pycalphad](https://pycalphad.org), which is installed separately:

```py
from pycalphad import Database, binplot, variables as v

db = Database.from_string(tdb, fmt="tdb")
binplot(
    db,
    ["NI", "RE"],
    list(db.phases),
    {v.X("RE"): (0, 1, 0.01), v.T: (300, 3600, 10), v.P: 101325, v.N: 1},
)
```

## Notes on FairChem (Meta) models {#fairchem-notes}

The FAIRChem models provided by Meta require extra authentication via HuggingFace:
1. Request access to the UMA models [via HuggingFace](https://huggingface.co/facebook/UMA). You will need to set up a HuggingFace account. You will need to receive approval for the UMA models to proceed.
2. Install the HuggingFace CLI with `pip install 'huggingface_hub'`.
3. Run `huggingface-cli login` from a shell to authenticate your session. You will need to set up an access token.
4. You can now use the FAIRChem calculators. The general syntax for setting up a FAIRChem calculator in `atomate2` is:
```py
calculator_kwargs = {
    "predict_unit": {"model_name": "uma-s-1p1"},
    "task_name": "omat",
}
```

`atomate2` will then set up a `FAIRChemCalculator`:
```py
from atomate2.forcefields.utils import MLFF, _DEFAULT_CALCULATOR_KWARGS
from fairchem.core import FAIRChemCalculator, pretrained_mlip

predict_unit_kwargs = calculator_kwargs.pop(
    "predict_unit", _DEFAULT_CALCULATOR_KWARGS[MLFF.FAIRChem]["predict_unit"]
)
calculator = FAIRChemCalculator(
    pretrained_mlip.get_predict_unit(predict_unit_kwargs),
    **{k: v for k, v in calculator_kwargs.items() if k != "predict_unit"},
)
```

The default in `atomate2` is the OMat24 model with `uma-s-1p1`.

## Notes on MACE-Field {#mace-field-notes}

[MACE-Field](https://doi.org/10.1103/b116-xy8k) gives the Born effective charges and the high-frequency dielectric tensor of inorganic crystals.
`ForceFieldDielectricMaker` computes them.
The phonon workflows use them for the non-analytical correction when `ForceFieldDielectricMaker` is set as `born_maker`.

MACE-Field is a fork of MACE that installs as `mace_torch`, so it replaces MACE.
Install it with `pip install git+https://github.com/mdi-group/mace-field.git@45d5c5fa7b40a155855b3d155df1760e36849e64`, the commit atomate2 is tested with.
The fork also runs the MACE foundation models, so the forces can come from MACE-OMAT in the same environment.
The model file has to be downloaded from the [MACE-Field releases](https://github.com/mdi-group/mace-field/releases).
There is no default model, so `model` must be set.
Give its absolute path, since each job runs in its own folder:

```py
from pymatgen.core import Lattice, Structure

from atomate2.forcefields.flows.phonons import PhononMaker
from atomate2.forcefields.jobs import ForceFieldDielectricMaker

structure = Structure.from_spacegroup(
    "Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
).get_primitive_structure()

flow = PhononMaker.from_force_field_name(
    "MACE_MP_0",
    calculator_kwargs={"model": "medium-omat-0"},
    born_maker=ForceFieldDielectricMaker(
        calculator_kwargs={"model": "/path/to/MACEField-MH-0-omat-dielectric.model"}
    ),
).make(structure)
```

Born charges from VASP can be passed to `make` as `born` and `epsilon_static`.
They must be in the same order as the sites of the structure.

Limitations:

- Both tensors are computed at zero electric field by default. The dielectric tensor is the clamped-ion one, 1 + chi, with chi the electronic susceptibility of the model.
- The model has the heads "mp-dielectric", "mp-ferroelectric" and "pt_head". "mp-dielectric" is used by default.
- The model runs on the CPU unless `device` is set in `calculator_kwargs`.
- The fork reports itself as `mace-torch` 0.3.15, so this is the `forcefield_version` in the output.

```{warning}
This feature is new and has not been tested widely.
It might still change in future versions.
```

We compared the "mp-dielectric" head with the DFPT results in the Materials Project's Harmonic Phonon Database.
The 50 materials have no Materials Project dielectric data, so they are likely not in the training data of MACE-Field.
The Born charges are off by 0.26 e on average, about 11%.
The high-frequency dielectric constants are off by 8% on average, and by more than 20% for 2 of the 50 materials.
