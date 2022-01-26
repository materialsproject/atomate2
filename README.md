# atomate2

<a href="https://github.com/materialsproject/atomate2/actions?query=workflow%3Atesting"><img alt="code coverage" src="https://img.shields.io/github/workflow/status/materialsproject/atomate2/testing?label=tests"></a>
<a href="https://codecov.io/gh/materialsproject/atomate2/"><img alt="code coverage" src="https://img.shields.io/codecov/c/gh/materialsproject/atomate2"></a>
<a href="https://pypi.org/project/atomate2"><img alt="pypi version" src="https://img.shields.io/pypi/v/atomate2?color=blue"></a>
<img alt="supported python versions" src="https://img.shields.io/pypi/pyversions/atomate2">

**👉 [Full Documentation][docs] 👈**

Atomate2 is a free, open-source software for performing complex materials science
workflows using simple Python functions. Features of atomate2 include

- It is built on open-source libraries: [pymatgen], [custodian], [jobflow], and
  [FireWorks].
- A library of "standard" workflows to compute a wide variety of desired materials
  properties.
- The ability scale from a single material, to 100 materials, or 100,000 materials.
- Easy routes to modifying and chaining workflows together.
- It can build large databases of output properties that you can query, analyze, and
  share in a systematic way.
- It automatically keeps meticulous records of jobs, their directories, runtime
  parameters, and more.

**Note**: Atomate2 is primarily built to work with the [VASP] electronic structure
software, but we are actively working on adding more codes.

## Workflows

Some of the workflows available in atomate2 are:

- electronic band structures
- electronic transport using [AMSET]
- full elastic tensor
- dielectric tensor

It is easy to customise and compose any of the above workflows.

## Quick start

Workflows in atomate2 written using the [jobflow] library. Workflows are generated using
`Maker` objects, that have a consistent API for modifying input settings and chaining
workflows together.  Below, we demonstrate how to run a band structure workflow
(see the [documentation][RelaxBandStructure] for more details). In total, 4 VASP
calculations will be performed:

1. A structural optimisation.
2. A self-consistent static calculation on the relaxed geometry.
3. A non-self-consistent calculation on a uniform k-point mesh (for the density of
   states).
4. A non-self-consistent calculation on a high symmetry k-point path (for the line mode
   band structure).

```python
from atomate2.vasp.flows.core import RelaxBandStructureMaker
from jobflow import run_locally
from pymatgen.core import Structure

# construct a rock salt MgO structure
mgo_structure = Structure(
    lattice=[[0, 2.13, 2.13], [2.13, 0, 2.13], [2.13, 2.13, 0]],
    species=["Mg", "O"],
    coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
)

# make a band structure flow to optimise the structure and obtain the band structure
bandstructure_flow = RelaxBandStructureMaker().make(mgo_structure)

# run the job
run_locally(bandstructure_flow, create_folders=True)
```

In this example, we run execute the workflow immediately. In many cases, you might want
to perform calculations on several materials simulatenously. To achieve this, all
atomate2 workflows can be run using the [FireWorks] software. See the
[documentation][atomate2_fireworks] for more details.

## Installation

Atomate2 is a Python 3.8+ library and can be installed using pip. Full installation
and configuration instructions are provided in the [installation tutorial][installation].

## Tutorials

The documentation includes comprehensive tutorials and reference information to get you
started:

- [Introduction to running workflows][running-workflows]
- [Using atomate2 with FireWorks][atomate2_fireworks]
- [List of VASP workflows][vasp_workflows]

## Need help?

Ask questions about atomate2 on the [atomate2 support forum][help-forum].
If you've found an issue with atomate2, please submit a bug report on [GitHub Issues][issues].

## What’s new?

Track changes to atomate2 through the [changelog][changelog].

## Contributing

We greatly appreciate any contributions in the form of a pull request.
Additional information on contributing to atomate2 can be found [here][contributing].
We maintain a list of all contributors [here][contributors].

## License

Atomate2 is released under a modified BSD license; the full text can be found [here][license].

## Acknowledgements

Atomate2 was designed and developed by Alex Ganose.

A full list of all contributors can be found [here][contributors].

[maggma]: https://materialsproject.github.io/maggma/
[pymatgen]: https://pymatgen.org
[fireworks]: https://materialsproject.github.io/fireworks/
[jobflow]: https://materialsproject.github.io/jobflow/
[custodian]: https://materialsproject.github.io/custodian/
[VASP]: https://www.vasp.at
[AMSET]: https://hackingmaterials.lbl.gov/amset/
[help-forum]: https://matsci.org/c/atomate
[issues]: https://github.com/materialsproject/atomate2/issues
[changelog]: https://materialsproject.github.io/atomate2/user/changelog.html
[installation]: https://materialsproject.github.io/atomate2/user/install.html
[contributing]: https://materialsproject.github.io/atomate2/user/contributing.html
[contributors]: https://materialsproject.github.io/atomate2/user/contributors.html
[license]: https://raw.githubusercontent.com/materialsproject/atomate2/main/LICENSE
[running-workflows]: https://materialsproject.github.io/atomate2/user/running-workflows.html
[atomate2_fireworks]: https://materialsproject.github.io/atomate2/user/fireworks.html
[vasp_workflows]: https://materialsproject.github.io/atomate2/user/codes/vasp.html
[RelaxBandStructure]: https://materialsproject.github.io/atomate2/user/codes/vasp.html#relax-and-band-structure
[docs]: https://materialsproject.github.io/atomate2/
