# atomate2

[![tests](https://img.shields.io/github/actions/workflow/status/materialsproject/atomate2/testing.yml?branch=main&label=tests)](https://github.com/materialsproject/atomate2/actions?query=workflow%3Atesting)
[![code coverage](https://img.shields.io/codecov/c/gh/materialsproject/atomate2)](https://codecov.io/gh/materialsproject/atomate2)
[![pypi version](https://img.shields.io/pypi/v/atomate2?color=blue)](https://pypi.org/project/atomate2)
![supported python versions](https://img.shields.io/pypi/pyversions/atomate2)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281/zenodo.10677081-blue?logo=Zenodo&logoColor=white)](https://zenodo.org/records/10677081)
[![This project supports Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)

[Documentation][docs] | [PyPI][pypi] | [GitHub][github]

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

## Workflows

Some of the workflows available in atomate2 are:

- electronic band structures
- elastic, dielectric, and piezoelectric tensors
- one-shot electron-phonon interactions
- electronic transport using [AMSET]
- phonons using [phonopy]
- defect formation energy diagrams
- [Lobster] bonding analysis with [lobsterpy]

It is easy to customise and compose any of the above workflows.

## Quick start

Workflows in atomate2 are written using the [jobflow] library. Workflows are generated using
`Maker` objects which have a consistent API for modifying input settings and chaining
workflows together. Below, we demonstrate how to run a band structure workflow
(see the [documentation][RelaxBandStructure] for more details). In total, 4 VASP
calculations will be performed:

1. A structural optimisation.
2. A self-consistent static calculation on the relaxed geometry.
3. A non-self-consistent calculation on a uniform k-point mesh (for the density of
   states).
4. A non-self-consistent calculation on a high symmetry k-point path (for the line mode
   band structure).

```py
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

# run the flow
run_locally(bandstructure_flow, create_folders=True)
```

Before the above code can run successfully, you'll need to

- tell pymatgen where to [find your pseudopotential files](https://pymatgen.org/installation.html#potcar-setup)
- tell atomate2 where to find your VASP binary
- (optionally) prepare an external database to store the job output

See the [installation] steps for details how to set all of this up.

In this example, we execute the workflow immediately. In many cases, you might want
to perform calculations on several materials simultaneously. To achieve this, all
atomate2 workflows can be run using the [FireWorks] software. See the
[documentation][atomate2_fireworks] for more details.

## Installation

Atomate2 is a Python 3.10+ library and can be installed using pip. Full installation
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

The development of atomate2 has benefited from many people across several research groups.
A full list of contributors can be found [here][contributors].

## Citing atomate2

A journal submission for `atomate2` is planned. In the meantime, please use [`citation.cff`](citation.cff) and the [Zenodo record](https://zenodo.org/badge/latestdoi/306414371) to cite `atomate2`.

```bib
@software{ganose_atomate2_2024,
  author = {Ganose, Alex and Riebesell, Janosh and George, J. and Shen, Jimmy and S. Rosen, Andrew and Ashok Naik, Aakash and nwinner and Wen, Mingjian and rdguha1995 and Kuner, Matthew and Petretto, Guido and Zhu, Zhuoying and Horton, Matthew and Sahasrabuddhe, Hrushikesh and Kaplan, Aaron and Schmidt, Jonathan and Ertural, Christina and Kingsbury, Ryan and McDermott, Matt and Goodall, Rhys and Bonkowski, Alexander and Purcell, Thomas and Zügner, Daniel and Qi, Ji},
  doi = {10.5281/zenodo.10677081},
  license = {cc-by-4.0},
  month = jan,
  title = {atomate2},
  url = {https://github.com/materialsproject/atomate2},
  version = {0.0.13},
  year = {2024}
}
```

[pymatgen]: https://pymatgen.org
[fireworks]: https://materialsproject.github.io/fireworks/
[jobflow]: https://materialsproject.github.io/jobflow/
[custodian]: https://materialsproject.github.io/custodian/
[VASP]: https://www.vasp.at
[AMSET]: https://hackingmaterials.lbl.gov/amset/
[help-forum]: https://matsci.org/c/atomate
[issues]: https://github.com/materialsproject/atomate2/issues
[changelog]: https://materialsproject.github.io/atomate2/about/changelog.html
[installation]: https://materialsproject.github.io/atomate2/user/install.html
[contributing]: https://materialsproject.github.io/atomate2/about/contributing.html
[contributors]: https://materialsproject.github.io/atomate2/about/contributors.html
[license]: https://raw.githubusercontent.com/materialsproject/atomate2/main/LICENSE
[running-workflows]: https://materialsproject.github.io/atomate2/user/running-workflows.html
[atomate2_fireworks]: https://materialsproject.github.io/atomate2/user/fireworks.html
[vasp_workflows]: https://materialsproject.github.io/atomate2/user/codes/vasp.html
[RelaxBandStructure]: https://materialsproject.github.io/atomate2/user/codes/vasp.html#relax-and-band-structure
[Lobster]: http://www.cohp.de
[lobsterpy]: https://github.com/JaGeo/LobsterPy
[phonopy]: https://github.com/phonopy/phonopy
[docs]: https://materialsproject.github.io/atomate2/
[github]: https://github.com/materialsproject/atomate2
[pypi]: https://pypi.org/project/atomate2
