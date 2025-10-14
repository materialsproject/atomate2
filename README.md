# atomate2

[![tests](https://img.shields.io/github/actions/workflow/status/materialsproject/atomate2/testing.yml?branch=main&label=tests)](https://github.com/materialsproject/atomate2/actions?query=workflow%3Atesting)
[![code coverage](https://img.shields.io/codecov/c/gh/materialsproject/atomate2)](https://codecov.io/gh/materialsproject/atomate2)
[![pypi version](https://img.shields.io/pypi/v/atomate2?color=blue)](https://pypi.org/project/atomate2)
![supported python versions](https://img.shields.io/pypi/pyversions/atomate2)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281/zenodo.10677080-blue?logo=Zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.15603088)
[![This project supports Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)
[![PyPI](https://img.shields.io/pypi/dm/atomate2.svg?maxAge=2592000)](https://pypi.python.org/pypi/atomate2)

[Documentation][docs] | [PyPI][pypi] | [GitHub][github]

Atomate2 is a free, open-source software for performing complex materials science
workflows using simple Python functions. Features of atomate2 include

- It is built on open-source libraries: [pymatgen], [custodian], [jobflow], and
  [jobflow-remote] or [FireWorks].
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
atomate2 workflows can be run using the [jobflow-remote] or [FireWorks] software. See the
[jobflow-remote-specific documentation][atomate2-jobflow-remote] or [fireworks-specific documentation][atomate2_fireworks] for more details.

## Installation

Atomate2 is a Python 3.10+ library and can be installed using pip. Full installation
and configuration instructions are provided in the [installation tutorial][installation].

## Tutorials

The documentation includes comprehensive tutorials and reference information to get you
started:

- [Introduction to running workflows][running-workflows]
- [Using atomate2 with FireWorks][atomate2_fireworks]
- [Overview of key concepts][key-concepts]
- [List of VASP workflows][vasp_workflows]
- [Executable tutorials for different workflows][tutorials]

In March 2025, the first dedicated school on atomate2 (including the workflow language jobflow and the workflow manager jobflow-remote) took place, and one can access the video material here:

- [Jobflow and Jobflow-remote][videotutorial1]
- [atomate2][videotutorial2]
- [Advanced Workflows in atomate2: Part 1][videotutorial3]
- [Advanced Workflows in atomate2: Part 2][videotutorial4]

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

If you use atomate2, please cite the [following article](https://doi.org/10.1039/D5DD00019J):

```bib
@article{ganose2025_atomate2,
	title = {Atomate2: modular workflows for materials science},
	author = {Ganose, Alex M. and Sahasrabuddhe, Hrushikesh and Asta, Mark and Beck, Kevin and Biswas, Tathagata and Bonkowski, Alexander and Bustamante, Joana and Chen, Xin and Chiang, Yuan and Chrzan, Daryl C. and Clary, Jacob and Cohen, Orion A. and Ertural, Christina and Gallant, Max C. and George, Janine and Gerits, Sophie and Goodall, Rhys E. A. and Guha, Rishabh D. and Hautier, Geoffroy and Horton, Matthew and Inizan, T. J. and Kaplan, Aaron D. and Kingsbury, Ryan S. and Kuner, Matthew C. and Li, Bryant and Linn, Xavier and McDermott, Matthew J. and Mohanakrishnan, Rohith Srinivaas and Naik, Aakash A. and Neaton, Jeffrey B. and Parmar, Shehan M. and Persson, Kristin A. and Petretto, Guido and Purcell, Thomas A. R. and Ricci, Francesco and Rich, Benjamin and Riebesell, Janosh and Rignanese, Gian-Marco and Rosen, Andrew S. and Scheffler, Matthias and Schmidt, Jonathan and Shen, Jimmy-Xuan and Sobolev, Andrei and Sundararaman, Ravishankar and Tezak, Cooper and Trinquet, Victor and Varley, Joel B. and Vigil-Fowler, Derek and Wang, Duo and Waroquiers, David and Wen, Mingjian and Yang, Han and Zheng, Hui and Zheng, Jiongzhi and Zhu, Zhuoying and Jain, Anubhav},
	year = {2025},
	journal = {Digital Discovery},
	doi = {10.1039/D5DD00019J},
	url = {https://doi.org/10.1039/D5DD00019J},
	urldate = {2025-07-01},
}
```

[pymatgen]: https://pymatgen.org
[fireworks]: https://materialsproject.github.io/fireworks/
[jobflow]: https://materialsproject.github.io/jobflow/
[jobflow-remote]: https://github.com/Matgenix/jobflow-remote
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
[key-concepts]: https://materialsproject.github.io/atomate2/user/key_concepts_overview.html#key-concepts-in-atomate2-job-flow-makers-inputset-taskdocument-and-builder
[atomate2_fireworks]: https://materialsproject.github.io/atomate2/user/fireworks.html
[atomate2-jobflow-remote]: https://materialsproject.github.io/atomate2/user/jobflow-remote.html
[vasp_workflows]: https://materialsproject.github.io/atomate2/user/codes/vasp.html
[tutorials]: https://materialsproject.github.io/atomate2/tutorials/tutorials.html
[RelaxBandStructure]: https://materialsproject.github.io/atomate2/user/codes/vasp.html#relax-and-band-structure
[Lobster]: http://www.cohp.de
[lobsterpy]: https://github.com/JaGeo/LobsterPy
[phonopy]: https://github.com/phonopy/phonopy
[docs]: https://materialsproject.github.io/atomate2/
[github]: https://github.com/materialsproject/atomate2
[pypi]: https://pypi.org/project/atomate2
[videotutorial1]: https://lhumos.org/collection/0/680bb4d7e4b0f0d2028027ce
[videotutorial2]: https://lhumos.org/collection/0/680bb4d3e4b0f0d2028027c9
[videotutorial3]: https://lhumos.org/collection/0/680bb4d0e4b0f0d2028027c5
[videotutorial4]: https://lhumos.org/collection/0/680bb4c7e4b0f0d2028027c1
