

### Installing Atomate2 with OpenMM

```bash
# setting up our conda environment
>>> conda create -n atomate2 python=3.11
>>> conda activate atomate2
>>> git
# installing atomate2
>>> git clone https://github.com/orionarcher/atomate2.git
>>> cd atomate2
>>> git branch openff
>>> git checkout openff
>>> git pull origin openff
>>> pip install -e .
# installing classical_md dependencies
>>> conda install -c conda-forge --file .github/classical_md_requirements.txt
```


### Testing installation
```bash
# make sure that tests are passing for CUDA
>>> python -m openmm.testInstallation
```

### Example Running on Perlmutter

```python
from atomate2.classical_md.core import generate_interchange
import numpy as np
from jobflow import run_locally, Flow
from atomate2.classical_md.openmm.flows.core import AnnealMaker, ProductionMaker
from atomate2.classical_md.openmm.jobs.core import (
    EnergyMinimizationMaker,
    NPTMaker,
    NVTMaker,
)
from pymatgen.core.structure import Molecule

charges = np.array([1.34, -0.39, -0.39, -0.39, -0.39, -0.39, -0.39])

positions_str = """
7

P 0.0 0.0 0.0
F 1.6 0.0 0.0
F -1.6 0.0 0.0
F 0.0 1.6 0.0
F 0.0 -1.6 0.0
F 0.0 0.0 1.6
F 0.0 0.0 -1.6
"""

# Create the Molecule object from the positions string
molecule = Molecule.from_str(positions_str, fmt="xyz")

mol_specs_dicts = [
    {"smile": "C1COC(=O)O1", "count": 100, "name": "EC"},
    {"smile": "CCOC(=O)OC", "count": 100, "name": "EMC"},
    {
        "smile": "F[P-](F)(F)(F)(F)F",
        "count": 50,
        "name": "PF6",
        "partial_charges": charges,
        "geometry": molecule,
        "charge_scaling": 0.8,
        "charge_method": "RESP",
    },
    {"smile": "[Li+]", "count": 50, "name": "Li", "charge_scaling": 0.8},
]

setup = generate_interchange(mol_specs_dicts, 1.3)


production_maker = ProductionMaker(
    name="test_production",
    energy_maker=EnergyMinimizationMaker(
        platform_name="CUDA",
        platform_properties={"DeviceIndex": "0"},
    ),
    npt_maker=NPTMaker(n_steps=100000),
    anneal_maker=AnnealMaker.from_temps_and_steps(n_steps=1500000),
    nvt_maker=NVTMaker(n_steps=2000000),
)

production_flow = production_maker.make(
    setup.output.interchange,
    prev_task=setup.output,
    output_dir="/pscratch/sd/o/oac/scratch",
)

run_locally(Flow([setup, production_flow]), ensure_success=True)
```
