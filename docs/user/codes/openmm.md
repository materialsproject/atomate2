## Installing Atomate2 from source with OpenMM

```bash
# setting up our conda environment
>>> conda create -n atomate2 python=3.11
>>> conda activate atomate2

# installing atomate2
>>> pip install git+https://github.com/orionarcher/atomate2.git

# installing classical_md dependencies
>>> conda install -c conda-forge --file .github/classical_md_requirements.txt
```

Alternatively, if you anticipate regularly updating
atomate2 from source (which at this point, you should),
you can clone the repository and install from source.

``` bash
# installing atomate2
>>> git clone https://github.com/orionarcher/atomate2.git
>>> cd atomate2
>>> git branch openff
>>> git checkout openff
>>> git pull origin openff
>>> pip install -e .
```

To test the openmm installation, you can run the following command. If
you intend to run on GPU, make sure that the tests are passing for CUDA.

```bash
>>> python -m openmm.testInstallation
```

## Understanding Atomate2 OpenMM

Atomate2 is really just a collection of jobflow workflows relevant to
materials science. In all the workflows, we pass our system of interest
between different jobs to perform the desired simulation. Representing the
intermediate state of a classical molecular dynamics simulation, however,
is challenging. While the intermediate representation between stages of
a periodic DFT simulation can include just the elements, xyz coordinates,
and box vectors, classical molecular dynamics systems must also include
velocities and forces. The latter is particularly challenging because
all MD engines represent forces differently. Rather than implement our
own representation, we use the `openff.interchange.Interchange` object,
which catalogs the necessary system properties and interfaces with a
variety of MD engines. This is the object that we pass between stages of
a classical MD simulation and it is the starting point of our workflow.

### Pouring a Glass of Wine

The first job we need to create generates the `Interchange` object.
To specify the system of interest, we use give it the SMILES strings,
counts, and names (optional) of the molecules we want to include.


```python
from atomate2.classical_md.core import generate_interchange

mol_specs_dicts = [
    {"smile": "O", "count": 200, "name": "water"},
    {"smile": "CCO", "count": 10, "name": "ethanol"},
    {"smile": "C1=C(C=C(C(=C1O)O)O)C(=O)O", "count": 1, "name": "gallic_acid"},
]

gallic_interchange_job = generate_interchange(mol_specs_dicts, 1.3)
```

If you are wondering what arguments are allowed in the dictionaries, check
out the `create_mol_spec` function in the `atomate2.classical_md.utils`
module. Under the hood, this is being called on each mol_spec dict.
Meaning the code below is functionally identical to the code above.


```python
from atomate2.classical_md.utils import create_mol_spec

mols_specs = [create_mol_spec(**mol_spec_dict) for mol_spec_dict in mol_specs_dicts]

generate_interchange(mols_specs, 1.3)
```

In a more complex simulation we might want to scale the ion charges
and include custom partial charges. An example with the Gen2
electrolyte is shown below. This yields the `elyte_interchange_job`
object, which we can pass to the next stage of the simulation.

NOTE: It's actually mandatory to include partial charges
for PF6- here, the built in partial charge method fails.


```python
import numpy as np
from pymatgen.core.structure import Molecule


pf6 = Molecule(
    ["P", "F", "F", "F", "F", "F", "F"],
    [
        [0.0, 0.0, 0.0],
        [1.6, 0.0, 0.0],
        [-1.6, 0.0, 0.0],
        [0.0, 1.6, 0.0],
        [0.0, -1.6, 0.0],
        [0.0, 0.0, 1.6],
        [0.0, 0.0, -1.6],
    ],
)
pf6_charges = np.array([1.34, -0.39, -0.39, -0.39, -0.39, -0.39, -0.39])

mol_specs_dicts = [
    {"smile": "C1COC(=O)O1", "count": 100, "name": "EC"},
    {"smile": "CCOC(=O)OC", "count": 100, "name": "EMC"},
    {
        "smile": "F[P-](F)(F)(F)(F)F",
        "count": 50,
        "name": "PF6",
        "partial_charges": pf6_charges,
        "geometry": pf6,
        "charge_scaling": 0.8,
        "charge_method": "RESP",
    },
    {"smile": "[Li+]", "count": 50, "name": "Li", "charge_scaling": 0.8},
]

elyte_interchange_job = generate_interchange(mol_specs_dicts, 1.3)
```

### The basic simulation

To run a production simulation, we will create a production flow,
link it to our `elyte_interchange_job`, and then run both locally.

In jobflow, jobs and flows are created by
[Makers](https://materialsproject.github.io/jobflow/tutorials/6-makers.html),
which can then be linked into more complex flows. The production maker links
together makers for energy minimization, pressure equilibration, annealing, and
a nvt simulation. The anneal maker itself creates a flow that links together nvt
and tempchange makers (it uses the `from_temps_and_steps` method to save us from
creating three more jobs manually). When linked up the `generate_interchange`
job this yields a production ready molecular dynamics workflow.


```python
from atomate2.classical_md.openmm.flows.core import AnnealMaker, ProductionMaker
from atomate2.classical_md.openmm.jobs.core import (
    EnergyMinimizationMaker,
    NPTMaker,
    NVTMaker,
)
from jobflow import Flow, run_locally


production_maker = ProductionMaker(
    name="production_flow",
    energy_maker=EnergyMinimizationMaker(traj_interval=10, state_interval=10),
    npt_maker=NPTMaker(n_steps=100),
    anneal_maker=AnnealMaker.from_temps_and_steps(n_steps=150),
    nvt_maker=NVTMaker(n_steps=100),
)

production_flow = production_maker.make(
    elyte_interchange_job.output.interchange,
    prev_task=elyte_interchange_job.output,
    output_dir="./tutorial_system",
)

run_locally(Flow([elyte_interchange_job, production_flow]))
```


When the above code is executed, you should expect to see something like this:

```
/tutorial_system
├── state.csv
├── state2.csv
├── state3.csv
├── state4.csv
├── state5.csv
├── state6.csv
├── taskdoc.json
├── trajectory.dcd
├── trajectory2.dcd
├── trajectory3.dcd
├── trajectory4.dcd
├── trajectory5.dcd
├── trajectory6.dcd
```

We see that each job saved a separate state and trajectory file. There are 6
because the `AnnealMaker` creates 3 sub-jobs and the `EnergyMinimizationMaker`
does not report anything. We also see a `taskdoc.json` file, which contains the
metadata for the entire workflow. This is needed when we later want to do
downstream analysis in `emmet`.

## More Options

Atomate2 OpenMM supports running a variety of workflows with different
configurations. Below we dig in to some of the more advanced options.

### Configuring the Simulation

All OpenMM jobs, i.e. anything in `atomate2.classical_md.openmm.jobs`, inherits
from the `BaseOpenMMMaker` class. `BaseOpenMMMaker` is highly configurable, you
can change the timestep, temperature, reporting frequencies, output types, and
a range of other properties. See the docstring for the full list of options.

Note that when instantiating the `ProductionMaker` above, we only set the
`traj_interval` and `state_interval` once, inside `EnergyMinimizationMaker`.
This is a key feature: all makers will inherit attributes from the previous
maker if they are not explicitly reset. This allows you to set the timestep
once and have it apply to all stages of the simulation. More explicitly,
the value inheritance is as follows: 1) any explicitly set value, 2)
the value from the previous maker, 3) the default value, shown below.


```python
from atomate2.classical_md.openmm.jobs.base import OPENMM_MAKER_DEFAULTS

print(OPENMM_MAKER_DEFAULTS)

{
    "step_size": 0.001,
    "temperature": 298,
    "friction_coefficient": 1,
    "platform_name": "CPU",
    "platform_properties": {},
    "state_interval": 1000,
    "state_file_name": "state",
    "traj_interval": 10000,
    "wrap_traj": False,
    "report_velocities": False,
    "traj_file_name": "trajectory",
    "traj_file_type": "dcd",
    "embed_traj": False,
}
```

Perhaps we want to embed the trajectory in the taskdoc, so that it
can be saved to the database, but only for our final run so we don't
waste space. AND we also want to add some tags, so we can identify
the simulation in our database more easily. Finally, we want to run
for much longer, more appropriate for a real production workflow.

```python
production_maker = ProductionMaker(
    name="production_flow",
    energy_maker=EnergyMinimizationMaker(traj_interval=0),
    npt_maker=NPTMaker(n_steps=1000000),
    anneal_maker=AnnealMaker.from_temps_and_steps(n_steps=1500000),
    nvt_maker=NVTMaker(
        n_steps=5000000, traj_interval=10000, embed_traj=True, tags=["production"]
    ),
)

production_flow = production_maker.make(
    elyte_interchange_job.output.interchange,
    prev_task=elyte_interchange_job.output,
    output_dir="./tutorial_system",
)

run_locally(Flow([elyte_interchange_job, production_flow]))
```



### Running with Databases

Before trying this, you should have a basic understanding of JobFlow
and [Stores](https://materialsproject.github.io/jobflow/stores.html).

To log OpenMM results to a database, you'll need to set up both a MongoStore,
for taskdocs, and blob storage, for trajectories. Here, I'll show you the
correct jobflow.yaml file to use the MongoDB storage and MinIO S3 storage
provided by NERSC. To get this up, you'll need to contact NERSC to get accounts
on their MongoDB and MinIO services. Then you can follow the instructions in
the [Stores](https://materialsproject.github.io/jobflow/stores.html) tutorial
to link jobflow to your databases. Your `jobflow.yaml` should look like this:

```yaml
JOB_STORE:
  docs_store:
    type: MongoStore
    database: DATABASE
    collection_name: atomate2_docs # suggested
    host: mongodb05.nersc.gov
    port: 27017
    username: USERNAME
    password: PASSWORD

  additional_stores:
      data:
          type: S3Store
          index:
              type: MongoStore
              database: DATABASE
              collection_name: atomate2_blobs_index # suggested
              host: mongodb05.nersc.gov
              port: 27017
              username: USERNAME
              password: PASSWORD
              key: blob_uuid
          bucket: oac
          s3_profile: oac
          s3_resource_kwargs:
              verify: false
          endpoint_url: https://next-gen-minio.materialsproject.org/
          key: blob_uuid
```

NOTE: This can work with any MongoDB and S3 storage, not just NERSC's.

As shown in the production example above, you'll need to set the `embed_traj`
property to `True` in any makers where you want to save the trajectory to
the database. Otherwise, the trajectory will only be saved locally.

Rather than use `jobflow.yaml`, you could also create the stores in
Python and pass the stores to the `run_locally` function. This is shown
below for completeness but the prior method is usually recommended.

<details>
    <summary>Configuring a JobStore in Python</summary>

    ```python
    from jobflow import run_locally, JobStore
    from maggma.stores import MongoStore, S3Store, MemoryStore

    md_doc_store = MongoStore(
        username="USERNAME",
        password="PASSWORD",
        database="DATABASE",
        collection_name="atomate2_docs",  # suggested
        host="mongodb05.nersc.gov",
        port=27017,
    )

    md_blob_index = MongoStore(
        username="USERNAME",
        password="PASSWORD",
        database="DATABASE",
        collection_name="atomate2_blobs_index",  # suggested
        host="mongodb05.nersc.gov",
        port=27017,
        key="blob_uuid",
    )

    md_blob_store = S3Store(
        index=md_blob_index,
        bucket="BUCKET",
        s3_profile="PROFILE",
        endpoint_url="https://next-gen-minio.materialsproject.org",
        key="blob_uuid",
    )

    wf = []  # set up whatever workflow you'd like to run

    # run the flow with our custom store
    run_locally(
        wf,
        store=JobStore(md_doc_store, additional_stores={"data": md_blob_store}),
        ensure_success=True,
    )
    ```
</details>


### Running on GPU(s)

Running on a GPU is nearly as simple as running on a CPU. The only difference
is that you need to specify the `platform_properties` argument in the
`EnergyMinimizationMaker` with the `DeviceIndex` of the GPU you want to use.


```python
production_maker = ProductionMaker(
    name="test_production",
    energy_maker=EnergyMinimizationMaker(
        platform_name="CUDA",
        platform_properties={"DeviceIndex": "0"},
    ),
    npt_maker=NPTMaker(n_steps=100),
    anneal_maker=AnnealMaker.from_temps_and_steps(n_steps=150),
    nvt_maker=NVTMaker(n_steps=1000),
)
```

Some systems (notably perlmutter) have multiple GPUs available on a
single node. To fully leverage the compute, you'll need to distribute
4 simulations across the 4 GPUs. A simple way to do this is with MPI.

First you'll need to install mpi4py.

```bash
>>> conda install mpi4py
```

Then you can modify and run the following script to distribute the work across the GPUs.


```python
# other imports

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

list_of_mol_spec_lists = []
# logic to add four mol_spec_lists to list_of_mol_spec_lists


flows = []
for i in range(4):
    device_index = i
    mol_specs = list_of_mol_spec_lists[i]

    setup = generate_interchange(mol_specs, 1.0)

    production_maker = ProductionMaker(
        name="test_production",
        energy_maker=EnergyMinimizationMaker(
            platform_name="CUDA",
            platform_properties={"DeviceIndex": str(device_index)},
        ),
        npt_maker=NPTMaker(n_steps=200000),
        anneal_maker=AnnealMaker.from_temps_and_steps(n_steps=1500000),
        nvt_maker=NVTMaker(n_steps=5000000, embed_traj=True),
    )

    production_flow = production_maker.make(
        setup.output.interchange,
        prev_task=setup.output,
        output_dir=f"/pscratch/sd/o/oac/openmm_runs/{i}",
    )
    flows.append(Flow([setup, production_flow]))

# this script will run four times, each with a different rank, thus distributing the work across the four GPUs.
run_locally(flows[rank], ensure_success=True)
```
