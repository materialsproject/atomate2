# Installing Atomate2 from source with OpenMM

```bash
# setting up our conda environment
>>> conda create -n atomate2 python=3.11
>>> conda activate atomate2

# installing atomate2
>>> pip install git+https://github.com/orionarcher/atomate2

# installing classical_md dependencies
>>> conda install -c conda-forge --file .github/classical_md_requirements.txt
```

Alternatively, if you anticipate regularly updating
atomate2 from source (which at this point, you should),
you can clone the repository and install from source.

``` bash
# installing atomate2
>>> git clone https://github.com/orionarcher/atomate2
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

### Setting up the system

The first job we need to create generates the `Interchange` object.
To specify the system of interest, we use give it the SMILES strings,
counts, and names (optional) of the molecules we want to include.

```python
from atomate2.openff.core import generate_interchange

mol_specs_dicts = [
    {"smiles": "O", "count": 200, "name": "water"},
    {"smiles": "CCO", "count": 10, "name": "ethanol"},
    {"smiles": "C1=C(C=C(C(=C1O)O)O)C(=O)O", "count": 1, "name": "gallic_acid"},
]

gallic_interchange_job = generate_interchange(mol_specs_dicts, 1.3)
```

If you are wondering what arguments are allowed in the dictionaries, check
out the `create_mol_spec` function in the `atomate2.openff.utils`
module. Under the hood, this is being called on each mol_spec dict.
Meaning the code below is functionally identical to the code above.

```python
from atomate2.openff.utils import create_mol_spec

mols_specs = [create_mol_spec(**mol_spec_dict) for mol_spec_dict in mol_specs_dicts]

generate_interchange(mols_specs, 1.3)
```

In a more complex simulation we might want to scale the ion charges
and include custom partial charges. An example with a EC:EMC:LiPF6
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
    {"smiles": "C1COC(=O)O1", "count": 100, "name": "EC"},
    {"smiles": "CCOC(=O)OC", "count": 100, "name": "EMC"},
    {
        "smiles": "F[P-](F)(F)(F)(F)F",
        "count": 50,
        "name": "PF6",
        "partial_charges": pf6_charges,
        "geometry": pf6,
        "charge_scaling": 0.8,
        "charge_method": "RESP",
    },
    {"smiles": "[Li+]", "count": 50, "name": "Li", "charge_scaling": 0.8},
]

elyte_interchange_job = generate_interchange(mol_specs_dicts, 1.3)
```

### Running a basic simulation

To run a production simulation, we will create a production flow,
link it to our `elyte_interchange_job`, and then run both locally.

In jobflow, jobs and flows are created by
[Makers](https://materialsproject.github.io/jobflow/tutorials/6-makers.html),
which can then be linked into more complex flows. Here, `OpenMMFlowMaker` links
together makers for energy minimization, pressure equilibration, annealing,
and a nvt simulation. The annealing step is a subflow that saves us from manually
instantiating three separate jobs.

Finally, we create our production flow and link to the `generate_interchange` job,
yielding a production ready molecular dynamics workflow.

```python
from atomate2.openmm.flows.core import OpenMMFlowMaker
from atomate2.openmm.jobs.core import (
    EnergyMinimizationMaker,
    NPTMaker,
    NVTMaker,
)
from jobflow import Flow, run_locally


production_maker = OpenMMFlowMaker(
    name="production_flow",
    makers=[
        EnergyMinimizationMaker(traj_interval=10, state_interval=10),
        NPTMaker(n_steps=100),
        OpenMMFlowMaker.anneal_flow(n_steps=150),
        NVTMaker(n_steps=100),
    ],
)

production_flow = production_maker.make(
    elyte_interchange_job.output.interchange,
    prev_dir=elyte_interchange_job.output.dir_name,
    output_dir="./tutorial_system",
)

run_locally(Flow([elyte_interchange_job, production_flow]))
```

Above, we are running a very short simulation (350 steps total) and reporting out
the trajectory and state information very frequently. For a more realistic
simulation, see the "Configuring the Simulation" section below.

When the above code is executed, you should expect to see this within the
`tutorial_system` directory:

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

Each job saved a separate state and trajectory file. There are 6 because
the anneal flow creates 3 sub-jobs and the `EnergyMinimizationMaker`
does not report anything. The `taskdoc.json` file contains the metadata
for the entire workflow.

Awesome! At this point, we've run a workflow and could start analyzing
our data. Before we get there though, let's go through some of the
other simulation options available.

## Digging Deeper

Atomate2 OpenMM supports running a variety of workflows with different
configurations. Below we dig in to some of the more advanced options.

### Configuring the Simulation

<details>
<summary>Learn more about the configuration of OpenMM simulations</summary>

All OpenMM jobs, i.e. anything in `atomate2.openmm.jobs`, inherits
from the `BaseOpenMMMaker` class. `BaseOpenMMMaker` is highly configurable, you
can change the timestep, temperature, reporting frequencies, output types, and
a range of other properties. See the docstring for the full list of options.

Note that when instantiating the `OpenMMFlowMaker` above, we only set the
`traj_interval` and `state_interval` once, inside `EnergyMinimizationMaker`.
This is a key feature: all makers will inherit attributes from the previous
maker if they are not explicitly reset. This allows you to set the timestep
once and have it apply to all stages of the simulation. The value inheritance
is as follows: 1) any explicitly set value, 2) the value from the previous
maker, 3) the default value (as shown below).

```python
from atomate2.openmm.jobs.base import OPENMM_MAKER_DEFAULTS

print(OPENMM_MAKER_DEFAULTS)
```

```py
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
production_maker = OpenMMFlowMaker(
    name="production_flow",
    tags=["tutorial_production_flow"],
    makers=[
        EnergyMinimizationMaker(traj_interval=0),
        NPTMaker(n_steps=1000000),
        OpenMMFlowMaker.anneal_flow(n_steps=1500000),
        NVTMaker(n_steps=5000000, traj_interval=10000, embed_traj=True),
    ],
)

production_flow = production_maker.make(
    elyte_interchange_job.output.interchange,
    prev_dir=elyte_interchange_job.output.dir_name,
    output_dir="./tutorial_system",
)

run_locally(Flow([elyte_interchange_job, production_flow]))
```

</details>

### Running with Databases

<details>
<summary>Learn to upload your MD data to databases</summary>

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
Python and pass the stores to the `run_locally` function. This is a bit
more code, so usually the prior method is preferred.

```python
from jobflow import run_locally, JobStore
from maggma.stores import MongoStore, S3Store

mongo_info = {
    "username": "USERNAME",
    "password": "PASSWORD",
    "database": "DATABASE",
    "host": "mongodb05.nersc.gov",
}

md_doc_store = MongoStore(**mongo_info, collection_name="atomate2_docs")

md_blob_index = MongoStore(
    **mongo_info,
    collection_name="atomate2_blobs_index",
    key="blob_uuid",
)

md_blob_store = S3Store(
    index=md_blob_index,
    bucket="BUCKET",
    s3_profile="PROFILE",
    endpoint_url="https://next-gen-minio.materialsproject.org",
    key="blob_uuid",
)

# run our previous flow with the new stores
run_locally(
    Flow([elyte_interchange_job, production_flow]),
    store=JobStore(md_doc_store, additional_stores={"data": md_blob_store}),
    ensure_success=True,
)
```

</details>

### Running on GPUs

<details>
<summary>Learn to accelerate MD simulations with GPUs</summary>

Running on a GPU is nearly as simple as running on a CPU. The only difference
is that you need to specify the `platform_properties` argument in the
`EnergyMinimizationMaker` with the `DeviceIndex` of the GPU you want to use.

```python
production_maker = OpenMMFlowMaker(
    name="test_production",
    makers=[
        EnergyMinimizationMaker(
            platform_name="CUDA",
            platform_properties={"DeviceIndex": "0"},
        ),
        NPTMaker(),
        OpenMMFlowMaker.anneal_flow(),
        NVTMaker(),
    ],
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

    production_maker = OpenMMFlowMaker(
        name="test_production",
        makers=[
            EnergyMinimizationMaker(
                platform_name="CUDA",
                platform_properties={"DeviceIndex": str(device_index)},
            ),
            NPTMaker(),
            OpenMMFlowMaker.anneal_flow(),
            NVTMaker(),
        ],
    )

    production_flow = production_maker.make(
        setup.output.interchange,
        prev_dir=setup.output.dir_name,
        output_dir=f"/pscratch/sd/o/oac/openmm_runs/{i}",
    )
    flows.append(Flow([setup, production_flow]))

# this script will run four times, each with a different rank, thus distributing the work across the four GPUs.
run_locally(flows[rank], ensure_success=True)
```

</details>

## Analysis with Emmet

For now, you'll need to make sure you have a particular emmet branch installed.
Later the builders will be integrated into `main`.

```bash
pip install git+https://github.com/orionarcher/emmet@md_builders
```

### Analyzing Local Data

<details>
<summary>Learn to analyze your data without a database</summary>

Emmet will give us a solid head start on analyzing our data even without touching
a database. Below, we use emmet to create a [MDAnalysis Universe](https://docs.mdanalysis.org/stable/documentation_pages/core/universe.html#module-MDAnalysis.core.universe)
and a [SolvationAnalysis Solute](https://solvation-analysis.readthedocs.io/en/latest/api/solute.html).
From here, we can do all sorts of very cool analysis, but that's beyond the
scope of this tutorial. Consult the tutorials in SolvationAnalysis and MDAnalysis
for more information.

```python
from atomate2.openff.core import ClassicalMDTaskDocument
from emmet.builders.classical_md.utils import create_universe, create_solute
from openff.interchange import Interchange

ec_emc_taskdoc = ClassicalMDTaskDocument.parse_file("tutorial_system/taskdoc.json")
interchange = Interchange.parse_raw(ec_emc_taskdoc.interchange)
mol_specs = ec_emc_taskdoc.mol_specs

u = create_universe(
    interchange,
    mol_specs,
    str("tutorial_system/trajectory5.dcd"),
    traj_format="DCD",
)

solute = create_solute(u, solute_name="Li", networking_solvents=["PF6"])
```

</details>

### Setting up builders

<details>
<summary>Connect with your databases</summary>

If you followed the instructions above to set up a database, you can
use the `ElectrolyteBuilder` to perform the same analysis as above.

First, we'll need to create the stores where are data is located,
these should match the stores you used when running your flow.

```python
from maggma.stores import MongoStore, S3Store

mongo_info = {
    "username": "USERNAME",
    "password": "PASSWORD",
    "database": "DATABASE",
    "host": "mongodb05.nersc.gov",
}
s3_info = {
    "bucket": "BUCKET",
    "s3_profile": "PROFILE",
    "endpoint_url": "https://next-gen-minio.materialsproject.org",
}

md_docs = MongoStore(**mongo_info, collection_name="atomate2_docs")
md_blob_index = MongoStore(
    **mongo_info,
    collection_name="atomate2_blobs_index",
    key="blob_uuid",
)
md_blob_store = S3Store(
    **s3_info,
    index=md_blob_index,
    key="blob_uuid",
)
```

Now we create our Emmet builder and connect to it. We
will include a query that will only select jobs with
the tag "tutorial_production_flow" that we used earlier.

```python
from emmet.builders.classical_md.openmm.core import ElectrolyteBuilder

builder = ElectrolyteBuilder(
    md_docs, md_blob_store, query={"output.tags": "tutorial_production_flow"}
)
builder.connect()
```

<details>
<summary>Here are some more convenient queries.</summary>

Here are some more convenient queries we could use!

```python
# query jobs from a specific day
april_16 = {"completed_at": {"$regex": "^2024-04-16"}}
may = {"completed_at": {"$regex": "^2024-05"}}


# query a particular set of jobs
job_uuids = [
    "3d7b4db4-85e5-48a5-9585-07b37910720f",
    "4202b18f-f156-4705-8ca6-ac2a08093174",
    "187d9466-c359-4013-9e25-8b4ece6e3ecf",
]
my_specific_jobs = {"uuid": {"$in": job_uuids}}
```

</details>

</details>

### Analyzing systems individually

<details>

<summary>Download and explore systems one-by-one</summary>

To analyze a specific system, you'll need the uuid of the taskdoc you want to
analyze. We can find the uuids of all the taskdocs in our builder by
retrieving the items and extracting the uuids.

```python
items = builder.get_items()
uuids = [item["uuid"] for item in items]
```

This, however, can quickly get confusing once you have many jobs.
At this point, I would highly recommend starting to use an application that
makes it easier to view and navigate MongoDB databases. I recommend
[Studio3T](https://robomongo.org/) or [DataGrip](https://www.jetbrains.com/datagrip/).

Now we again use our builder to create a `Universe` and `Solute`. This time
`instatiate_universe` downloads the trajectory, saves it locally, and uses
it to create a `Universe`.

```python
# a query that will grab
tutorial_query = {"tags": "tutorial_production_flow"}

u = builder.instantiate_universe(uuid, "directory/to/store/trajectory")

solute = create_solute(
    u,
    solute_name="Li",
    networking_solvents=["PF6"],
    fallback_radius=3,
)
```

</details>

### Automated analysis with builders

<details>
<summary>Do it all for all the systems!</summary>

Finally, we'll put the H in high-throughput molecular dynamics. Below,
we create Stores to hold our `SolvationDocs` and `CalculationDocs` and
execute the builder on all of our jobs!

Later, there will also be `TransportDocs`, `EquilibrationDocs` and more.
Aggregating most of what you might want to know about an MD simulation.

```python
solvation_docs = MongoStore(**mongo_info, collection_name="solvation_docs")
calculation_docs = MongoStore(**mongo_info, collection_name="calculation_docs")
builder = ElectrolyteBuilder(md_docs, md_blob_store, solvation_docs, calculation_docs)

builder.connect()
items = builder.get_items()
processed_docs = builder.process_items(items)
builder.update_targets(processed_docs)
```

</details>
