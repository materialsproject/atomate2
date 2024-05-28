(installation)=

# Installation

## Introduction

This guide will get you up and running in an environment for running high-throughput
workflows with atomate2. atomate2 is built on the `pymatgen`, `custodian`, `jobflow`, and
`FireWorks` libraries. Briefly:

- [`pymatgen`] is used to create input files and analyze the output of materials science codes.
- [`custodian`] runs your simulation code (e.g., VASP) and performs error checking/handling
  and checkpointing.
- [`jobflow`] is used to design computational workflows.
- [`FireWorks`] (optional) is used to manage and execute workflows on HPC machines.

Running and writing your own workflows are covered in later tutorials. For now, these
topics will be covered in enough depth to get you set up and to help you know where to
troubleshoot if you're having problems.

Note that this installation tutorial is VASP-centric since almost all functionality
currently in atomate2 pertains to VASP.

[`pymatgen`]: http://pymatgen.org
[`custodian`]: https://materialsproject.github.io/custodian
[`fireworks`]: https://materialsproject.github.io/fireworks
[`jobflow`]: https://materialsproject.github.io/jobflow

### Objectives

- Install and configure atomate2 on your computing cluster.
- Validate the installation with a test workflow.

### Installation checklist

Completing everything on this checklist should result in a fully functioning
environment.

1. [Prerequisites](#prerequisites)
1. [Create a directory scaffold](#create-a-directory-scaffold-for-atomate2)
1. [Create a conda environment](#create-a-conda-environment)
1. [Install Python packages](#install-python-packages)
1. [Configure output database](#configure-calculation-output-database)
1. [Configure pymatgen](#configure-pymatgen)
1. [Run a test workflow](#run-a-test-workflow)

## Prerequisites

Before you install, you need to make sure that your "worker" computer (where the
simulations will be run, often a computing cluster) that will execute workflows can
(i) run the base simulation packages (e.g., VASP, LAMMPS, FEFF, etc) and (ii) connect
to a MongoDB database. For (i), make sure you have the appropriate licenses and
compilation to run the simulation packages that are needed. For (ii), make sure your
computing center doesn't have firewalls that prevent database access. Typically,
academic computing clusters as well as systems with a MOM-node style architecture
(e.g., NERSC) are OK.

### VASP

To get access to VASP on supercomputing resources typically requires that you're added
to a user group on the system you work on after your license is verified. Ensure that
you have access to the VASP executable and that it is functional before starting this
tutorial.

### MongoDB

[MongoDB](https://docs.mongodb.com/manual) is a NoSQL database that stores each database
entry as a document, which is represented in JSON format (the formatting is similar to
a dictionary in Python). Atomate2 uses MongoDB to:

- Create a database of calculation results.
- Store the workflows that you want to run as well as their state details (through
  FireWorks - optional).

MongoDB must be running and available to accept connections whenever you're running
workflows. Thus, it is strongly recommended that you have a server to run MongoDB or
(simpler) use a hosting service. Your options are:

1. Use a commercial service to host your MongoDB instance. These are typically the
  easiest to use and offer high-quality service but require payment for larger
  databases. [MongoDB Atlas](https://www.mongodb.com/cloud/atlas) offers a free 500 MB
  server which is certainly enough to get started for small to medium-sized projects, and
  it is easy to upgrade or migrate your database if you exceed the free allocation.
1. Contact your supercomputing center to see if they offer MongoDB hosting (e.g., NERSC
  has this, Google "request NERSC MongoDB database").
1. Self-host a MongoDB server.

If you're just starting, we suggest option 1 (with a free plan) or 2
(if available to you). The third option will require you to open up network settings to
accept outside connections properly which can sometimes be tricky.

Next, create a new database and set up an account with admin access. Keep a record of
your credentials - we will configure `jobflow` to connect to them in a later step. Also,
make sure you note down the hostname and port for the MongoDB instance.

```{note}
The computers that perform the calculations must have access to your MongoDB server.
Some computing resources have firewalls blocking connections. Although this is not a
problem for most computing centers that allow such connections (particularly from
MOM-style nodes, e.g. at NERSC, SDSC, etc.), but some of the more security-sensitive
centers (e.g., LLNL, PNNL, ARCHER) will run into issues. If you run into connection
issues later in this tutorial, some options are:

- Contact your computing center to review their security policy to allow connections
  from your MongoDB server (best resolution).
- Host your Mongo database on a machine that you're able to securely connect to,
  e.g. on the supercomputing network itself (ask a system administrator for help).
- Use a proxy service to forward connections from the MongoDB --> login node -->
  compute node (you might try, for example, [the mongo-proxy tool](https://github.com/bakks/mongo-proxy).
- Set up an ssh tunnel to forward connections from allowed machines (the tunnel must
  be kept alive at all times you're running workflows).
```

## Create a directory scaffold for atomate2

Installing atomate2 includes the installation of codes, configuration files, and various
binaries and libraries. Thus, it is useful to create a directory structure that
organizes all these items.

1. Log in to the compute cluster and create a directory in a spot on disk that has
   relatively fast access from compute nodes _and_ that is only accessible by yourself
   or your collaborators. Your environment and configuration files will go here,
   including database credentials. We will call this place `<<INSTALL_DIR>>`. A good
   name might simply be `atomate2`.

2. Now you should scaffold the rest of your `<<INSTALL_DIR>>` for the things we are
   going to do next. Run `mkdir -p atomate2/{config,logs}` to create directories named
   `logs` and `config` so your directory structure looks like:

```text
atomate2
├── config
└── logs
```

## Create a conda environment

```{note}
Make sure to create a Python 3.8+ environment as recent versions of atomate2 only
support Python 3.8 and higher.
```

We highly recommend that you organize your installation of the atomate2 and the other
Python codes using a conda virtual environment. Some of the main benefits are:

- Different Python projects that have conflicting packages can coexist on the same
  machine.
- Different versions of Python can exist on the same machine and be managed more easily
  (e.g. Python 2 and Python 3).
- You have full rights and control over the environment. On computing resources,
  this solves permissions issues with installing and modifying packages.

The easiest way to get a Python virtual environment is to use the `conda` tool.
Most clusters (e.g., NESRC) have [Anaconda](https://continuum.io) installed already
which provides access to the `conda` binary. If the `conda` tool is not available, you can
install it by following the installation instructions for
[Miniconda](https://docs.conda.io/en/latest/miniconda.html). To set up your conda environment:

1. Create a new conda environment called atomate2 with Python 3.9 using
   `conda create -n atomate2 python=3.9`.
2. Activate your environment by running `conda activate atomate2`. Now, when you use
   the command `python`, you'll be using the version of `python` in the atomate2
   conda environment folder.
3. Consider adding `conda activate atomate2` to your .bashrc or .bash_profile file so
   that it is run whenever you log in. Otherwise, note that you must call this command
   after every login before you can do work on your atomate project.

## Install Python packages

Next, we will download and install all of the atomate2-related Python packages.

To install the packages run:

```bash
pip install atomate2
```

If you would like to use more specialized capabilities of `atomate2` such as the phonon, Lobster or force field workflows, you would need to run one of

```bash
pip install atomate2[phonons]
pip install atomate2[lobster]
pip install atomate2[forcefields]
```

See [`pyproject.toml`](https://github.com/materialsproject/atomate2/blob/main/pyproject.toml) for all available optional dependency sets. More detailed instructions can be found under [dev installation](../dev/dev_install.md).

## Configure calculation output database

The next step is to configure your MongoDB database that will be used to store
calculation outputs.

```{note}
All of the paths here must be *absolute paths*. For example, the absolute path that
refers to `<<INSTALL_DIR>>` might be `/global/homes/u/username/atomate` (don't
use the relative directory `~/atomate`).
```

```{warning}
**Passwords will be stored in plain text!** These files should be stored in a place
that is not accessible by unauthorized users. Also, you should make random passwords
that are unique only to these databases.
```

Create the following files in `<<INSTALL_DIR>>/config`.

### `jobflow.yaml`

The `jobflow.yaml` file contains the credentials of the MongoDB server that will store
calculation outputs. The `jobflow.yaml` file requires you to enter the basic database
information as well as what to call the main collection that results are kept in (e.g.
`outputs`). Note that you should replace the whole `<<PROPERTY>>` definition with
your own settings.

```yaml
JOB_STORE:
  docs_store:
    type: MongoStore
    database: <<DB_NAME>>
    host: <<HOSTNAME>>
    port: <<PORT>>
    username: <<USERNAME>>
    password: <<PASSWORD>>
    collection_name: outputs
  additional_stores:
    data:
      type: GridFSStore
      database: <<DB_NAME>>
      host: <<HOSTNAME>>
      port: <<PORT>>
      username: <<USERNAME>>
      password: <<PASSWORD>>
      collection_name: outputs_blobs
```

````{note}
If you're using a MongoDB hosted on Atlas (using the free plan linked above) the
connection format is slightly different. Instead your `jobflow.yaml` file should
contain the following.

```yaml
JOB_STORE:
  docs_store:
    type: MongoURIStore
    uri: mongodb+srv://<<USERNAME>>:<<PASSWORD>>@<<HOST>>/<<DB_NAME>>?retryWrites=true&w=majority
    collection_name: outputs
  additional_stores:
    data:
      type: GridFSURIStore
      uri: mongodb+srv://<<USERNAME>>:<<PASSWORD>>@<<HOST>>/<<DB_NAME>>?retryWrites=true&w=majority
      collection_name: outputs_blobs
```

The URI key may be different based on the Atlas database you deployed. You can
see the template for the URI string by clicking on "Databases" (under "Deployment"
in the left hand menu) then "Connect" then "Connect your application". Select
Python as the driver and 3.12 as the version. The connection string should now be
displayed in the box.

Note that the username and password are not your login account details for Atlas.
Instead you must add a new database user by selecting "Database Access" (under
"Security" in the left hand menu) and then "Add a new database user".

Secondly, Atlas only allows connections from known IP addresses. You must therefore
add the IP address of your cluster (and any other computers you'll be connecting
from) by clicking "Network Access" (under "Security" in the left hand menu) and then
"Add IP address".
````

Atomate2 uses two database collections, one for small documents (such as elastic
tensors, structures, and energies) called the `docs` store and another for large
documents such as band structures and density of states called the `data` store.

Due to inherent limitations in MongoDB (individual documents cannot be larger than 16
Mb), we use GridFS to store large data. GridFS sits on top of MongoDB and
therefore doesn't require any further configuration on your part. However, other
storage types are available (such as Amazon S3). For more information please read
[](advanced_storage).

### atomate2.yaml

The `atomate2.yaml` file controls all atomate2 settings. You can see the full list
of available settings in the {obj}`.Atomate2Settings` docs. For now, we will just
configure the commands used to run VASP.

Write the `atomate2.yaml` file with the following content,

```yaml
VASP_CMD: <<VASP_CMD>>
```

This is the command that you would use to run VASP with parallelization
(`srun -n 16 vasp`, `ibrun -n 16 vasp`, `mpirun -n 16 vasp`, ...).

### Finishing up

The directory structure of `<<INSTALL_DIR>>/config` should now look like

```txt
config
├── jobflow.yaml
└── atomate2.yaml
```

The last thing to configure atomate2 is to add the following lines to your
.bashrc / .bash_profile file to set an environment variable telling atomate2 and `jobflow`
where to find the config files.

```bash
export ATOMATE2_CONFIG_FILE="<<INSTALL_DIR>>/config/atomate2.yaml"
export JOBFLOW_CONFIG_FILE="<<INSTALL_DIR>>/config/jobflow.yaml"
```

where `<<INSTALL_DIR>>` is your installation directory.

## Configure pymatgen

If you're planning to run VASP, the last configuration step is to configure `pymatgen` to
(required) find the pseudopotentials for VASP and (optional) set up your API key from
the [Materials Project].

### Pseudopotentials

The pseudopotentials should be available on the compute machine. Follow the
[pseudopotential installation instructions in the `pymatgen` documentation](https://pymatgen.org/installation.html#potcar-setup)
and then return to this tutorial.

### Materials Project API key

You can get an API key from the [Materials Project] by logging in and going to your
[Dashboard](materials project). Add this also to
your `~/.config/.pmgrc.yaml` so that it looks like the following

```yaml
PMG_VASP_PSP_DIR: <<INSTALL_DIR>>/pps
PMG_MAPI_KEY: <<YOUR_API_KEY>>
```

You can generate this file and set these values using the `pymatgen` CLI:

```bash
pmg config --add PMG_VASP_PSP_DIR /abs/path/to/psp PMG_MAPI_KEY your_api_key
```

[materials project]: https://materialsproject.org/dashboard

## Run a test workflow

To make sure that everything is set up correctly and in place, we'll finally run a
simple (but real) test workflow. We will first define a Python script to run the
workflow. Next, we'll submit a job to run the script. Finally, we'll examine the
database to check the job output. In this tutorial, we will be submitting an individual
workflow manually. If you want to manage and execute many workflows simultaneously
this can be achieved using the FireWorks package and is covered in
[](atomate2_fireWorks).

This particular workflow will only run a single calculation that optimizes a crystal
structure (not very exciting). In the subsequent tutorials, we'll run more complex
workflows.

### Define the workflow

Workflows are written using the `jobflow` software. Essentially, individual stages of
a workflow are simple Python functions. Jobflow provides a way to connect jobs in a natural way.
For more details on connecting jobs see: [](connecting_vasp_jobs).

Go to the directory where you would like your calculations to run (i.e., your scratch
or work directory) and create a file called `relax.py` containing:

```py
from atomate2.vasp.jobs.core import RelaxMaker
from jobflow import run_locally
from pymatgen.core import Structure

# construct an FCC silicon structure
si_structure = Structure(
    lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
    species=["Si", "Si"],
    coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
)

# make a relax job to optimise the structure
relax_job = RelaxMaker().make(si_structure)

# run the job
run_locally(relax_job, create_folders=True)
```

The `run_locally` function is a `jobflow` command that will execute the workflow on
the current computing resource.

### Submit the workflow

Next, make a job submission script called `job.sh` containing:

```bash
conda activate atomate2
python relax.py
```

The job submission script should include all the headers specific to your HPC resource.
For example, if your machine uses the Grid Engine scheduler for submitting and running
jobs, your script would look something like:

```bash
#!/bin/bash -l
#$ -N relax_si
#$ -P my_project
#$ -l h_rt=1:00:00
#$ -l mem=4G
#$ -pe mpi 16
#$ -cwd

# ensure you load the modules to run VASP, e.g., module load vasp

conda activate atomate2
python relax.py
```

Finally, submit the job to the queue using the normal scheduler command. For example
on the Grid Engine scheduler, this would be using `qsub job.sh`.

### Analyzing the results

Once the job is finished, you can connect to the output database and check the job
output.

```py
from jobflow import SETTINGS

store = SETTINGS.JOB_STORE

# connect to the job store
store.connect()

# query the job store
result = store.query_one(
    {"output.formula_pretty": "Si"}, properties=["output.output.energy_per_atom"]
)
print(result)
```

We query the database using the MongoDB query language. You can also connect to the
database using graphical tools, such as [Robo3T](https://robomongo.org) to explore your
results.

The outputs of VASP calculations always have the same set of keys. This structure is
called a schema. You can see the VASP calculation scheme in the {obj}`~atomate2.vasp.schemas.task.TaskDocument`
section of the documentation.

### Next steps

That's it! You've completed the installation tutorial!

See the following pages for more information on the topics we covered here:

- To see how to run and customize the existing Workflows in atomate2, try the
  [](running_workflows) tutorial (suggested next step).
- To see how to manage and execute many workflows at once, try the
  [](atomate2_fireWorks) tutorial.

## Troubleshooting and FAQ

### My job failed

Check the job error files in the launch directory for any errors. Also, check the job
standard output for a full log of the workflow execution and to check for a Python
traceback.

### I honestly tried everything I can to solve my problem. I still need help

There is a [support forum for atomate2](https://discuss.matsci.org/c/atomate).
