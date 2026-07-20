Installation
============

`atomate2siesta` can be installed in a few easy steps. Follow these instructions to get started with the package and ensure all dependencies are properly installed.

Step 1: Install Using pip
-------------------------

You can install `atomate2siesta` directly from its Git repository using `pip`. Run the following command in your terminal:

.. code-block:: bash

   pip install atomate2[siesta]

or, to install from a local clone in editable / development mode:

.. code-block:: bash

   git clone https://github.com/materialsproject/atomate2.git
   cd atomate2
   pip install -e ".[siesta]"

.. important::
   Always include the ``[siesta]`` extra. Installing the base package
   (``pip install .`` / ``pip install -e .``) does **not** pull in the
   SIESTA-specific dependencies, and importing ``atomate2.siesta`` will then fail
   with a ``ModuleNotFoundError`` such as ``No module named 'sisl'`` or
   ``No module named 'pyfiglet'``. The ``[siesta]`` extra installs:

   - ``sisl>=0.16.2`` — SIESTA structure / FDF handling
   - ``pyfiglet`` — CLI banner
   - ``questionary>=2.0.0`` — interactive CLI prompts
   - ``rich`` — formatted terminal output
   - ``colorama``
   - ``seaborn`` — plotting
   - ``atomate2[ase, phonons]`` — ASE + phonopy / seekpath for phonon workflows

.. note::
   you can use `hash -r` command if cli not working

Step 2: Automatic Installation of Required Dependencies
-------------------------------------------------------

The package depends on several other tools to work correctly. All the dependencies will be installed automatically.


Step 3: Verify Your Installation
--------------------------------

Once installed, you can verify that `atomate2siesta` is correctly installed by importing it in a Python environment:

.. code-block:: python

   import atomate2.siesta

If no errors occur, the installation was successful.

Step 3-1 Optional: Install Additional Tools
-------------------------------------------

Depending on your workflow, you may need to install additional tools or dependencies to enhance the functionality of `atomate2siesta`. For example, if you plan on running high-throughput workflows or using custom SIESTA pseudopotentials, you might want to install:

- **FireWorks**: A workflow management system for running high-throughput simulations. You can install it using:

  .. code-block:: bash

     pip install fireworks


- **jobflow-remote**: This package allows you to manage remote job execution within `jobflow` workflows, which is essential for running simulations on remote HPC clusters. Install it using:

  .. code-block:: bash

     pip install jobflow-remote

Step 4: Setting Up jobflow-remote
---------------------------------

To use `jobflow-remote` for managing remote jobs within your `atomate2siesta` workflows, you'll need to configure it according to your HPC environment. Here's a brief example setup:

1. **Create a Configuration File**: `jobflow-remote` requires a configuration YAML file. For example:

   .. code-block:: yaml

      name: my_hpc_cluster
      host: hpc.mydomain.com
      username: your_username
      ssh_key: ~/.ssh/id_rsa
      job_script:
         command: 'sbatch'
         script: |
            #!/bin/bash
            #SBATCH --job-name=myjob
            #SBATCH --time=00:30:00
            module load siesta
            siesta < siesta.fdf > siesta.out


Step 5: Project Configuration
-----------------------------

Generate a project configuration file using the following command:

.. code-block:: bash

   jf project generate YOUR_PROJECT_NAME

This will create a YAML file (e.g., `std.yaml`) where you can define your workers, queue, and jobstore configuration.

Example Configuration:

.. code-block:: yaml

   name: std
   workers:
     example_worker:
       type: remote
       scheduler_type: slurm
       work_dir: /path/to/run/folder
       pre_run: source /path/to/python/environment/activate
       timeout_execute: 60
       host: remote.host.net
       user: bob
   queue:
     store:
       type: MongoStore
       host: localhost
       database: db_name
       username: bob
       password: secret_password
       collection_name: jobs
   jobstore:
     docs_store:
       type: MongoStore
       database: db_name
       host: host.mongodb.com
       port: 27017
       username: bob
       password: secret_password
       collection_name: outputs
     additional_stores:
       data:
         type: GridFSStore
         database: db_name
         host: host.mongodb.com
         port: 27017
         username: bob
         password: secret_password
         collection_name: outputs_blobs

Step 6: Run Jobs with jobflow-remote
------------------------------------

Once the project is set up, you can use the `jf runner` command to start the job execution:

.. code-block:: bash

   jf runner start

To check job statuses:

.. code-block:: bash

   jf job list

To stop the runner:

.. code-block:: bash

   jf runner stop

You can fetch job results using:

.. code-block:: bash

   jf job output JOB_ID

3. **Submit Remote Jobs**: You can now submit jobs to your remote cluster in `atomate2siesta` workflows:

   .. code-block:: python

      from jobflow_remote import submit_flow
      resources = {"nodes": 1,"partition": "batch","ntasks":12, "time":"24:00:00", "job_name":"testing",}
      project = "atlas"
      submit_flow(relax_job,project=project,worker=worker,resources=resources)


This setup allows `atomate2siesta` workflows to seamlessly run on remote HPC clusters using jobflow-remote.

NOTE: Environment Setup
-------------------------

If you are using conda, you can set up a consistent environment across machines. On one machine, export the environment using:

.. code-block:: bash

   conda env export > jobflow_env.yaml

Then, create the same environment on another machine:

.. code-block:: bash

   conda env create -n env_name --file jobflow_env.yaml
