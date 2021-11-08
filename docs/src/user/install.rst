.. _installation tutorial:

============
Installation
============

Introduction
============

This guide will get you up and running in an environment for running high-throughput
workflows with atomate2. atomate2 is built on the pymatgen, custodian, jobflow, and
FireWorks libraries. Briefly:

* pymatgen_ is used create input files and analyze the output of materials science codes.
* custodian_ runs your simulation code (e.g., VASP) and performs error checking/handling
  and checkpointing.
* jobflow_ is used to design computational workflows.
* FireWorks_ (optional) is used to manage and execute workflows on HPC machines.

Running and writing your own workflows are covered in later tutorials. For now, these
topics will be covered in enough depth to get you set up and to help you know where to
troubleshoot if you are having problems.

Note that this installation tutorial is VASP-centric since almost all functionality
currently in atomate2 pertains to VASP.

.. _pymatgen: http://pymatgen.org
.. _custodian: https://materialsproject.github.io/custodian/
.. _FireWorks: https://materialsproject.github.io/fireworks/
.. _jobflow: https://materialsproject.github.io/jobflow/

Objectives
----------

* Install and configure atomate2 on your computing cluster.
* Validate the installation with a test workflow.

Installation checklist
----------------------

Completing everything on this checklist should result in a fully functioning
environment.

1. Prerequisites_
#. `Create a directory scaffold for atomate2`_
#. `Create a conda environment`_
#. `Install Python packages`_
#. `Configure jobflow`_
#. `Configure pymatgen`_
#. `Run a test workflow`_

.. _Prerequisites:

Prerequisites
=============

Before you install, you need to make sure that your "worker" computer (where the
simulations will be run, often a computing cluster) that will execute workflows can
(i) run the base simulation packages (e.g., VASP, LAMMPs, FEFF, etc) and (ii) connect
to a MongoDB database. For (i), make sure you have the appropriate licenses and
compilation to run the simulation packages that are needed. For (ii), make sure your
computing center doesn't have firewalls that prevent database access. Typically,
academic computing clusters as well as systems with a MOM-node style architecture
(e.g., NERSC) are OK.

VASP
----

To get access to VASP on supercomputing resources typically requires that you are added
to a user group on the system you work on after your license is verified. Ensure that
you have access to the VASP executable and that it is functional before starting this
tutorial.

MongoDB
-------

MongoDB_ is a NoSQL database that stores each database entry as a document, which is
represented in the JSON format (the formatting is similar to a dictionary in Python).
Atomate2 uses MongoDB to:

* to create database of calculation results.
* store the workflows that you want to run as well as their state details (through
  FireWorks - optional).

MongoDB must be running and available to accept connections whenever you are running
workflows. Thus, it is strongly recommended that you have a server to run MongoDB or
(simpler) use a hosting service. Your options are:

* use a commercial service to host your MongoDB instance. These are typically the
  easiest to use and offer high quality service but require payment for larger
  databases. `MongoDB Atlas <https://www.mongodb.com/cloud/atlas>`_ offers free 500 MB
  which is certainly enough to get started for small to medium size projects, and it is
  easy to upgrade or migrate your database if you do exceed the free allocation.
* contact your supercomputing center to see if they offer MongoDB hosting (e.g., NERSC
  has this, Google "request NERSC MongoDB database")
* self-host a MongoDB server

If you are just starting, we suggest the first (with a free plan) or second option
(if available to you). The third option will require you to open up network settings to
accept outside connections properly which can sometimes be tricky.

Next, create a new database and set up two new username/password combinations:

- an admin user
- a read-only user

Keep a record of your credentials - we will configure jobflow to connect to them in a
later step. Also make sure you note down the hostname and port for the MongoDB instance.

.. note::

    The computers that perform the calculations must have access to your MongoDB server.
    Some computing resources have firewalls blocking connections. Although this is not a
    problem for most computing centers that allow such connections (particularly from
    MOM-style nodes, e.g. at NERSC, SDSC, etc.), but some of the more security-sensitive
    centers (e.g., LLNL, PNNL, ARCHER) will run into issues. If you run into connection
    issues later in this tutorial, some options are:

    * contact your computing center to review their security policy to allow connections
      from your MongoDB server (best resolution)
    * host your Mongo database on a machine that you are able to securely connect to,
      e.g. on the supercomputing network itself (ask a system administrator for help)
    * use a proxy service to forward connections from the MongoDB --> login node -->
      compute node (you might try, for example, `the mongo-proxy tool
      <https://github.com/bakks/mongo-proxy>`_).
    * set up an ssh tunnel to forward connections from allowed machines (the tunnel must
      be kept alive at all times you are running workflows)


.. _MongoDB: https://docs.mongodb.com/manual/

.. _Create a directory scaffold for atomate2:

Create a directory scaffold for atomate2
========================================

Installing atomate2 includes installation of codes, configuration files, and various
binaries and libraries. Thus, it is useful to create a directory structure that
organizes all these items.

1. Log in to the compute cluster and create a directory in a spot on disk that has
   relatively fast access from compute nodes *and* that is only accessible by yourself
   or your collaborators. Your environment and configuration files will go here,
   including database credentials. We will call this place ``<<INSTALL_DIR>>``. A good
   name might simply be ``atomate2``.

#. Now you should scaffold the rest of your ``<<INSTALL_DIR>>`` for the things we are
   going to do next. Create a directories named ``logs``, and ``config`` so your
   directory structure looks like:

    ::

        atomate2
        ├── config
        └── logs

.. _Create a conda environment:

Create a conda environment
==========================

.. note::

   Make sure to create a Python 3.7+ environment as recent versions of atomate2 only
   support Python 3.7 and higher.

We highly recommended that you organize your installation of the atomate2 and the other
Python codes using a conda virtual environment. Some of the main benefits are:

- Different Python projects that have conflicting packages can coexist on the same
  machine.
- Different versions of Python can exist on the same machine and be managed more easily
  (e.g. Python 2 and Python 3).
- You have full rights and control over the environment. On computing resources,
  this solves permissions issues with installing and modifying packages.

The easiest way to get a Python virtual environment is to use the ``conda`` tool.
Most clusters (e.g., NESRC) have Anaconda_ installed already which provides access to
the ``conda`` binary. If the ``conda`` tool is not available, you can install it by
following the installation instructions for Miniconda_. To set up your conda
environment:

#. Create a new conda environment called atomate2 with Python 3.9 using
   ``conda create -n atomate2 python=3.9``.

#. Activate your environment by running ``conda activate atomate2``. Now, when you use
   the command ``python``, you'll be using the version of ``python`` in the atomate2
   conda environment folder.

#. Consider adding ``conda activate atomate2`` to your .bashrc or .bash_profile file so
   that it is run whenever you log in. Otherwise, note that you must call this command
   after every log in before you can do work on your atomate project.

.. _Anaconda: https://www.continuum.io
.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html

.. _Install Python packages:

Install Python packages
=======================

Next, we will download and install all of the atomate2-related Python packages.

To install the packages run::

    pip install atomate2

.. _conda: https://conda.io/docs/using/pkgs.html
.. _PyPI: https://pypi.python.org/pypi

.. _Configure jobflow:

Configure calculation output database
=====================================

The next step is to configure your mongoDB database that will be used to store
calculation outputs.

.. note::

   All of the paths here must be *absolute paths*. For example, the absolute path that
   refers to ``<<INSTALL_DIR>>`` might be ``/global/homes/u/username/atomate`` (don't
   use the relative directory ``~/atomate``).

.. warning::

    **Passwords will be stored in plain text!** These files should be stored in a place
    that is not accessible by unauthorized users. Also, you should make random passwords
    that are unique only to these databases.

Create the following files in ``<<INSTALL_DIR>>/config``.

jobflow.yaml
------------

The ``jobflow.yaml`` file contains the credentials of the MongoDB server that will store
calculation outputs. The ``jobflow.json`` file requires you to enter the basic database
information as well as what to call the main collection that results are kept in (e.g.
``ouputs``). Note that you should replace the whole ``<<PROPERTY>>`` definition with
your own settings.

.. code-block:: yaml

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

Atomate2 uses two database collections, one for small documents (such as elastic
tensors, structures, and energies) called the ``docs`` store and another for large
documents such as band structures and density of states called the ``data`` store.

Due to inherent limitations in MongoDB (individual documents cannot be larger than 16
Mb), we use GridFS to store large data. GridFS sits on top of MongoDB and
therefore doesn't require any further configuration on your part. However, other
storage types are available (such as Amazon S3). For more information please read
:ref:`advanced_storage`.

atomate2.yaml
-------------

The ``atomate2.yaml`` file controls all atomate2 settings. You can see the full list
of available settings in the :obj:`.Atomate2Settings`. docs. For now, we will just
configure the commands used to run VASP.

Write the ``atomate2.yaml`` file with the following content,

.. code-block:: yaml

    VASP_CMD: <<VASP_CMD>>

The is the command that you would use to run VASP with parallelization
(``srun -n 16 vasp``, ``ibrun -n 16 vasp``, ``mpirun -n 16 vasp``, ...).

Finishing up
------------

The directory structure of ``<<INSTALL_DIR>>/config`` should now look like

::

    config
    ├── jobflow.yaml
    ├── atomate2.yaml

The last thing we need to do to configure atomate2 is add the following lines to your
.bashrc / .bash_profile file to set an environment variable telling atomate2 and jobflow
where to find the config files.

.. code-block:: bash

    export ATOMATE2_CONFIG_FILE=<<INSTALL_DIR>>/config/atomate2.yaml
    export JOBFLOW_CONFIG_FILE=<<INSTALL_DIR>>/config/jobflow.yaml

where ``<<INSTALL_DIR>>`` is your installation directory.

.. _Configure pymatgen:

Configure pymatgen
==================

If you are planning to run VASP, the last configuration step is to configure pymatgen to
(required) find the pseudopotentials for VASP and (optional) set up your API key from
the `Materials Project`_.

Pseudopotentials
----------------

The psuedopotentials should be available on the compute machine. Follow the
`pseudopotential installation instructions in the pymatgen documentation <https://pymatgen.org/installation.html#potcar-setup>`_
and then return to this tutorial.

Materials Project API key
-------------------------

You can get an API key from the `Materials Project`_ by logging in and going to your
`Dashboard`_. Add this also to your ``.pmgrc.yaml`` so that it looks like the following

.. code-block:: yaml

    PMG_VASP_PSP_DIR: <<INSTALL_DIR>>/pps
    PMG_MAPI_KEY: <<YOUR_API_KEY>>

.. _Materials Project: https://materialsproject.org/dashboard
.. _Dashboard: https://materialsproject.org/dashboard

.. _Run a test workflow:

Run a test workflow
===================

To make sure that everything is set up correctly and in place, we'll finally run a
simple (but real) test workflow. Two methods to create workflows are (i) using atomate2's
command line utility ``atwf`` or (ii) by creating workflows in Python. For the most
part, we recommend using method (ii), the Python interface, since it is more powerful
and also simple to use. However, in order to get started without any programming, we'll
stick to method (i), the command line, using ``atwf`` to construct a workflow. Note that
we'll discuss the Python interface more in the :ref:`running workflows tutorial` and
provide details on writing custom workflows in the :ref:`creating workflows`.

Ideally you set up a Materials Project API key in the `Configure pymatgen`_ section,
otherwise you will need to provide a POSCAR for the structure you want to run. In
addition, there are two different methods to use ``atwf`` - one using a library of
preset functions for constructing workflows and another with a library of files for
constructing workflows.

This particular workflow will only run a single calculation that optimizes a crystal
structure (not very exciting). In the subsequent tutorials, we'll run more complex
workflows.

Add a workflow
--------------

Below are 4 different options for adding a workflow to the database. You only need to execute one of the below commands; note that it doesn't matter at this point whether you are loading the workflow from a file or from a Python function.

* Option 1 (you set up a Materials Project API key, and want to load the workflow using a file): ``atwf add -l vasp -s optimize_only.yaml -m mp-149 -c '{"vasp_cmd": ">>vasp_cmd<<", "db_file": ">>db_file<<"}'``
* Option 2 (you set up a Materials Project API key, and want to load the workflow using a Python function): ``atwf add -l vasp -p wf_structure_optimization -m mp-149``
* Option 3 (you will load the structure from a POSCAR file, and want to load the workflow using a file): ``atwf add -l vasp -s optimize_only.yaml POSCAR -c '{"vasp_cmd": ">>vasp_cmd<<", "db_file": ">>db_file<<"}'``
* Option 4 (you will load the structure from a POSCAR file, and want to load the workflow using a Python function): ``atwf add -l vasp -p wf_structure_optimization POSCAR``

All of these function specify (i) a type of workflow and (ii) the structure to feed into that workflow.

* The ``-l vasp`` option states to use the ``vasp`` library of workflows.
* The ``-s optimize_only.yaml`` sets the specification of the workflow using the ``optimize_only.yaml`` file in `this directory <https://github.com/hackingmaterials/atomate/blob/main/atomate/vasp/workflows/base/library/>`_. Alternatively, the ``-p wf_structure_optimization`` sets the workflow specification using the preset Python function located in `this module <https://github.com/hackingmaterials/atomate/blob/main/atomate/vasp/workflows/presets/core.py>`_. For now, it's probably best not to worry about the distinction but to know that both libraries of workflows are available to you.
* The ``-c`` option is used in file-based workflows to make sure that one uses the ``vasp_cmd`` and ``db_file`` that are specified in ``my_fworker.yaml`` that you specified earlier. In the preset workflows, it is the default behavior to take these parameters from the ``my_fworker.yaml`` so this option is not needed.

Verify the workflow
-------------------

These commands added a workflow for running a single structure optimization FireWork to your LaunchPad. You can verify that by using FireWorks' ``lpad`` utility:

.. code-block:: bash

    lpad get_wflows

which should return:

.. code-block:: bash

    [
        {
            "state": "READY",
            "name": "Si--1",
            "created_on": "2015-12-30T18:00:00.000000",
            "states_list": "REA"
        },
    ]

Note that the ``lpad`` command is from FireWorks and has many functions. As simple modifications to the above command, you can also try ``lpad get_wflows -d more`` (or if you are very curious, ``lpad get_wflows -d all``). You can use ``lpad get_wflows -h`` to see a list of all available modifications and ``lpad -h`` to see all possible commands.

If this works, congrats! You've added a workflow (in this case, just a single calculation) to the FireWorks database.

Submit the workflow
-------------------

To launch this FireWork through queue, go to the directory where you would like your calculations to run (e.g. your scratch or work directories) and run the command

.. code-block:: bash

    qlaunch rapidfire -m 1

There are lots of things to note here:

* The ``-m 1`` means to keep a maximum of 1 job in the queue to prevent submitting too many jobs. As with all FireWorks commands, you can get more options using ``qlaunch rapidfire -h`` or simply ``qlaunch -h``.
* The qlaunch mode specified above is the simplest and most general way to get started. It will end up creating a somewhat nested directory structure, but this will make more sense when there are many calculations to run.
* One other option for qlaunch is "reservation mode", i.e., ``qlaunch -r rapidfire``. There are many nice things about this mode - you'll get pretty queue job names that represent your calculated composition and task type (these are really nice to see specifically which calculations are queued) and you'll have more options for tailoring specific queue parameters to specific jobs. In addition, reservation mode will automatically stop submitting jobs to the queue depending on how many jobs you have in the database so you don't need to use the ``-m 1`` parameter (this is usually desirable and nice, although in some cases it's better to submit to the queue first and add jobs to the database later which reservation mode doesn't support). However, reservation mode does add its own complications and we do not recommend starting with it (in many if not most cases, it's not worth switching at all). If you are interested by this option, consult the FireWorks documentation for more details.
* If you want to run directly on your computing platform rather than through a queue, use ``rlaunch rapidfire`` instead of the ``qlaunch`` command (go through the FireWorks documentation to understand the details).

If all went well, you can check that the FireWork is in the queue by using the commands for your queue system (e.g. ``squeue`` or ``qstat``). When the job finally starts running, you will see the state of the workflow as running using the command ``lpad get_wflows -d more``.

Analyzing the results
---------------------

Once this FireWorks is launched and is completed, you can use pymatgen-db to check that it was entered into your results database by running

.. code-block:: bash

    mgdb query -c <<INSTALL_DIR>>/config/db.json --props task_id formula_pretty output.energy_per_atom

This time, ``<<INSTALL_DIR>>`` can be relative. You should have seen the energy per atom you calculated for Si.

Note that the ``mgdb`` tools is only one way to see the results. You can connect to your MongoDB and explore the results using any MongoDB analysis tool. In later tools, we'll also demonstrate how various Python classes in atomate also help in retrieving and analyzing data. For now, the ``mgdb`` command is a simple way to get basic properties.

You can also check that the workflow is marked as completed in your FireWorks database:

.. code-block:: bash

    lpad get_wflows -d more

which will show the state of the workflow as COMPLETED.

Next steps
----------

That's it! You've completed the installation tutorial!

See the following pages for more information on the topics we covered here:

* To see how to run and customize the existing Workflows and FireWorks try the :ref:`running workflows tutorial` (suggested next step)
* For submitting jobs to the queue in reservation mode see the `FireWorks advanced queue submission tutorial`_
* For using pymatgen-db to query your database see the `pymatgen-db documentation`_


.. _FireWorks advanced queue submission tutorial: https://materialsproject.github.io/fireworks/queue_tutorial_pt2.html
.. _pymatgen-db documentation: https://materialsproject.github.io/pymatgen-db/

Troubleshooting and FAQ:
========================

Q: I can't connect to my LaunchPad database
-------------------------------------------

:A: Make sure the right LaunchPad file is getting selected

  Adding the following line to your ``FW_config.yaml`` will cause the line to be printed every time that configuration is selected

  ::

    ECHO_TEST: Database at <<INSTALL_DIR>>/config/FW_config.yaml is getting selected.

  Then running ``lpad version`` should give the following result if that configuration file is being chosen

  ::

    $ lpad version

    Database at <<INSTALL_DIR>>/config/FW_config.yaml is getting selected.
    FireWorks version: x.y.z
    located in: <<INSTALL_DIR>>/atomate_env/lib/python3.6/site-packages/fireworks

  If it's not being found, check that ``echo $FW_CONFIG_FILE`` returns the location of that file (you could use ``cat $FW_CONFIG_FILE`` to check the contents)

:A: Double check all of the configuration settings in ``my_launchpad.yaml``

:A: Have you had success connecting before? Is there a firewall blocking your connection?

:A: You can try following the tutorials of FireWorks which will go through this process in a little more detail.


Q: My job fizzled!
------------------

:A: Check the ``*_structure_optimization.out`` and ``*_structure_optimization.error`` in the launch directory for any errors. Also check the ``FW.json`` to check for a Python traceback.


Q: I made a mistake using reservation mode, how do I cancel my job?
-------------------------------------------------------------------

:A: One drawback of using the reservation mode (the ``-r`` in ``qlaunch -r rapidfire``) is that you have to cancel your job in two places: the queue and the LaunchPad. To cancel the job in the queue, use whatever command you usually would (e.g. ``scancel`` or ``qdel``). To cancel or rerun the FireWork, run

    .. code-block:: bash

        lpad defuse_fws -i 1

    or

    .. code-block:: bash

        lpad rerun_fws -i 1

    where `-i 1` means to make perfom the operations on the FireWork at index 1. Run ``lpad -h`` to see all of the options.

The non-reservation mode for qlaunching requires a little less maintenance with certain tradeoffs, which are detailed in the FireWorks documentation.

Q: I honestly tried everything I can to solve my problem. I still need help!
----------------------------------------------------------------------------

:A: There is a support forum for atomate: https://discuss.matsci.org/c/atomate


You can install atomate2 with ``pip`` or from source.
