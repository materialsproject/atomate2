=======================================
Database CLI Reference
=======================================

The ``atomate2siesta-database`` command-line interface provides tools for testing, managing, and querying MongoDB databases used for atomate2siesta workflow storage.

Installation
============

The CLI is automatically installed with atomate2siesta:

.. code-block:: bash

   pip install -e ".[dev,tests,docs]"

Quick Start
===========

.. code-block:: bash

   # Test MongoDB connection
   atomate2siesta-database test

   # Show configuration examples
   atomate2siesta-database config

   # List recent calculation results
   atomate2siesta-database list --limit 10

   # Query calculations for a specific formula
   atomate2siesta-database query Si

   # Show comprehensive database statistics
   atomate2siesta-database stats

Commands
========

test
----

Test MongoDB connection and display database statistics.

**Usage:**

.. code-block:: bash

   atomate2siesta-database test [OPTIONS]

**Options:**

* ``--host TEXT``: MongoDB host address (default: localhost)
* ``--port INTEGER``: MongoDB port (default: 27017)
* ``--database TEXT``: Database name (default: atomate2siesta)
* ``--collection TEXT``: Collection name (default: tasks)

**Examples:**

.. code-block:: bash

   # Test local MongoDB
   atomate2siesta-database test

   # Test remote MongoDB
   atomate2siesta-database test --host db.example.com --port 27018

   # Test custom database
   atomate2siesta-database test --database my_calculations --collection results

**Output:**

* Connection status (PyMongo and Maggma)
* Document count
* Database size
* Storage size
* Index count

----

list
----

List recent documents in the database with key metadata.

**Usage:**

.. code-block:: bash

   atomate2siesta-database list [OPTIONS]

**Options:**

* ``--host TEXT``: MongoDB host address (default: localhost)
* ``--port INTEGER``: MongoDB port (default: 27017)
* ``--database TEXT``: Database name (default: atomate2siesta)
* ``--collection TEXT``: Collection name (default: tasks)
* ``--limit INTEGER``: Maximum number of documents to display (default: 10)

**Examples:**

.. code-block:: bash

   # List 10 most recent calculations
   atomate2siesta-database list

   # List 50 most recent with custom database
   atomate2siesta-database list --limit 50 --database production_db

   # List from remote server
   atomate2siesta-database list --host server.com --port 27018

**Output Table Columns:**

* UUID (first 36 characters)
* Chemical formula
* Calculation state (successful, failed, etc.)
* Total energy (eV)
* Calculation type

----

query
-----

Query documents with flexible filters and display detailed results.

**Usage:**

.. code-block:: bash

   atomate2siesta-database query [OPTIONS] [FORMULA]

**Arguments:**

* ``FORMULA``: Chemical formula to search for (e.g., Si, Fe2O3, Al2O3) - optional

**Connection Options:**

* ``--host TEXT``: MongoDB host address (default: localhost)
* ``--port INTEGER``: MongoDB port (default: 27017)
* ``--database TEXT``: Database name (default: atomate2siesta)
* ``--collection TEXT``: Collection name (default: tasks)

**Query Filter Options:**

* ``--formula TEXT``: Filter by chemical formula
* ``--state TEXT``: Filter by calculation state (successful, failed, etc.)
* ``--calc-type TEXT``: Filter by calculation type (relax, static, bands, etc.)
* ``--energy-min FLOAT``: Minimum energy threshold (eV)
* ``--energy-max FLOAT``: Maximum energy threshold (eV)
* ``--latest INTEGER``: Show N most recent calculations (sorted by completion time)

**Export Options:**

* ``--export TEXT``: Export format (json or csv)
* ``--output TEXT``: Output filename (without extension, default: query_results)

**Examples:**

.. code-block:: bash

   # Query all Silicon calculations
   atomate2siesta-database query Si

   # Query by state
   atomate2siesta-database query --state successful

   # Query by calculation type
   atomate2siesta-database query --calc-type relax

   # Query by energy range
   atomate2siesta-database query --energy-min -200 --energy-max -100

   # Show 10 most recent calculations
   atomate2siesta-database query --latest 10

   # Export to JSON
   atomate2siesta-database query Si --export json --output silicon_results

   # Export to CSV
   atomate2siesta-database query --state successful --export csv

   # Combine multiple filters
   atomate2siesta-database query --formula Si --state successful --calc-type relax

   # Query from remote server with filters
   atomate2siesta-database query --host db.example.com --latest 20 --export json

**Output Table Columns:**

* UUID
* Formula
* State
* Calc Type
* Energy (eV)
* K-points
* Basis
* Mesh Cutoff (Ry)
* Completion Time

----

stats
-----

Display comprehensive statistics for all collections in the database.

**Usage:**

.. code-block:: bash

   atomate2siesta-database stats [OPTIONS]

**Options:**

* ``--host TEXT``: MongoDB host address (default: localhost)
* ``--port INTEGER``: MongoDB port (default: 27017)
* ``--database TEXT``: Database name (default: atomate2siesta)

**Examples:**

.. code-block:: bash

   # Show statistics for local database
   atomate2siesta-database stats

   # Show statistics for production database
   atomate2siesta-database stats --database production_siesta

   # Show statistics for remote database
   atomate2siesta-database stats --host db.example.com --database research

**Output:**

* Collection-level statistics (documents, size, avg doc size)
* Database-level statistics (total data size, storage size, indexes)
* Summary totals

----

clear
-----

Clear all documents from a collection (USE WITH CAUTION).

**Usage:**

.. code-block:: bash

   atomate2siesta-database clear [OPTIONS]

**Options:**

* ``--host TEXT``: MongoDB host address (default: localhost)
* ``--port INTEGER``: MongoDB port (default: 27017)
* ``--database TEXT``: Database name (default: atomate2siesta)
* ``--collection TEXT``: Collection name (default: tasks)
* ``--force``: Skip confirmation prompt

**Examples:**

.. code-block:: bash

   # Clear with confirmation prompt
   atomate2siesta-database clear

   # Clear without confirmation (dangerous!)
   atomate2siesta-database clear --force

   # Clear specific collection
   atomate2siesta-database clear --collection test_results

   # Clear remote database collection
   atomate2siesta-database clear --host db.example.com --collection old_data

.. warning::

   This command permanently deletes all documents in the specified collection. Use with extreme caution, especially with ``--force`` flag.

----

config
------

Show example configuration files for jobflow and Python usage.

**Usage:**

.. code-block:: bash

   atomate2siesta-database config

**Output:**

* Example ``~/.jobflow.yaml`` configuration
* Example Python code for using MongoStore
* Setup instructions

**Example:**

.. code-block:: bash

   atomate2siesta-database config > my-database-setup.md

Common Use Cases
================

Initial Setup and Testing
--------------------------

.. code-block:: bash

   # 1. Start MongoDB
   brew services start mongodb-community  # macOS
   sudo systemctl start mongodb          # Linux

   # 2. Test connection
   atomate2siesta-database test

   # 3. Show configuration examples
   atomate2siesta-database config

   # 4. Verify empty database
   atomate2siesta-database stats

Monitoring Running Workflows
-----------------------------

.. code-block:: bash

   # Check recent calculations
   atomate2siesta-database list --limit 20

   # Monitor specific material
   atomate2siesta-database query Si

   # Check overall statistics
   atomate2siesta-database stats

Production Database Management
-------------------------------

.. code-block:: bash

   # Monitor production database
   atomate2siesta-database stats --database production_siesta

   # List recent successful calculations
   atomate2siesta-database list --database production_siesta --limit 50

   # Query specific materials
   atomate2siesta-database query TiO2 --database production_siesta

Development and Testing
-----------------------

.. code-block:: bash

   # Test against development database
   atomate2siesta-database test --database dev_siesta

   # Clear test database
   atomate2siesta-database clear --database test_siesta --force

   # Verify cleanup
   atomate2siesta-database stats --database test_siesta

Remote Database Access
----------------------

.. code-block:: bash

   # Connect to remote MongoDB via SSH tunnel
   ssh -L 27017:localhost:27017 user@remote-server

   # In another terminal, query remote database
   atomate2siesta-database test --host localhost --port 27017
   atomate2siesta-database list --host localhost --limit 100
   atomate2siesta-database stats --host localhost

Integration with Python
=======================

The CLI complements Python-based database operations:

.. code-block:: python

   from maggma.stores import MongoStore
   from atomate2.siesta.jobs.core import RelaxMaker
   from jobflow import run_locally
   from pymatgen.core import Structure

   # Create store
   store = MongoStore(
       database="atomate2siesta",
       collection_name="tasks",
       host="localhost",
       port=27017
   )

   # Run calculation with database storage
   structure = Structure.from_file("structure.cif")
   job = RelaxMaker.fixed_cell_relaxation().make(structure)
   results = run_locally(job, create_folders=True, store=store)

   # Then use CLI to query results
   # $ atomate2siesta-database list --limit 1
   # $ atomate2siesta-database query Si

Error Handling
==============

Connection Failed
-----------------

.. code-block:: bash

   $ atomate2siesta-database test
   Connection failed: [Errno 61] Connection refused

**Solution:** Start MongoDB service:

* macOS: ``brew services start mongodb-community``
* Linux: ``sudo systemctl start mongodb``

Module Not Found
----------------

.. code-block:: bash

   $ atomate2siesta-database test
   pymongo not installed. Run: pip install pymongo

**Solution:** Install required packages:

.. code-block:: bash

   pip install pymongo maggma

Permission Denied
-----------------

.. code-block:: bash

   $ atomate2siesta-database test
   Error: authentication failed

**Solution:** Provide credentials in connection string or configure MongoDB authentication properly.

Environment Variables
=====================

You can set default values using environment variables:

.. code-block:: bash

   export MONGO_HOST=localhost
   export MONGO_PORT=27017
   export MONGO_DATABASE=atomate2siesta

.. note::

   Command-line options always override environment variables.

Tips and Best Practices
========================

1. **Regular Monitoring**: Use ``stats`` command regularly to monitor database growth
2. **Backup Before Clear**: Always backup before using ``clear`` command
3. **Use Descriptive Databases**: Use different database names for dev/test/production
4. **Index Important Fields**: Create indexes for frequently queried fields
5. **SSH Tunnels for Remote Access**: Use SSH tunneling for secure remote database access
6. **Test Connection First**: Always run ``test`` before attempting operations on a new database

See Also
========

* Tutorial 13: Database Storage
* `MongoDB Documentation <https://docs.mongodb.com/>`_
* `Maggma Documentation <https://materialsproject.github.io/maggma/>`_
* `Jobflow Database Guide <https://materialsproject.github.io/jobflow/>`_
