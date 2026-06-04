Troubleshooting Guide
=====================

This guide helps you resolve common issues when using atomate2siesta.

.. contents:: Quick Navigation
   :local:
   :depth: 2

Installation Issues
-------------------

Package Installation Fails
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom**: ``pip install`` fails with dependency errors

**Common Causes**:

1. **Outdated pip**

   .. code-block:: bash

      # Solution: Update pip
      pip install --upgrade pip setuptools wheel

2. **Python version mismatch**

   .. code-block:: bash

      # Check Python version (requires 3.9+)
      python --version

      # Use correct Python version
      python3.9 -m pip install -e .

3. **Missing build dependencies**

   .. code-block:: bash

      # Install build essentials (Linux)
      sudo apt-get install build-essential python3-dev

      # Install build essentials (macOS)
      xcode-select --install

Import Errors
~~~~~~~~~~~~~

**Symptom**: ``ModuleNotFoundError: No module named 'atomate2.siesta'``

**Solutions**:

1. **Verify installation**:

   .. code-block:: bash

      pip list | grep atomate2

2. **Reinstall in development mode**:

   .. code-block:: bash

      cd /path/to/atomate2siesta
      pip install -e .

3. **Check Python environment**:

   .. code-block:: bash

      which python
      python -c "import sys; print(sys.path)"

Configuration Issues
--------------------

Configuration File Not Found
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom**: Settings not loaded, defaults used instead

**Cause**: Configuration file at wrong location or wrong name

**Solution**:

.. code-block:: bash

   # Correct location and name (NO hyphen after atomate2)
   ~/.atomate2siesta.yaml

   # NOT these:
   # ~/.atomate2-siesta.yaml  ❌ (wrong: has hyphen)
   # ~/.atomate2/siesta.yaml  ❌ (wrong: subdirectory)

**Verify configuration is loaded**:

.. code-block:: python

   from atomate2.siesta import SETTINGS
   print(SETTINGS)

SIESTA Command Not Found
~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom**: ``FileNotFoundError: SIESTA executable not found``

**Solutions**:

1. **Use full path to SIESTA**:

   .. code-block:: yaml

      # ~/.atomate2siesta.yaml
      SIESTA_CMD: "/usr/local/bin/siesta < siesta.fdf > siesta.out"

2. **Add SIESTA to PATH**:

   .. code-block:: bash

      # Add to ~/.bashrc or ~/.zshrc
      export PATH="/path/to/siesta/bin:$PATH"

3. **For MPI runs**:

   .. code-block:: yaml

      SIESTA_CMD: "mpirun -np 4 /path/to/siesta < siesta.fdf > siesta.out"

Pseudopotential Issues
~~~~~~~~~~~~~~~~~~~~~~

**Symptom**: ``PseudopotentialError: Pseudopotential not found for element X``

**Solutions**:

1. **Set pseudopotential path**:

   .. code-block:: yaml

      # ~/.atomate2siesta.yaml
      SIESTA_PP_PATH: "/path/to/pseudopotentials"

2. **Download pseudopotentials**:

   .. code-block:: bash

      # List available pseudopotential sets
      atomate2siesta-pseudos available

      # Install a pseudopotential set
      atomate2siesta-pseudos install psf

3. **Verify pseudopotentials exist**:

   .. code-block:: bash

      ls $SIESTA_PP_PATH/
      # Should show .psf or .vps files

SIESTA Calculation Errors
--------------------------

SCF Not Converged
~~~~~~~~~~~~~~~~~

**Symptom**: ``SCF not converged after N iterations``

This is the **most common** SIESTA error. Multiple solutions available:

**Solution 1: Automatic Error Handling (Recommended)**

Enable custodian for automatic SCF recovery:

.. code-block:: python

   from atomate2.siesta.jobs.core import RelaxMaker

   maker = RelaxMaker.fixed_cell_relaxation(
       use_custodian=True,  # Enable automatic error handling
       custodian_max_errors=15,  # Allow up to 15 correction attempts
   )

Custodian will automatically try 5 progressive correction strategies:

1. Reduce mixer weight (0.1 → 0.05 → 0.01)
2. Increase Pulay history (5 → 8 → 12)
3. Switch mixer method (Pulay → Linear)
4. Tighten tolerances
5. Restart from previous DM

**Solution 2: Manual Parameter Adjustment**

If not using custodian, adjust SCF parameters manually:

.. code-block:: python

   maker = RelaxMaker.fixed_cell_relaxation(
       user_params={
           "SCF.Mixer.Weight": 0.05,      # Reduce from default 0.1
           "SCF.Mixer.History": 8,         # Increase from default 5
           "SCF.Mix.First": True,          # Mix from first iteration
           "DM.Tolerance": 1e-5,           # Tighten tolerance
           "DM.NumberPulay": 8,            # More Pulay mixing
       }
   )

**Solution 3: Improve Initial Guess**

.. code-block:: python

   maker = RelaxMaker.fixed_cell_relaxation(
       user_params={
           "DM.UseSaveDM": True,           # Use previous DM if available
           "DM.MixingWeight": 0.05,        # Gentle mixing
           "ElectronicTemperature": "100 K",  # Small smearing
       }
   )

**Solution 4: System-Specific Strategies**

For **magnetic systems**:

.. code-block:: python

   user_params={
       "SpinPolarized": True,
       "DM.InitSpin": [[1, 1], [2, -1]],  # Initial spin configuration
       "SCF.Mixer.Weight": 0.01,           # Very gentle mixing
   }

For **metals**:

.. code-block:: python

   user_params={
       "OccupationFunction": "MP",         # Methfessel-Paxton
       "OccupationMPOrder": 1,
       "ElectronicTemperature": "300 K",   # Appropriate smearing
   }

For **large band gap systems**:

.. code-block:: python

   user_params={
       "OccupationFunction": "FD",         # Fermi-Dirac
       "ElectronicTemperature": "25 K",    # Small smearing
   }

**Solution 5: Check Physical Setup**

- **k-point mesh too coarse**: Increase k-points
- **Basis set issues**: Try different ``PAO.BasisSize`` (SZ, DZ, DZP, TZP)
- **Initial structure**: Ensure structure is reasonable

Memory Errors
~~~~~~~~~~~~~

**Symptom**: ``MemoryError`` or SIESTA killed by system

**Solutions**:

1. **Enable automatic memory handling**:

   .. code-block:: python

      maker = RelaxMaker.fixed_cell_relaxation(
          use_custodian=True,
      )

      # Custodian will automatically:
      # - Reduce k-point mesh
      # - Reduce basis size
      # - Disable unnecessary outputs

2. **Reduce memory manually**:

   .. code-block:: python

      user_params={
          "SaveRho": False,              # Don't save charge density
          "SaveDeltaRho": False,         # Don't save delta charge
          "SaveElectrostaticPotential": False,
          "WriteDenchar": False,
          "PAO.BasisSize": "SZ",         # Smaller basis set
      }

3. **Use more processors with less memory per process**:

   .. code-block:: yaml

      # ~/.atomate2siesta.yaml
      SIESTA_CMD: "mpirun -np 8 siesta < siesta.fdf > siesta.out"

Walltime/Timeout Errors
~~~~~~~~~~~~~~~~~~~~~~~

**Symptom**: Calculation killed due to time limit

**Solutions**:

1. **Enable automatic restart**:

   .. code-block:: python

      maker = RelaxMaker.fixed_cell_relaxation(
          use_custodian=True,
      )

      # Custodian will automatically restart from DM/XV files

2. **Set checkpoint files**:

   .. code-block:: python

      user_params={
          "DM.UseSaveDM": True,
          "UseSaveXV": True,
          "MD.UseSaveXV": True,
          "MD.UseSaveCG": True,
      }

3. **Reduce calculation cost**:

   .. code-block:: python

      user_params={
          "MaxSCFIterations": 100,       # Limit SCF iterations
          "MD.MaxSteps": 50,              # Limit MD/relax steps
      }

Negative Eigenvalues
~~~~~~~~~~~~~~~~~~~~

**Symptom**: ``WARNING: Negative eigenvalues detected``

**Cause**: Insufficient basis set or numerical issues

**Solutions**:

1. **Increase basis quality**:

   .. code-block:: python

      user_params={
          "PAO.BasisSize": "DZP",        # From SZ or DZ
          "PAO.EnergyShift": "0.01 Ry",  # Tighter confinement
      }

2. **Increase mesh cutoff**:

   .. code-block:: python

      user_params={
          "Mesh.Cutoff": "300 Ry",       # Increase from default
      }

3. **Use custodian** (handles automatically):

   .. code-block:: python

      maker = RelaxMaker.fixed_cell_relaxation(use_custodian=True)

Geometry Relaxation Not Converging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom**: Forces not decreasing, relaxation oscillating

**Solutions**:

1. **Adjust force tolerance**:

   .. code-block:: python

      user_params={
          "MD.MaxForceTol": "0.04 eV/Ang",  # Default
          "MD.MaxStressTol": "1.0 GPa",      # For cell relaxation
      }

2. **Change relaxation algorithm**:

   .. code-block:: python

      user_params={
          "MD.TypeOfRun": "CG",           # Try Conjugate Gradient
          # or
          "MD.TypeOfRun": "FIRE",         # Try FIRE algorithm
      }

3. **Use smaller steps**:

   .. code-block:: python

      user_params={
          "MD.MaxCGDispl": "0.1 Ang",     # Smaller displacement
          "MD.MaxStressTol": "0.5 GPa",   # Tighter stress tolerance
      }

File Format Issues
------------------

Structure File Conversion Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom**: Cannot read XV, FDF, or XSF files

**Solution**: Use the structure conversion powerup:

.. code-block:: python

   from atomate2.siesta.powerups import siesta_to_pymatgen
   from atomate2.siesta.jobs.core import RelaxMaker

   # Read SIESTA structure file
   job = RelaxMaker().make("input.XV")
   job = siesta_to_pymatgen(job, "input.XV")

**Supported formats**: XV, FDF, XSF, CIF, POSCAR

See: ``tutorials/07-advanced-features/04-structure-conversion/``

Invalid FDF Syntax
~~~~~~~~~~~~~~~~~~

**Symptom**: SIESTA fails with ``FDF syntax error``

**Cause**: Incorrect parameter format in ``user_params``

**Solution**: Check FDF parameter format:

.. code-block:: python

   # Correct formats
   user_params={
       "Mesh.Cutoff": "300 Ry",        # With units
       "PAO.BasisSize": "DZP",         # String
       "SCF.Mixer.Weight": 0.05,       # Float
       "a2s_kpts": [4, 4, 4],              # List for k-points
       "SpinPolarized": True,          # Boolean
   }

**Common mistakes**:

.. code-block:: python

   # Wrong - missing units
   "Mesh.Cutoff": 300                  # ❌ Should be "300 Ry"

   # Wrong - wrong case
   "pao.basissize": "DZP"             # ❌ Should be "PAO.BasisSize"

   # Wrong - Python bool instead of string
   "SpinPolarized": "True"            # ❌ Should be True (Python bool)

Workflow Execution Issues
--------------------------

Job Directory Conflicts
~~~~~~~~~~~~~~~~~~~~~~~

**Symptom**: ``FileExistsError: job directory already exists``

**Solutions**:

1. **Use unique names**:

   .. code-block:: python

      from datetime import datetime

      job.update_config(
          name=f"relax_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
      )

2. **Clean old directories**:

   .. code-block:: bash

      rm -rf job_*  # Caution: removes all job directories

3. **Use database storage** (recommended for production):

   See: ``tutorials/04-infrastructure/01-database-storage/``

Jobflow Run Errors
~~~~~~~~~~~~~~~~~~

**Symptom**: ``run_locally()`` fails unexpectedly

**Solutions**:

1. **Enable verbose logging**:

   .. code-block:: python

      import logging
      logging.basicConfig(level=logging.DEBUG)

      result = run_locally(job, create_folders=True)

2. **Check job validity**:

   .. code-block:: python

      from jobflow import Flow

      # Validate job
      print(job.name)
      print(job.function)
      print(job.uuid)

3. **Run in debug mode**:

   .. code-block:: python

      # Create folders but don't run
      flow = Flow([job])
      flow.update_metadata({"dry_run": True})

Remote Execution Issues
~~~~~~~~~~~~~~~~~~~~~~~

**Symptom**: Jobs fail on HPC cluster

**Solutions**:

1. **Verify cluster setup**:

   .. code-block:: bash

      atomate2siesta-cluster status

2. **Check jobflow-remote configuration**:

   .. code-block:: bash

      atomate2siesta-jobflow-remote test

3. **Test locally first**:

   .. code-block:: python

      # Always test workflows locally before submitting to cluster
      result = run_locally(job, create_folders=True)

See: ``tutorials/04-infrastructure/02-job-submission/``

Performance Issues
------------------

Calculation Too Slow
~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Calculation taking much longer than expected

**Solutions**:

1. **Check k-point mesh**:

   .. code-block:: python

      # Too dense
      user_params={"a2s_kpts": [12, 12, 12]}  # Reduce to [6, 6, 6]

2. **Optimize parallelization**:

   .. code-block:: yaml

      # ~/.atomate2siesta.yaml
      # Try different processor counts
      SIESTA_CMD: "mpirun -np 8 siesta < siesta.fdf > siesta.out"

3. **Use efficiency options**:

   .. code-block:: python

      user_params={
          "DirectPhi": True,           # Faster for large systems
          "SaveMemory": True,          # Trade speed for memory
      }

See: ``tutorials/07-advanced-features/14-efficiency-options/``

Convergence Studies Taking Forever
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Solution**: Use dry-run mode to preview first:

.. code-block:: python

   from atomate2.siesta.flows.convergence import KpointsConvergenceFlowMaker

   maker = KpointsConvergenceFlowMaker(
       kpoints_list=[[2,2,2], [4,4,4], [6,6,6], [8,8,8]],
       dry_run=True,  # Just generate inputs, don't run
   )
   workflow = maker.make(structure)

See: ``tutorials/04-infrastructure/05-dry-run-preview/``

Database Issues
---------------

MongoDB Connection Failed
~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom**: ``ConnectionError: Could not connect to MongoDB``

**Solutions**:

1. **Test connection**:

   .. code-block:: bash

      atomate2siesta-database test

2. **Check MongoDB is running**:

   .. code-block:: bash

      # Linux/macOS
      sudo systemctl status mongod

      # Start if not running
      sudo systemctl start mongod

3. **Verify connection string**:

   .. code-block:: yaml

      # ~/.atomate2siesta.yaml
      SIESTA_STORE: "mongodb://localhost:27017/atomate2siesta"

Query Returns No Results
~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom**: Database query finds nothing despite calculations running

**Cause**: Results not stored in database

**Solution**: Ensure you're using database store:

.. code-block:: python

   from maggma.stores import MongoStore
   from jobflow import JobStore

   # Create store
   store = JobStore(
       MongoStore(
           database="atomate2siesta",
           collection_name="tasks",
           host="localhost",
           port=27017,
       )
   )

   # Run with store
   result = run_locally(job, store=store, create_folders=True)

See: ``tutorials/04-infrastructure/01-database-storage/``

Debugging Tips
--------------

Enable Detailed Logging
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import logging

   # Set to DEBUG for maximum verbosity
   logging.basicConfig(
       level=logging.DEBUG,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
   )

Inspect SIESTA Output Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Check SIESTA output
   cat job_*/siesta.out | grep -i "error\|warning\|fatal"

   # Check final energy
   grep "Total =" job_*/siesta.out

   # Check SCF convergence
   grep "SCF" job_*/siesta.out | tail -20

Use Dry-Run Mode
~~~~~~~~~~~~~~~~

Preview workflow without running expensive calculations:

.. code-block:: python

   maker = RelaxMaker.fixed_cell_relaxation(dry_run=True)
   job = maker.make(structure)
   result = run_locally(job, create_folders=True)

   # Check generated inputs
   cat job_*/siesta.fdf

Check Input Files
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from atomate2.siesta.sets.tiers import Tier1InputSet

   input_set = Tier1InputSet(structure=structure)
   input_set.write_input("test_inputs")

   # Manually inspect files in test_inputs/

Getting Help
------------

Still Stuck?
~~~~~~~~~~~~

1. **Search existing issues**: https://github.com/arsalan-akhtar/atomate2siesta/issues

2. **Check tutorials**: Most common workflows have dedicated tutorials

3. **Open a new issue** with:

   - Minimal reproducible example
   - Full error message
   - SIESTA version
   - Python version
   - Relevant configuration

4. **Include diagnostic information**:

   .. code-block:: python

      # Run this and include output
      from atomate2.siesta import SETTINGS
      import sys
      print(f"Python: {sys.version}")
      print(f"Settings: {SETTINGS}")

Common Error Messages Quick Reference
--------------------------------------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Error Message
     - Solution Section
   * - ``SCF not converged``
     - `SCF Not Converged`_
   * - ``MemoryError``
     - `Memory Errors`_
   * - ``SIESTA executable not found``
     - `SIESTA Command Not Found`_
   * - ``Pseudopotential not found``
     - `Pseudopotential Issues`_
   * - ``Negative eigenvalues``
     - `Negative Eigenvalues`_
   * - ``FileExistsError: job directory``
     - `Job Directory Conflicts`_
   * - ``ConnectionError: MongoDB``
     - `MongoDB Connection Failed`_
   * - ``FDF syntax error``
     - `Invalid FDF Syntax`_
   * - ``Forces not converging``
     - `Geometry Relaxation Not Converging`_

Additional Resources
--------------------

- **Full Documentation**: :doc:`index`
- **Tutorials**: ``tutorials/README.md``
- **Custodian Guide**: :doc:`custodian`
- **CLI Tools**: :doc:`cli-tools`
- **GitHub Issues**: https://github.com/arsalan-akhtar/atomate2siesta/issues
