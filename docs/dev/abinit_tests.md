
# Writing ABINIT Tests

## Considerations

Atomate2 includes tools to help write tests for ABINIT workflows. The primary
considerations with the atomate2 testing environment are listed below.

### Pseudopotentials

ABINIT heavily relies on pseudo potential tables accessible through abipy. These
tables are large in size or can be downloaded on the fly at input creation time.
Therefore, a smaller pseudopotential table is included for just a few elements.
Structures to be used for testing should be based on the missing elements should
be added to this pseudopotential table.

Note that information from the real pseudopotential files is used in the creation
of the jobs and flows, hence fake pseudopotentials are not an option here.


### File sizes

The files produced by ABINIT are generally large and would overwhelm the size of the
atomate2 repository if not managed carefully. For example, density (DEN) and
wavefunction (WFK) files can easily be ten's of megabytes which can quickly add up.

To overcome this, we only include essential ABINIT output files in the atomate2 test
folder. For example, DEN, WFK and other density information is not needed in most
instances. For these outputs files which can be required inputs for some jobs, fake
files are generated in the test folder and the linking copying of the files is checked
using these fake files. These fake files contain the information whether they are
a regular file or a symbolic link to another regular file.

### ABINIT execution

We cannot run ABINIT on the testing server due to the computational expense. Furthermore,
different versions/compilations of ABINIT may yield slightly different total energies
which are not important for our tests – we only test that (i) inputs are written
correctly, (ii) outputs are parsed correctly, and (iii) jobs are connected together
properly.

This is achieved by "mocking" ABINIT execution. Instead of running ABINIT, we copy reference
output files into the current directory and then proceed with running the workflow.

Note that it is still possible to run integration tests where ABINIT is executed by
passing the `--abinit-integration` option to pytest:

```bash
pytest --abinit-integration
```

When executing tests with the real abinit, larger deviations are expected depending on
ABINIT version, compilation options, etc.

## Generation of new tests

Atomate2 provides an automatic procedure to prepare ABINIT data (reference files)
for use in atomate2 tests. It does this by:

- Preparing a standard maker file that will be used to generate the reference files as
  well as to run the tests.
- Create the flow or job using the maker file and a structure file (cif file or other).
- Copying ABINIT inputs and outputs into the correct directory structure and creating
  the fake input and output files for large files when relevant.
- Providing a template unit test that is configured for the specific workflow.

There are four stages to generating the test data:

### 1. Create the maker file

The `atm dev abinit-script-maker` command allows to prepare a template `create_maker.py`
script in the current directory. You should adapt this file for the maker you intend
to test. Try to make sure to use parameters that allow the generation to be executed
relatively fast. Additionally, with the integration testing capability for ABINIT
workflows, the faster the workflows can run, the better.

After adapting the `create_maker.py` script for the maker to be tested, you should run
it:

```bash
python create_maker.py
```

This will generate a `maker.json` containing the serialized version of the maker together
with additional metadata information, inter alia the string of the `create_maker.py` script
itself, the author and author mail (extracted from git config information), date of
generation, ...

### 2. Generate the reference files

The `atm dev abinit-generate-reference` command runs the workflow for a given structure
in the current directory using `jobflow`'s `run_locally` option. This will execute the
different abinit jobs of the flow in separated run folders, and dump an `outputs.json`
file with all the outputs of the flow.

Note that the structure is specified either implicitly in an `initial_structure.json`
file:

```bash
atm dev abinit-generate-reference
```

or explicitly, e.g. as a path to a CIF file:

```bash
atm dev abinit-generate-reference /path/to/structure.cif
```

When an explicit structure file is passed to the `atm dev abinit-generate-reference`
command, the structure is dumped to an `initial_structure.json` file.

### 3. Copy files to the test_data folder

Now that the flow has been executed, the generated input and output files have to be
copied to the tests/test_data/abinit folder. This is achieved using the
`atm dev abinit-test-data` command:

```bash
atm dev abinit-test-data TEST_NAME
```

You should change `TEST_NAME` to be a name for the workflow test. Note, `TEST_NAME` should not
contain spaces or punctuation. For example, the band structure workflow test data was
genenerated using `atm dev vasp-test-data Si_band_structure`.

This will automatically detect whether the Maker is a Job Maker or a Flow Maker and
copy files in the corresponding `tests/test_data/abinit/jobs/NameOfMaker/TEST_NAME`
or `tests/test_data/abinit/flows/NameOfMaker/TEST_NAME` directory. It will create
the `NameOfMaker/TEST_NAME` directory structure and copy the information about the
Maker and initial structure, i.e. `maker.json`, `initial_structure.json` and
`make_info.json` if present.

Each job of the flow has its own directory in the `TEST_NAME` directory,
with one directory for each "restart" (i.e. index of the job). The directory
for a given ABINIT run, hereafter referenced as `REF_RUN_FOLDER` thus has the
following structure:

`tests/test_data/abinit/jobs_OR_flows/NameOfMaker/TEST_NAME/JOB_NAME/JOB_INDEX`

where `JOB_NAME` is the name of the job and `JOB_INDEX` is the index of the job
(usually "1" unless the job is restarted).

**Note:** For the script to run successfully, every job in the workflow must have
a unique name. For example, there cannot be two calculations called "relax".
Instead you should ensure they are named something like "relax 1" and "relax 2".

Each `REF_RUN_FOLDER` contains:
- A folder called "inputs" with the run.abi and abinit_input.json, as well as with the
  indata, outdata and tmpdata directories. The indata directory potentially contains
  the reference fake input files needed for the job to be executed (e.g. a fake link to a
  previous DEN file).
- A folder called "outputs" with the run.abo, run.err, run.log, as well as with the
  indata, outdata and tmpdata directories. In the indata, outdata and tmpdata directories,
  the large files are replaced by fake reference files while the necessary files for the
  workflow test execution are present.

### 4. Write the test

The `atm dev abinit-test-data` command also generates a template test method that is
configured to use the test data that was just generated. In this template test method,
the maker, the initial_structure and the reference paths (i.e. the mapping from the job
name and job index to the reference job folder) are automatically loaded from the test
folder.

Add `assert` statements to validate the workflow outputs.
