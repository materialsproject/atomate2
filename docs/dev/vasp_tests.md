
# Writing VASP Tests

## Considerations

Atomate2 includes tools to help write tests for VASP workflows. The primary
considerations with the atomate2 testing environment are listed below.

### Pseudopotentials

We cannot include any POTCAR files with atomate2 as they are copyrighted material.

To overcome this, the reference test data includes POTCAR.spec files that only
contain the pseudopotential name and not the data.

### File sizes

The files produced by VASP are generally large and would overwhelm the size of the
atomate2 repository if not managed carefully. For example, CHGCAR files can easily be
ten's of megabytes which can quickly add up.

To overcome this, we only include essential VASP output files in the atomate2 test
folder. For example, CHGCAR, LOCPOT, and other density information is not needed in most
instances. One exception is non-self-consistent band structures where the charge density
must be copied from a static calculation. Any other example is in the amset workflow,
where the WAVECAR is needed to extract the wavefunction coefficients.

### VASP execution

We cannot run VASP on the testing server due to the computational expense. Furthermore,
different versions/compilations of VASP may yield slightly different total energies
which are not important for our tests – we only test that (i) inputs are written
correctly, (ii) outputs are parsed correctly, and (iii) jobs are connected together
properly.

This is achieved by "mocking" VASP execution. Instead of running VASP, we copy reference
output files into the current directory and then proceed with running the workflow.

## The atomate2 dev command

Atomate2 provides the `atm dev vasp-test-data` command that automatically prepares
VASP data for use in atomate2 tests. It does this by:

- Copying VASP inputs and outputs into the correct directory structure.
- Converting POTCAR files to POTCAR.spec files.
- Removing large and unnecessary VASP files.
- Providing a template unit test that is configured for the specific workflow.

There are four stages to generating the test data:

## 1. Run the VASP workflow to generate reference outputs

Ensure that you are on a machine that can run VASP. Create a python file that contains
the code to run your workflow. We recommend adjusting the VASP settings so that the
files generated are not too large and can be run quickly. E.g., by reducing the k-point
mesh density or energy cutoff.

The script should also contain some additional code that will allow
`atm dev vasp-test-data` to process the reference data. Below we give an example
used to generate the elastic constant workflow test data.

```py
from atomate2.vasp.flows.elastic import ElasticMaker
from atomate2.vasp.powerups import update_user_kpoints_settings
from pymatgen.core import Structure
from jobflow import run_locally, JobStore
from maggma.stores.mongolike import MemoryStore
from monty.serialization import dumpfn

# silicon structure
si_structure = Structure(
    lattice=[
        [3.348898, 0.0, 1.933487],
        [1.116299, 3.157372, 1.933487],
        [0.0, 0.0, 3.866975],
    ],
    species=["Si", "Si"],
    coords=[[0.25, 0.25, 0.25], [0, 0, 0]],
)

# generate the flow and reduce the k-point mesh for the relaxation jobs
flow = ElasticMaker().make(si_structure)
flow = update_user_kpoints_settings(flow, {"grid_density": 100}, name_filter="relax")

# run the workflow using a custom store so that we can easily compile test data
store = JobStore(MemoryStore(), additional_stores={"data": MemoryStore()})
run_locally(flow, store=store, create_folders=True)

# dump all of the job outputs to the outputs.json file in the current directory
outputs = list(store.query(load=True))
dumpfn(outputs, "outputs.json")
```

You should edit the part where the flow is generated but leave the rest of the code
the same. You should now run the script in a folder and generate the outputs.json file.

## 2. Compile the test data

The next stage is to compile the calculation data into the correct format. For each
VASP job in the workflow, there should be a folder with the name of the job that
contains:

- A folder called "inputs" with the INCAR, POTCAR.spec, POSCAR, and KPOINTS files. Note
  that the KPOINTS file is optional and won't be present if KSPACING is set in the INCAR.
- A folder called "outputs" with the vasprun.xml, OUTCAR, json log files and any other
  output files needed for the workflow to run (e.g., CHGCAR file for band structure
  workflows).

To generate this folder run the following command in the folder containing the
outputs.json file.

```bash
atm dev vasp-test-data WF_NAME
```

You should change WF_NAME to be a name for the workflow. Note, WF_NAME should not
contain spaces or punctuation. For example, the elastic constant workflow test data was generated using `atm dev vasp-test-data Si_elastic`.

This will generate a folder in the current directory called "WF_NAME" that contains
the folders in the correct format.

````{note}
By default, the script will only copy POTCAR, POSCAR, CONTCAR, KPOINTS, INCAR,
vasprun.xml, OUTCAR and JSON files to the WF_NAME folder. If additional files are
needed for specific steps of the workflow you need to copy them in manually. A
mapping from jobflow calculation folder to job folder in WF_NAME is given at the
top of the `atm dev vasp-test-data` script output. E.g., it will look something
like

```
A mapping from the original job folders to the formatted folders is:
  /Users/alex/atomate2/job_2021-11-08-17-24-31-799852-28250  ->  Si_elastic/tight_relax_1
  /Users/alex/atomate2/job_2021-11-08-17-25-14-718901-28808  ->  Si_elastic/tight_relax_2
  /Users/alex/atomate2/job_2021-11-08-17-25-38-237201-15341  ->  Si_elastic/elastic_relax_6_6
  /Users/alex/atomate2/job_2021-11-08-17-26-12-877896-35631  ->  Si_elastic/elastic_relax_5_6
  /Users/alex/atomate2/job_2021-11-08-17-26-47-215837-12883  ->  Si_elastic/elastic_relax_4_6
  /Users/alex/atomate2/job_2021-11-08-17-27-11-602937-71135  ->  Si_elastic/elastic_relax_3_6
  /Users/alex/atomate2/job_2021-11-08-17-27-45-722573-61724  ->  Si_elastic/elastic_relax_2_6
  /Users/alex/atomate2/job_2021-11-08-17-28-10-286137-10861  ->  Si_elastic/elastic_relax_1_6
```
````

```{warning}
For the script to run successfully, every job in your workflow must have a unique
name. For example, there cannot be two calculations called "relax". Instead you
should ensure they are named something like "relax 1" and "relax 2".
```

## 3. Copy the test data folder into atomate2

You can now copy the WF_NAME folder into the atomate2 test files. VASP test files live
in `atomate2/tests/test_data/vasp`. Ensure that a workflow with that name doesn't
already exist in the folder.

## 4. Write the test

The `atm dev vasp-test-data` also generates an example test that is configured to
use the test data we just generated.

The most important part is the section that mocks VASP and configures which checks
to perform on the input files. For the elastic constant workflow, it looks something like
this:

```py
# mapping from job name to directory containing test files
ref_paths = {
    "elastic relax 1/6": "Si_elastic/elastic_relax_1_6",
    "elastic relax 2/6": "Si_elastic/elastic_relax_2_6",
    "elastic relax 3/6": "Si_elastic/elastic_relax_3_6",
    "elastic relax 4/6": "Si_elastic/elastic_relax_4_6",
    "elastic relax 5/6": "Si_elastic/elastic_relax_5_6",
    "elastic relax 6/6": "Si_elastic/elastic_relax_6_6",
    "tight relax 1": "Si_elastic/tight_relax_1",
    "tight relax 2": "Si_elastic/tight_relax_2",
}

# settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
fake_run_vasp_kwargs = {
    "elastic relax 1/6": {"incar_settings": ["NSW", "ISMEAR"]},
    "elastic relax 2/6": {"incar_settings": ["NSW", "ISMEAR"]},
    "elastic relax 3/6": {"incar_settings": ["NSW", "ISMEAR"]},
    "elastic relax 4/6": {"incar_settings": ["NSW", "ISMEAR"]},
    "elastic relax 5/6": {"incar_settings": ["NSW", "ISMEAR"]},
    "elastic relax 6/6": {"incar_settings": ["NSW", "ISMEAR"]},
    "tight relax 1": {"incar_settings": ["NSW", "ISMEAR"]},
    "tight relax 2": {"incar_settings": ["NSW", "ISMEAR"]},
}

# automatically use fake VASP and write POTCAR.spec during the test
mock_vasp(ref_paths, fake_run_vasp_kwargs)
```

The `ref_paths` variable contains the mapping from job name to test folder.
The `fake_run_vasp_kwargs` contains the settings that will get passed to the
`fake_run_vasp` function in the `atomate2/tests/vasp/conftest.py` file. This
variable controls which INCAR settings are checked in the reference INCAR and the INCAR
generated by atomate2 during the test. You should update these settings to include
the important parameters for the jobs in your workflow. I.e., if it is a relaxation
then the value of NSW is important.

Finally, the call too `mock_vasp` configures the test such that:

1. POTCAR files will be written as POTCAR.spec files.
2. The `fake_run_vasp` function will be called instead of the {obj}`.run_vasp`
   function. `fake_run_vasp` is responsible for checking the correct inputs are
   written (by comparing against the files in the "inputs" folder) and copying in the
   reference files from the "outputs" folder for each job.

After `mock_vasp` is called, you should edit the generate and run the workflow.
Ensure that the workflow is generated in exactly the same was as in step 1. E.g.,
if you altered the k-point density when generating the test data, you must also alter
the k-point density during the test.

Finally, you should add `assert` statements to validate the workflow outputs. As an
example, the full elastic workflow test is reproduced below.

```py
def test_elastic(mock_vasp, clean_dir):
    import numpy as np
    from jobflow import run_locally

    from atomate2.common.schemas.elastic import ElasticDocument
    from atomate2.vasp.flows.elastic import ElasticMaker
    from atomate2.vasp.powerups import update_user_kpoints_settings

    # mapping from job name to directory containing test files
    ref_paths = {
        "elastic relax 1/6": "Si_elastic/elastic_relax_1_6",
        "elastic relax 2/6": "Si_elastic/elastic_relax_2_6",
        "elastic relax 3/6": "Si_elastic/elastic_relax_3_6",
        "elastic relax 4/6": "Si_elastic/elastic_relax_4_6",
        "elastic relax 5/6": "Si_elastic/elastic_relax_5_6",
        "elastic relax 6/6": "Si_elastic/elastic_relax_6_6",
        "tight relax 1": "Si_elastic/tight_relax_1",
        "tight relax 2": "Si_elastic/tight_relax_2",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "elastic relax 1/6": {"incar_settings": ["NSW", "ISMEAR"]},
        "elastic relax 2/6": {"incar_settings": ["NSW", "ISMEAR"]},
        "elastic relax 3/6": {"incar_settings": ["NSW", "ISMEAR"]},
        "elastic relax 4/6": {"incar_settings": ["NSW", "ISMEAR"]},
        "elastic relax 5/6": {"incar_settings": ["NSW", "ISMEAR"]},
        "elastic relax 6/6": {"incar_settings": ["NSW", "ISMEAR"]},
        "tight relax 1": {"incar_settings": ["NSW", "ISMEAR"]},
        "tight relax 2": {"incar_settings": ["NSW", "ISMEAR"]},
    }

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    # generate flow
    si_structure = Structure(
        lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )

    # generate the flow and reduce the k-point mesh for the relaxation jobs
    flow = ElasticMaker().make(si_structure)
    flow = update_user_kpoints_settings(
        flow, {"grid_density": 100}, name_filter="relax"
    )

    # run the flow and ensure that it finished running successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    # validate workflow outputs
    elastic_output = responses[flow.jobs[-1].uuid][1].output
    assert isinstance(elastic_output, ElasticDocument)
    assert np.allclose(
        elastic_output.elastic_tensor.ieee_format,
        [
            [155.7923, 54.8871, 54.8871, 0.0, 0.0, 0.0],
            [54.8871, 155.7923, 54.8871, 0.0, 0.0, 0.0],
            [54.8871, 54.8871, 155.7923, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 31.5356, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 31.5356, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 31.5356],
        ],
        atol=1e-3,
    )
```

Note that the `mock_vasp` and `clean_dir` arguments to the test function are
[pytest fixtures](https://docs.pytest.org/en/6.2.x/fixture.html) and are essential
for the test to run successfully.

```{warning}
For `mock_vasp` to work correctly, all imports needed for the test must be
imported in the test function itself (rather than at the top of the file).
```
