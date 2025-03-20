(atomate2_jobflow-remote)=
# Using atomate2 with jobflow-remote

Please see [jobflow-remote][jobflow-remote] for how to install the software.
Jobflow-remote allows you to easily submit and manage thousands of jobs at once and has several submission
options including remote submission.

Once you have constructed your atomate2 workflow, you can easily submit it with jobflow-remote:

```py
from atomate2.vasp.flows.core import RelaxBandStructureMaker
from atomate2.vasp.powerups import add_metadata_to_flow
from jobflow_remote import submit_flow
from pymatgen.core import Structure

resources = {"nodes": 1, "partition": "micro", "time": "00:55:00", "ntasks": 48}

# construct a rock salt MgO structure
mgo_structure = Structure(
    lattice=[[0, 2.13, 2.13], [2.13, 0, 2.13], [2.13, 2.13, 0]],
    species=["Mg", "O"],
    coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
)

# make a band structure flow to optimise the structure and obtain the band structure
bandstructure_flow = RelaxBandStructureMaker().make(mgo_structure)

# (Optional) add metadata to the flow task document.
# Could be useful to filter specific results from the database.
# For e.g., adding material project ID for the compound, use following lines
bandstructure_flow = add_metadata_to_flow(
    flow=bandstructure_flow,
    additional_fields={"mp_id": "mp-190"},
)

submit_flow(bandstructure_flow, worker="my_worker", resources=resources, project="my_project")
```





[jobflow-remote]: https://materialsproject.github.io/fireworks/