(atomate2_fireworks)=

# Using atomate2 with FireWorks

This tutorial will document how to use atomate2 with [FireWorks][fireworks].
FireWorks allows you to easily submit and manage thousands of jobs at once.

Follow the [FireWorks Setup][fireworks_instructions]
in the Jobflow documentation to install FireWorks.

Once you have constructed your workflow using atomate2, you can convert it to a
FireWorks workflow using the {obj}`~jobflow.managers.fireworks.flow_to_workflow` function.
The workflow can then be submitted to the launchpad in the usual way. For example, to
submit an MgO band structure workflow using FireWorks:

```python
from fireworks import LaunchPad
from atomate2.vasp.flows.core import RelaxBandStructureMaker
from jobflow.managers.fireworks import flow_to_workflow
from pymatgen.core import Structure

# construct a rock salt MgO structure
mgo_structure = Structure(
    lattice=[[0, 2.13, 2.13], [2.13, 0, 2.13], [2.13, 2.13, 0]],
    species=["Mg", "O"],
    coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
)

# make a band structure flow to optimise the structure and obtain the band structure
bandstructure_flow = RelaxBandStructureMaker().make(mgo_structure)

# convert the flow to a fireworks WorkFlow object
wf = flow_to_workflow(bandstructure_flow)

# submit the workflow to the FireWorks launchpad
lpad = LaunchPad.auto_load()
lpad.add_wf(wf)
```

Additional details about interfacing Jobflow-based packages with FireWorks can be found in the [Running Jobflow with FireWorks](fw_guide) guide.

[fireworks]: https://materialsproject.github.io/fireworks/
[fireworks_instructions]: https://materialsproject.github.io/jobflow/install_fireworks.html
[fw_guide]: https://materialsproject.github.io/jobflow/tutorials/8-fireworks.html
