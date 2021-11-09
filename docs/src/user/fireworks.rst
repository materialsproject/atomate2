.. _atomate2_fireworks:

=============================
Using atomate2 with FireWorks
=============================

This tutorial will document how to configure atomate2 with FireWorks_. FireWorks allows
you to easily submit and manage thousands of jobs at once.

.. _FireWorks: https://materialsproject.github.io/fireworks/

For now, follow the
`FireWorks instructions <https://atomate.org/installation.html#configure-database-connections-and-computing-center-parameters>`_
in the atomate1 documentation but ignore the parts pertaining to atomate1.

Once you have constructed your workflow using atomate2, you can convert it to a
FireWorks workflow using the :obj:`~jobflow.managers.fireworks.flow_to_workflow` function.
The workflow can then be submitted to the launchpad in the usual way. For example, to
submit an MgO band structure workflow using FireWorks:

.. code-block:: python

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
    bandstructure_flow = RelaxBandStructureMaker().make(si_structure)

    # convert the flow to a fireworks WorkFlow object
    wf = flow_to_workflow(bandstructure_flow)

    # submit the workflow to the FireWorks launchpad
    lpad = LaunchPad.auto_load()
    lpad.add_wf(wf)
