(atomate2_execution)=

# Executing atomate2 workflows on remote resources

It is increasingly common to want to run workflows across many different
clusters and diverse hardware.
`atomate2`, via Jobflow, is compatible with several execution backends that make this
possible.

Jobflow workflows can be executed using either

- [FireWorks][fireworks]
- [Jobflow-Remote][jobflow-remote]

which each has its own strengths and weaknesses.
This document will focus on the broad strokes, and full tutorials for setting up
FireWorks and Jobflow-Remote can be found under the [Jobflow FireWorks Guide][fw_guide] 
and [Using atomate2 with FireWorks][fireworks].

Both FireWorks and Jobflow-Remote orchestrate Jobflow workflows via an
intermediate database (the so-called Launchpad for FireWorks, and the queue store for Jobflow-Remote). 

FireWorks has a centralised server and worker model, whereby remote compute resources ("FireWorkers") connect directly to the an intermediate database to request jobs, execute them, and then return the results. 
Typically, this is achieved by submitting a job to the HPC queue on the cluster that invokes the `qlaunch` script to ask the database for a task to perform. This job can be submitted in such a way that the queue is kept full, by submitting an HPC job that essentially submits a copy of itself in so-called "infinite" mode.

[fireworks]: https://materialsproject.github.io/fireworks/
[jobflow-remote]: https://matgenix.github.io/jobflow-remote/
[fireworks_instructions]: https://materialsproject.github.io/jobflow/install_fireworks.html
[fw_guide]: https://materialsproject.github.io/jobflow/tutorials/8-fireworks.html
