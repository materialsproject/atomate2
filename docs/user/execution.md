(atomate2_execution)=

# Executing atomate2 workflows on remote resources

It is increasingly common to want to run workflows across many different clusters and diverse hardware.
`atomate2`, via Jobflow, is compatible with several execution backends that make this possible.

`atomate2` (and more generally, Jobflow) workflows can be executed on remote systems using either:

- [FireWorks][fireworks]
- [Jobflow-Remote][jobflow-remote]

Each approach has strengths and weaknesses.
This document will focus on the broad strokes, with full tutorials for setting up
FireWorks and Jobflow-Remote can be found under the [Jobflow FireWorks Guide][fw_guide] 
and [Using atomate2 with FireWorks][fireworks].

FireWorks and Jobflow-Remote can orchestrate Jobflow workflows via an intermediate database (the so-called Launchpad for FireWorks, and the queue store for Jobflow-Remote). 
They both make use of MongoDB as the database backend, which requires a persistent server, or at least a local workstation, to be running in perpituity.

FireWorks has a centralised server and worker model, where remote compute resources ("FireWorkers") connect directly to the intermediate database to request jobs, execute them, and then serialize and return the results. 
Typically, this is achieved by submitting a job to the HPC queue on the cluster that invokes the `qlaunch` script to ask the database for a task to perform. 
This job can then optionally be created in such a way that the queue is kept full, by way of a `qlaunch` HPC job that essentially submits a copy of itself in so-called "infinite" mode.
In the not-uncommon case where direct connection from a compute node to a remote database is not possible, FireWorks also offers an "offline" mode where only the login node needs to be able to make the connection.

Jobflow-Remote works slightly differently; a separate "Runner" process (or daemon) monitors the queue store database for new jobs (created by locally executing a Python script containing a `jobflow_remote.submit_flow` call).
This "Runner" (and the local environment that the workflow script was called from) needs to "know" about the various compute resources ("workers") available for executing the workflow, as well as configuration for how to connect to them, and default settings for the HPC job (e.g., which queueing system, project budget, queue partition etc. to submit to).
The "Runner" process then takes a job from the database, resolves its dependencies, generates an HPC queue submission script (where appropriate), copies the required files to the remote worker (in a specialised directory for that specific workflow run), and then submits the job to the queue.

Once the job has been submitted, both FireWorks and the Jobflow "Runner" then monitor the HPC queue and update the queue store with the current state of the job (and will attempt to retry on various error states); if the the job runs successfully, the serialized results will be transferred back into the corresponding database.

There are benefits to each of the different approaches taken by FireWorks and
Jobflow-Remote respectively.

- FireWorks supports batching workflows into single HPC jobs with "rapidfire" mode. This can be very beneficial in cases where compute resources allow very long-running node reservations to particular users, but comes with the cost that it is more likely a given job may not finish during the given walltime limits (e.g., a new workflow could be executed without enough time left in the HPC job).
- Jobflow-Remote makes it easier to have fine-grained control over the compute resources (e.g., number of cores, walltime, compute environment) available to a given step in a workflow (e.g., 
- FireWorks is a mature software package that predates Jobflow. It has been battle-tested for many years in production workflows. As such, FireWorks v2 now has a very stable API but is unlikely to add new features. Jobflow-Remote is relatively new, and is still under active development. As it is designed directly with Jobflow in mind, it can adapt to the latest features and trends in the field (e.g., support for multi-factor authentication on clusters, simpler configuration, more flexible job serialization).

[fireworks]: https://materialsproject.github.io/fireworks/
[jobflow-remote]: https://matgenix.github.io/jobflow-remote/
[fireworks_instructions]: https://materialsproject.github.io/jobflow/install_fireworks.html
[fw_guide]: https://materialsproject.github.io/jobflow/tutorials/8-fireworks.html
