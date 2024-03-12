# Atomate 1 vs 2

This document contains introductory context for people coming from atomate 1.
One of atomate2's core ideas is to allow scaling from a single material, to 100 materials, or 100,000 materials. Therefore, both local submission options and a connection to workflow managers such as FireWorks exist. We plan to support more workflow managers in the future to further ease job submission.

## Relation between managers running the actual jobs and the workflow as written in `atomate2`

There is no leakage between job manager and the workflow definition in `atomate2`. For example, Fireworks is not a required dependency of `atomate2` or `jobflow`. Any `atomate2` workflow can be run using the local manager or an extensible set of external providers, Fireworks just one among them. E.g. all tests are run with mocked calls to the executables using the local manager.

## Do I need to write separate codes for different managers?

If you are adding a new manager option beyond local or FireWorks, you'll need to write a new converter in `jobflow` that takes a job or flow and converts it to the analogous object(s) for the manager you wish to use. This does not impact the `atomate2` code in any way.

Typically, the workflow is as follows:

1. Write a workflow in `atomate2` that represents a job or flow.
2. Import the job or flow in your script/notebook from `atomate2`.
3. Convert the job or flow to manager-compatible objects using a convenience function in `jobflow`.
4. Define the specs for the new object type based on your desired resources.
5. Dispatch the jobs.

## What if a workflow manager stops being maintained?

`atomate2` and `jobflow` remain unaffected in such a case. The user can choose a different manager to suit their needs. This ensures that workflow definition and dispatch are fully decoupled.

In an ideal world, `jobflow` would offer multiple manager options, allowing users to run `atomate2` codes as per their preference. This full decoupling is one of the most powerful features of this stack.
