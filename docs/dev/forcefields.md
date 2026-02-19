# Maintaining the machine learning forcefields module

Some of these points are already noted in the `pyproject.toml`. This goes into a bit more depth.

## Overview

`atomate2` contains a convenience interface to many common machine learning interatomic forcefields (MLFFs) via the atomic simulation environment (ASE). In the literature, these may also be known as machine learning interatomic potentials (MLIPs), or, when specifically referring to MLIPs with coverage of most of the periodic table, foundation potentials (FPs). 

There is both an `ase` module in `atomate2`, based around general `ase` `Calculator`s, and a `forcefields`-specific module which has a higher number of workflows.

The `ase` module should be used to manage high-level tasks, such as geometry optimization, molecular dynamics, and nudged elastic band. Any further developments to these tools in `ase` should also warrant updates in this module in `atomate2`. For example, when `ase` rolled out the `MTKNPT` NPT MD barostat as a replacement for the default barostat, this was also made the default in `atomate2`.

The `forcefields` library should be used to develop concrete implementations of workflows, e.g., harmonic phonon, Grüneisen parameter, ApproxNEB.

## Dependency Chaos

The individual MLFFs in `atomate2` often have conflicting dependencies. This makes testing and managing a consistent, relatively up-to-date testing environment challenging.
We want to avoid pinning MLFF libraries at older versions, because this may break their API within `atomate2`, or lead to drift in test data as models evolve.

Thus, it is likely that the `pyproject.toml` contains multiple optional dependencies under the header `strict-forcefields-*`. These groupings are used to ensure the most recent version of a MLFF library is installed in CI testing, with acceptable dependencies. The names of these groups can change over time, but the names should be chosen to be informative as to why they exist: ex., `strict-forcefields-e3nn-limited` to indicate that these MLFFs need an older version of `e3nn`, or `strict-forcefields-generic` to indicate that no strong dependency limitation is observed.

When updating these groupings, it is critical to ensure that you also update the `.github/workflows/testing.yml` testing workflow. You will see that the different forcefield dependency groups are tested separately.

When adding a new MLFF and tests for it (if possible), you must ensure that appropriate `pytest.mark.skipif` decorators are applied if that MLFF package is not installed. A `mlff_is_installed` boolean check is included in `tests/forcefields/conftest.py` for convenience in writing these skip test markers. See `tests/forcefields/test_jobs.py` for examples.

## Testing limitations

Some MLFFs, like FAIRChem, have access restrictions on them which prohibit running tests in CI. For these, we should also likely create tests for continuing development even if they are not run in CI.

Other MLFFs, like GAP or Nequip, are more generic architectures and require specific potential files to describe certain chemical spaces. Contributors adding new architectures which require these potential fields should submit minimal potential files (as small as possible to test, accuracy is not important here) to run tests for these.