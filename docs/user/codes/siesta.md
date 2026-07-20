(codes.siesta)=

# SIESTA

Atomate2 provides a comprehensive set of workflows for the
[SIESTA](https://siesta-project.org) density functional theory code. SIESTA uses a
basis of numerical atomic orbitals, which makes it efficient for large systems and a
natural fit for high-throughput materials simulations.

The SIESTA integration covers structural relaxation, band structure and density of
states, equation of state, elastic constants, phonons (with phonopy), nudged elastic
band, surface energies and adsorption, point defects, and electrocatalysis, together
with a tier-based input system, a recipe book of one-line workflows, and a custodian
error-handling layer.

## Configuration

These workflows require SIESTA to be installed and available on your path. All settings
that control SIESTA execution can be set using the `~/.atomate2.yaml` configuration file
or using environment variables (with the `ATOMATE2_` prefix). For more details on
configuring atomate2, see the [Installation page](installation).

The most important settings to consider are:

- `SIESTA_CMD`: The command used to run SIESTA, e.g.
  `mpirun -n 16 siesta < input.fdf > siesta.out`.
- `SIESTA_PP_PATH`: The directory containing the pseudopotential files (`.psf` / `.psml`)
  used to write SIESTA input.
- `FLOS_PATH`: The path to the [flos](https://github.com/siesta-project/flos) Lua
  library, required by the Lua-driven workflows (NEB and Lua relaxation).
- `VIBRA_CMD`, `OPTICAL_INPUT_CMD`, `OPTICAL_CMD`: Commands for the auxiliary SIESTA
  utilities used by the vibrational and optical workflows.
- `SIESTA_ZIP_FILES`: Whether to gzip large output files after a calculation completes.
- `SIESTA_SHOW_BANNER`, `SIESTA_SHOW_DOCSTRINGS`, `SIESTA_SHOW_PARAMETER_EVOLUTION`:
  Display options controlling the Rich console output (welcome banner, FlowMaker
  docstring panels, and parameter-evolution tables).

```{note}
The standalone `atomate2siesta` distribution reads the same settings from a
`~/.atomate2siesta.yaml` file. When SIESTA is used through atomate2, the settings live
in the standard `~/.atomate2.yaml` alongside the other code settings.
```

## Documentation

The pages below document the SIESTA workflows in detail, from a first calculation
through to advanced, publication-quality studies.

```{toctree}
:caption: Getting started
:maxdepth: 1
/siesta/introduction
/siesta/installation
/siesta/usage
/siesta/fdf-parameters
/siesta/schemas
/siesta/troubleshooting
```

```{toctree}
:caption: Command-line tools
:maxdepth: 1
/siesta/cli-tools
/siesta/cli-database
/siesta/cli-cluster-setup
/siesta/cli-jobflow-remote
/siesta/jobflow-remote-rerun-failed-jobs
/siesta/siesta-pseudos
/siesta/siesta-inputs
```

```{toctree}
:caption: Key features
:maxdepth: 1
/siesta/features
/siesta/makers-vs-flowmakers
/siesta/recipe-book
/siesta/tier-system
/siesta/tier-system-clarification
/siesta/tier-defaults-explained
/siesta/module-registry-explained
/siesta/custodian
/siesta/advanced-workflows
/siesta/defaults
```

```{toctree}
:caption: Tutorials
:maxdepth: 1
/siesta/tutorials/index
/siesta/tutorials-index
/siesta/tutorials-md/QUICKSTART
/siesta/tutorials-md/README
```

```{toctree}
:caption: Reference
:maxdepth: 1
/siesta/api/modules
/siesta/contributing
/siesta/changelog
```
