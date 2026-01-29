(codes.forcefields)=

# Machine Learning forcefields / interatomic potentials

`atomate2` includes an interface to a few common machine learning interatomic potentials (MLIPs), also known variously as machine learning forcefields (MLFFs), or foundation potentials (FPs) for universal variants.

Most of `Maker` classes using the forcefields inherit from `atomate2.forcefields.utils.ForceFieldMixin` to specify which forcefield to use.
The `ForceFieldMixin` mixin provides the following configurable parameters:

- `force_field_name`: Name of the forcefield to use.
- `calculator_kwargs`: Keyword arguments to pass to the corresponding ASE calculator.

These parameters are passed to `atomate2.forcefields.utils.ase_calculator()` to instantiate the appropriate ASE calculator.

The `force_field_name` should be either one of predefined `atomate2.forcefields.utils.MLFF` (or its string equivalent) or a dictionary decodable as a class or function for ASE calculator as follows.

## Using predefined forcefields supported via `atomate2.forcefields.utils.MLFF`

Support is provided for the following models, which can be selected using `atomate2.forcefields.utils.MLFF`, as shown in the table below.
**You need only install packages for the forcefields you wish to use.**

| Forcefield Name | `MLFF` | Reference | Description |
| ---- | ---- | ---- | ---- |
| CHGNet | `CHGNet` | [10.1038/s42256-023-00716-3](https://doi.org/10.1038/s42256-023-00716-3) | Available via the `chgnet` and `matgl` packages |
| DeepMD | `MLFF.DeepMD` | [10.1103/PhysRevB.108.L180104](https://doi.org/10.1103/PhysRevB.108.L180104) | The Deep Potential model used for this test is `UniPero`, a universal interatomic potential for perovskite oxides. It can be downloaded [here](https://github.com/sliutheorygroup/UniPero) |
| Gaussian Approximation Potential (GAP) | `GAP` | [10.1103/PhysRevLett.104.136403](https://doi.org/10.1103/PhysRevLett.104.136403) |  Relies on `quippy-ase` package |
| M3GNet | `M3GNet` | [10.1038/s43588-022-00349-3](https://doi.org/10.1038/s43588-022-00349-3) | Relies on `matgl` package |
| MACE-MP-0 | `MACE` or `MACE_MP_0` (recommended) | [10.1063/5.0297006](https://doi.org/10.1063/5.0297006) | Relies on `mace_torch` and optionally `torch_dftd` packages |
| MACE-MP-0b3 | `MACE_MP_0B3` | [10.1063/5.0297006](https://doi.org/10.1063/5.0297006) | Relies on `mace_torch` and optionally `torch_dftd` packages |
| MACE-MPA-0 | `MACE_MPA_0` | [10.1063/5.0297006](https://doi.org/10.1063/5.0297006) | Relies on `mace_torch` and optionally `torch_dftd` packages |
| MatPES-PBE | `MATPES_PBE` | [10.48550/arXiv.2503.04070](https://doi.org/10.48550/arXiv.2503.04070) | Relies on `matgl`. Defaults to TensorNet architecture, but can also use M3GNet or CHGNet architectures via kwargs. See `atomate2.forcefields.utils._DEFAULT_CALCULATOR_KWARGS` for more options. |
| MatPES-r<sup>2</sup>SCAN | `MATPES_R2SCAN`| [10.48550/arXiv.2503.04070](https://doi.org/10.48550/arXiv.2503.04070) | Relies on `matgl`. Defaults to TensorNet architecture, but can also use M3GNet or CHGNet architectures via kwargs. See `atomate2.forcefields.utils._DEFAULT_CALCULATOR_KWARGS` for more options. |
| Neuroevolution Potential (NEP) | `NEP` | [10.1103/PhysRevB.104.104309](https://doi.org/10.1103/PhysRevB.104.104309) | Relies on `calorine` package |
| Neural Equivariant Interatomic Potentials (Nequip) | `Nequip` | [10.1038/s41467-022-29939-5](https://doi.org/10.1038/s41467-022-29939-5) | Relies on the `nequip` package |
| SevenNet | `SevenNet` | [10.1021/acs.jctc.4c00190](https://doi.org/10.1021/acs.jctc.4c00190) | Relies on the `sevenn` package |

## Using custom forcefields by dictionary

`force_field_name` also accepts a MSONable dictionary for specifying a custom ASE calculator class or function [^calculator-meta-type-annotation].
For example, a `Job` created with the following code snippet instantiates `chgnet.model.dynamics.CHGNetCalculator` as the ASE calculator:
```python
job = ForceFieldStaticMaker(
    force_field_name={
        "@module": "chgnet.model.dynamics",
        "@callable": "CHGNetCalculator",
    }
).make(structure)
```

[^calculator-meta-type-annotation]: In this context, the type annotation of the decoded dict should be either `Type[Calculator]` or `Callable[..., Calculator]`, where `Calculator` is from `ase.calculators.calculator`.
