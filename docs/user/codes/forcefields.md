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

Support is provided for the following models, which can be selected using `atomate2.forcefields.utils.MLFF`, as shown in the table below (in alphabetical order):
**You need only install packages for the forcefields you wish to use.**

| Forcefield Name | `MLFF` | Reference | Description |
| ---- | ---- | ---- | ---- |
| Allegro | `Allegro` | [10.1038/s41467-023-36329-y](https://doi.org/10.1038/s41467-023-36329-y) | Requires the `nequip-allegro` package |
| CHGNet | `CHGNet` | [10.1038/s42256-023-00716-3](https://doi.org/10.1038/s42256-023-00716-3) | Available via the `chgnet` and `matgl` packages |
| DeepMD | `DeepMD` | [10.1103/PhysRevB.108.L180104](https://doi.org/10.1103/PhysRevB.108.L180104) | The Deep Potential model used for this test is `UniPero`, a universal interatomic potential for perovskite oxides. It can be downloaded [here](https://github.com/sliutheorygroup/UniPero) |
| FAIRChem | `FAIRChem` | [Meta's FAIRChem Github](https://github.com/facebookresearch/fairchem) | Proprietary, requires extra authentication. See notes below. |
| Gaussian Approximation Potential (GAP) | `GAP` | [10.1103/PhysRevLett.104.136403](https://doi.org/10.1103/PhysRevLett.104.136403) |  Relies on `quippy-ase` package |
| M3GNet | `M3GNet` | [10.1038/s43588-022-00349-3](https://doi.org/10.1038/s43588-022-00349-3) | Relies on `matgl` package |
| MACE-MP-0 | `MACE` or `MACE_MP_0` (recommended) | [10.1063/5.0297006](https://doi.org/10.1063/5.0297006) | Relies on `mace_torch` and optionally `torch_dftd` packages |
| MACE-MP-0b3 | `MACE_MP_0B3` | [10.1063/5.0297006](https://doi.org/10.1063/5.0297006) | Relies on `mace_torch` and optionally `torch_dftd` packages |
| MACE-MPA-0 | `MACE_MPA_0` | [10.1063/5.0297006](https://doi.org/10.1063/5.0297006) | Relies on `mace_torch` and optionally `torch_dftd` packages |
| MatPES-PBE | `MATPES_PBE` | [10.48550/arXiv.2503.04070](https://doi.org/10.48550/arXiv.2503.04070) | Relies on `matgl`. Defaults to TensorNet architecture, but can also use M3GNet or CHGNet architectures via kwargs. See `atomate2.forcefields.utils._DEFAULT_CALCULATOR_KWARGS` for more options. |
| MatPES-r<sup>2</sup>SCAN | `MATPES_R2SCAN`| [10.48550/arXiv.2503.04070](https://doi.org/10.48550/arXiv.2503.04070) | Relies on `matgl`. Defaults to TensorNet architecture, but can also use M3GNet or CHGNet architectures via kwargs. See `atomate2.forcefields.utils._DEFAULT_CALCULATOR_KWARGS` for more options. |
| MatterSim | `MatterSim` | [arXiv:2405.04967](https://arxiv.org/abs/2405.04967) | Requires the `mattersim` package |
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

## Notes on FairChem (Meta) models {#fairchem-notes}

The FAIRChem models provided by Meta require extra authentication via HuggingFace:
1. Request access to the UMA models [via HuggingFace](https://huggingface.co/facebook/UMA). You will need to set up a HuggingFace account. You will need to receive approval for the UMA models to proceed.
2. Install the HuggingFace CLI with `pip install 'huggingface_hub'`.
3. Run `huggingface-cli login` from a shell to authenticate your session. You will need to set up an access token.
4. You can now use the FAIRChem calculators. The general syntax for setting up a FAIRChem calculator in `atomate2` is:
```py
calculator_kwargs = {
    "predict_unit": {"model_name": "uma-s-1p1"},
    "task_name": "omat",
}
```

`atomate2` will then set up a `FAIRChemCalculator`:
```py
from atomate2.forcefields.utils import MLFF, _DEFAULT_CALCULATOR_KWARGS
from fairchem.core import FAIRChemCalculator, pretrained_mlip

predict_unit_kwargs = calculator_kwargs.pop(
    "predict_unit", _DEFAULT_CALCULATOR_KWARGS[MLFF.FAIRChem]["predict_unit"]
)
calculator = FAIRChemCalculator(
    pretrained_mlip.get_predict_unit(predict_unit_kwargs),
    **{k: v for k, v in calculator_kwargs.items() if k != "predict_unit"},
)
```

The default in `atomate2` is the OMat24 model with `uma-s-1p1`.
