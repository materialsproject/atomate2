"""NEB (Nudged Elastic Band) workflow implementations.

This package provides three types of NEB workflows:

1. **AseNebFlowMaker**: Full ASE-based NEB with Python optimization
   - Uses optimizers (PER_IMAGE_LBFGS/FIRE/BFGS) with SIESTA as calculator
   - Persistent folder approach (custodian-style)
   - Default: PER_IMAGE_LBFGS (each image has its own L-BFGS, like Lua FLOS)
   - Best for: Complex reaction paths, climbing image NEB

2. **NebDirectFlowMaker**: FLOS/Lua-based NEB from two structures
   - Uses SIESTA's Lua scripting (neb.lua)
   - Simpler setup, runs entirely within SIESTA
   - Best for: Quick NEB calculations, SIESTA-native workflows

3. **NebVacancyExchangeFlowMaker**: Specialized for atom swapping
   - Generates initial/final by swapping two atoms
   - Uses FLOS/Lua NEB
   - Best for: Vacancy diffusion, atom exchange barriers

Available Optimizers (AseNebFlowMaker)
--------------------------------------
**Per-image optimizers** (recommended - like Lua FLOS):

- **PER_IMAGE_LBFGS** (default): Each NEB image has its own L-BFGS optimizer
  with limited memory (~20 steps). Best balance of convergence and memory.

- **PER_IMAGE_BFGS**: Each NEB image has its own full BFGS optimizer with
  complete inverse Hessian. May converge faster for small systems but uses
  more memory (O(n²) per image).

**Global optimizers** (treat all images together):

- **FIRE**: No Hessian, handles oscillations well. Good fallback option.

- **LBFGS/BFGS**: Global optimizer for all images combined. Can oscillate
  due to mixing curvature information from different regions.

Example
-------
>>> from atomate2.siesta.flows.neb import AseNebFlowMaker
>>> from pymatgen.core import Structure
>>>
>>> initial = Structure.from_file("initial.cif")
>>> final = Structure.from_file("final.cif")
>>>
>>> # Default: PER_IMAGE_LBFGS (recommended)
>>> maker = AseNebFlowMaker(number_of_images=7, fmax=0.05)
>>> flow = maker.make(initial_structure=initial, final_structure=final)
>>>
>>> # Alternative: FIRE optimizer
>>> maker = AseNebFlowMaker(number_of_images=7, optimizer="FIRE", fmax=0.05)
>>> flow = maker.make(initial_structure=initial, final_structure=final)
"""

# Import from split modules
from atomate2.siesta.flows.neb.ase_neb import (
    AseNebFlowMaker,
    PerImageBFGS,
    PerImageLBFGS,
)
from atomate2.siesta.flows.neb.direct import NebDirectFlowMaker
from atomate2.siesta.flows.neb.vacancy_exchange import NebVacancyExchangeFlowMaker

__all__ = [
    "AseNebFlowMaker",
    "NebDirectFlowMaker",
    "NebVacancyExchangeFlowMaker",
    "PerImageBFGS",
    "PerImageLBFGS",
]
