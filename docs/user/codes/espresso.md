(codes.espresso)=

# Quantum ESPRESSO

`atomate2` runs `pw.x` and `ph.x` of [Quantum ESPRESSO](https://www.quantum-espresso.org).
The inputs are written and the outputs are read with the Quantum ESPRESSO I/O of ASE, so install `atomate2` with `pip install 'atomate2[ase]'`.

## Settings

Set these in `~/.atomate2.yaml`:

```yaml
ESPRESSO_PW_CMD: mpirun -n 16 pw.x
ESPRESSO_PH_CMD: mpirun -n 16 ph.x
ESPRESSO_PSEUDO_DIR: /path/to/pseudopotentials
```

`ESPRESSO_PSEUDO_DIR` holds the pseudopotential files and one JSON file per pseudopotential family, in the format of the [SSSP libraries](https://www.materialscloud.org/discover/sssp).
For each element, the JSON file gives the name of the pseudopotential file and the recommended cutoffs in Ry.
For example, the entry of Si in SSSP 1.3.0 PBE efficiency includes:

```json
{"Si": {"filename": "Si.pbe-n-rrkjus_psl.1.0.0.UPF", "cutoff_wfc": 30.0, "cutoff_rho": 240.0}}
```

The default family is `SSSP_1.3.0_PBEsol_efficiency`, so `ESPRESSO_PSEUDO_DIR` needs `SSSP_1.3.0_PBEsol_efficiency.json` and its pseudopotential files.
The calculation uses the largest cutoffs of the elements in the structure.

## Jobs

| Maker | Programs | Output |
| ---- | ---- | ---- |
| `EspressoStaticMaker` | `pw.x` (`scf`) | energy, forces, and stress |
| `EspressoRelaxMaker` | `pw.x` (`vc-relax`) | relaxed structure, energy, forces, and stress |
| `EspressoDielectricMaker` | `pw.x` (`scf`), then `ph.x` at Γ | dielectric tensor and Born effective charges |

All jobs return an `EspressoTaskDoc` with the input namelists and the output in eV, eV/Å, and kbar.
The k-point grid is centered on Γ, with a spacing of at most `kspacing` (0.15 1/Å by default, including the factor of 2π, as for `KSPACING` in VASP).
The relaxation stops when the energy change is below 1e-6 Ry, the forces are below 1e-4 Ry/Bohr, and the pressure is below 0.1 kbar.
A job fails if `pw.x` did not converge the self-consistency or the relaxation, since `pw.x` then still stops with exit code 0.
Sites with a nonzero `magmom` property start a spin-polarized calculation.
`pw.x` does not write the wavefunctions (`disk_io="nowf"`), except in the dielectric job, where `ph.x` needs them.
The dielectric job runs `pw.x` with fixed occupations and without the `magmom` site property, so it needs a nonmagnetic insulator.

Change the `pw.x` settings with `user_input_data`, grouped by namelist:

```python
from atomate2.espresso.jobs import EspressoStaticMaker
from atomate2.espresso.sets import EspressoInputGenerator

maker = EspressoStaticMaker(
    input_set_generator=EspressoInputGenerator(
        kspacing=0.2, user_input_data={"system": {"degauss": 0.02}}
    )
)
```
