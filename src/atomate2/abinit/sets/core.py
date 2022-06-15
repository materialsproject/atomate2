"""Module defining core Abinit input set generators."""

from dataclasses import dataclass, field
from typing import Optional

from abipy.abio.factories import (
    dos_from_gsinput,
    ebands_from_gsinput,
    ion_ioncell_relax_input,
    scf_input,
)
from abipy.abio.input_tags import MOLECULAR_DYNAMICS, NSCF, RELAX, SCF

from atomate2.abinit.files import load_abinit_input
from atomate2.abinit.sets.base import AbinitInputGenerator

__all__ = [
    "StaticSetGenerator",
    "NonSCFSetGenerator",
    "RelaxSetGenerator",
]


@dataclass
class GroundStateSetGenerator(AbinitInputGenerator):
    """Common class for ground-state generators."""

    calc_type: str = "ground_state"

    kppa: Optional[float] = None
    ecut: Optional[float] = None
    pawecutdg: Optional[float] = None
    nband: Optional[int] = None
    accuracy: str = "normal"
    spin_mode: str = "polarized"
    smearing: str = "fermi_dirac:0.1 eV"
    charge: float = 0.0
    scf_algorithm: Optional[str] = None
    shift_mode: str = "Monkhorst-Pack"

    restart_from_deps: tuple = (f"{SCF}|{RELAX}|{MOLECULAR_DYNAMICS}:WFK|DEN",)


@dataclass
class StaticSetGenerator(GroundStateSetGenerator):
    """Class to generate Abinit static input sets."""

    calc_type: str = "static"

    def get_abinit_input(
        self,
        structure=None,
        pseudos=None,
        prev_outputs=None,
        kppa=GroundStateSetGenerator.kppa,
        ecut=GroundStateSetGenerator.ecut,
        pawecutdg=GroundStateSetGenerator.pawecutdg,
        nband=GroundStateSetGenerator.nband,
        accuracy=GroundStateSetGenerator.accuracy,
        spin_mode=GroundStateSetGenerator.spin_mode,
        smearing=GroundStateSetGenerator.smearing,
        charge=GroundStateSetGenerator.charge,
        scf_algorithm=GroundStateSetGenerator.scf_algorithm,
        shift_mode=GroundStateSetGenerator.shift_mode,
    ):
        """Get AbinitInput object for static calculation."""
        if structure is None:
            raise RuntimeError("Structure is mandatory for StaticSet generation.")
        if prev_outputs is not None:
            raise RuntimeError(
                "Previous outputs not allowed for StaticSetGenerator. "
                "To restart from a previous static or otherwise scf "
                "(e.g. relaxation) calculation, use restart_from argument of "
                "get_input_set method instead."
            )

        return scf_input(
            structure=structure,
            pseudos=pseudos,
            kppa=kppa,
            ecut=ecut,
            pawecutdg=pawecutdg,
            nband=nband,
            accuracy=accuracy,
            spin_mode=spin_mode,
            smearing=smearing,
            charge=charge,
            scf_algorithm=scf_algorithm,
            shift_mode=shift_mode,
        )

    def on_restart(self, abinit_input):
        """Perform updates of AbinitInput upon restart.

        In this case, a static calculation can be started from a relaxation one.
        The relaxation-like variables need to be removed from the AbinitInput.
        """
        # Always remove relaxation-like variables so that if we make the SCF job
        # starting from a previous relaxation or molecular dynamics job, the
        # structure will indeed be static.
        abinit_input.pop_vars(["ionmov", "optcell", "ntime"])


@dataclass
class NonSCFSetGenerator(AbinitInputGenerator):
    """Class to generate Abinit non-SCF input sets."""

    calc_type: str = "nscf"

    nband: Optional[int] = None
    accuracy: str = "normal"

    # TODO: how to switch between ndivsm and line_density determination of kpoints ?
    #  One way could be to set the convention in the settings somehow ?
    #  e.g. we could say that by default line_density is used (to do the same as vasp)
    #  Then if the user tries to set ndivsm explicitly (either in the __init__ of the
    #  generator or by setting the attribute after initialization), one should probably
    #  raise an error saying that this parameter is only used when the other convention
    #  is used. And vice versa of course ...
    #  Same holds for reciprocal_density vs kppa.
    ndivsm: int = 15
    # TODO: if we want to use line_density, we should make a "converter" from
    #  line_density to ndivsm (as ndivsm is the standard approach to perform Non Scf
    #  calculations on a line band structure in abinit).
    # line_density: float = 20

    # TODO: how to switch between reciprocal_density and kppa determination of kpoints ?
    kppa: float = 1000
    # TODO: make a converter from kppa to reciprocal_density ?
    #  or just use the vasp KPoints object generated using reciprocal_density
    #  to set up the grid manually in abinit ?
    # reciprocal_density: float = 100

    mode: str = "line"

    restart_from_deps: tuple = (f"{NSCF}:WFK",)
    prev_outputs_deps: tuple = (f"{SCF}:DEN",)

    # class variables
    # params: ClassVar[tuple] = (
    #     "nband",
    #     "ndivsm",
    #     "kppa",
    #     "accuracy",
    # )

    def __post_init__(self):
        """Ensure mode is set correctly."""
        self.mode = self.mode.lower()

        supported_modes = ("line", "uniform")
        if self.mode not in supported_modes:
            raise ValueError(f"Supported modes are: {', '.join(supported_modes)}")

    def get_abinit_input(
        self,
        structure=None,
        pseudos=None,
        prev_outputs=None,
        nband=nband,
        accuracy=accuracy,
        ndivsm=ndivsm,
        kppa=kppa,
        mode=mode,
    ):
        """Get AbinitInput object for Non-SCF calculation."""
        if prev_outputs is None:
            raise RuntimeError(
                "No previous_outputs. Cannot perform non-SCF calculation."
            )
        if len(prev_outputs) != 1:
            raise RuntimeError(
                "Should have exactly one previous output (an SCF calculation)."
            )
        prev_output = prev_outputs[0]
        previous_abinit_input = load_abinit_input(prev_output)
        if structure is not None:
            if structure != previous_abinit_input.structure:
                raise RuntimeError(
                    "Structure is provided in non-SCF input set generator but "
                    "is not the same as the one from the previous (SCF) input set."
                )

        # params = self.get_params(instance_or_class=self, kwargs=kwargs, prev_gen=None)
        if mode == "line":
            return ebands_from_gsinput(
                gs_input=previous_abinit_input,
                nband=nband,
                ndivsm=ndivsm,
                accuracy=accuracy,
            )
        elif mode == "uniform":
            uniform_input = dos_from_gsinput(
                gs_input=previous_abinit_input,
                dos_kppa=kppa,
                nband=nband,
                accuracy=accuracy,
                pdos=False,
            )
            return uniform_input
        else:
            raise RuntimeError(
                f"'{self.mode}' is wrong mode for {self.__class__.__name__}."
            )


@dataclass
class NonScfWfqInputGenerator(AbinitInputGenerator):
    """Input set generator for Non-Scf Wfq calculations."""

    calc_type: str = "nscf_wfq"

    wfq_tol: dict = field(default_factory=lambda: {"tolwfr": 1.0e-18})

    restart_from_deps: tuple = (f"{NSCF}:WFQ",)
    prev_outputs_deps: tuple = (f"{SCF}:DEN",)

    def get_abinit_input(
        self, structure=None, pseudos=None, prev_outputs=None, qpt=None, **kwargs
    ):
        """Get AbinitInput object for Non-SCF Wfq calculation."""
        if qpt is None:
            raise RuntimeError(
                "Should provide q-point at which non-SCF Wfq calculation "
                "has to be done."
            )
        if structure is not None:
            # TODO: maybe just check that the structure is the same as the one
            #  in the previous_input_set ?
            raise RuntimeError(
                "Structure should not be set in a non-SCF Wfq input set. "
                "It should come directly from a previous (SCF) input set."
            )
        if prev_outputs is None:
            raise RuntimeError(
                "No previous_outputs. Cannot perform non-SCF Wfq calculation."
            )
        if len(prev_outputs) != 1:
            raise RuntimeError(
                "Should have exactly one previous output (an SCF calculation)."
            )
        prev_output = prev_outputs[0]
        wfq_input = load_abinit_input(prev_output)
        wfq_input.set_vars(kptopt=3, nqpt=1, iscf=-2, qpt=qpt, **self.wfq_tol)
        return wfq_input


@dataclass
class DdkInputGenerator(AbinitInputGenerator):
    """Input set generator for Non-Scf Wfq calculations."""

    calc_type: str = "ddk"

    def get_abinit_input(
        self, structure=None, pseudos=None, prev_outputs=None, **kwargs
    ):
        raise NotImplementedError()


@dataclass
class RelaxSetGenerator(GroundStateSetGenerator):
    """Class to generate Abinit relaxation input sets."""

    calc_type: str = "relaxation"

    relax_cell: bool = True
    tolmxf: float = 5.0e-5

    # # class variables
    # params: ClassVar[tuple] = (
    #     "kppa",
    #     "ecut",
    #     "pawecutdg",
    #     "nband",
    #     "accuracy",
    #     "spin_mode",
    #     "smearing",
    #     "charge",
    #     "scf_algorithm",
    #     "shift_mode",
    #     "relax_cell",
    #     "tolmxf",
    # )

    def get_abinit_input(
        self,
        structure=None,
        pseudos=None,
        prev_outputs=None,
        kppa=GroundStateSetGenerator.kppa,
        ecut=GroundStateSetGenerator.ecut,
        pawecutdg=GroundStateSetGenerator.pawecutdg,
        nband=GroundStateSetGenerator.nband,
        accuracy=GroundStateSetGenerator.accuracy,
        spin_mode=GroundStateSetGenerator.spin_mode,
        smearing=GroundStateSetGenerator.smearing,
        charge=GroundStateSetGenerator.charge,
        scf_algorithm=GroundStateSetGenerator.scf_algorithm,
        shift_mode=GroundStateSetGenerator.shift_mode,
        relax_cell=relax_cell,
        tolmxf=tolmxf,
        **kwargs,
    ):
        if structure is None:
            raise RuntimeError("Structure is mandatory for RelaxSet generation.")
        if prev_outputs is not None:
            raise RuntimeError(
                "Previous outputs not allowed for RelaxSetGenerator. "
                "To restart from a previous static or otherwise scf "
                "(e.g. relaxation) calculation, use restart_from argument of "
                "get_input_set method instead."
            )
        if kwargs.get("atoms_constraints", None) is not None:
            raise NotImplementedError("Atoms constraints not implemented.")

        ind = 1 if relax_cell else 0
        relax_input = ion_ioncell_relax_input(
            structure,
            pseudos=pseudos,
            kppa=kppa,
            nband=nband,
            ecut=ecut,
            pawecutdg=pawecutdg,
            accuracy=accuracy,
            spin_mode=spin_mode,
            smearing=smearing,
            charge=charge,
            scf_algorithm=scf_algorithm,
            shift_mode=shift_mode,
        )[ind]
        relax_input["tolmxf"] = tolmxf

        return relax_input
