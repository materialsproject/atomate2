import pytest
from jobflow import Flow, OutputReference
from pymatgen.core import Lattice, Structure

from atomate2.forcefields.flows.finite_temperature_phonons import (
    MLFFMDVaspStaticFiniteTemperaturePhononMaker,
    VaspMDMLFFStaticFiniteTemperaturePhononMaker,
)
from atomate2.forcefields.jobs import ForceFieldStaticMaker
from atomate2.forcefields.md import ForceFieldMDMaker
from atomate2.vasp.flows.finite_temperature_phonons import FiniteTemperaturePhononMaker
from atomate2.vasp.jobs.core import StaticMaker
from atomate2.vasp.jobs.md import MDMaker
from atomate2.vasp.sets.core import LangevinMDSetGenerator, MDSetGenerator


@pytest.fixture
def nio_supercell():
    """A 3x3x3 supercell of rock-salt NiO, which gets +U in VASP."""
    structure = Structure(
        Lattice.cubic(4.17), ["Ni", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    return structure * (3, 3, 3)


@pytest.mark.parametrize(
    ("maker_cls", "md_code", "code"),
    [
        (FiniteTemperaturePhononMaker, "vasp", "vasp"),
        (VaspMDMLFFStaticFiniteTemperaturePhononMaker, "vasp", "forcefields"),
        (MLFFMDVaspStaticFiniteTemperaturePhononMaker, "forcefields", "vasp"),
    ],
)
def test_vasp_flows(si_structure, maker_cls, md_code, code):
    maker = maker_cls(md_runs=2)
    assert (maker.md_code, maker.code) == (md_code, code)
    flow = maker.make(si_structure)
    names = [job.name for job in flow.jobs]
    # the Born charges are computed when the statics use VASP, as in the
    # pheasy phonon workflows
    born_names = ["dielectric"] if code == "vasp" else []
    md_names = ["molecular dynamics" if md_code == "vasp" else "ASE MD"] * 2
    assert names == [
        "double relax",
        "get_supercell_size",
        *born_names,
        "get_md_supercell",
        f"{md_names[0]} 1/2",
        "get_md_restart_structure",
        f"{md_names[1]} 2/2",
        "select_md_snapshots",
        "run_phonon_displacements",
        "fit_finite_temperature_phonons",
    ]
    relax = flow.jobs[0]
    reference, md_1, restart, md_2, snapshots, statics, fit = flow.jobs[-7:]
    assert isinstance(relax, Flow)

    # a diagonal supercell
    supercell = flow.jobs[1]
    assert supercell.function_args[1:] == (12.0, None)
    assert supercell.function_kwargs["force_diagonal"]

    # the first MD starts from the undisplaced supercell, the second from the
    # end of the first, and both get the relaxation directory
    assert md_1.function_args[0].uuid == reference.uuid
    assert md_2.function_args[0].uuid == restart.uuid
    assert restart.function_args == (md_1.output.dir_name, md_code, reference.output)
    for md_job in (md_1, md_2):
        assert md_job.function_kwargs["prev_dir"].uuid == relax.output.uuid
    assert maker.get_md_steps() == [4000, 4000]

    args = snapshots.function_args
    assert [ref.uuid for ref in args[0]] == [md_1.uuid, md_2.uuid]
    assert args[1:] == (md_code, reference.output, 300.0, 1.0, 1.0, 50)

    # the statics get the snapshots followed by the undisplaced supercell, and
    # only VASP statics get the relaxation directory
    static_kwargs = statics.function_kwargs
    assert static_kwargs["displacements"].uuid == snapshots.uuid
    assert static_kwargs["prev_dir"].uuid == relax.output.uuid
    assert static_kwargs["prev_dir_argname"] == ("prev_dir" if code == "vasp" else None)
    assert static_kwargs["phonon_maker"] == maker.phonon_displacement_maker

    fit_kwargs = fit.function_kwargs
    assert fit_kwargs["md_code"] == md_code
    assert fit_kwargs["code"] == code
    assert fit_kwargs["md_uuids"] == [md_1.uuid, md_2.uuid]
    assert fit_kwargs["displacement_data"].uuid == statics.uuid
    assert fit_kwargs["optimization_run_uuid"] == relax.output.uuid
    assert fit_kwargs["rotational_sum_rule"] == "BHH"
    assert fit_kwargs["alpha_min"] == -6
    if code == "forcefields":
        assert fit_kwargs["force_field_name"] == "MACE-MP-0"
        assert fit_kwargs["force_field_kwargs"] == {"model": "medium"}
        assert fit_kwargs["md_force_field_name"] is None
    else:
        assert fit_kwargs["force_field_name"] is None
    if md_code == "forcefields":
        assert fit_kwargs["md_force_field_name"] == "MACE-MP-0"
    if code == "vasp":
        born = flow.jobs[2]
        assert fit_kwargs["born"].uuid == born.uuid
        assert fit_kwargs["epsilon_static"].uuid == born.uuid
        assert fit_kwargs["born_run_uuid"] == born.uuid
        assert born.function_kwargs["prev_dir"].uuid == relax.output.uuid
    else:
        assert fit_kwargs["born"] is None
        assert fit_kwargs["born_run_uuid"] is None
    assert fit.output.uuid == flow.output.uuid


def test_flow_from_output_reference():
    """The structure can be the output of an earlier job."""
    structure = OutputReference("1234", attributes=(("a", "structure"),))
    flow = FiniteTemperaturePhononMaker().make(structure)
    assert flow.jobs[0].name == "double relax"
    assert [job.name for job in flow.jobs][4] == "molecular dynamics"


def test_non_diagonal_supercell_matrix(si_structure):
    with pytest.raises(ValueError, match="diagonal supercell matrix"):
        FiniteTemperaturePhononMaker().make(
            si_structure, supercell_matrix=[[1, 1, 0], [0, 1, 0], [0, 0, 1]]
        )


def test_relax_md_and_static_settings_match(nio_supercell, test_dir):
    """The relaxation, the MD and the statics must see the same Hamiltonian."""
    maker = FiniteTemperaturePhononMaker(md_time=4.0, md_time_step=0.5)
    md_generator = maker.get_md_maker(maker.get_md_steps()[0]).input_set_generator
    static_generator = maker.phonon_displacement_maker.input_set_generator
    relax_generator = maker.bulk_relax_maker.relax_maker1.input_set_generator
    # alternating magnetic moments on Ni, as carried over from a relaxation
    magmoms = [2.0, -2.0] * 13 + [2.0] + [0.0] * 27
    magnetic = nio_supercell.copy(site_properties={"magmom": magmoms})
    sets = {
        "md": md_generator.get_input_set(magnetic, potcar_spec=True),
        "static": static_generator.get_input_set(magnetic, potcar_spec=True),
        "relax": relax_generator.get_input_set(magnetic, potcar_spec=True),
    }
    for key in ("GGA", "ISPIN", "MAGMOM", "LDAU", "LDAUU", "LDAUL", "ISMEAR", "SIGMA"):
        values = {name: input_set.incar.get(key) for name, input_set in sets.items()}
        assert values["md"] == values["static"] == values["relax"], key
    assert sets["md"].incar["GGA"] == "Ps"
    assert sets["md"].incar["LDAUU"] == [6.2, 0]
    assert sets["md"].incar["MAGMOM"] == magmoms
    assert sets["md"].kpoints.kpts == sets["static"].kpoints.kpts
    assert sets["relax"].kpoints.kpts == sets["static"].kpoints.kpts
    for key in ("ENCUT", "ENAUG", "PREC"):
        assert sets["relax"].incar[key] == sets["static"].incar[key], key

    # 4 ps at 0.5 fs, every step written to XDATCAR
    md_incar, static_incar = sets["md"].incar, sets["static"].incar
    assert md_incar["NSW"] == 8000
    assert md_incar["POTIM"] == 0.5
    assert md_incar["NBLOCK"] == 1
    assert md_incar["TEBEG"] == md_incar["TEEND"] == 300
    assert (md_incar["IBRION"], md_incar["ISIF"]) == (0, 2)
    assert (md_incar["MDALGO"], md_incar["SMASS"]) == (2, 0)
    assert md_incar["ENCUT"] == 500
    assert static_incar["ENCUT"] == 600
    assert static_incar["EDIFF"] == 1e-7
    assert static_incar["NSW"] == 0

    # the same holds after a previous calculation sets the spin
    si = Structure.from_file(test_dir / "structures" / "Si.cif") * (3, 3, 3)
    prev_dir = test_dir / "vasp" / "Si_pheasy" / "tight_relax_2" / "outputs"
    prev_sets = {
        name: generator.get_input_set(si, prev_dir=prev_dir, potcar_spec=True)
        for name, generator in (
            ("md", md_generator),
            ("static", static_generator),
            ("relax", relax_generator),
        )
    }
    for key in ("ISPIN", "MAGMOM", "ISMEAR", "SIGMA"):
        assert prev_sets["md"].incar.get(key) == prev_sets["static"].incar.get(key)
    assert prev_sets["md"].kpoints.kpts == prev_sets["static"].kpoints.kpts
    assert prev_sets["relax"].kpoints.kpts == prev_sets["static"].kpoints.kpts


def test_langevin_thermostat(nio_supercell):
    maker = FiniteTemperaturePhononMaker(thermostat="langevin", temperature=600)
    md_maker = maker.get_md_maker(100)
    assert isinstance(md_maker.input_set_generator, LangevinMDSetGenerator)
    incar = md_maker.input_set_generator.get_input_set(
        nio_supercell, potcar_spec=True
    ).incar
    assert incar["MDALGO"] == 3
    assert incar["LANGEVIN_GAMMA"] == [10.0, 10.0]
    assert incar["TEBEG"] == incar["TEEND"] == 600
    assert incar["ENCUT"] == 500
    # the maker of the flow is not changed
    assert type(maker.md_maker.input_set_generator) is MDSetGenerator

    # a Langevin MD maker needs the Langevin thermostat
    maker = FiniteTemperaturePhononMaker(md_maker=md_maker)
    with pytest.raises(ValueError, match="Set thermostat to 'langevin'"):
        maker.get_md_maker(100)
    maker = FiniteTemperaturePhononMaker(md_maker=md_maker, thermostat="langevin")
    assert maker.get_md_maker(100).input_set_generator.nsteps == 100


def test_md_maker_checks():
    maker = FiniteTemperaturePhononMaker(
        md_maker=MDMaker(
            input_set_generator=MDSetGenerator(user_incar_settings={"POTIM": 2})
        )
    )
    with pytest.raises(ValueError, match=r"The flow sets \['POTIM'\]"):
        maker.get_md_maker(10)

    maker = FiniteTemperaturePhononMaker(
        md_maker=MDMaker(
            input_set_generator=MDSetGenerator(user_incar_settings={"NBLOCK": 10})
        )
    )
    with pytest.raises(ValueError, match="NBLOCK must be 1"):
        maker.get_md_maker(10)

    with pytest.raises(TypeError, match="VASP MDMaker with an MDSetGenerator"):
        FiniteTemperaturePhononMaker(md_maker=StaticMaker()).get_md_maker(10)


def test_from_force_field_name():
    maker = VaspMDMLFFStaticFiniteTemperaturePhononMaker.from_force_field_name(
        "MACE-MP-0",
        calculator_kwargs={"model": "medium-omat-0"},
        temperature=500.0,
        phonon_displacement_maker=StaticMaker(),
    )
    assert maker.name == ("VASP MD MACE_MP_0 Static Finite Temperature Phonon Maker")
    assert maker.temperature == 500.0
    assert isinstance(maker.md_maker, MDMaker)
    # the maker built for the force field replaces the one given
    displacement_maker = maker.phonon_displacement_maker
    assert isinstance(displacement_maker, ForceFieldStaticMaker)
    assert displacement_maker.calculator_kwargs["model"] == "medium-omat-0"
    assert maker.born_maker is None

    maker = MLFFMDVaspStaticFiniteTemperaturePhononMaker.from_force_field_name(
        "MACE-MP-0"
    )
    assert maker.name == "MACE_MP_0 MD VASP Static Finite Temperature Phonon Maker"
    assert isinstance(maker.md_maker, ForceFieldMDMaker)
    assert (maker.md_code, maker.code) == ("forcefields", "vasp")
    assert type(maker.born_maker).__name__ == "DielectricMaker"
