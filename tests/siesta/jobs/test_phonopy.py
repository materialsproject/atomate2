"""
Tests for phonopy integration jobs.

These tests validate:
- PhonopyMaker workflow creation
- Supercell generation and displacement creation
- Thermal properties calculation
- Force calculation workflow
- Integration with SIESTA makers
- Serialization and edge cases
"""

from jobflow import Flow

from atomate2.siesta.jobs.phonon.phonopy import PhonopyMaker
from atomate2.siesta.jobs.core import StaticMaker, RelaxMaker


class TestPhonopyMaker:
    """Tests for PhonopyMaker workflow."""

    def test_default_phonopy_maker(self):
        """Test creation of default PhonopyMaker."""
        maker = PhonopyMaker()

        assert maker.name == "phonopy"
        assert maker.displacement == 0.01
        assert maker.symprec == 1e-5
        assert maker.relax_maker is None
        assert isinstance(maker.static_maker, StaticMaker)
        assert maker.min_length is None
        assert maker.prefer_90_degrees is True
        assert maker.use_symmetry is True
        assert maker.create_thermal_properties is True
        assert maker.t_step == 10
        assert maker.t_max == 1000
        assert maker.t_min == 0
        assert maker.mesh == (50, 50, 50)

    def test_phonopy_maker_with_custom_params(self):
        """Test PhonopyMaker with custom parameters."""
        maker = PhonopyMaker(
            name="custom_phonon",
            displacement=0.02,
            symprec=1e-4,
            min_length=15.0,
            prefer_90_degrees=False,
            use_symmetry=False,
        )

        assert maker.name == "custom_phonon"
        assert maker.displacement == 0.02
        assert maker.symprec == 1e-4
        assert maker.min_length == 15.0
        assert maker.prefer_90_degrees is False
        assert maker.use_symmetry is False

    def test_phonopy_maker_with_supercell_matrix(self):
        """Test PhonopyMaker with explicit supercell matrix."""
        supercell = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        maker = PhonopyMaker(supercell_matrix=supercell)

        assert maker.supercell_matrix == supercell

    def test_phonopy_maker_with_relax_maker(self):
        """Test PhonopyMaker with initial relaxation."""
        relax_maker = RelaxMaker.fixed_cell_relaxation()
        maker = PhonopyMaker(relax_maker=relax_maker)

        assert maker.relax_maker == relax_maker
        assert isinstance(maker.relax_maker, RelaxMaker)

    def test_phonopy_maker_with_custom_static_maker(self):
        """Test PhonopyMaker with custom static maker."""
        static_maker = StaticMaker()
        maker = PhonopyMaker(static_maker=static_maker)

        assert maker.static_maker == static_maker

    def test_phonopy_maker_thermal_properties_settings(self):
        """Test PhonopyMaker thermal properties parameters."""
        maker = PhonopyMaker(
            create_thermal_properties=True,
            t_step=5,
            t_min=100,
            t_max=500,
        )

        assert maker.create_thermal_properties is True
        assert maker.t_step == 5
        assert maker.t_min == 100
        assert maker.t_max == 500

    def test_phonopy_maker_mesh_setting(self):
        """Test PhonopyMaker q-point mesh parameter."""
        maker = PhonopyMaker(mesh=(100, 100, 100))

        assert maker.mesh == (100, 100, 100)

    def test_phonopy_maker_make_flow(self, si_structure):
        """Test that PhonopyMaker creates a valid flow."""
        maker = PhonopyMaker(
            min_length=10.0,  # Small supercell for testing
        )

        flow = maker.make(si_structure)

        assert isinstance(flow, Flow)
        assert flow.name == "phonopy"
        # Flow should have: generate displacements, forces, analysis
        assert len(flow) >= 3

    def test_phonopy_maker_make_flow_with_relaxation(self, si_structure):
        """Test PhonopyMaker flow with initial relaxation."""
        relax_maker = RelaxMaker.fixed_cell_relaxation()
        maker = PhonopyMaker(
            relax_maker=relax_maker,
            min_length=10.0,
        )

        flow = maker.make(si_structure)

        assert isinstance(flow, Flow)
        # Should have: relax, generate displacements, forces, analysis
        assert len(flow) >= 4

    def test_phonopy_maker_job_naming(self, si_structure):
        """Test that jobs have correct naming."""
        maker = PhonopyMaker(min_length=10.0)
        flow = maker.make(si_structure)

        job_names = [job.name for job in flow]

        # Check for expected job name patterns
        assert any("displacement" in name.lower() for name in job_names)
        assert any("analysis" in name.lower() for name in job_names)

    def test_phonopy_maker_serialization(self):
        """Test PhonopyMaker serialization."""
        maker = PhonopyMaker(
            displacement=0.02,
            min_length=15.0,
            use_symmetry=False,
        )

        # Serialize
        maker_dict = maker.as_dict()
        assert isinstance(maker_dict, dict)
        assert maker_dict["@class"] == "PhonopyMaker"

        # Deserialize
        maker_restored = maker.from_dict(maker_dict)
        assert isinstance(maker_restored, PhonopyMaker)
        assert maker_restored.displacement == 0.02
        assert maker_restored.min_length == 15.0
        assert maker_restored.use_symmetry is False


class TestPhonopySupercellGeneration:
    """Tests for supercell generation behavior."""

    def test_phonopy_maker_auto_supercell(self, si_structure):
        """Test automatic supercell generation."""
        maker = PhonopyMaker(
            min_length=12.0,
            supercell_matrix=None,  # Will be auto-generated
        )

        flow = maker.make(si_structure)
        assert isinstance(flow, Flow)

    def test_phonopy_maker_explicit_supercell(self, si_structure):
        """Test with explicit supercell matrix."""
        supercell = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        maker = PhonopyMaker(supercell_matrix=supercell)

        flow = maker.make(si_structure)
        assert isinstance(flow, Flow)

    def test_phonopy_maker_with_supercell_override(self, si_structure):
        """Test overriding supercell matrix in make() call."""
        maker = PhonopyMaker(supercell_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        # Override with larger supercell
        override_supercell = [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
        flow = maker.make(si_structure, supercell_matrix=override_supercell)

        assert isinstance(flow, Flow)

    def test_phonopy_maker_different_supercell_sizes(self, si_structure):
        """Test different supercell sizes."""
        supercells = [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # 1x1x1
            [[2, 0, 0], [0, 2, 0], [0, 0, 2]],  # 2x2x2
            [[3, 0, 0], [0, 3, 0], [0, 0, 3]],  # 3x3x3
        ]

        for sc in supercells:
            maker = PhonopyMaker(supercell_matrix=sc)
            flow = maker.make(si_structure)
            assert isinstance(flow, Flow)


class TestPhonopyDisplacementSettings:
    """Tests for displacement parameters."""

    def test_phonopy_maker_displacement_values(self, si_structure):
        """Test different displacement values."""
        displacements = [0.005, 0.01, 0.02, 0.03]

        for disp in displacements:
            maker = PhonopyMaker(
                displacement=disp,
                min_length=10.0,
            )
            flow = maker.make(si_structure)
            assert isinstance(flow, Flow)

    def test_phonopy_maker_symmetry_settings(self, si_structure):
        """Test with/without symmetry."""
        for use_sym in [True, False]:
            maker = PhonopyMaker(
                use_symmetry=use_sym,
                min_length=10.0,
            )
            flow = maker.make(si_structure)
            assert isinstance(flow, Flow)

    def test_phonopy_maker_symprec_values(self, si_structure):
        """Test different symmetry precision values."""
        symprecs = [1e-3, 1e-4, 1e-5, 1e-6]

        for symprec in symprecs:
            maker = PhonopyMaker(
                symprec=symprec,
                min_length=10.0,
            )
            flow = maker.make(si_structure)
            assert isinstance(flow, Flow)


class TestPhonopyThermalProperties:
    """Tests for thermal properties calculation."""

    def test_phonopy_maker_with_thermal_properties(self, si_structure):
        """Test with thermal properties enabled."""
        maker = PhonopyMaker(
            create_thermal_properties=True,
            min_length=10.0,
        )

        flow = maker.make(si_structure)
        assert isinstance(flow, Flow)
        assert maker.create_thermal_properties is True

    def test_phonopy_maker_without_thermal_properties(self, si_structure):
        """Test with thermal properties disabled."""
        maker = PhonopyMaker(
            create_thermal_properties=False,
            min_length=10.0,
        )

        flow = maker.make(si_structure)
        assert isinstance(flow, Flow)
        assert maker.create_thermal_properties is False

    def test_phonopy_maker_temperature_range(self, si_structure):
        """Test custom temperature range."""
        maker = PhonopyMaker(
            create_thermal_properties=True,
            t_min=0,
            t_max=1500,
            t_step=20,
            min_length=10.0,
        )

        flow = maker.make(si_structure)
        assert isinstance(flow, Flow)
        assert maker.t_min == 0
        assert maker.t_max == 1500
        assert maker.t_step == 20

    def test_phonopy_maker_different_meshes(self, si_structure):
        """Test different q-point mesh densities."""
        meshes = [
            (20, 20, 20),
            (50, 50, 50),
            (100, 100, 100),
        ]

        for mesh in meshes:
            maker = PhonopyMaker(
                mesh=mesh,
                min_length=10.0,
            )
            flow = maker.make(si_structure)
            assert isinstance(flow, Flow)


class TestPhonopyIntegration:
    """Integration tests for phonopy workflows."""

    def test_all_phonopy_makers_create_valid_flows(self, si_structure):
        """Test that different phonopy configurations create valid flows."""
        makers = [
            PhonopyMaker(min_length=10.0),
            PhonopyMaker(supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]]),
            PhonopyMaker(min_length=10.0, use_symmetry=False),
            PhonopyMaker(min_length=10.0, create_thermal_properties=False),
        ]

        for maker in makers:
            flow = maker.make(si_structure)
            assert isinstance(flow, Flow)
            assert len(flow) > 0

    def test_phonopy_makers_with_different_structures(self, si_structure, al_structure):
        """Test phonopy makers work with different structures."""
        structures = [si_structure, al_structure]
        maker = PhonopyMaker(min_length=10.0)

        for structure in structures:
            flow = maker.make(structure)
            assert isinstance(flow, Flow)

    def test_phonopy_flow_output_references(self, si_structure):
        """Test that flows have proper output handling."""
        maker = PhonopyMaker(min_length=10.0)
        flow = maker.make(si_structure)

        # Flow should be iterable
        jobs = list(flow)
        assert len(jobs) > 0

        # Each job should have a name
        for job in jobs:
            assert hasattr(job, "name")
            assert hasattr(job, "function")

    def test_phonopy_with_relax_and_static_makers(self, si_structure):
        """Test phonopy with custom relax and static makers."""
        relax_maker = RelaxMaker.fixed_cell_relaxation()
        static_maker = StaticMaker()

        maker = PhonopyMaker(
            relax_maker=relax_maker,
            static_maker=static_maker,
            min_length=10.0,
        )

        flow = maker.make(si_structure)
        assert isinstance(flow, Flow)
        assert len(flow) >= 4  # relax + displacements + forces + analysis


class TestPhonopyEdgeCases:
    """Test edge cases and error handling."""

    def test_phonopy_with_none_prev_dir(self, si_structure):
        """Test phonopy with None as prev_dir."""
        maker = PhonopyMaker(min_length=10.0)

        flow = maker.make(si_structure, prev_dir=None)
        assert isinstance(flow, Flow)

    def test_multiple_flows_from_same_maker(self, si_structure, al_structure):
        """Test creating multiple flows from the same maker."""
        maker = PhonopyMaker(min_length=10.0)

        flow1 = maker.make(si_structure)
        flow2 = maker.make(al_structure)

        # Flows should be independent
        assert flow1 is not flow2
        assert isinstance(flow1, Flow)
        assert isinstance(flow2, Flow)

    def test_maker_modification_doesnt_affect_flows(self, si_structure):
        """Test that modifying maker after flow creation doesn't affect flow."""
        maker = PhonopyMaker(displacement=0.01, min_length=10.0)
        flow1 = maker.make(si_structure)

        # Modify maker
        maker.displacement = 0.02

        flow2 = maker.make(si_structure)

        # Flows should be independent
        assert flow1 is not flow2

    def test_phonopy_with_very_small_structure(self):
        """Test phonopy with minimal structure."""
        from pymatgen.core import Structure, Lattice

        # Create minimal structure (single atom cubic)
        lattice = Lattice.cubic(3.0)
        structure = Structure(lattice, ["Si"], [[0.0, 0.0, 0.0]])

        maker = PhonopyMaker(min_length=6.0)  # 2x2x2 supercell
        flow = maker.make(structure)

        assert isinstance(flow, Flow)


class TestPhonopyMakerSerialization:
    """Test serialization of phonopy makers."""

    def test_all_phonopy_makers_serializable(self):
        """Test that all phonopy configurations can be serialized."""
        makers = [
            ("basic", PhonopyMaker()),
            ("custom_displacement", PhonopyMaker(displacement=0.02)),
            (
                "custom_supercell",
                PhonopyMaker(supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]]),
            ),
            ("no_symmetry", PhonopyMaker(use_symmetry=False)),
            (
                "no_thermal",
                PhonopyMaker(create_thermal_properties=False),
            ),
        ]

        for name, maker in makers:
            # Serialize
            maker_dict = maker.as_dict()
            assert isinstance(maker_dict, dict), f"{name} failed to serialize"

            # Deserialize
            maker_restored = maker.from_dict(maker_dict)
            assert maker_restored is not None, f"{name} failed to deserialize"

    def test_phonopy_serialization_preserves_parameters(self):
        """Test that serialization preserves all parameters."""
        maker = PhonopyMaker(
            name="custom_phonon",
            supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            displacement=0.02,
            symprec=1e-4,
            min_length=15.0,
            prefer_90_degrees=False,
            use_symmetry=False,
            create_thermal_properties=True,
            t_step=5,
            t_max=800,
            t_min=50,
            mesh=(80, 80, 80),
        )

        # Serialize and deserialize
        maker_dict = maker.as_dict()
        maker_restored = maker.from_dict(maker_dict)

        # Check all parameters preserved
        assert maker_restored.name == "custom_phonon"
        assert maker_restored.supercell_matrix == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        assert maker_restored.displacement == 0.02
        assert maker_restored.symprec == 1e-4
        assert maker_restored.min_length == 15.0
        assert maker_restored.prefer_90_degrees is False
        assert maker_restored.use_symmetry is False
        assert maker_restored.create_thermal_properties is True
        assert maker_restored.t_step == 5
        assert maker_restored.t_max == 800
        assert maker_restored.t_min == 50
        # Tuples may convert to lists during serialization
        assert list(maker_restored.mesh) == [80, 80, 80]


class TestPhonopyFlowComposition:
    """Test flow composition for phonopy workflows."""

    def test_phonopy_flow_structure(self, si_structure):
        """Test the structure of phonopy flows."""
        maker = PhonopyMaker(min_length=10.0)
        flow = maker.make(si_structure)

        jobs = list(flow)

        # Should have displacement generation, forces, and analysis
        assert len(jobs) >= 3

        # Check job names
        job_names = [job.name for job in jobs]
        assert any("displacement" in name.lower() for name in job_names)
        assert any("analysis" in name.lower() for name in job_names)

    def test_phonopy_with_custom_name(self, si_structure):
        """Test creating phonopy flows with custom names."""
        maker = PhonopyMaker(
            name="My Phonon Calculation",
            min_length=10.0,
        )

        flow = maker.make(si_structure)
        assert flow.name == "My Phonon Calculation"

    def test_phonopy_flow_with_relax(self, si_structure):
        """Test phonopy flow structure with relaxation."""
        relax_maker = RelaxMaker.fixed_cell_relaxation()
        maker = PhonopyMaker(
            relax_maker=relax_maker,
            min_length=10.0,
        )

        flow = maker.make(si_structure)
        jobs = list(flow)

        # Should have relax + displacement + forces + analysis
        assert len(jobs) >= 4

        # First job should be relaxation
        assert "relax" in jobs[0].name.lower()


class TestPhonopyParameterValidation:
    """Test parameter validation for phonopy makers."""

    def test_valid_displacement_values(self, si_structure):
        """Test that valid displacement values are accepted."""
        valid_displacements = [0.005, 0.01, 0.015, 0.02, 0.03]

        for disp in valid_displacements:
            maker = PhonopyMaker(displacement=disp, min_length=10.0)
            flow = maker.make(si_structure)
            assert isinstance(flow, Flow)

    def test_valid_temperature_ranges(self, si_structure):
        """Test different temperature ranges."""
        temp_ranges = [
            (0, 500, 10),
            (0, 1000, 10),
            (100, 1500, 20),
            (0, 2000, 50),
        ]

        for t_min, t_max, t_step in temp_ranges:
            maker = PhonopyMaker(
                t_min=t_min,
                t_max=t_max,
                t_step=t_step,
                min_length=10.0,
            )
            flow = maker.make(si_structure)
            assert isinstance(flow, Flow)

    def test_valid_symprec_values(self, si_structure):
        """Test different symmetry precision values."""
        symprecs = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

        for symprec in symprecs:
            maker = PhonopyMaker(symprec=symprec, min_length=10.0)
            flow = maker.make(si_structure)
            assert isinstance(flow, Flow)


class TestGeneratePhononDisplacements:
    """Tests for generate_phonon_displacements job function using .original pattern."""

    def test_generate_displacements_basic(self, si_structure):
        """Test basic displacement generation."""
        from atomate2.siesta.jobs.phonon.phonopy import generate_phonon_displacements

        result = generate_phonon_displacements.original(
            structure=si_structure,
            supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            displacement=0.01,
            symprec=1e-5,
            use_symmetry=True,
        )

        # Check output structure
        assert isinstance(result, dict)
        assert "displaced_structures" in result
        assert "supercell_matrix" in result
        assert "phonopy_settings" in result

        # Check displaced structures
        assert isinstance(result["displaced_structures"], list)
        assert len(result["displaced_structures"]) > 0

        # Each should be a Structure
        from pymatgen.core import Structure

        for struct in result["displaced_structures"]:
            assert isinstance(struct, Structure)

    def test_generate_displacements_with_symmetry(self, si_structure):
        """Test that symmetry reduces number of displacements."""
        from atomate2.siesta.jobs.phonon.phonopy import generate_phonon_displacements

        # With symmetry
        result_sym = generate_phonon_displacements.original(
            structure=si_structure,
            supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            displacement=0.01,
            use_symmetry=True,
        )

        # Without symmetry
        result_no_sym = generate_phonon_displacements.original(
            structure=si_structure,
            supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            displacement=0.01,
            use_symmetry=False,
        )

        # Symmetry should reduce displacements
        n_sym = len(result_sym["displaced_structures"])
        n_no_sym = len(result_no_sym["displaced_structures"])

        assert n_sym > 0
        assert n_no_sym > 0
        assert n_sym <= n_no_sym  # Symmetry reduces or keeps same

    def test_generate_displacements_auto_supercell(self, si_structure):
        """Test automatic supercell generation from min_length."""
        from atomate2.siesta.jobs.phonon.phonopy import generate_phonon_displacements

        result = generate_phonon_displacements.original(
            structure=si_structure,
            supercell_matrix=None,  # Auto-generate
            min_length=10.0,
            displacement=0.01,
        )

        # Should generate supercell matrix automatically
        assert "supercell_matrix" in result
        assert result["supercell_matrix"] is not None
        assert len(result["displaced_structures"]) > 0

    def test_generate_displacements_different_displacement_values(self, si_structure):
        """Test different displacement values."""
        from atomate2.siesta.jobs.phonon.phonopy import generate_phonon_displacements

        displacements = [0.005, 0.01, 0.02, 0.03]

        for disp in displacements:
            result = generate_phonon_displacements.original(
                structure=si_structure,
                supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
                displacement=disp,
            )

            assert len(result["displaced_structures"]) > 0
            assert result["phonopy_settings"]["displacement"] == disp

    def test_generate_displacements_phonopy_settings(self, si_structure):
        """Test that phonopy settings are correctly stored."""
        from atomate2.siesta.jobs.phonon.phonopy import generate_phonon_displacements

        supercell = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        displacement = 0.015
        symprec = 1e-4
        use_symmetry = False

        result = generate_phonon_displacements.original(
            structure=si_structure,
            supercell_matrix=supercell,
            displacement=displacement,
            symprec=symprec,
            use_symmetry=use_symmetry,
        )

        settings = result["phonopy_settings"]
        assert settings["supercell_matrix"] == supercell
        assert settings["displacement"] == displacement
        assert settings["symprec"] == symprec
        assert settings["use_symmetry"] == use_symmetry

    def test_generate_displacements_supercell_size(self, si_structure):
        """Test different supercell sizes."""
        from atomate2.siesta.jobs.phonon.phonopy import generate_phonon_displacements

        supercells = [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
        ]

        for sc in supercells:
            result = generate_phonon_displacements.original(
                structure=si_structure,
                supercell_matrix=sc,
                displacement=0.01,
            )

            # Should generate displacements for all supercell sizes
            assert len(result["displaced_structures"]) > 0
            assert result["supercell_matrix"] == sc

    def test_generate_displacements_structure_properties(self, si_structure):
        """Test that displaced structures have correct properties."""
        from atomate2.siesta.jobs.phonon.phonopy import generate_phonon_displacements

        result = generate_phonon_displacements.original(
            structure=si_structure,
            supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            displacement=0.01,
        )

        # Original structure should have 2 Si atoms (diamond cubic)
        # 2x2x2 supercell should have 2 * 2^3 = 16 Si atoms
        expected_atoms = len(si_structure) * 8  # 2x2x2 = 8

        for struct in result["displaced_structures"]:
            # Each displaced structure should be a supercell
            assert len(struct) == expected_atoms
            # Should maintain same composition (all Si)
            assert (
                struct.composition.reduced_formula
                == si_structure.composition.reduced_formula
            )

    def test_generate_displacements_different_structures(
        self, si_structure, al_structure
    ):
        """Test displacement generation for different structures."""
        from atomate2.siesta.jobs.phonon.phonopy import generate_phonon_displacements

        structures = [si_structure, al_structure]

        for struct in structures:
            result = generate_phonon_displacements.original(
                structure=struct,
                supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
                displacement=0.01,
            )

            assert len(result["displaced_structures"]) > 0
            assert "phonopy_settings" in result

    def test_generate_displacements_prefer_90_degrees(self, si_structure):
        """Test prefer_90_degrees parameter."""
        from atomate2.siesta.jobs.phonon.phonopy import generate_phonon_displacements

        # Test both True and False
        for prefer_90 in [True, False]:
            result = generate_phonon_displacements.original(
                structure=si_structure,
                supercell_matrix=None,  # Auto-generate
                min_length=10.0,
                prefer_90_degrees=prefer_90,
            )

            assert len(result["displaced_structures"]) > 0

    def test_generate_displacements_minimal_structure(self):
        """Test with minimal single-atom structure."""
        from pymatgen.core import Structure, Lattice
        from atomate2.siesta.jobs.phonon.phonopy import generate_phonon_displacements

        # Create minimal structure (single atom cubic)
        lattice = Lattice.cubic(3.0)
        structure = Structure(lattice, ["Si"], [[0.0, 0.0, 0.0]])

        result = generate_phonon_displacements.original(
            structure=structure,
            supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            displacement=0.01,
        )

        # Should work even with single atom
        assert len(result["displaced_structures"]) > 0

    def test_generate_displacements_different_symprec(self, si_structure):
        """Test different symmetry precision values."""
        from atomate2.siesta.jobs.phonon.phonopy import generate_phonon_displacements

        symprecs = [1e-3, 1e-4, 1e-5, 1e-6]

        for symprec in symprecs:
            result = generate_phonon_displacements.original(
                structure=si_structure,
                supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
                displacement=0.01,
                symprec=symprec,
                use_symmetry=True,
            )

            assert len(result["displaced_structures"]) > 0
            assert result["phonopy_settings"]["symprec"] == symprec

    def test_generate_displacements_non_cubic_supercell(self, si_structure):
        """Test with non-cubic supercell matrices."""
        from atomate2.siesta.jobs.phonon.phonopy import generate_phonon_displacements

        non_cubic_supercells = [
            [[2, 0, 0], [0, 2, 0], [0, 0, 1]],  # Different z
            [[1, 0, 0], [0, 2, 0], [0, 0, 2]],  # Different x
            [[3, 0, 0], [0, 1, 0], [0, 0, 1]],  # Elongated x
        ]

        for sc in non_cubic_supercells:
            result = generate_phonon_displacements.original(
                structure=si_structure,
                supercell_matrix=sc,
                displacement=0.01,
            )

            assert len(result["displaced_structures"]) > 0

    def test_generate_displacements_return_value_types(self, si_structure):
        """Test that return values have correct types."""
        from atomate2.siesta.jobs.phonon.phonopy import generate_phonon_displacements
        from pymatgen.core import Structure

        result = generate_phonon_displacements.original(
            structure=si_structure,
            supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            displacement=0.01,
        )

        # Check types
        assert isinstance(result, dict)
        assert isinstance(result["displaced_structures"], list)
        assert isinstance(result["supercell_matrix"], list)
        assert isinstance(result["phonopy_settings"], dict)

        # Check nested types
        for struct in result["displaced_structures"]:
            assert isinstance(struct, Structure)

        for row in result["supercell_matrix"]:
            assert isinstance(row, list)

    def test_generate_displacements_count_consistency(self, si_structure):
        """Test that same parameters give same number of displacements."""
        from atomate2.siesta.jobs.phonon.phonopy import generate_phonon_displacements

        # Run twice with same parameters
        result1 = generate_phonon_displacements.original(
            structure=si_structure,
            supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            displacement=0.01,
            symprec=1e-5,
            use_symmetry=True,
        )

        result2 = generate_phonon_displacements.original(
            structure=si_structure,
            supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            displacement=0.01,
            symprec=1e-5,
            use_symmetry=True,
        )

        # Should generate same number of displacements
        assert len(result1["displaced_structures"]) == len(
            result2["displaced_structures"]
        )
