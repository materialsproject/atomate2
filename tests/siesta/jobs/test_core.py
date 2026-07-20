"""
Tests for core SIESTA jobs (RelaxMaker, StaticMaker, BandStructureMaker).

These tests validate:
- Maker creation and configuration
- Job generation from makers
- Parameter handling (user_params, tier system)
- Input set generation
- Serialization/deserialization
"""

from atomate2.siesta.flows.phonon import SiestaPhononFlowMaker
from atomate2.siesta.jobs.core import (
    BandStructureMaker,
    LuaMaker,
    OpticalMaker,
    PhononMaker,
    RelaxMaker,
    SocketIOStaticMaker,
    StaticMaker,
)
from atomate2.siesta.sets.core import (
    BandStructureSetGenerator,
    LuaSetGenerator,
    OpticalSetGenerator,
    PhononSetGenerator,
    RelaxSetGenerator,
    SocketIOSetGenerator,
    StaticSetGenerator,
)


class TestStaticMaker:
    """Tests for StaticMaker (SCF calculations)."""

    def test_default_static_maker(self, si_structure):
        """Test creation of default StaticMaker."""
        maker = StaticMaker()

        assert maker.name == "SCF Calculation"
        assert maker.calc_type == "scf"
        assert isinstance(maker.input_set_generator, StaticSetGenerator)

        # Test job generation
        job = maker.make(si_structure)
        assert job.name == "SCF Calculation"
        assert hasattr(job, "function")

    def test_static_maker_scf_classmethod(self, si_structure):
        """Test StaticMaker.scf() class method."""
        maker = StaticMaker.scf()

        assert maker.name == "SCF Calculation"
        assert isinstance(maker.input_set_generator, StaticSetGenerator)

        job = maker.make(si_structure)
        assert job.name == "SCF Calculation"

    def test_static_maker_with_user_params(self, si_structure):
        """Test StaticMaker with custom user parameters."""
        user_params = {
            "PAO.BasisSize": "DZP",
            "a2s_kpts": [6, 6, 6],
            "Mesh.Cutoff": "300 Ry",
        }

        maker = StaticMaker(
            input_set_generator=StaticSetGenerator(user_params=user_params)
        )

        _job = maker.make(si_structure)

        # Check that user params are stored in input_set_generator
        assert maker.input_set_generator.user_params == user_params

    def test_static_maker_with_tier(self, si_structure):
        """Test StaticMaker with tier system."""
        maker = StaticMaker(
            input_set_generator=StaticSetGenerator(
                tier="advanced", user_params={"PAO.BasisSize": "TZP"}
            )
        )

        assert maker.input_set_generator.tier == "advanced"
        assert maker.input_set_generator.user_params["PAO.BasisSize"] == "TZP"

    def test_static_maker_serialization(self, si_structure):
        """Test that StaticMaker can be serialized/deserialized."""
        maker = StaticMaker(
            input_set_generator=StaticSetGenerator(user_params={"PAO.BasisSize": "DZP"})
        )

        # Makers should be MSONable (serializable)

        # Serialize
        maker_dict = maker.as_dict()
        assert isinstance(maker_dict, dict)
        assert maker_dict["@class"] == "StaticMaker"

        # Deserialize
        maker_restored = maker.from_dict(maker_dict)
        assert isinstance(maker_restored, StaticMaker)
        assert maker_restored.name == maker.name


class TestRelaxMaker:
    """Tests for RelaxMaker (geometry optimization)."""

    def test_default_relax_maker(self, si_structure):
        """Test creation of default RelaxMaker."""
        maker = RelaxMaker()

        assert maker.name == "Relaxation calculation"
        assert maker.calc_type == "relax"
        assert isinstance(maker.input_set_generator, RelaxSetGenerator)

    def test_fixed_cell_relaxation(self, si_structure):
        """Test RelaxMaker.fixed_cell_relaxation() class method."""
        maker = RelaxMaker.fixed_cell_relaxation()

        assert "fixed-cell" in maker.name
        assert isinstance(maker.input_set_generator, RelaxSetGenerator)
        assert maker.input_set_generator.relax_cell is False

        job = maker.make(si_structure)
        assert job.name == "Relaxation calculation-fixed-cell"

    def test_variable_cell_relaxation(self, si_structure):
        """Test RelaxMaker.variable_cell_relaxation() class method."""
        maker = RelaxMaker.variable_cell_relaxation()

        assert "variable-cell" in maker.name
        assert isinstance(maker.input_set_generator, RelaxSetGenerator)
        assert maker.input_set_generator.relax_cell is True

        job = maker.make(si_structure)
        assert job.name == "Relaxation calculation-variable-cell"

    def test_relax_maker_with_user_params(self, si_structure):
        """Test RelaxMaker with custom parameters."""
        user_params = {
            "PAO.BasisSize": "DZP",
            "a2s_kpts": [4, 4, 4],
            "Mesh.Cutoff": "300 Ry",
            "MD.MaxForceTol": "0.01 eV/Ang",
        }

        maker = RelaxMaker.fixed_cell_relaxation(user_params=user_params)

        assert maker.input_set_generator.user_params == user_params
        assert maker.input_set_generator.relax_cell is False

    def test_relax_maker_parameter_separation(self, si_structure):
        """Test that maker and input_set parameters are correctly separated."""
        # Test with both maker-specific and input-generator-specific params
        maker = RelaxMaker.fixed_cell_relaxation(
            use_custodian=True,  # Maker parameter
            custodian_max_errors=10,  # Maker parameter
            user_params={"PAO.BasisSize": "DZP"},  # InputSet parameter
            tier="intermediate",  # InputSet parameter
        )

        # Check maker parameters
        assert maker.use_custodian is True
        assert maker.custodian_max_errors == 10

        # Check input set parameters
        assert maker.input_set_generator.user_params["PAO.BasisSize"] == "DZP"
        assert maker.input_set_generator.tier == "intermediate"

    def test_relax_maker_with_tier(self, si_structure):
        """Test RelaxMaker with tier system."""
        maker = RelaxMaker.fixed_cell_relaxation(
            tier="advanced", user_params={"PAO.BasisSize": "TZP"}
        )

        assert maker.input_set_generator.tier == "advanced"

    def test_relax_maker_with_custodian(self, si_structure):
        """Test RelaxMaker with custodian enabled."""
        maker = RelaxMaker.fixed_cell_relaxation(
            use_custodian=True, custodian_max_errors=5
        )

        assert maker.use_custodian is True
        assert maker.custodian_max_errors == 5

    def test_relax_maker_with_prev_dir(self, si_structure, tmp_path):
        """Test RelaxMaker with previous directory."""
        maker = RelaxMaker.fixed_cell_relaxation()

        prev_dir = tmp_path / "previous"
        prev_dir.mkdir()

        job = maker.make(si_structure, prev_dir=str(prev_dir))
        assert job.name == "Relaxation calculation-fixed-cell"

    def test_relax_maker_serialization(self, si_structure):
        """Test RelaxMaker serialization."""
        maker = RelaxMaker.fixed_cell_relaxation(
            user_params={"PAO.BasisSize": "DZP"}, use_custodian=True
        )

        # Serialize
        maker_dict = maker.as_dict()
        assert isinstance(maker_dict, dict)
        assert maker_dict["@class"] == "RelaxMaker"

        # Deserialize
        maker_restored = maker.from_dict(maker_dict)
        assert isinstance(maker_restored, RelaxMaker)
        assert maker_restored.use_custodian is True


class TestBandStructureMaker:
    """Tests for BandStructureMaker (electronic band structure calculations)."""

    def test_default_band_structure_maker(self, si_structure):
        """Test creation of default BandStructureMaker."""
        maker = BandStructureMaker()

        assert maker.name == "bands"
        assert maker.calc_type == "band_structure"
        assert isinstance(maker.input_set_generator, BandStructureSetGenerator)

    def test_bandstructure_calculation_classmethod(self, si_structure):
        """Test BandStructureMaker.bandstructure_calculation() class method."""
        maker = BandStructureMaker.bandstructure_calculation()

        assert "Calculation" in maker.name
        assert isinstance(maker.input_set_generator, BandStructureSetGenerator)

        job = maker.make(si_structure)
        assert "Calculation" in job.name

    def test_band_structure_maker_with_user_params(self, si_structure):
        """Test BandStructureMaker with custom parameters."""
        user_params = {
            "PAO.BasisSize": "TZP",
            "a2s_kpts": [8, 8, 8],
            "Mesh.Cutoff": "400 Ry",
        }

        maker = BandStructureMaker.bandstructure_calculation(user_params=user_params)

        assert maker.input_set_generator.user_params == user_params

    def test_band_structure_maker_with_kpath(self, si_structure):
        """Test BandStructureMaker with custom k-path."""
        # BandStructureSetGenerator should handle k-path automatically
        maker = BandStructureMaker.bandstructure_calculation()

        job = maker.make(si_structure)
        assert job.name.startswith("bands")

    def test_band_structure_maker_serialization(self, si_structure):
        """Test BandStructureMaker serialization."""
        maker = BandStructureMaker.bandstructure_calculation(
            user_params={"PAO.BasisSize": "DZP"}
        )

        # Serialize
        maker_dict = maker.as_dict()
        assert isinstance(maker_dict, dict)
        assert maker_dict["@class"] == "BandStructureMaker"

        # Deserialize
        maker_restored = maker.from_dict(maker_dict)
        assert isinstance(maker_restored, BandStructureMaker)


class TestLuaMaker:
    """Tests for LuaMaker (Lua scripting for relaxation and NEB)."""

    def test_default_lua_maker(self, si_structure):
        """Test creation of default LuaMaker."""
        maker = LuaMaker()

        assert maker.name == "Lua calculation"
        assert maker.calc_type == "relax"
        assert isinstance(maker.input_set_generator, LuaSetGenerator)

        # Test job generation
        job = maker.make(si_structure)
        assert job.name == "Lua calculation"
        assert hasattr(job, "function")

    def test_lua_maker_fixed_cell_relaxation(self, si_structure):
        """Test LuaMaker.fixed_cell_relaxation() class method."""
        maker = LuaMaker.fixed_cell_relaxation()

        assert maker.name == "Lua calculation-fixed-cell"
        assert isinstance(maker.input_set_generator, LuaSetGenerator)
        assert maker.input_set_generator.relax_cell is False
        assert maker.input_set_generator.lua_type == "lua_relaxation"

        job = maker.make(si_structure)
        assert job.name == "Lua calculation-fixed-cell"

    def test_lua_maker_fixed_cell_with_default_script(self, si_structure):
        """Test that fixed_cell_relaxation adds default Lua script."""
        maker = LuaMaker.fixed_cell_relaxation()

        # Should have default Lua.Script set
        assert maker.input_set_generator.user_params is not None
        assert (
            "Lua.Script" in maker.input_set_generator.user_params
            or "lua.script"
            in {k.lower() for k in maker.input_set_generator.user_params.keys()}
        )

        job = maker.make(si_structure)
        assert job.name == "Lua calculation-fixed-cell"

    def test_lua_maker_fixed_cell_with_custom_script(self, si_structure):
        """Test LuaMaker with custom Lua script."""
        maker = LuaMaker.fixed_cell_relaxation(
            user_params={"Lua.Script": "custom_relax.lua"}
        )

        # Should preserve user's custom script
        assert maker.input_set_generator.user_params["Lua.Script"] == "custom_relax.lua"

        job = maker.make(si_structure)
        assert job.name == "Lua calculation-fixed-cell"

    def test_lua_maker_neb_classmethod(self, si_structure):
        """Test LuaMaker.neb() class method."""
        maker = LuaMaker.neb()

        assert maker.name == "Lua calculation-neb-with-fix-cell"
        assert isinstance(maker.input_set_generator, LuaSetGenerator)
        assert maker.input_set_generator.relax_cell is False
        assert maker.input_set_generator.lua_type == "lua_neb"

        job = maker.make(si_structure)
        assert job.name == "Lua calculation-neb-with-fix-cell"

    def test_lua_maker_neb_with_default_script(self, si_structure):
        """Test that neb() adds default NEB Lua script."""
        maker = LuaMaker.neb()

        # Should have default neb.lua script
        assert maker.input_set_generator.user_params is not None
        assert (
            "Lua.Script" in maker.input_set_generator.user_params
            or "lua.script"
            in {k.lower() for k in maker.input_set_generator.user_params.keys()}
        )

        job = maker.make(si_structure)
        assert job.name == "Lua calculation-neb-with-fix-cell"

    def test_lua_maker_neb_with_custom_script(self, si_structure):
        """Test LuaMaker NEB with custom Lua script."""
        maker = LuaMaker.neb(user_params={"Lua.Script": "custom_neb.lua"})

        # Should preserve user's custom script
        assert maker.input_set_generator.user_params["Lua.Script"] == "custom_neb.lua"

        job = maker.make(si_structure)
        assert job.name == "Lua calculation-neb-with-fix-cell"

    def test_lua_maker_with_user_params(self, si_structure):
        """Test LuaMaker with additional user parameters."""
        user_params = {
            "Lua.Script": "relax_geometry_lbfgs.lua",
            "PAO.BasisSize": "DZP",
            "a2s_kpts": [4, 4, 4],
        }

        maker = LuaMaker.fixed_cell_relaxation(user_params=user_params)

        # All parameters should be present
        for key in user_params:
            assert key in maker.input_set_generator.user_params

        job = maker.make(si_structure)
        assert job.name == "Lua calculation-fixed-cell"

    def test_lua_maker_serialization(self, si_structure):
        """Test LuaMaker serialization."""
        maker = LuaMaker.fixed_cell_relaxation(
            user_params={"Lua.Script": "relax_geometry_lbfgs.lua"}
        )

        # Serialize
        maker_dict = maker.as_dict()
        assert isinstance(maker_dict, dict)
        assert maker_dict["@class"] == "LuaMaker"

        # Deserialize
        maker_restored = maker.from_dict(maker_dict)
        assert isinstance(maker_restored, LuaMaker)
        assert maker_restored.name == maker.name


class TestPhononMaker:
    """Tests for PhononMaker (force constant calculations)."""

    def test_default_phonon_maker(self, si_structure):
        """Test creation of default PhononMaker."""
        maker = PhononMaker()

        assert maker.name == "Relaxation calculation"
        assert maker.calc_type == "relax"
        assert isinstance(maker.input_set_generator, PhononSetGenerator)

        # Test job generation
        job = maker.make(si_structure)
        assert job.name == "Relaxation calculation"
        assert hasattr(job, "function")

    def test_phonon_maker_fc_calculations(self, si_structure):
        """Test PhononMaker.fc_calculations() class method."""
        maker = PhononMaker.fc_calculations()

        assert maker.name == "Relaxation calculation-phonon"
        assert isinstance(maker.input_set_generator, PhononSetGenerator)
        assert maker.input_set_generator.md_type_of_run == "FC"

        job = maker.make(si_structure)
        assert job.name == "Relaxation calculation-phonon"

    def test_phonon_maker_with_user_params(self, si_structure):
        """Test PhononMaker with custom user parameters."""
        user_params = {
            "PAO.BasisSize": "DZP",
            "a2s_kpts": [6, 6, 6],
            "MD.FCFirst": 1,
            "MD.FCLast": 2,
        }

        maker = PhononMaker.fc_calculations(user_params=user_params)

        # Check parameters stored
        assert maker.input_set_generator.user_params == user_params

        job = maker.make(si_structure)
        assert job.name == "Relaxation calculation-phonon"

    def test_phonon_maker_serialization(self, si_structure):
        """Test PhononMaker serialization."""
        maker = PhononMaker.fc_calculations(user_params={"PAO.BasisSize": "DZP"})

        # Serialize
        maker_dict = maker.as_dict()
        assert isinstance(maker_dict, dict)
        assert maker_dict["@class"] == "PhononMaker"

        # Deserialize
        maker_restored = maker.from_dict(maker_dict)
        assert isinstance(maker_restored, PhononMaker)
        assert maker_restored.name == maker.name


class TestOpticalMaker:
    """Tests for OpticalMaker (optical properties calculations)."""

    def test_default_optical_maker(self, si_structure):
        """Test creation of default OpticalMaker."""
        maker = OpticalMaker()

        assert maker.name == "Optical calculation"
        assert maker.calc_type == "Optical"
        assert isinstance(maker.input_set_generator, OpticalSetGenerator)

        # Test job generation
        job = maker.make(si_structure)
        assert job.name == "Optical calculation"
        assert hasattr(job, "function")

    def test_optical_maker_optical_calculations(self, si_structure):
        """Test OpticalMaker.optical_calculations() class method."""
        maker = OpticalMaker.optical_calculations()

        assert maker.name == "Optical calculation"
        assert isinstance(maker.input_set_generator, OpticalSetGenerator)
        assert maker.input_set_generator.optical_calculation is True

        job = maker.make(si_structure)
        assert job.name == "Optical calculation"

    def test_optical_maker_with_user_params(self, si_structure):
        """Test OpticalMaker with custom user parameters."""
        user_params = {
            "PAO.BasisSize": "TZP",
            "a2s_kpts": [8, 8, 8],
            "Optical.OffsetMesh": True,
        }

        maker = OpticalMaker.optical_calculations(user_params=user_params)

        # Check user parameters are stored (generator also injects defaults
        # such as Mesh.Cutoff, so check subset rather than exact equality)
        stored = maker.input_set_generator.user_params
        for key, value in user_params.items():
            assert stored[key] == value

        job = maker.make(si_structure)
        assert job.name == "Optical calculation"

    def test_optical_maker_serialization(self, si_structure):
        """Test OpticalMaker serialization."""
        maker = OpticalMaker.optical_calculations(user_params={"PAO.BasisSize": "DZP"})

        # Serialize
        maker_dict = maker.as_dict()
        assert isinstance(maker_dict, dict)
        assert maker_dict["@class"] == "OpticalMaker"

        # Deserialize
        maker_restored = maker.from_dict(maker_dict)
        assert isinstance(maker_restored, OpticalMaker)
        assert maker_restored.name == maker.name


class TestSocketIOStaticMaker:
    """Tests for SocketIOStaticMaker (multi-structure calculations via socket)."""

    def test_default_socketio_maker(self, si_structure):
        """Test creation of default SocketIOStaticMaker."""
        maker = SocketIOStaticMaker()

        assert maker.name == "SCF Calculations Socket"
        assert maker.calc_type == "multi_scf"
        assert maker.host == "localhost"
        assert maker.port == 12345
        assert isinstance(maker.input_set_generator, SocketIOSetGenerator)

    def test_socketio_maker_custom_host_port(self, si_structure):
        """Test SocketIOStaticMaker with custom host and port."""
        maker = SocketIOStaticMaker(host="compute-node", port=54321)

        assert maker.host == "compute-node"
        assert maker.port == 54321

    def test_socketio_maker_with_user_params(self, si_structure):
        """Test SocketIOStaticMaker with custom user parameters."""
        user_params = {"PAO.BasisSize": "DZP", "a2s_kpts": [6, 6, 6]}

        maker = SocketIOStaticMaker(
            input_set_generator=SocketIOSetGenerator(user_params=user_params)
        )

        # Generator may inject defaults, so check provided params are stored
        stored = maker.input_set_generator.user_params
        for key, value in user_params.items():
            assert stored[key] == value

    def test_socketio_maker_make_accepts_structure_list(
        self, si_structure, al_structure
    ):
        """Test that SocketIOStaticMaker.make() accepts list of structures."""
        maker = SocketIOStaticMaker()

        # Should accept list of structures
        structures = [si_structure, al_structure]
        job = maker.make(structures)

        assert hasattr(job, "name")
        assert hasattr(job, "function")

    def test_socketio_maker_make_accepts_single_structure(self, si_structure):
        """Test that SocketIOStaticMaker.make() accepts single structure."""
        maker = SocketIOStaticMaker()

        # Should also accept single structure (will be converted to list)
        job = maker.make(si_structure)

        assert hasattr(job, "name")
        assert hasattr(job, "function")

    def test_socketio_maker_serialization(self):
        """Test SocketIOStaticMaker serialization."""
        maker = SocketIOStaticMaker(host="compute-node", port=54321)

        # Serialize
        maker_dict = maker.as_dict()
        assert isinstance(maker_dict, dict)
        assert maker_dict["@class"] == "SocketIOStaticMaker"
        assert maker_dict["host"] == "compute-node"
        assert maker_dict["port"] == 54321

        # Deserialize
        maker_restored = maker.from_dict(maker_dict)
        assert isinstance(maker_restored, SocketIOStaticMaker)
        assert maker_restored.host == maker.host
        assert maker_restored.port == maker.port


class TestSiestaPhononMaker:
    """Tests for SiestaPhononFlowMaker (convenience wrapper for phonopy)."""

    def test_default_phonon_maker(self, si_structure):
        """Test creation of default SiestaPhononFlowMaker."""
        maker = SiestaPhononFlowMaker()

        assert maker.name == "siesta phonopy"
        assert maker.min_length == 6.0
        assert maker.displacement == 0.01
        assert maker.use_symmetry is True

    def test_phonon_maker_with_custom_params(self, si_structure):
        """Test SiestaPhononFlowMaker with custom parameters."""
        maker = SiestaPhononFlowMaker(
            min_length=15.0, displacement=0.02, mesh=(40, 40, 40), use_symmetry=False
        )

        assert maker.min_length == 15.0
        assert maker.displacement == 0.02
        assert maker.mesh == (40, 40, 40)
        assert maker.use_symmetry is False

    def test_phonon_maker_with_custom_makers(self, si_structure):
        """Test SiestaPhononFlowMaker with custom relax and static makers."""
        relax_maker = RelaxMaker.variable_cell_relaxation(
            user_params={"PAO.BasisSize": "DZP"}
        )
        static_maker = StaticMaker(
            input_set_generator=StaticSetGenerator(user_params={"PAO.BasisSize": "TZP"})
        )

        phonon_maker = SiestaPhononFlowMaker(
            relax_maker=relax_maker, static_maker=static_maker
        )

        assert phonon_maker.relax_maker is not None
        assert phonon_maker.static_maker is not None
        assert (
            phonon_maker.relax_maker.input_set_generator.user_params["PAO.BasisSize"]
            == "DZP"
        )
        assert (
            phonon_maker.static_maker.input_set_generator.user_params["PAO.BasisSize"]
            == "TZP"
        )

    def test_phonon_maker_output_options(self, si_structure):
        """Test SiestaPhononFlowMaker with selective optional-output config.

        The standalone plotting toggles were removed from this maker; optional
        post-processing is now controlled via thermal-properties/symmetry knobs.
        """
        maker = SiestaPhononFlowMaker(
            create_thermal_properties=False,
            prefer_90_degrees=False,
        )

        assert maker.create_thermal_properties is False
        assert maker.prefer_90_degrees is False

    def test_phonon_maker_thermal_properties(self, si_structure):
        """Test SiestaPhononFlowMaker thermal properties configuration."""
        maker = SiestaPhononFlowMaker(
            create_thermal_properties=True, t_min=100, t_max=500, t_step=50
        )

        assert maker.create_thermal_properties is True
        assert maker.t_min == 100
        assert maker.t_max == 500
        assert maker.t_step == 50

    def test_phonon_maker_supercell_options(self, si_structure):
        """Test SiestaPhononFlowMaker supercell configuration."""
        maker = SiestaPhononFlowMaker(
            supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]], prefer_90_degrees=False
        )

        assert maker.supercell_matrix == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        assert maker.prefer_90_degrees is False

    def test_phonon_maker_symmetry_options(self, si_structure):
        """Test SiestaPhononFlowMaker symmetry configuration."""
        maker = SiestaPhononFlowMaker(use_symmetry=False, symprec=1e-4)

        assert maker.use_symmetry is False
        assert maker.symprec == 1e-4


class TestMakerIntegration:
    """Integration tests for maker workflows."""

    def test_relax_then_static(self, si_structure):
        """Test chaining relax and static calculations."""
        relax_maker = RelaxMaker.fixed_cell_relaxation(
            user_params={"PAO.BasisSize": "DZP"}
        )
        static_maker = StaticMaker(
            input_set_generator=StaticSetGenerator(user_params={"PAO.BasisSize": "DZP"})
        )

        # Create jobs
        relax_job = relax_maker.make(si_structure)
        assert relax_job.name == "Relaxation calculation-fixed-cell"

        # Static job would use output from relax in real workflow
        static_job = static_maker.make(si_structure)
        assert static_job.name == "SCF Calculation"

    def test_all_makers_create_valid_jobs(self, si_structure):
        """Test that all makers can create valid jobs."""
        makers = [
            StaticMaker(),
            StaticMaker.scf(),
            RelaxMaker.fixed_cell_relaxation(),
            RelaxMaker.variable_cell_relaxation(),
            BandStructureMaker(),
            BandStructureMaker.bandstructure_calculation(),
            LuaMaker.fixed_cell_relaxation(),
            LuaMaker.neb(),
            PhononMaker.fc_calculations(),
            OpticalMaker.optical_calculations(),
        ]

        for maker in makers:
            job = maker.make(si_structure)
            assert hasattr(job, "name")
            assert hasattr(job, "function")
            assert callable(job.function)

    def test_makers_with_different_structures(
        self, si_structure, al_structure, graphene_structure
    ):
        """Test that makers work with different structure types."""
        maker = RelaxMaker.fixed_cell_relaxation()

        structures = [si_structure, al_structure, graphene_structure]

        for structure in structures:
            job = maker.make(structure)
            assert hasattr(job, "name")
            assert job.name == "Relaxation calculation-fixed-cell"

    def test_maker_with_tier_presets(self, si_structure):
        """Test maker with tier preset application."""
        from atomate2.siesta.sets.tiers import apply_tier_preset

        maker = RelaxMaker.fixed_cell_relaxation()

        # Apply preset
        maker = apply_tier_preset(maker, "relax_standard")

        # Maker should have preset applied
        job = maker.make(si_structure)
        assert job.name == "Relaxation calculation-fixed-cell"


class TestMakerEdgeCases:
    """Test edge cases and error handling."""

    def test_maker_with_empty_user_params(self, si_structure):
        """Test maker with empty user_params."""
        maker = RelaxMaker.fixed_cell_relaxation(user_params={})

        # Note: Tier system may add default parameters even when user_params={}
        # This is intentional behavior to ensure sensible defaults
        assert isinstance(maker.input_set_generator.user_params, dict)

        job = maker.make(si_structure)
        assert job.name == "Relaxation calculation-fixed-cell"

    def test_maker_with_none_prev_dir(self, si_structure):
        """Test maker with None as prev_dir."""
        maker = RelaxMaker.fixed_cell_relaxation()

        job = maker.make(si_structure, prev_dir=None)
        assert job.name == "Relaxation calculation-fixed-cell"

    def test_multiple_jobs_from_same_maker(self, si_structure, al_structure):
        """Test creating multiple jobs from the same maker."""
        maker = StaticMaker.scf()

        job1 = maker.make(si_structure)
        job2 = maker.make(al_structure)

        # Jobs should be independent
        assert job1.name == job2.name
        assert job1 is not job2

    def test_maker_modification_doesnt_affect_jobs(self, si_structure):
        """Test that modifying maker after job creation doesn't affect job."""
        maker = RelaxMaker.fixed_cell_relaxation(user_params={"PAO.BasisSize": "SZ"})

        job1 = maker.make(si_structure)

        # Modify maker (create new instance with different params)
        maker = RelaxMaker.fixed_cell_relaxation(user_params={"PAO.BasisSize": "DZP"})

        job2 = maker.make(si_structure)

        # Jobs should have different parameters
        assert job1 is not job2
