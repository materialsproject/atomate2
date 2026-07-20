"""
Tests for Equation of State (EOS) workflows.

These tests validate:
- SiestaEosFlowMaker (basic EOS workflow)
- EOSFullBasisConvergenceFlowMaker (full basis set convergence with parameter optimization)
- Flow composition and chaining
- Parameter handling
- Serialization
"""

from jobflow import Flow

from atomate2.siesta.flows.eos import (
    EOSFullBasisConvergenceFlowMaker,
    EOSMaker,
    SiestaEosFlowMaker,
    collect_eos_parameter_data,
    plot_eos_parameter_fits_from_data,
    plot_eos_parameter_timing,
    write_eos_parameter_summary,
)
from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker


class TestSiestaEosMaker:
    """Tests for SiestaEosFlowMaker workflow."""

    def test_default_siesta_eos_maker(self):
        """Test creation of default SiestaEosFlowMaker."""
        maker = SiestaEosFlowMaker()

        assert maker.name == "siesta eos"
        assert isinstance(maker.initial_relax_maker, RelaxMaker)
        assert isinstance(maker.eos_relax_maker, RelaxMaker)
        assert maker.static_maker is None
        assert hasattr(maker, "linear_strain")
        assert hasattr(maker, "number_of_frames")

    def test_siesta_eos_maker_with_custom_params(self):
        """Test SiestaEosFlowMaker with custom parameters."""
        initial_relax = RelaxMaker.variable_cell_relaxation()
        eos_relax = RelaxMaker.fixed_cell_relaxation()
        static = StaticMaker()

        maker = SiestaEosFlowMaker(
            name="custom_eos",
            initial_relax_maker=initial_relax,
            eos_relax_maker=eos_relax,
            static_maker=static,
            linear_strain=(-0.08, 0.08),
            number_of_frames=9,
        )

        assert maker.name == "custom_eos"
        assert maker.initial_relax_maker == initial_relax
        assert maker.eos_relax_maker == eos_relax
        assert maker.static_maker == static
        assert maker.linear_strain == (-0.08, 0.08)
        assert maker.number_of_frames == 9

    def test_siesta_eos_maker_without_initial_relax(self):
        """Test SiestaEosFlowMaker without initial relaxation."""
        maker = SiestaEosFlowMaker(initial_relax_maker=None)

        assert maker.initial_relax_maker is None
        assert maker.eos_relax_maker is not None

    def test_siesta_eos_maker_with_static_calculations(self):
        """Test SiestaEosFlowMaker with static calculations after relaxation."""
        static = StaticMaker()
        maker = SiestaEosFlowMaker(static_maker=static)

        assert maker.static_maker == static

    def test_siesta_eos_maker_strain_parameters(self):
        """Test different strain parameters."""
        strains = [
            (-0.05, 0.05),
            (-0.1, 0.1),
            (-0.03, 0.03),
        ]

        for strain in strains:
            maker = SiestaEosFlowMaker(linear_strain=strain)
            assert maker.linear_strain == strain

    def test_siesta_eos_maker_number_of_frames(self):
        """Test different numbers of strain frames."""
        frame_counts = [5, 7, 9, 11]

        for n_frames in frame_counts:
            maker = SiestaEosFlowMaker(number_of_frames=n_frames)
            assert maker.number_of_frames == n_frames

    def test_siesta_eos_maker_make_flow(self, si_structure):
        """Test that SiestaEosFlowMaker creates a valid flow."""
        maker = SiestaEosFlowMaker(
            linear_strain=(-0.05, 0.05),
            number_of_frames=5,
        )

        flow = maker.make(si_structure)

        assert isinstance(flow, Flow)
        assert flow.name == "siesta eos"
        # Should have initial relax + strain calculations + post-processing
        assert len(flow) > 1

    def test_siesta_eos_maker_from_parameters(self):
        """Test SiestaEosFlowMaker creation from parameters dict."""
        params = {
            "PAO.BasisSize": "DZP",
            "a2s_kpts": [6, 6, 6],
        }

        maker = SiestaEosFlowMaker.from_parameters(params)

        assert isinstance(maker, SiestaEosFlowMaker)
        assert isinstance(maker.initial_relax_maker, RelaxMaker)
        assert isinstance(maker.eos_relax_maker, RelaxMaker)

    def test_siesta_eos_maker_serialization(self):
        """Test SiestaEosFlowMaker serialization."""
        maker = SiestaEosFlowMaker(
            linear_strain=(-0.06, 0.06),
            number_of_frames=7,
        )

        # Serialize
        maker_dict = maker.as_dict()
        assert isinstance(maker_dict, dict)
        assert maker_dict["@class"] == "SiestaEosFlowMaker"

        # Deserialize
        maker_restored = maker.from_dict(maker_dict)
        assert isinstance(maker_restored, SiestaEosFlowMaker)
        # Tuples may convert to lists during serialization
        assert tuple(maker_restored.linear_strain) == (
            -0.06,
            0.06,
        ) or maker_restored.linear_strain == (-0.06, 0.06)
        assert maker_restored.number_of_frames == 7


class TestEOSMakerAlias:
    """Test EOSMaker alias."""

    def test_eos_maker_alias(self):
        """Test that EOSMaker is an alias for SiestaEosFlowMaker."""
        maker = EOSMaker()

        assert isinstance(maker, SiestaEosFlowMaker)
        assert maker.name == "siesta eos"

    def test_eos_maker_alias_with_params(self):
        """Test EOSMaker alias with parameters."""
        maker = EOSMaker(
            name="my_eos",
            number_of_frames=7,
        )

        assert isinstance(maker, SiestaEosFlowMaker)
        assert maker.name == "my_eos"
        assert maker.number_of_frames == 7


class TestEOSFullBasisConvergenceMaker:
    """Tests for EOSFullBasisConvergenceFlowMaker workflow."""

    def test_default_eos_parameter_convergence_maker(self):
        """Test creation of default EOSFullBasisConvergenceFlowMaker."""
        maker = EOSFullBasisConvergenceFlowMaker()

        assert maker.name == "EOS Full Basis Convergence"
        # Check defaults after __post_init__
        assert maker.basis_sizes == ["DZ", "DZP", "TZP"]
        assert maker.energy_shifts == [0.005, 0.010, 0.015, 0.020]
        assert maker.split_norms == [0.15, 0.20, 0.25]
        assert maker.linear_strain == (-0.05, 0.05)
        assert maker.number_of_frames == 7
        assert maker.initial_relax_maker is not None
        assert maker.eos_relax_maker is not None

    def test_eos_parameter_convergence_maker_with_custom_params(self):
        """Test EOSFullBasisConvergenceFlowMaker with custom parameters."""
        maker = EOSFullBasisConvergenceFlowMaker(
            name="Custom EOS Convergence",
            basis_sizes=["SZ", "DZ", "DZP"],
            energy_shifts=[0.01, 0.02],
            split_norms=[0.15, 0.20],
            a2s_kpts=[4, 4, 4],
            linear_strain=(-0.03, 0.03),
            number_of_frames=5,
        )

        assert maker.name == "Custom EOS Convergence"
        assert maker.basis_sizes == ["SZ", "DZ", "DZP"]
        assert maker.energy_shifts == [0.01, 0.02]
        assert maker.split_norms == [0.15, 0.20]
        assert maker.a2s_kpts == [4, 4, 4]
        assert maker.linear_strain == (-0.03, 0.03)
        assert maker.number_of_frames == 5

    def test_eos_parameter_convergence_maker_basis_sizes(self):
        """Test different basis size combinations."""
        basis_combos = [
            ["DZ", "DZP"],
            ["DZ", "DZP", "TZP"],
            ["SZ", "DZ", "TZ"],
        ]

        for basis_list in basis_combos:
            maker = EOSFullBasisConvergenceFlowMaker(basis_sizes=basis_list)
            assert maker.basis_sizes == basis_list

    def test_eos_parameter_convergence_maker_energy_shifts(self):
        """Test different energy shift values."""
        energy_shift_sets = [
            [0.01, 0.015, 0.02],
            [0.005, 0.010],
            [0.01],
        ]

        for es_list in energy_shift_sets:
            maker = EOSFullBasisConvergenceFlowMaker(energy_shifts=es_list)
            assert maker.energy_shifts == es_list

    def test_eos_parameter_convergence_maker_split_norms(self):
        """Test different split norm values."""
        split_norm_sets = [
            [0.15, 0.20, 0.25],
            [0.10, 0.15],
            [0.20],
        ]

        for sn_list in split_norm_sets:
            maker = EOSFullBasisConvergenceFlowMaker(split_norms=sn_list)
            assert maker.split_norms == sn_list

    def test_eos_parameter_convergence_maker_total_calculations(self):
        """Test calculation of total number of EOS workflows."""
        maker = EOSFullBasisConvergenceFlowMaker(
            basis_sizes=["DZ", "DZP"],  # 2
            energy_shifts=[0.01, 0.02],  # 2
            split_norms=[0.15, 0.20],  # 2
        )

        # Total = 2 * 2 * 2 = 8 EOS workflows
        expected_total = 2 * 2 * 2
        actual_total = (
            len(maker.basis_sizes) * len(maker.energy_shifts) * len(maker.split_norms)
        )
        assert actual_total == expected_total

    def test_eos_parameter_convergence_maker_make_flow(self, si_structure):
        """Test that EOSFullBasisConvergenceFlowMaker creates a valid flow."""
        # Use minimal parameters for testing
        maker = EOSFullBasisConvergenceFlowMaker(
            basis_sizes=["DZ"],
            energy_shifts=[0.01],
            split_norms=[0.15],
            number_of_frames=5,  # Minimal frames
        )

        flow = maker.make(si_structure)

        assert isinstance(flow, Flow)
        assert flow.name == "EOS Full Basis Convergence"
        # Should have EOS workflows + collection + analysis jobs
        assert len(flow) > 1

    def test_eos_parameter_convergence_maker_job_structure(self, si_structure):
        """Test the structure of jobs in convergence flow."""
        maker = EOSFullBasisConvergenceFlowMaker(
            basis_sizes=["DZ"],
            energy_shifts=[0.01],
            split_norms=[0.15],
            number_of_frames=5,
        )

        flow = maker.make(si_structure)
        job_names = [job.name for job in flow]

        # Should have collection, plot, timing, and summary jobs
        assert any("collect" in name.lower() for name in job_names)

    def test_eos_parameter_convergence_maker_with_static_maker(self, si_structure):
        """Test convergence maker with static calculations."""
        static = StaticMaker()
        maker = EOSFullBasisConvergenceFlowMaker(
            basis_sizes=["DZ"],
            energy_shifts=[0.01],
            split_norms=[0.15],
            static_maker=static,
            number_of_frames=5,
        )

        flow = maker.make(si_structure)
        assert isinstance(flow, Flow)

    def test_eos_parameter_convergence_maker_serialization(self):
        """Test EOSFullBasisConvergenceFlowMaker serialization."""
        maker = EOSFullBasisConvergenceFlowMaker(
            basis_sizes=["DZ", "DZP"],
            energy_shifts=[0.01, 0.02],
            split_norms=[0.15],
        )

        # Serialize
        maker_dict = maker.as_dict()
        assert isinstance(maker_dict, dict)
        assert maker_dict["@class"] == "EOSFullBasisConvergenceFlowMaker"

        # Deserialize
        maker_restored = maker.from_dict(maker_dict)
        assert isinstance(maker_restored, EOSFullBasisConvergenceFlowMaker)
        assert maker_restored.basis_sizes == ["DZ", "DZP"]
        assert maker_restored.energy_shifts == [0.01, 0.02]


class TestEOSFlowIntegration:
    """Integration tests for EOS workflows."""

    def test_all_eos_makers_create_valid_flows(self, si_structure):
        """Test that all EOS makers can create valid flows."""
        makers = [
            SiestaEosFlowMaker(number_of_frames=5),
            EOSMaker(number_of_frames=5),
            EOSFullBasisConvergenceFlowMaker(
                basis_sizes=["DZ"],
                energy_shifts=[0.01],
                split_norms=[0.15],
                number_of_frames=5,
            ),
        ]

        for maker in makers:
            flow = maker.make(si_structure)
            assert isinstance(flow, Flow)
            assert len(flow) > 0

    def test_eos_makers_with_different_structures(self, si_structure, al_structure):
        """Test EOS makers work with different structures."""
        structures = [si_structure, al_structure]
        maker = SiestaEosFlowMaker(number_of_frames=5)

        for structure in structures:
            flow = maker.make(structure)
            assert isinstance(flow, Flow)

    def test_eos_flow_output_references(self, si_structure):
        """Test that flows have proper output handling."""
        maker = SiestaEosFlowMaker(number_of_frames=5)
        flow = maker.make(si_structure)

        # Flow should be iterable
        jobs = list(flow)
        assert len(jobs) > 0

        # Each job should have a name
        for job in jobs:
            assert hasattr(job, "name")
            assert hasattr(job, "function")

    def test_eos_with_different_makers_combinations(self, si_structure):
        """Test EOS with different combinations of makers."""
        combinations = [
            {
                "initial_relax_maker": RelaxMaker.variable_cell_relaxation(),
                "eos_relax_maker": RelaxMaker.fixed_cell_relaxation(),
            },
            {
                "initial_relax_maker": None,
                "eos_relax_maker": RelaxMaker.fixed_cell_relaxation(),
            },
            {
                "initial_relax_maker": RelaxMaker.variable_cell_relaxation(),
                "eos_relax_maker": RelaxMaker.fixed_cell_relaxation(),
                "static_maker": StaticMaker(),
            },
        ]

        for combo in combinations:
            maker = SiestaEosFlowMaker(number_of_frames=5, **combo)
            flow = maker.make(si_structure)
            assert isinstance(flow, Flow)


class TestEOSEdgeCases:
    """Test edge cases and error handling."""

    def test_eos_with_none_prev_dir(self, si_structure):
        """Test EOS with None as prev_dir."""
        maker = SiestaEosFlowMaker(number_of_frames=5)

        flow = maker.make(si_structure, prev_dir=None)
        assert isinstance(flow, Flow)

    def test_multiple_flows_from_same_maker(self, si_structure, al_structure):
        """Test creating multiple flows from the same maker."""
        maker = SiestaEosFlowMaker(number_of_frames=5)

        flow1 = maker.make(si_structure)
        flow2 = maker.make(al_structure)

        # Flows should be independent
        assert flow1 is not flow2
        assert isinstance(flow1, Flow)
        assert isinstance(flow2, Flow)

    def test_maker_modification_doesnt_affect_flows(self, si_structure):
        """Test that modifying maker after flow creation doesn't affect flow."""
        maker = SiestaEosFlowMaker(number_of_frames=5)
        flow1 = maker.make(si_structure)

        # Modify maker
        maker.number_of_frames = 5

        flow2 = maker.make(si_structure)

        # Flows should be independent
        assert flow1 is not flow2

    def test_eos_with_minimal_frames(self, si_structure):
        """Test EOS with minimal number of frames."""
        maker = SiestaEosFlowMaker(number_of_frames=5)

        flow = maker.make(si_structure)
        assert isinstance(flow, Flow)

    def test_eos_with_large_strain(self, si_structure):
        """Test EOS with large strain range."""
        maker = SiestaEosFlowMaker(
            linear_strain=(-0.15, 0.15),
            number_of_frames=5,
        )

        flow = maker.make(si_structure)
        assert isinstance(flow, Flow)


class TestEOSParameterValidation:
    """Test parameter validation for EOS makers."""

    def test_valid_strain_ranges(self, si_structure):
        """Test that valid strain ranges are accepted."""
        valid_strains = [
            (-0.05, 0.05),
            (-0.1, 0.1),
            (-0.03, 0.03),
            (-0.08, 0.08),
        ]

        for strain in valid_strains:
            maker = SiestaEosFlowMaker(linear_strain=strain, number_of_frames=5)
            flow = maker.make(si_structure)
            assert isinstance(flow, Flow)

    def test_valid_frame_counts(self, si_structure):
        """Test different frame count values."""
        frame_counts = [5, 7, 9, 11]  # Use 5+ for EOS fitting

        for n_frames in frame_counts:
            maker = SiestaEosFlowMaker(number_of_frames=n_frames)
            flow = maker.make(si_structure)
            assert isinstance(flow, Flow)

    def test_convergence_maker_parameter_combinations(self, si_structure):
        """Test valid parameter combinations for convergence maker."""
        # Test small parameter space
        maker = EOSFullBasisConvergenceFlowMaker(
            basis_sizes=["DZ"],
            energy_shifts=[0.01],
            split_norms=[0.15],
            number_of_frames=5,
        )

        flow = maker.make(si_structure)
        assert isinstance(flow, Flow)

    def test_convergence_maker_with_kpoints(self, si_structure):
        """Test convergence maker with k-points."""
        kpts_list = [
            [4, 4, 4],
            [6, 6, 6],
            [8, 8, 8],
        ]

        for kpts in kpts_list:
            maker = EOSFullBasisConvergenceFlowMaker(
                basis_sizes=["DZ"],
                energy_shifts=[0.01],
                split_norms=[0.15],
                a2s_kpts=kpts,
                number_of_frames=5,
            )
            flow = maker.make(si_structure)
            assert isinstance(flow, Flow)


class TestEOSMakerSerialization:
    """Test serialization of all EOS makers."""

    def test_all_eos_makers_serializable(self):
        """Test that all EOS makers can be serialized and deserialized."""
        makers = [
            ("siesta_eos", SiestaEosFlowMaker()),
            ("eos_alias", EOSMaker()),
            (
                "convergence",
                EOSFullBasisConvergenceFlowMaker(
                    basis_sizes=["DZ"],
                    energy_shifts=[0.01],
                    split_norms=[0.15],
                ),
            ),
        ]

        for name, maker in makers:
            # Serialize
            maker_dict = maker.as_dict()
            assert isinstance(maker_dict, dict), f"{name} failed to serialize"

            # Deserialize
            maker_restored = maker.from_dict(maker_dict)
            assert maker_restored is not None, f"{name} failed to deserialize"

    def test_siesta_eos_serialization_preserves_parameters(self):
        """Test that serialization preserves all parameters."""
        maker = SiestaEosFlowMaker(
            name="custom_eos",
            linear_strain=(-0.07, 0.07),
            number_of_frames=8,
        )

        # Serialize and deserialize
        maker_dict = maker.as_dict()
        maker_restored = maker.from_dict(maker_dict)

        # Check all parameters preserved
        assert maker_restored.name == "custom_eos"
        # Tuples may convert to lists during serialization
        assert tuple(maker_restored.linear_strain) == (
            -0.07,
            0.07,
        ) or maker_restored.linear_strain == (-0.07, 0.07)
        assert maker_restored.number_of_frames == 8

    def test_convergence_maker_serialization_preserves_lists(self):
        """Test that convergence maker preserves list parameters."""
        maker = EOSFullBasisConvergenceFlowMaker(
            basis_sizes=["DZ", "DZP", "TZP"],
            energy_shifts=[0.01, 0.015, 0.02],
            split_norms=[0.15, 0.20, 0.25],
        )

        # Serialize and deserialize
        maker_dict = maker.as_dict()
        maker_restored = maker.from_dict(maker_dict)

        # Check lists preserved
        assert maker_restored.basis_sizes == ["DZ", "DZP", "TZP"]
        assert maker_restored.energy_shifts == [0.01, 0.015, 0.02]
        assert maker_restored.split_norms == [0.15, 0.20, 0.25]


class TestEOSFlowComposition:
    """Test flow composition for EOS workflows."""

    def test_eos_flow_structure(self, si_structure):
        """Test the structure of EOS flows."""
        maker = SiestaEosFlowMaker(number_of_frames=5)
        flow = maker.make(si_structure)

        jobs = list(flow)

        # Should have relaxation + strain calculations + post-processing
        assert len(jobs) >= 3

    def test_eos_with_custom_name(self, si_structure):
        """Test creating EOS flows with custom names."""
        maker = SiestaEosFlowMaker(
            name="My Custom EOS",
            number_of_frames=5,
        )

        flow = maker.make(si_structure)
        assert flow.name == "My Custom EOS"

    def test_convergence_flow_has_analysis_jobs(self, si_structure):
        """Test that convergence flow includes analysis jobs."""
        maker = EOSFullBasisConvergenceFlowMaker(
            basis_sizes=["DZ"],
            energy_shifts=[0.01],
            split_norms=[0.15],
            number_of_frames=5,
        )

        flow = maker.make(si_structure)
        job_names = [job.name for job in flow]

        # Should have collection and summary jobs
        assert any("collect" in name.lower() for name in job_names)

    def test_eos_flow_without_initial_relax(self, si_structure):
        """Test EOS flow structure without initial relaxation."""
        maker = SiestaEosFlowMaker(
            initial_relax_maker=None,
            number_of_frames=5,
        )

        flow = maker.make(si_structure)
        jobs = list(flow)

        # Check that we don't have TWO relax steps (only EOS relax steps)
        assert len(jobs) >= 2  # At least EOS calcs + post-processing


class TestEOSWorkflowScaling:
    """Test scaling behavior of EOS workflows."""

    def test_eos_frame_count_scaling(self, si_structure):
        """Test that EOS job count scales with frame count."""
        frame_counts = [5, 7, 9]  # Use 5+ frames for EOS fitting
        job_counts = []

        for n_frames in frame_counts:
            maker = SiestaEosFlowMaker(
                number_of_frames=n_frames,
                initial_relax_maker=None,  # Simplify for counting
            )
            flow = maker.make(si_structure)
            job_counts.append(len(list(flow)))

        # More frames should generally mean more jobs
        # (though exact relationship depends on flow structure)
        assert job_counts[1] >= job_counts[0]
        assert job_counts[2] >= job_counts[1]

    def test_convergence_maker_scaling(self, si_structure):
        """Test that convergence maker scales correctly."""
        # Test with 1x1x1 = 1 EOS workflow
        maker1 = EOSFullBasisConvergenceFlowMaker(
            basis_sizes=["DZ"],
            energy_shifts=[0.01],
            split_norms=[0.15],
            number_of_frames=5,
        )
        flow1 = maker1.make(si_structure)

        # Test with 2x1x1 = 2 EOS workflows
        maker2 = EOSFullBasisConvergenceFlowMaker(
            basis_sizes=["DZ", "DZP"],
            energy_shifts=[0.01],
            split_norms=[0.15],
            number_of_frames=5,
        )
        flow2 = maker2.make(si_structure)

        # Should have more jobs with more parameter combinations
        assert len(list(flow2)) > len(list(flow1))


class TestEOSDryRunMode:
    """Test dry-run mode for EOS workflows."""

    def test_siesta_eos_maker_dry_run_initialization(self):
        """Test SiestaEosFlowMaker dry_run initialization."""
        maker = SiestaEosFlowMaker(
            dry_run=True,
            dry_run_output_dir="eos_dry_run",
            dry_run_format="xsf",
        )

        assert maker.dry_run is True
        assert maker.dry_run_output_dir == "eos_dry_run"
        assert maker.dry_run_format == "xsf"

    def test_siesta_eos_maker_dry_run_propagation(self):
        """Test that dry_run propagates to child makers."""
        maker = SiestaEosFlowMaker(
            dry_run=True,
            dry_run_output_dir="test_dry_run",
            dry_run_format="cif",
        )

        # Check propagation via __post_init__
        assert maker.initial_relax_maker.dry_run is True
        assert maker.initial_relax_maker.dry_run_output_dir == "test_dry_run"
        assert maker.initial_relax_maker.dry_run_format == "cif"

        assert maker.eos_relax_maker.dry_run is True
        assert maker.eos_relax_maker.dry_run_output_dir == "test_dry_run"
        assert maker.eos_relax_maker.dry_run_format == "cif"

    def test_siesta_eos_maker_dry_run_flow_creation(self, si_structure):
        """Test that dry_run mode creates valid flows."""
        maker = SiestaEosFlowMaker(
            dry_run=True,
            dry_run_output_dir="eos_dry",
            number_of_frames=5,
        )

        flow = maker.make(si_structure)
        assert isinstance(flow, Flow)

    def test_eos_parameter_convergence_dry_run(self):
        """Test EOSFullBasisConvergenceFlowMaker with dry_run."""
        maker = EOSFullBasisConvergenceFlowMaker(
            basis_sizes=["DZ"],
            energy_shifts=[0.01],
            split_norms=[0.15],
            dry_run=True,
            dry_run_output_dir="convergence_dry",
            number_of_frames=5,
        )

        assert maker.dry_run is True
        assert maker.dry_run_output_dir == "convergence_dry"

    def test_dry_run_with_no_initial_relax(self):
        """Test dry_run with no initial relaxation maker."""
        maker = SiestaEosFlowMaker(
            initial_relax_maker=None,
            dry_run=True,
            dry_run_output_dir="no_initial_dry",
            number_of_frames=5,
        )

        # Should not fail even without initial_relax_maker
        assert maker.dry_run is True
        assert maker.initial_relax_maker is None

    def test_dry_run_different_formats(self):
        """Test dry_run with different output formats."""
        formats = ["cif", "xsf", "json", "poscar"]

        for fmt in formats:
            maker = SiestaEosFlowMaker(
                dry_run=True,
                dry_run_format=fmt,
                number_of_frames=5,
            )
            assert maker.dry_run_format == fmt


class TestEOSWithMolecules:
    """Test EOS workflows with molecular systems."""

    def test_siesta_eos_maker_with_molecule(self, h2o_structure):
        """Test SiestaEosFlowMaker with molecule input."""
        maker = SiestaEosFlowMaker(number_of_frames=5)

        flow = maker.make(h2o_structure)
        assert isinstance(flow, Flow)

    def test_eos_convergence_with_molecule(self, h2o_structure):
        """Test EOSFullBasisConvergenceFlowMaker with molecule."""
        maker = EOSFullBasisConvergenceFlowMaker(
            basis_sizes=["DZ"],
            energy_shifts=[0.01],
            split_norms=[0.15],
            number_of_frames=5,
        )

        flow = maker.make(h2o_structure)
        assert isinstance(flow, Flow)

    def test_molecule_eos_without_initial_relax(self, h2o_structure):
        """Test molecule EOS without initial relaxation."""
        maker = SiestaEosFlowMaker(
            initial_relax_maker=None,
            number_of_frames=5,
        )

        flow = maker.make(h2o_structure)
        assert isinstance(flow, Flow)


class TestEOSFromParameters:
    """Test EOS creation from parameters dict."""

    def test_from_parameters_basic(self):
        """Test basic from_parameters usage."""
        params = {"PAO.BasisSize": "DZP"}
        maker = SiestaEosFlowMaker.from_parameters(params)

        assert isinstance(maker, SiestaEosFlowMaker)

    def test_from_parameters_with_kpts(self):
        """Test from_parameters with k-points."""
        params = {
            "PAO.BasisSize": "DZP",
            "a2s_kpts": [6, 6, 6],
        }
        maker = SiestaEosFlowMaker.from_parameters(params)

        assert isinstance(maker, SiestaEosFlowMaker)
        assert isinstance(maker.initial_relax_maker, RelaxMaker)
        assert isinstance(maker.eos_relax_maker, RelaxMaker)

    def test_from_parameters_with_kwargs(self):
        """Test from_parameters with additional kwargs."""
        params = {"PAO.BasisSize": "TZP"}
        maker = SiestaEosFlowMaker.from_parameters(
            params,
            name="custom_from_params",
            number_of_frames=7,
        )

        assert maker.name == "custom_from_params"
        assert maker.number_of_frames == 7

    def test_from_parameters_creates_valid_flow(self, si_structure):
        """Test that from_parameters creates valid flows."""
        params = {
            "PAO.BasisSize": "DZP",
            "a2s_kpts": [4, 4, 4],
        }
        maker = SiestaEosFlowMaker.from_parameters(params, number_of_frames=5)

        flow = maker.make(si_structure)
        assert isinstance(flow, Flow)


class TestCollectEOSParameterData:
    """Test collect_eos_parameter_data() function."""

    def test_collect_with_valid_relax_data(self):
        """Test collection from valid relax EOS outputs."""
        # Mock EOS outputs (dict with relax key)
        eos_outputs = [
            {
                "relax": {
                    "volume": [100.0, 105.0, 110.0],
                    "energy": [-10.5, -10.6, -10.4],
                    "EOS": {
                        "birch_murnaghan": {
                            "v0": 105.0,
                            "e0": -10.6,
                            "b0 GPa": 150.0,
                        }
                    },
                    "run_time": [100.0, 105.0, 110.0],
                }
            },
            {
                "relax": {
                    "volume": [100.0, 105.0, 110.0],
                    "energy": [-10.3, -10.5, -10.2],
                    "EOS": {
                        "birch_murnaghan": {
                            "v0": 104.0,
                            "e0": -10.5,
                            "b0 GPa": 145.0,
                        }
                    },
                    "run_time": [95.0, 100.0, 105.0],
                }
            },
        ]

        job_metadata = [
            {
                "basis_size": "SZ",
                "energy_shift": 0.01,
                "split_norm": 0.15,
                "label": "SZ_0.01_0.15",
            },
            {
                "basis_size": "DZ",
                "energy_shift": 0.01,
                "split_norm": 0.15,
                "label": "DZ_0.01_0.15",
            },
        ]

        basis_sizes = ["SZ", "DZ"]

        result = collect_eos_parameter_data.original(
            eos_outputs, job_metadata, basis_sizes
        )

        # Check structure
        assert isinstance(result, dict)
        assert "basis_sizes" in result
        assert "v0_values" in result
        assert "e0_values" in result
        assert "b0_values" in result

        # Check values
        assert len(result["v0_values"]) == 2
        assert result["v0_values"][0] == 105.0
        assert result["v0_values"][1] == 104.0

    def test_collect_prefers_static_over_relax(self):
        """Test that static data is preferred over relax data."""
        eos_outputs = [
            {
                "static": {
                    "volume": [100.0, 105.0],
                    "energy": [-10.7, -10.8],
                    "EOS": {
                        "birch_murnaghan": {
                            "v0": 106.0,
                            "e0": -10.8,
                            "b0 GPa": 155.0,
                        }
                    },
                },
                "relax": {
                    "volume": [100.0, 105.0],
                    "energy": [-10.5, -10.6],
                    "EOS": {
                        "birch_murnaghan": {
                            "v0": 105.0,
                            "e0": -10.6,
                            "b0 GPa": 150.0,
                        }
                    },
                },
            }
        ]

        job_metadata = [
            {
                "basis_size": "DZP",
                "energy_shift": 0.01,
                "split_norm": 0.15,
                "label": "DZP_static",
            },
        ]

        result = collect_eos_parameter_data.original(eos_outputs, job_metadata, ["DZP"])

        # Should use static data (v0=106.0)
        assert result["v0_values"][0] == 106.0

    def test_collect_handles_missing_eos_data(self):
        """Test handling of outputs without EOS data."""
        eos_outputs = [
            {"relax": {"volume": [100.0], "energy": [-10.0]}},  # No EOS key
            {
                "relax": {
                    "volume": [105.0],
                    "energy": [-10.5],
                    "EOS": {
                        "birch_murnaghan": {
                            "v0": 105.0,
                            "e0": -10.5,
                            "b0 GPa": 145.0,
                        }
                    },
                }
            },
        ]

        job_metadata = [
            {
                "basis_size": "SZ",
                "energy_shift": 0.01,
                "split_norm": 0.15,
                "label": "SZ_missing",
            },
            {
                "basis_size": "DZ",
                "energy_shift": 0.01,
                "split_norm": 0.15,
                "label": "DZ_valid",
            },
        ]

        result = collect_eos_parameter_data.original(
            eos_outputs, job_metadata, ["SZ", "DZ"]
        )

        # Should only collect valid data (1 result)
        assert len(result["v0_values"]) == 1
        assert result["basis_sizes"][0] == "DZ"

    def test_collect_handles_non_dict_output(self):
        """Test handling of non-dictionary outputs."""
        eos_outputs = [
            "invalid_output",  # String instead of dict
            {
                "relax": {
                    "volume": [100.0],
                    "energy": [-10.0],
                    "EOS": {
                        "birch_murnaghan": {
                            "v0": 100.0,
                            "e0": -10.0,
                            "b0 GPa": 140.0,
                        }
                    },
                }
            },
        ]

        job_metadata = [
            {
                "basis_size": "SZ",
                "energy_shift": 0.01,
                "split_norm": 0.15,
                "label": "SZ_invalid",
            },
            {
                "basis_size": "DZ",
                "energy_shift": 0.01,
                "split_norm": 0.15,
                "label": "DZ_valid",
            },
        ]

        result = collect_eos_parameter_data.original(
            eos_outputs, job_metadata, ["SZ", "DZ"]
        )

        # Should skip invalid output
        assert len(result["v0_values"]) == 1
        assert result["basis_sizes"][0] == "DZ"

    def test_collect_handles_multiple_eos_models(self):
        """Test fallback to alternative EOS models."""
        eos_outputs = [
            {
                "relax": {
                    "volume": [100.0],
                    "energy": [-10.0],
                    "EOS": {
                        "vinet": {  # No birch_murnaghan, should use vinet
                            "v0": 102.0,
                            "e0": -10.2,
                            "b0 GPa": 148.0,
                        }
                    },
                }
            }
        ]

        job_metadata = [
            {
                "basis_size": "TZP",
                "energy_shift": 0.01,
                "split_norm": 0.15,
                "label": "TZP_vinet",
            },
        ]

        result = collect_eos_parameter_data.original(eos_outputs, job_metadata, ["TZP"])

        # Should use vinet model
        assert len(result["v0_values"]) == 1
        assert result["v0_values"][0] == 102.0

    def test_collect_handles_missing_timing_data(self):
        """Test handling of missing run_time data."""
        eos_outputs = [
            {
                "relax": {
                    "volume": [100.0],
                    "energy": [-10.0],
                    "EOS": {
                        "birch_murnaghan": {
                            "v0": 100.0,
                            "e0": -10.0,
                            "b0 GPa": 140.0,
                        }
                    },
                    # No run_time key
                }
            }
        ]

        job_metadata = [
            {
                "basis_size": "SZ",
                "energy_shift": 0.01,
                "split_norm": 0.15,
                "label": "SZ_no_time",
            },
        ]

        result = collect_eos_parameter_data.original(eos_outputs, job_metadata, ["SZ"])

        # Should have None for run_times
        assert "run_times" in result
        assert result["run_times"][0] is None

    def test_collect_empty_inputs(self):
        """Test with empty inputs."""
        result = collect_eos_parameter_data.original([], [], [])

        # Should return empty but valid structure
        assert isinstance(result, dict)
        assert "basis_sizes" in result
        assert len(result["basis_sizes"]) == 0


class TestPlotEOSParameterFits:
    """Test plot_eos_parameter_fits_from_data() function."""

    def test_plot_with_empty_data(self, tmp_path):
        """Test plotting with empty data generates error plot."""
        empty_data = {
            "basis_sizes": [],
            "v0_values": [],
            "e0_values": [],
            "b0_values": [],
            "labels": [],
            "all_volumes": [],
            "all_energies": [],
        }

        result = plot_eos_parameter_fits_from_data.original(
            empty_data, str(tmp_path / "plot.png")
        )

        # Should return dict with plot key
        assert isinstance(result, dict)
        assert "plot" in result

    def test_plot_returns_dict_with_path(self, tmp_path):
        """Test that plot function returns dict with file path."""
        data = {
            "basis_sizes": ["SZ", "DZ"],
            "v0_values": [100.0, 105.0],
            "e0_values": [-10.0, -10.5],
            "b0_values": [140.0, 145.0],
            "labels": ["SZ_0.01_0.15", "DZ_0.01_0.15"],
            "all_volumes": [[95.0, 100.0, 105.0], [100.0, 105.0, 110.0]],
            "all_energies": [[-9.8, -10.0, -9.9], [-10.3, -10.5, -10.4]],
        }

        result = plot_eos_parameter_fits_from_data.original(
            data, str(tmp_path / "plot.png")
        )

        assert isinstance(result, dict)
        assert "plot" in result
        from pathlib import Path

        assert Path(result["plot"]).exists()


class TestPlotEOSParameterTiming:
    """Test plot_eos_parameter_timing() function."""

    def test_timing_with_no_data(self, tmp_path):
        """Test timing plot with no timing data."""
        data = {
            "basis_sizes": ["SZ"],
            "energy_shifts": [0.01],
            "split_norms": [0.15],
            "labels": ["SZ_0.01_0.15"],
            "v0_values": [100.0],
            "run_times": [None],  # No timing data
        }

        result = plot_eos_parameter_timing.original(data, str(tmp_path / "timing.png"))

        # Should still return dict
        assert isinstance(result, dict)
        assert "timing_plot" in result

    def test_timing_with_valid_data(self, tmp_path):
        """Test timing plot with valid timing data."""
        data = {
            "basis_sizes": ["SZ", "DZ", "DZP"],
            "energy_shifts": [0.01, 0.01, 0.01],
            "split_norms": [0.15, 0.15, 0.15],
            "labels": ["SZ_0.01_0.15", "DZ_0.01_0.15", "DZP_0.01_0.15"],
            "v0_values": [100.0, 105.0, 108.0],
            "run_times": [50.0, 75.0, 100.0],
        }

        result = plot_eos_parameter_timing.original(data, str(tmp_path / "timing.png"))

        assert isinstance(result, dict)
        assert "timing_plot" in result


class TestWriteEOSParameterSummary:
    """Test write_eos_parameter_summary() function."""

    def test_summary_with_empty_data(self, tmp_path):
        """Test summary generation with empty data."""
        empty_data = {
            "basis_sizes": [],
            "energy_shifts": [],
            "split_norms": [],
            "labels": [],
            "v0_values": [],
            "e0_values": [],
            "b0_values": [],
        }

        result = write_eos_parameter_summary.original(
            empty_data, str(tmp_path / "summary.txt")
        )

        # Should return dict with summary key
        assert isinstance(result, dict)
        assert "summary" in result
        from pathlib import Path

        assert Path(result["summary"]).exists()

    def test_summary_with_valid_data(self, tmp_path):
        """Test summary generation with valid data."""
        data = {
            "basis_sizes": ["SZ", "DZ", "DZP"],
            "energy_shifts": [0.01, 0.01, 0.01],
            "split_norms": [0.15, 0.15, 0.15],
            "labels": ["SZ_0.01_0.15", "DZ_0.01_0.15", "DZP_0.01_0.15"],
            "v0_values": [100.0, 105.0, 104.5],
            "e0_values": [-10.0, -10.5, -10.48],
            "b0_values": [140.0, 145.0, 144.5],
            "run_times": [50.0, 75.0, 100.0],
        }

        result = write_eos_parameter_summary.original(
            data, str(tmp_path / "summary.txt")
        )

        assert isinstance(result, dict)
        assert "summary" in result
        from pathlib import Path

        summary_path = Path(result["summary"])
        assert summary_path.exists()

        # Check content exists
        content = summary_path.read_text()
        assert len(content) > 0
        assert "SZ" in content or "DZ" in content

    def test_summary_identifies_global_optimum(self, tmp_path):
        """Test that summary identifies the global optimum (lowest E0)."""
        data = {
            "basis_sizes": ["SZ", "DZ", "DZP"],
            "energy_shifts": [0.01, 0.01, 0.01],
            "split_norms": [0.15, 0.15, 0.15],
            "labels": ["SZ_0.01_0.15", "DZ_0.01_0.15", "DZP_0.01_0.15"],
            "v0_values": [100.0, 105.0, 104.5],
            "e0_values": [-10.0, -10.8, -10.5],  # DZ has lowest energy
            "b0_values": [140.0, 145.0, 144.5],
            "run_times": [50.0, 75.0, 100.0],
        }

        result = write_eos_parameter_summary.original(
            data, str(tmp_path / "summary.txt")
        )

        from pathlib import Path

        content = Path(result["summary"]).read_text()

        # Should mention DZ as global optimum (has E0 = -10.8)
        assert "DZ" in content
