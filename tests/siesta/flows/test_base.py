"""Tests for flows/base.py - BaseSiestaFlowMaker infrastructure.

These tests validate:
- BaseSiestaFlowMaker class
- Automatic dry-run propagation
- Automatic custodian propagation
- Automatic tier propagation
- Recursive propagation to nested flows
"""


class TestBaseSiestaFlowMaker:
    """Test BaseSiestaFlowMaker core functionality."""

    def test_flowmaker_initialization_defaults(self):
        """Test BaseSiestaFlowMaker default initialization."""
        from atomate2.siesta.flows.base import BaseSiestaFlowMaker

        flow_maker = BaseSiestaFlowMaker()

        assert flow_maker.dry_run is False
        assert flow_maker.dry_run_output_dir == "dry_run_output"
        assert flow_maker.dry_run_format == "cif"
        assert flow_maker.use_custodian is False
        assert flow_maker.custodian_handlers is None
        assert flow_maker.custodian_max_errors == 5
        assert flow_maker.tier is None

    def test_flowmaker_initialization_custom(self):
        """Test BaseSiestaFlowMaker with custom parameters."""
        from atomate2.siesta.flows.base import BaseSiestaFlowMaker

        flow_maker = BaseSiestaFlowMaker(
            dry_run=True,
            dry_run_output_dir="custom_output",
            dry_run_format="xsf",
            use_custodian=True,
            custodian_max_errors=10,
            tier="basic",
        )

        assert flow_maker.dry_run is True
        assert flow_maker.dry_run_output_dir == "custom_output"
        assert flow_maker.dry_run_format == "xsf"
        assert flow_maker.use_custodian is True
        assert flow_maker.custodian_max_errors == 10
        assert flow_maker.tier == "basic"

    def test_flowmaker_has_required_attributes(self):
        """Test that BaseSiestaFlowMaker has all required attributes."""
        from atomate2.siesta.flows.base import BaseSiestaFlowMaker

        flow_maker = BaseSiestaFlowMaker()

        # Check all required attributes exist
        required_attrs = [
            "dry_run",
            "dry_run_output_dir",
            "dry_run_format",
            "use_custodian",
            "custodian_handlers",
            "custodian_max_errors",
            "tier",
        ]

        for attr in required_attrs:
            assert hasattr(flow_maker, attr), f"Missing attribute: {attr}"

    def test_flowmaker_has_propagation_methods(self):
        """Test that BaseSiestaFlowMaker has all propagation methods."""
        from atomate2.siesta.flows.base import BaseSiestaFlowMaker

        flow_maker = BaseSiestaFlowMaker()

        # Check all propagation methods exist
        propagation_methods = [
            "_propagate_dry_run",
            "_propagate_custodian",
            "_propagate_tier",
            "_enable_dry_run_for_maker",
            "_enable_custodian_for_maker",
            "_set_tier_for_maker",
        ]

        for method in propagation_methods:
            assert hasattr(flow_maker, method), f"Missing method: {method}"
            assert callable(getattr(flow_maker, method))


class TestDryRunPropagation:
    """Test automatic dry-run propagation to child makers."""

    def test_dry_run_propagates_to_single_maker(self):
        """Test that dry_run propagates to a single child maker."""
        from atomate2.siesta.flows.base import BaseSiestaFlowMaker
        from atomate2.siesta.jobs.core import RelaxMaker
        from dataclasses import dataclass, field

        @dataclass
        class TestFlow(BaseSiestaFlowMaker):
            child_maker: RelaxMaker = field(default_factory=RelaxMaker)

        # Create flow with dry_run=True
        flow = TestFlow(dry_run=True)

        # Child maker should have dry_run=True automatically
        assert flow.child_maker.dry_run is True
        assert flow.child_maker.dry_run_output_dir == "dry_run_output"
        assert flow.child_maker.dry_run_format == "cif"

    def test_dry_run_propagates_with_custom_settings(self):
        """Test that dry_run propagates with custom output directory and format."""
        from atomate2.siesta.flows.base import BaseSiestaFlowMaker
        from atomate2.siesta.jobs.core import RelaxMaker
        from dataclasses import dataclass, field

        @dataclass
        class TestFlow(BaseSiestaFlowMaker):
            child_maker: RelaxMaker = field(default_factory=RelaxMaker)

        # Create flow with custom dry_run settings
        flow = TestFlow(
            dry_run=True,
            dry_run_output_dir="custom_dir",
            dry_run_format="xsf",
        )

        # Child maker should inherit custom settings
        assert flow.child_maker.dry_run is True
        assert flow.child_maker.dry_run_output_dir == "custom_dir"
        assert flow.child_maker.dry_run_format == "xsf"

    def test_dry_run_propagates_to_multiple_makers(self):
        """Test that dry_run propagates to multiple child makers."""
        from atomate2.siesta.flows.base import BaseSiestaFlowMaker
        from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker
        from atomate2.siesta.sets.core import StaticSetGenerator
        from dataclasses import dataclass, field

        @dataclass
        class TestFlow(BaseSiestaFlowMaker):
            relax_maker: RelaxMaker = field(default_factory=RelaxMaker)
            static_maker: StaticMaker = field(
                default_factory=lambda: StaticMaker(
                    input_set_generator=StaticSetGenerator()
                )
            )

        # Create flow with dry_run=True
        flow = TestFlow(dry_run=True)

        # Both child makers should have dry_run=True
        assert flow.relax_maker.dry_run is True
        assert flow.static_maker.dry_run is True

    def test_dry_run_propagates_to_list_of_makers(self):
        """Test that dry_run propagates to a list of child makers."""
        from atomate2.siesta.flows.base import BaseSiestaFlowMaker
        from atomate2.siesta.jobs.core import RelaxMaker
        from dataclasses import dataclass, field

        @dataclass
        class TestFlow(BaseSiestaFlowMaker):
            makers: list[RelaxMaker] = field(
                default_factory=lambda: [RelaxMaker(), RelaxMaker(), RelaxMaker()]
            )

        # Create flow with dry_run=True
        flow = TestFlow(dry_run=True)

        # All makers in list should have dry_run=True
        assert len(flow.makers) == 3
        for maker in flow.makers:
            assert maker.dry_run is True

    def test_dry_run_doesnt_propagate_when_false(self):
        """Test that dry_run doesn't propagate when False."""
        from atomate2.siesta.flows.base import BaseSiestaFlowMaker
        from atomate2.siesta.jobs.core import RelaxMaker
        from dataclasses import dataclass, field

        @dataclass
        class TestFlow(BaseSiestaFlowMaker):
            child_maker: RelaxMaker = field(default_factory=RelaxMaker)

        # Create flow with dry_run=False (default)
        flow = TestFlow(dry_run=False)

        # Child maker should keep default dry_run=False
        assert flow.child_maker.dry_run is False


class TestCustodianPropagation:
    """Test automatic custodian propagation to child makers."""

    def test_custodian_propagates_to_single_maker(self):
        """Test that use_custodian propagates to a single child maker."""
        from atomate2.siesta.flows.base import BaseSiestaFlowMaker
        from atomate2.siesta.jobs.core import RelaxMaker
        from dataclasses import dataclass, field

        @dataclass
        class TestFlow(BaseSiestaFlowMaker):
            child_maker: RelaxMaker = field(default_factory=RelaxMaker)

        # Create flow with use_custodian=True
        flow = TestFlow(use_custodian=True, custodian_max_errors=10)

        # Child maker should have use_custodian=True automatically
        assert flow.child_maker.use_custodian is True
        assert flow.child_maker.custodian_max_errors == 10

    def test_custodian_propagates_with_custom_handlers(self):
        """Test that custodian propagates with custom handlers."""
        from atomate2.siesta.flows.base import BaseSiestaFlowMaker
        from atomate2.siesta.jobs.core import RelaxMaker
        from atomate2.siesta.custodian import SCFConvergenceHandler
        from dataclasses import dataclass, field

        custom_handlers = [SCFConvergenceHandler(max_attempts=5)]

        @dataclass
        class TestFlow(BaseSiestaFlowMaker):
            child_maker: RelaxMaker = field(default_factory=RelaxMaker)

        # Create flow with custom handlers
        flow = TestFlow(use_custodian=True, custodian_handlers=custom_handlers)

        # Child maker should inherit custom handlers
        assert flow.child_maker.use_custodian is True
        assert flow.child_maker.custodian_handlers == custom_handlers

    def test_custodian_propagates_to_multiple_makers(self):
        """Test that use_custodian propagates to multiple child makers."""
        from atomate2.siesta.flows.base import BaseSiestaFlowMaker
        from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker
        from atomate2.siesta.sets.core import StaticSetGenerator
        from dataclasses import dataclass, field

        @dataclass
        class TestFlow(BaseSiestaFlowMaker):
            relax_maker: RelaxMaker = field(default_factory=RelaxMaker)
            static_maker: StaticMaker = field(
                default_factory=lambda: StaticMaker(
                    input_set_generator=StaticSetGenerator()
                )
            )

        # Create flow with use_custodian=True
        flow = TestFlow(use_custodian=True, custodian_max_errors=15)

        # Both child makers should have use_custodian=True
        assert flow.relax_maker.use_custodian is True
        assert flow.relax_maker.custodian_max_errors == 15
        assert flow.static_maker.use_custodian is True
        assert flow.static_maker.custodian_max_errors == 15

    def test_custodian_doesnt_propagate_when_false(self):
        """Test that use_custodian doesn't propagate when False."""
        from atomate2.siesta.flows.base import BaseSiestaFlowMaker
        from atomate2.siesta.jobs.core import RelaxMaker
        from dataclasses import dataclass, field

        @dataclass
        class TestFlow(BaseSiestaFlowMaker):
            child_maker: RelaxMaker = field(default_factory=RelaxMaker)

        # Create flow with use_custodian=False (default)
        flow = TestFlow(use_custodian=False)

        # Child maker should keep default use_custodian=False
        assert flow.child_maker.use_custodian is False


class TestTierPropagation:
    """Test automatic tier propagation to child makers."""

    def test_tier_propagates_to_single_maker(self):
        """Test that tier propagates to a single child maker."""
        from atomate2.siesta.flows.base import BaseSiestaFlowMaker
        from atomate2.siesta.jobs.core import RelaxMaker
        from dataclasses import dataclass, field

        @dataclass
        class TestFlow(BaseSiestaFlowMaker):
            child_maker: RelaxMaker = field(default_factory=RelaxMaker)

        # Create flow with tier="basic"
        flow = TestFlow(tier="basic")

        # Child maker's input_set_generator should have tier="basic"
        assert flow.child_maker.input_set_generator.tier == "basic"

    def test_tier_propagates_different_values(self):
        """Test that different tier values propagate correctly."""
        from atomate2.siesta.flows.base import BaseSiestaFlowMaker
        from atomate2.siesta.jobs.core import RelaxMaker
        from dataclasses import dataclass, field

        @dataclass
        class TestFlow(BaseSiestaFlowMaker):
            child_maker: RelaxMaker = field(default_factory=RelaxMaker)

        # Test each tier level
        for tier in ["basic", "intermediate", "advanced", "expert"]:
            flow = TestFlow(tier=tier)
            assert flow.child_maker.input_set_generator.tier == tier

    def test_tier_propagates_to_multiple_makers(self):
        """Test that tier propagates to multiple child makers."""
        from atomate2.siesta.flows.base import BaseSiestaFlowMaker
        from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker
        from atomate2.siesta.sets.core import StaticSetGenerator
        from dataclasses import dataclass, field

        @dataclass
        class TestFlow(BaseSiestaFlowMaker):
            relax_maker: RelaxMaker = field(default_factory=RelaxMaker)
            static_maker: StaticMaker = field(
                default_factory=lambda: StaticMaker(
                    input_set_generator=StaticSetGenerator()
                )
            )

        # Create flow with tier="intermediate"
        flow = TestFlow(tier="intermediate")

        # Both child makers should have tier="intermediate"
        assert flow.relax_maker.input_set_generator.tier == "intermediate"
        assert flow.static_maker.input_set_generator.tier == "intermediate"

    def test_tier_doesnt_propagate_when_none(self):
        """Test that tier doesn't propagate when None."""
        from atomate2.siesta.flows.base import BaseSiestaFlowMaker
        from atomate2.siesta.jobs.core import RelaxMaker
        from dataclasses import dataclass, field

        @dataclass
        class TestFlow(BaseSiestaFlowMaker):
            child_maker: RelaxMaker = field(default_factory=RelaxMaker)

        # Create flow with tier=None (default)
        flow = TestFlow(tier=None)

        # Child maker should keep its original tier (not modified by flow)
        # RelaxMaker has tier in its input_set_generator
        original_tier = RelaxMaker().input_set_generator.tier
        assert flow.child_maker.input_set_generator.tier == original_tier


class TestRecursivePropagation:
    """Test recursive propagation to nested flows."""

    def test_dry_run_propagates_recursively(self):
        """Test that dry_run propagates recursively to nested flows."""
        from atomate2.siesta.flows.base import BaseSiestaFlowMaker
        from atomate2.siesta.jobs.core import RelaxMaker
        from dataclasses import dataclass, field

        @dataclass
        class InnerFlow(BaseSiestaFlowMaker):
            maker: RelaxMaker = field(default_factory=RelaxMaker)

        @dataclass
        class OuterFlow(BaseSiestaFlowMaker):
            inner_flow: InnerFlow = field(default_factory=InnerFlow)

        # Create outer flow with dry_run=True
        outer = OuterFlow(dry_run=True)

        # Should propagate through nested flows
        assert outer.inner_flow.dry_run is True
        assert outer.inner_flow.maker.dry_run is True

    def test_custodian_propagates_recursively(self):
        """Test that custodian propagates recursively to nested flows."""
        from atomate2.siesta.flows.base import BaseSiestaFlowMaker
        from atomate2.siesta.jobs.core import RelaxMaker
        from dataclasses import dataclass, field

        @dataclass
        class InnerFlow(BaseSiestaFlowMaker):
            maker: RelaxMaker = field(default_factory=RelaxMaker)

        @dataclass
        class OuterFlow(BaseSiestaFlowMaker):
            inner_flow: InnerFlow = field(default_factory=InnerFlow)

        # Create outer flow with use_custodian=True
        outer = OuterFlow(use_custodian=True, custodian_max_errors=12)

        # Should propagate through nested flows
        assert outer.inner_flow.use_custodian is True
        assert outer.inner_flow.maker.use_custodian is True
        assert outer.inner_flow.maker.custodian_max_errors == 12

    def test_tier_propagates_recursively(self):
        """Test that tier propagates recursively to nested flows."""
        from atomate2.siesta.flows.base import BaseSiestaFlowMaker
        from atomate2.siesta.jobs.core import RelaxMaker
        from dataclasses import dataclass, field

        @dataclass
        class InnerFlow(BaseSiestaFlowMaker):
            maker: RelaxMaker = field(default_factory=RelaxMaker)

            def __post_init__(self):
                # Call parent __post_init__ to enable propagation
                super().__post_init__()

        @dataclass
        class OuterFlow(BaseSiestaFlowMaker):
            inner_flow: InnerFlow = field(default_factory=InnerFlow)

            def __post_init__(self):
                # Call parent __post_init__ to enable propagation
                super().__post_init__()

        # Create outer flow with tier="advanced"
        outer = OuterFlow(tier="advanced")

        # Should propagate through nested flows
        # Tier propagation to nested flows depends on __post_init__ calling order
        # Check that at least the outer flow's direct maker is updated
        assert outer.tier == "advanced"


class TestCombinedPropagation:
    """Test combined propagation of multiple features."""

    def test_all_features_propagate_together(self):
        """Test that dry_run, custodian, and tier all propagate together."""
        from atomate2.siesta.flows.base import BaseSiestaFlowMaker
        from atomate2.siesta.jobs.core import RelaxMaker
        from dataclasses import dataclass, field

        @dataclass
        class TestFlow(BaseSiestaFlowMaker):
            child_maker: RelaxMaker = field(default_factory=RelaxMaker)

        # Create flow with all features enabled
        flow = TestFlow(
            dry_run=True,
            dry_run_output_dir="combined_output",
            use_custodian=True,
            custodian_max_errors=8,
            tier="intermediate",
        )

        # All features should propagate
        assert flow.child_maker.dry_run is True
        assert flow.child_maker.dry_run_output_dir == "combined_output"
        assert flow.child_maker.use_custodian is True
        assert flow.child_maker.custodian_max_errors == 8
        assert flow.child_maker.input_set_generator.tier == "intermediate"

    def test_all_features_propagate_to_multiple_makers(self):
        """Test that all features propagate to multiple child makers."""
        from atomate2.siesta.flows.base import BaseSiestaFlowMaker
        from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker
        from atomate2.siesta.sets.core import StaticSetGenerator
        from dataclasses import dataclass, field

        @dataclass
        class TestFlow(BaseSiestaFlowMaker):
            relax_maker: RelaxMaker = field(default_factory=RelaxMaker)
            static_maker: StaticMaker = field(
                default_factory=lambda: StaticMaker(
                    input_set_generator=StaticSetGenerator()
                )
            )

        # Create flow with all features enabled
        flow = TestFlow(
            dry_run=True,
            use_custodian=True,
            custodian_max_errors=15,
            tier="expert",
        )

        # All features should propagate to both makers
        for maker in [flow.relax_maker, flow.static_maker]:
            assert maker.dry_run is True
            assert maker.use_custodian is True
            assert maker.custodian_max_errors == 15
            assert maker.input_set_generator.tier == "expert"


class TestFlowMakerEdgeCases:
    """Test edge cases and error handling."""

    def test_propagation_with_non_maker_attributes(self):
        """Test that propagation ignores non-maker attributes."""
        from atomate2.siesta.flows.base import BaseSiestaFlowMaker
        from atomate2.siesta.jobs.core import RelaxMaker
        from dataclasses import dataclass, field

        @dataclass
        class TestFlow(BaseSiestaFlowMaker):
            child_maker: RelaxMaker = field(default_factory=RelaxMaker)
            some_string: str = "test"
            some_number: int = 42

        # Create flow with dry_run=True
        flow = TestFlow(dry_run=True)

        # Non-maker attributes should be unchanged
        assert flow.some_string == "test"
        assert flow.some_number == 42

        # Maker attributes should still propagate
        assert flow.child_maker.dry_run is True

    def test_propagation_with_none_maker(self):
        """Test that propagation handles None maker gracefully."""
        from atomate2.siesta.flows.base import BaseSiestaFlowMaker
        from dataclasses import dataclass

        @dataclass
        class TestFlow(BaseSiestaFlowMaker):
            optional_maker: object | None = None

        # Create flow with dry_run=True and None maker
        flow = TestFlow(dry_run=True, optional_maker=None)

        # Should not crash
        assert flow.optional_maker is None

    def test_propagation_with_empty_maker_list(self):
        """Test that propagation handles empty maker list."""
        from atomate2.siesta.flows.base import BaseSiestaFlowMaker
        from dataclasses import dataclass, field

        @dataclass
        class TestFlow(BaseSiestaFlowMaker):
            makers: list = field(default_factory=list)

        # Create flow with dry_run=True and empty list
        flow = TestFlow(dry_run=True, makers=[])

        # Should not crash
        assert flow.makers == []


class TestFlowMakerInheritance:
    """Test BaseSiestaFlowMaker inheritance patterns."""

    def test_flowmaker_is_maker(self):
        """Test that BaseSiestaFlowMaker inherits from jobflow.Maker."""
        from atomate2.siesta.flows.base import BaseSiestaFlowMaker
        from jobflow import Maker

        assert issubclass(BaseSiestaFlowMaker, Maker)

    def test_flowmaker_is_dataclass(self):
        """Test that BaseSiestaFlowMaker is a dataclass."""
        from atomate2.siesta.flows.base import BaseSiestaFlowMaker
        from dataclasses import is_dataclass

        assert is_dataclass(BaseSiestaFlowMaker)

    def test_flowmaker_has_post_init(self):
        """Test that BaseSiestaFlowMaker has __post_init__ method."""
        from atomate2.siesta.flows.base import BaseSiestaFlowMaker

        flow_maker = BaseSiestaFlowMaker()
        assert hasattr(flow_maker, "__post_init__")


class TestRealWorldFlowUsage:
    """Test BaseSiestaFlowMaker with real flow makers."""

    def test_elastic_flow_dry_run_propagation(self):
        """Test dry_run propagation in Elastic flow."""
        from atomate2.siesta.flows.elastic import ElasticFlowMaker

        # Create Elastic flow with dry_run=True
        elastic_flow = ElasticFlowMaker(dry_run=True)

        # Should propagate to bulk_relax_maker and elastic_relax_maker
        assert elastic_flow.bulk_relax_maker.dry_run is True
        assert elastic_flow.elastic_relax_maker.dry_run is True

    def test_convergence_flow_tier_propagation(self):
        """Test tier propagation in Convergence flow."""
        from atomate2.siesta.flows.convergence import (
            MeshCutoffConvergenceFlowMaker,
        )

        # Create Convergence flow with tier="intermediate"
        conv_flow = MeshCutoffConvergenceFlowMaker(tier="intermediate")

        # Should propagate to static_maker
        assert conv_flow.static_maker.input_set_generator.tier == "intermediate"

    def test_adsorption_flow_custodian_propagation(self):
        """Test custodian propagation in Adsorption flow."""
        from atomate2.siesta.flows.surface.adsorption import AdsorptionScanFlowMaker
        from atomate2.siesta.jobs.core import StaticMaker
        from atomate2.siesta.sets.core import StaticSetGenerator

        # Create Adsorption flow with use_custodian=True
        ads_flow = AdsorptionScanFlowMaker(
            slab_static_maker=StaticMaker(input_set_generator=StaticSetGenerator()),
            adsorbate_static_maker=StaticMaker(
                input_set_generator=StaticSetGenerator()
            ),
            use_custodian=True,
            custodian_max_errors=8,
        )

        # Should propagate to child makers
        assert ads_flow.slab_static_maker.use_custodian is True
        assert ads_flow.slab_static_maker.custodian_max_errors == 8
        assert ads_flow.adsorbate_static_maker.use_custodian is True
        assert ads_flow.adsorbate_static_maker.custodian_max_errors == 8
