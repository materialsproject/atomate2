"""
Tests for elastic constants workflows.

These tests validate:
- ElasticFlowMaker (elastic tensor calculation)
- Flow composition and chaining
- Parameter handling
- Serialization
"""

from jobflow import Flow

from atomate2.siesta.flows.elastic import ElasticFlowMaker, save_elastic_results_job
from atomate2.siesta.jobs.core import StaticMaker, RelaxMaker


class TestElasticMaker:
    """Tests for ElasticFlowMaker workflow."""

    def test_default_elastic_maker(self):
        """Test creation of default ElasticFlowMaker."""
        maker = ElasticFlowMaker()

        assert maker.name == "elastic"
        assert isinstance(maker.bulk_relax_maker, RelaxMaker)
        assert isinstance(maker.elastic_relax_maker, StaticMaker)
        assert maker.order == 2
        assert maker.sym_reduce is True
        assert maker.symprec == 1e-5

    def test_elastic_maker_with_custom_params(self):
        """Test ElasticFlowMaker with custom parameters."""
        bulk_relax = RelaxMaker.variable_cell_relaxation()
        elastic_relax = StaticMaker()

        maker = ElasticFlowMaker(
            name="custom_elastic",
            bulk_relax_maker=bulk_relax,
            elastic_relax_maker=elastic_relax,
            order=3,
            sym_reduce=False,
            symprec=1e-6,
        )

        assert maker.name == "custom_elastic"
        assert maker.bulk_relax_maker == bulk_relax
        assert maker.elastic_relax_maker == elastic_relax
        assert maker.order == 3
        assert maker.sym_reduce is False
        assert maker.symprec == 1e-6

    def test_elastic_maker_without_bulk_relax(self):
        """Test ElasticFlowMaker without initial bulk relaxation."""
        maker = ElasticFlowMaker(bulk_relax_maker=None)

        assert maker.bulk_relax_maker is None
        assert maker.elastic_relax_maker is not None

    def test_elastic_maker_order_2(self):
        """Test ElasticFlowMaker with order 2 (standard)."""
        maker = ElasticFlowMaker(order=2)
        assert maker.order == 2

    def test_elastic_maker_order_3(self):
        """Test ElasticFlowMaker with order 3 (third-order constants)."""
        maker = ElasticFlowMaker(order=3)
        assert maker.order == 3

    def test_elastic_maker_symmetry_reduction(self):
        """Test symmetry reduction parameters."""
        maker_with_sym = ElasticFlowMaker(sym_reduce=True, symprec=1e-5)
        assert maker_with_sym.sym_reduce is True
        assert maker_with_sym.symprec == 1e-5

        maker_no_sym = ElasticFlowMaker(sym_reduce=False)
        assert maker_no_sym.sym_reduce is False

    def test_elastic_maker_make_flow(self, si_structure):
        """Test that ElasticFlowMaker creates a valid flow."""
        maker = ElasticFlowMaker()
        flow = maker.make(si_structure)

        assert isinstance(flow, Flow)
        assert flow.name == "elastic"
        # Should have at least bulk relax + deformations + fit + save jobs
        assert len(flow) > 1

    def test_elastic_maker_stress_sign_correction(self):
        """Test stress sign correction property."""
        maker = ElasticFlowMaker()
        assert maker.stress_sign_correction == -1000.0

    def test_elastic_maker_prev_calc_dir_argname(self):
        """Test previous calculation directory argument name."""
        maker = ElasticFlowMaker()
        assert maker.prev_calc_dir_argname == "prev_dir"

    def test_elastic_maker_serialization(self):
        """Test ElasticFlowMaker serialization."""
        maker = ElasticFlowMaker(
            name="test_elastic",
            order=3,
            sym_reduce=False,
        )

        # Serialize
        maker_dict = maker.as_dict()
        assert isinstance(maker_dict, dict)
        assert maker_dict["@class"] == "ElasticFlowMaker"

        # Deserialize
        maker_restored = maker.from_dict(maker_dict)
        assert isinstance(maker_restored, ElasticFlowMaker)
        assert maker_restored.name == "test_elastic"
        assert maker_restored.order == 3
        assert maker_restored.sym_reduce is False


class TestElasticMakerCustomization:
    """Test customization options for ElasticFlowMaker."""

    def test_elastic_with_custom_bulk_relax_maker(self, si_structure):
        """Test ElasticFlowMaker with custom bulk relaxation maker."""
        custom_bulk = RelaxMaker.variable_cell_relaxation()

        maker = ElasticFlowMaker(bulk_relax_maker=custom_bulk)
        flow = maker.make(si_structure)

        assert isinstance(flow, Flow)
        assert maker.bulk_relax_maker == custom_bulk

    def test_elastic_with_custom_elastic_relax_maker(self, si_structure):
        """Test ElasticFlowMaker with custom elastic relaxation maker."""
        custom_elastic = StaticMaker()

        maker = ElasticFlowMaker(elastic_relax_maker=custom_elastic)
        flow = maker.make(si_structure)

        assert isinstance(flow, Flow)
        assert maker.elastic_relax_maker == custom_elastic

    def test_elastic_with_both_custom_makers(self, si_structure):
        """Test ElasticFlowMaker with both custom makers."""
        custom_bulk = RelaxMaker.variable_cell_relaxation()
        custom_elastic = StaticMaker()

        maker = ElasticFlowMaker(
            bulk_relax_maker=custom_bulk,
            elastic_relax_maker=custom_elastic,
        )

        flow = maker.make(si_structure)
        assert isinstance(flow, Flow)

    def test_elastic_with_max_failed_deformations(self, si_structure):
        """Test ElasticFlowMaker with max_failed_deformations parameter."""
        # Test with int
        maker1 = ElasticFlowMaker(max_failed_deformations=2)
        flow1 = maker1.make(si_structure)
        assert isinstance(flow1, Flow)

        # Test with float
        maker2 = ElasticFlowMaker(max_failed_deformations=0.1)
        flow2 = maker2.make(si_structure)
        assert isinstance(flow2, Flow)

        # Test with None
        maker3 = ElasticFlowMaker(max_failed_deformations=None)
        flow3 = maker3.make(si_structure)
        assert isinstance(flow3, Flow)

    def test_elastic_with_deformation_kwargs(self, si_structure):
        """Test ElasticFlowMaker with custom deformation generation kwargs."""
        maker = ElasticFlowMaker(
            generate_elastic_deformations_kwargs={"norm_strains": [0.01, 0.02]}
        )

        flow = maker.make(si_structure)
        assert isinstance(flow, Flow)

    def test_elastic_with_fitting_kwargs(self, si_structure):
        """Test ElasticFlowMaker with custom fitting kwargs."""
        maker = ElasticFlowMaker(
            fit_elastic_tensor_kwargs={"fitting_method": "pseudoinverse"}
        )

        flow = maker.make(si_structure)
        assert isinstance(flow, Flow)

    def test_elastic_with_task_document_kwargs(self, si_structure):
        """Test ElasticFlowMaker with custom task document kwargs."""
        maker = ElasticFlowMaker(
            task_document_kwargs={"fitting_method": "pseudoinverse"}
        )

        flow = maker.make(si_structure)
        assert isinstance(flow, Flow)


class TestElasticFlowIntegration:
    """Integration tests for elastic workflows."""

    def test_elastic_flow_with_different_structures(self, si_structure, al_structure):
        """Test elastic maker works with different structures."""
        structures = [si_structure, al_structure]
        maker = ElasticFlowMaker()

        for structure in structures:
            flow = maker.make(structure)
            assert isinstance(flow, Flow)

    def test_elastic_flow_output_references(self, si_structure):
        """Test that flows have proper output handling."""
        maker = ElasticFlowMaker()
        flow = maker.make(si_structure)

        # Flow should be iterable
        jobs = list(flow)
        assert len(jobs) > 0

        # Each job/flow should have a name
        for item in jobs:
            assert hasattr(item, "name")
            # Check if it's a Job (has function) or Flow (has jobs)
            assert (
                hasattr(item, "function")
                or hasattr(item, "jobs")
                or isinstance(item, Flow)
            )

    def test_elastic_flow_with_none_prev_dir(self, si_structure):
        """Test elastic flow with None as prev_dir."""
        maker = ElasticFlowMaker()

        flow = maker.make(si_structure, prev_dir=None)
        assert isinstance(flow, Flow)

    def test_multiple_flows_from_same_maker(self, si_structure, al_structure):
        """Test creating multiple flows from the same maker."""
        maker = ElasticFlowMaker()

        flow1 = maker.make(si_structure)
        flow2 = maker.make(al_structure)

        # Flows should be independent
        assert flow1 is not flow2
        assert isinstance(flow1, Flow)
        assert isinstance(flow2, Flow)


class TestElasticEdgeCases:
    """Test edge cases and error handling."""

    def test_elastic_with_different_symmetry_settings(self, si_structure):
        """Test elastic with different symmetry settings."""
        settings = [
            {"sym_reduce": True, "symprec": 1e-5},
            {"sym_reduce": True, "symprec": 1e-6},
            {"sym_reduce": False, "symprec": 1e-5},
        ]

        for setting in settings:
            maker = ElasticFlowMaker(**setting)
            flow = maker.make(si_structure)
            assert isinstance(flow, Flow)

    def test_elastic_with_different_orders(self, si_structure):
        """Test elastic with different tensor orders."""
        for order in [2, 3]:
            maker = ElasticFlowMaker(order=order)
            flow = maker.make(si_structure)
            assert isinstance(flow, Flow)
            assert maker.order == order

    def test_maker_modification_doesnt_affect_flows(self, si_structure):
        """Test that modifying maker after flow creation doesn't affect flow."""
        maker = ElasticFlowMaker(order=2)
        flow1 = maker.make(si_structure)

        # Modify maker
        maker.order = 3

        flow2 = maker.make(si_structure)

        # Flows should be independent
        assert flow1 is not flow2


class TestElasticParameterValidation:
    """Test parameter validation for elastic makers."""

    def test_valid_order_values(self, si_structure):
        """Test that valid order values are accepted."""
        for order in [2, 3]:
            maker = ElasticFlowMaker(order=order)
            flow = maker.make(si_structure)
            assert isinstance(flow, Flow)

    def test_valid_symprec_values(self, si_structure):
        """Test different symprec values."""
        symprec_values = [1e-3, 1e-4, 1e-5, 1e-6]

        for symprec in symprec_values:
            maker = ElasticFlowMaker(symprec=symprec)
            flow = maker.make(si_structure)
            assert isinstance(flow, Flow)

    def test_valid_sym_reduce_values(self, si_structure):
        """Test sym_reduce boolean values."""
        for sym_reduce in [True, False]:
            maker = ElasticFlowMaker(sym_reduce=sym_reduce)
            flow = maker.make(si_structure)
            assert isinstance(flow, Flow)


class TestElasticMakerSerialization:
    """Test serialization of ElasticFlowMaker."""

    def test_elastic_maker_serializable(self):
        """Test that ElasticFlowMaker can be serialized and deserialized."""
        maker = ElasticFlowMaker()

        # Serialize
        maker_dict = maker.as_dict()
        assert isinstance(maker_dict, dict)

        # Deserialize
        maker_restored = maker.from_dict(maker_dict)
        assert maker_restored is not None
        assert isinstance(maker_restored, ElasticFlowMaker)

    def test_elastic_serialization_preserves_parameters(self):
        """Test that serialization preserves all parameters."""
        maker = ElasticFlowMaker(
            name="custom_elastic",
            order=3,
            sym_reduce=False,
            symprec=1e-6,
        )

        # Serialize and deserialize
        maker_dict = maker.as_dict()
        maker_restored = maker.from_dict(maker_dict)

        # Check all parameters preserved
        assert maker_restored.name == "custom_elastic"
        assert maker_restored.order == 3
        assert maker_restored.sym_reduce is False
        assert maker_restored.symprec == 1e-6

    def test_elastic_serialization_with_custom_makers(self):
        """Test serialization with custom sub-makers."""
        custom_bulk = RelaxMaker.variable_cell_relaxation()
        custom_elastic = StaticMaker()

        maker = ElasticFlowMaker(
            bulk_relax_maker=custom_bulk,
            elastic_relax_maker=custom_elastic,
        )

        # Serialize and deserialize
        maker_dict = maker.as_dict()
        maker_restored = maker.from_dict(maker_dict)

        assert isinstance(maker_restored, ElasticFlowMaker)
        assert isinstance(maker_restored.bulk_relax_maker, RelaxMaker)
        assert isinstance(maker_restored.elastic_relax_maker, StaticMaker)


class TestElasticFlowComposition:
    """Test flow composition for elastic workflows."""

    def test_elastic_flow_structure(self, si_structure):
        """Test the structure of elastic flows."""
        maker = ElasticFlowMaker()
        flow = maker.make(si_structure)

        jobs = list(flow)

        # Should have bulk relax + deformations + fit + save
        assert len(jobs) >= 2

    def test_elastic_with_custom_name(self, si_structure):
        """Test creating elastic flows with custom names."""
        maker = ElasticFlowMaker(name="My Custom Elastic")

        flow = maker.make(si_structure)
        assert flow.name == "My Custom Elastic"

    def test_elastic_flow_without_bulk_relax(self, si_structure):
        """Test elastic flow structure without bulk relaxation."""
        maker = ElasticFlowMaker(bulk_relax_maker=None)

        flow = maker.make(si_structure)
        jobs = list(flow)

        # Should not have initial bulk relaxation job
        assert len(jobs) >= 1

    def test_elastic_flow_has_save_job(self, si_structure):
        """Test that elastic flow includes save results job."""
        maker = ElasticFlowMaker()
        flow = maker.make(si_structure)

        job_names = [job.name for job in flow]

        # Should have save_results job
        assert any("save" in name.lower() for name in job_names)


class TestElasticMakerProperties:
    """Test ElasticFlowMaker properties and methods."""

    def test_prev_calc_dir_argname_property(self):
        """Test prev_calc_dir_argname property."""
        maker = ElasticFlowMaker()
        assert maker.prev_calc_dir_argname == "prev_dir"

    def test_stress_sign_correction_property(self):
        """Test stress_sign_correction property."""
        maker = ElasticFlowMaker()
        # SIESTA stress correction factor
        assert maker.stress_sign_correction == -1000.0
        assert isinstance(maker.stress_sign_correction, float)

    def test_maker_has_required_attributes(self):
        """Test that maker has all required attributes."""
        maker = ElasticFlowMaker()

        required_attrs = [
            "name",
            "order",
            "sym_reduce",
            "symprec",
            "bulk_relax_maker",
            "elastic_relax_maker",
            "max_failed_deformations",
            "generate_elastic_deformations_kwargs",
            "fit_elastic_tensor_kwargs",
            "task_document_kwargs",
        ]

        for attr in required_attrs:
            assert hasattr(maker, attr)


class TestElasticWorkflowOptions:
    """Test various workflow configuration options."""

    def test_elastic_with_different_maker_combinations(self, si_structure):
        """Test elastic with different combinations of makers."""
        combinations = [
            {
                "bulk_relax_maker": RelaxMaker.variable_cell_relaxation(),
                "elastic_relax_maker": StaticMaker(),
            },
            {
                "bulk_relax_maker": None,
                "elastic_relax_maker": StaticMaker(),
            },
            {
                "bulk_relax_maker": RelaxMaker.variable_cell_relaxation(),
                "elastic_relax_maker": RelaxMaker.fixed_cell_relaxation(),
            },
        ]

        for combo in combinations:
            maker = ElasticFlowMaker(**combo)
            flow = maker.make(si_structure)
            assert isinstance(flow, Flow)

    def test_elastic_with_all_custom_options(self, si_structure):
        """Test elastic with all custom options."""
        maker = ElasticFlowMaker(
            name="fully_custom_elastic",
            order=3,
            sym_reduce=False,
            symprec=1e-6,
            bulk_relax_maker=RelaxMaker.variable_cell_relaxation(),
            elastic_relax_maker=StaticMaker(),
            max_failed_deformations=2,
            generate_elastic_deformations_kwargs={"norm_strains": [0.01]},
            fit_elastic_tensor_kwargs={"fitting_method": "pseudoinverse"},
            task_document_kwargs={},
        )

        flow = maker.make(si_structure)
        assert isinstance(flow, Flow)
        assert flow.name == "fully_custom_elastic"


class TestElasticDryRunMode:
    """Test dry-run mode for elastic workflows."""

    def test_elastic_maker_dry_run_initialization(self):
        """Test ElasticFlowMaker dry_run initialization."""
        maker = ElasticFlowMaker(
            dry_run=True,
            dry_run_output_dir="elastic_dry_run",
            dry_run_format="xsf",
        )

        assert maker.dry_run is True
        assert maker.dry_run_output_dir == "elastic_dry_run"
        assert maker.dry_run_format == "xsf"

    def test_elastic_maker_dry_run_propagation(self):
        """Test that dry_run propagates to child makers."""
        maker = ElasticFlowMaker(
            dry_run=True,
            dry_run_output_dir="test_dry_run",
            dry_run_format="cif",
        )

        # Check propagation via __post_init__
        assert maker.bulk_relax_maker.dry_run is True
        assert maker.bulk_relax_maker.dry_run_output_dir == "test_dry_run"
        assert maker.bulk_relax_maker.dry_run_format == "cif"

        assert maker.elastic_relax_maker.dry_run is True
        assert maker.elastic_relax_maker.dry_run_output_dir == "test_dry_run"
        assert maker.elastic_relax_maker.dry_run_format == "cif"

    def test_elastic_maker_dry_run_flow_creation(self, si_structure):
        """Test that dry_run mode creates valid flows."""
        maker = ElasticFlowMaker(
            dry_run=True,
            dry_run_output_dir="elastic_dry",
        )

        flow = maker.make(si_structure)
        assert isinstance(flow, Flow)

    def test_elastic_dry_run_with_no_bulk_relax(self):
        """Test dry_run with no bulk relaxation maker."""
        maker = ElasticFlowMaker(
            bulk_relax_maker=None,
            dry_run=True,
            dry_run_output_dir="no_bulk_dry",
        )

        # Should not fail even without bulk_relax_maker
        assert maker.dry_run is True
        assert maker.bulk_relax_maker is None

    def test_dry_run_different_formats(self):
        """Test dry_run with different output formats."""
        formats = ["cif", "xsf", "json", "poscar"]

        for fmt in formats:
            maker = ElasticFlowMaker(
                dry_run=True,
                dry_run_format=fmt,
            )
            assert maker.dry_run_format == fmt

    def test_dry_run_with_order_3(self, si_structure):
        """Test dry_run with third-order elastic constants."""
        maker = ElasticFlowMaker(
            order=3,
            dry_run=True,
            dry_run_output_dir="order3_dry",
        )

        flow = maker.make(si_structure)
        assert isinstance(flow, Flow)
        assert maker.order == 3


class TestElasticWithMolecules:
    """Test elastic workflows with molecular systems."""

    def test_elastic_maker_with_molecule(self, h2o_structure):
        """Test ElasticFlowMaker with molecule input."""
        maker = ElasticFlowMaker()

        flow = maker.make(h2o_structure)
        assert isinstance(flow, Flow)

    def test_elastic_without_bulk_relax_molecule(self, h2o_structure):
        """Test molecule elastic without bulk relaxation."""
        maker = ElasticFlowMaker(bulk_relax_maker=None)

        flow = maker.make(h2o_structure)
        assert isinstance(flow, Flow)


class TestElasticWorkflowDeformations:
    """Test deformation-related parameters."""

    def test_elastic_with_custom_strain_values(self, si_structure):
        """Test elastic with custom strain magnitudes."""
        maker = ElasticFlowMaker(
            generate_elastic_deformations_kwargs={"norm_strains": [0.01, 0.02]}
        )

        flow = maker.make(si_structure)
        assert isinstance(flow, Flow)

    def test_elastic_with_shear_deformations_only(self, si_structure):
        """Test elastic with only shear deformations."""
        maker = ElasticFlowMaker(
            generate_elastic_deformations_kwargs={
                "norm_strains": [0.01],
                "shear_strains": [0.02],
            }
        )

        flow = maker.make(si_structure)
        assert isinstance(flow, Flow)

    def test_elastic_with_max_failed_deformations(self, si_structure):
        """Test elastic with maximum failed deformations limit."""
        maker = ElasticFlowMaker(max_failed_deformations=5)

        flow = maker.make(si_structure)
        assert isinstance(flow, Flow)
        assert maker.max_failed_deformations == 5


class TestElasticFittingOptions:
    """Test tensor fitting options."""

    def test_elastic_with_pseudoinverse_fitting(self, si_structure):
        """Test elastic with pseudoinverse fitting method."""
        maker = ElasticFlowMaker(
            fit_elastic_tensor_kwargs={"fitting_method": "pseudoinverse"}
        )

        flow = maker.make(si_structure)
        assert isinstance(flow, Flow)

    def test_elastic_with_independent_fitting(self, si_structure):
        """Test elastic with independent fitting method."""
        maker = ElasticFlowMaker(
            fit_elastic_tensor_kwargs={"fitting_method": "independent"}
        )

        flow = maker.make(si_structure)
        assert isinstance(flow, Flow)


class TestElasticSymmetryHandling:
    """Test symmetry handling in elastic workflows."""

    def test_elastic_with_symmetry_reduction(self, si_structure):
        """Test elastic with symmetry reduction enabled."""
        maker = ElasticFlowMaker(sym_reduce=True, symprec=1e-5)

        flow = maker.make(si_structure)
        assert isinstance(flow, Flow)
        assert maker.sym_reduce is True

    def test_elastic_without_symmetry_reduction(self, si_structure):
        """Test elastic without symmetry reduction."""
        maker = ElasticFlowMaker(sym_reduce=False)

        flow = maker.make(si_structure)
        assert isinstance(flow, Flow)
        assert maker.sym_reduce is False

    def test_elastic_with_tight_symmetry_tolerance(self, si_structure):
        """Test elastic with tight symmetry tolerance."""
        maker = ElasticFlowMaker(sym_reduce=True, symprec=1e-7)

        flow = maker.make(si_structure)
        assert isinstance(flow, Flow)
        assert maker.symprec == 1e-7

    def test_elastic_with_loose_symmetry_tolerance(self, si_structure):
        """Test elastic with loose symmetry tolerance."""
        maker = ElasticFlowMaker(sym_reduce=True, symprec=1e-3)

        flow = maker.make(si_structure)
        assert isinstance(flow, Flow)
        assert maker.symprec == 1e-3


class TestSaveElasticResultsJob:
    """Test save_elastic_results_job() function."""

    def test_save_with_dict_elastic_doc(self, tmp_path):
        """Test saving elastic results from dictionary."""
        elastic_doc = {
            "formula_pretty": "Si",
            "elastic_tensor": {
                "ieee_format": [
                    [165.0, 64.0, 64.0, 0.0, 0.0, 0.0],
                    [64.0, 165.0, 64.0, 0.0, 0.0, 0.0],
                    [64.0, 64.0, 165.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 79.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 79.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 79.0],
                ]
            },
            "derived_properties": {
                "k_vrh": 97.67,
                "g_vrh": 66.53,
                "y_mod": 163.0,
                "homogeneous_poisson": 0.22,
            },
            "symmetry": {"crystal_system": "cubic", "symbol": "Fd-3m", "number": 227},
            "fitting_method": "finite_difference",
            "order": 2,
            "nsites": 2,
            "volume": 40.0,
            "density": 2.33,
        }

        result = save_elastic_results_job.original(elastic_doc, str(tmp_path))

        # Check return structure
        assert isinstance(result, dict)
        assert "json_file" in result
        assert "txt_file" in result

        # Check files exist
        from pathlib import Path

        assert Path(result["json_file"]).exists()
        assert Path(result["txt_file"]).exists()

    def test_save_creates_json_file(self, tmp_path):
        """Test that JSON file is created with correct structure."""
        elastic_doc = {
            "formula_pretty": "Al",
            "elastic_tensor": {"ieee_format": [[70.0] * 6] * 6},
            "derived_properties": {"k_vrh": 76.0, "g_vrh": 26.0},
            "symmetry": {"crystal_system": "cubic"},
        }

        result = save_elastic_results_job.original(elastic_doc, str(tmp_path))

        import json

        with open(result["json_file"]) as f:
            data = json.load(f)

        # Check JSON structure
        assert "metadata" in data
        assert "mechanical_properties" in data
        assert "elastic_tensor_ieee_GPa" in data
        assert data["metadata"]["formula"] == "Al"

    def test_save_creates_txt_summary(self, tmp_path):
        """Test that TXT summary file is created."""
        elastic_doc = {
            "formula_pretty": "Fe",
            "elastic_tensor": {"ieee_format": [[200.0] * 6] * 6},
            "derived_properties": {},
            "symmetry": {"crystal_system": "bcc"},
        }

        result = save_elastic_results_job.original(elastic_doc, str(tmp_path))

        from pathlib import Path

        content = Path(result["txt_file"]).read_text()

        # Check TXT content
        assert "ELASTIC CONSTANTS CALCULATION RESULTS" in content
        assert "Fe" in content
        assert "ELASTIC TENSOR" in content

    def test_save_with_negative_constants_warning(self, tmp_path):
        """Test detection of negative elastic constants."""
        elastic_doc = {
            "formula_pretty": "TestMaterial",
            "elastic_tensor": {
                "ieee_format": [
                    [-50.0, 10.0, 10.0, 0.0, 0.0, 0.0],  # Negative diagonal
                    [10.0, 100.0, 10.0, 0.0, 0.0, 0.0],
                    [10.0, 10.0, 100.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 50.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 50.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 50.0],
                ]
            },
            "derived_properties": {},
            "symmetry": {},
        }

        result = save_elastic_results_job.original(elastic_doc, str(tmp_path))

        from pathlib import Path

        content = Path(result["txt_file"]).read_text()

        # Should contain warning about negative constants
        assert "WARNING" in content or "negative" in content.lower()

    def test_save_with_all_positive_constants(self, tmp_path):
        """Test with all positive elastic constants (no warning)."""
        elastic_doc = {
            "formula_pretty": "Diamond",
            "elastic_tensor": {
                "ieee_format": [
                    [1076.0, 125.0, 125.0, 0.0, 0.0, 0.0],
                    [125.0, 1076.0, 125.0, 0.0, 0.0, 0.0],
                    [125.0, 125.0, 1076.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 578.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 578.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 578.0],
                ]
            },
            "derived_properties": {"k_vrh": 442.0, "g_vrh": 535.0},
            "symmetry": {},
        }

        result = save_elastic_results_job.original(elastic_doc, str(tmp_path))

        # Should complete successfully
        from pathlib import Path

        assert Path(result["json_file"]).exists()
        assert Path(result["txt_file"]).exists()

    def test_save_with_custom_output_folder(self, tmp_path):
        """Test saving to custom output folder."""
        output_folder = tmp_path / "custom_results"
        elastic_doc = {
            "formula_pretty": "Cu",
            "elastic_tensor": {"ieee_format": [[170.0] * 6] * 6},
            "derived_properties": {},
            "symmetry": {},
        }

        result = save_elastic_results_job.original(elastic_doc, str(output_folder))

        # Check files created in custom folder
        from pathlib import Path

        assert Path(result["json_file"]).parent == output_folder
        assert Path(result["txt_file"]).parent == output_folder
        assert output_folder.exists()

    def test_save_with_minimal_elastic_doc(self, tmp_path):
        """Test with minimal elastic document (missing optional fields)."""
        elastic_doc = {
            "formula_pretty": "Unknown",
            "elastic_tensor": {"ieee_format": []},
            "derived_properties": {},
            "symmetry": {},
        }

        result = save_elastic_results_job.original(elastic_doc, str(tmp_path))

        # Should still create files
        from pathlib import Path

        assert Path(result["json_file"]).exists()
        assert Path(result["txt_file"]).exists()

    def test_save_extracts_all_properties(self, tmp_path):
        """Test that all mechanical properties are extracted."""
        elastic_doc = {
            "formula_pretty": "GaN",
            "elastic_tensor": {"ieee_format": [[350.0] * 6] * 6},
            "derived_properties": {
                "k_vrh": 200.0,
                "k_voigt": 205.0,
                "k_reuss": 195.0,
                "g_vrh": 150.0,
                "g_voigt": 155.0,
                "g_reuss": 145.0,
                "y_mod": 350.0e9,  # Pa (source converts Pa -> GPa)
                "homogeneous_poisson": 0.25,
                "universal_anisotropy": 1.2,
            },
            "symmetry": {},
        }

        result = save_elastic_results_job.original(elastic_doc, str(tmp_path))

        import json

        with open(result["json_file"]) as f:
            data = json.load(f)

        props = data["mechanical_properties"]
        assert props["bulk_modulus_vrh_GPa"] == 200.0
        assert props["shear_modulus_vrh_GPa"] == 150.0
        assert props["youngs_modulus_GPa"] == 350.0
        assert props["poisson_ratio"] == 0.25

    def test_save_with_equilibrium_stress(self, tmp_path):
        """Test saving with equilibrium stress data."""
        elastic_doc = {
            "formula_pretty": "Ti",
            "elastic_tensor": {"ieee_format": [[160.0] * 6] * 6},
            "derived_properties": {},
            "symmetry": {},
            "eq_stress": [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]],
        }

        result = save_elastic_results_job.original(elastic_doc, str(tmp_path))

        from pathlib import Path

        content = Path(result["txt_file"]).read_text()

        # Should contain equilibrium stress section
        assert "EQUILIBRIUM STRESS" in content

    def test_save_handles_crystal_system_enum(self, tmp_path):
        """Test handling of crystal_system as enum or string."""
        from enum import Enum

        class CrystalSystem(Enum):
            CUBIC = "cubic"

        elastic_doc = {
            "formula_pretty": "NaCl",
            "elastic_tensor": {"ieee_format": [[50.0] * 6] * 6},
            "derived_properties": {},
            "symmetry": {"crystal_system": CrystalSystem.CUBIC},
        }

        result = save_elastic_results_job.original(elastic_doc, str(tmp_path))

        import json

        with open(result["json_file"]) as f:
            data = json.load(f)

        # Should convert enum to string
        assert data["metadata"]["crystal_system"] == "cubic"

    def test_save_with_pydantic_model(self, tmp_path):
        """Test saving with Pydantic model (has model_dump method)."""
        from unittest.mock import MagicMock

        # Create mock Pydantic model
        elastic_doc = MagicMock()
        elastic_doc.model_dump.return_value = {
            "formula_pretty": "MgO",
            "elastic_tensor": {"ieee_format": [[300.0] * 6] * 6},
            "derived_properties": {"k_vrh": 160.0},
            "symmetry": {},
        }

        result = save_elastic_results_job.original(elastic_doc, str(tmp_path))

        # Should call model_dump and create files
        elastic_doc.model_dump.assert_called_once()
        from pathlib import Path

        assert Path(result["json_file"]).exists()

    def test_save_timestamp_in_filenames(self, tmp_path):
        """Test that timestamp is included in filenames."""
        elastic_doc = {
            "formula_pretty": "Test",
            "elastic_tensor": {"ieee_format": []},
            "derived_properties": {},
            "symmetry": {},
        }

        result = save_elastic_results_job.original(elastic_doc, str(tmp_path))

        # Filenames should contain timestamp format (YYYYMMDD_HHMMSS)
        from pathlib import Path

        json_name = Path(result["json_file"]).name
        txt_name = Path(result["txt_file"]).name

        assert "elastic_results_" in json_name
        assert "elastic_summary_" in txt_name
        assert json_name.endswith(".json")
        assert txt_name.endswith(".txt")

    def test_save_tensor_formatting_in_txt(self, tmp_path):
        """Test that elastic tensor is properly formatted in TXT file."""
        elastic_doc = {
            "formula_pretty": "TestFormat",
            "elastic_tensor": {
                "ieee_format": [
                    [100.0, 50.0, 50.0, 0.0, 0.0, 0.0],
                    [50.0, 100.0, 50.0, 0.0, 0.0, 0.0],
                    [50.0, 50.0, 100.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 25.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 25.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 25.0],
                ]
            },
            "derived_properties": {},
            "symmetry": {},
        }

        result = save_elastic_results_job.original(elastic_doc, str(tmp_path))

        from pathlib import Path

        content = Path(result["txt_file"]).read_text()

        # Check tensor is formatted as 6x6 matrix
        assert "[1]" in content and "[6]" in content  # Row/column labels
        assert "100.0000" in content  # Formatted numbers

    def test_save_with_structure_information(self, tmp_path):
        """Test that structure information is included in output."""
        elastic_doc = {
            "formula_pretty": "ZnO",
            "elastic_tensor": {"ieee_format": [[200.0] * 6] * 6},
            "derived_properties": {},
            "symmetry": {},
            "nsites": 4,
            "volume": 47.62,
            "density": 5.61,
        }

        result = save_elastic_results_job.original(elastic_doc, str(tmp_path))

        from pathlib import Path

        content = Path(result["txt_file"]).read_text()

        # Check structure info is present
        assert "STRUCTURE INFORMATION" in content
        assert "47.62" in content  # Volume
        assert "5.61" in content  # Density
        assert "4" in content  # Number of sites
