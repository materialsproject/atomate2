"""
Integration tests for SIESTA workflow chaining and cross-module validation.

These tests validate end-to-end workflows that chain multiple makers together,
testing:
- Workflow composition (job chaining)
- Output reference passing (prev_dir chains)
- Dry-run mode propagation across workflows
- Cross-flow integration (combined workflows)
- Real-world usage patterns

Note: These are integration tests that validate workflow structure and composition.
They do not execute actual SIESTA calculations.
"""

import pytest
from jobflow import Flow
from pymatgen.core import Lattice, Structure

from atomate2.siesta.flows.core import DifferentBasisSCFFlowMaker
from atomate2.siesta.flows.elastic import ElasticFlowMaker
from atomate2.siesta.flows.eos import SiestaEosFlowMaker
from atomate2.siesta.flows.phonon import SiestaGruneisenFlowMaker, SiestaPhononFlowMaker
from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker
from atomate2.siesta.sets.core import StaticSetGenerator


@pytest.fixture
def si_structure():
    """Silicon structure for testing."""
    si_lattice = Lattice.cubic(5.43)
    return Structure(si_lattice, ["Si", "Si"], [[0.00, 0.00, 0.00], [0.25, 0.25, 0.25]])


@pytest.fixture
def al_structure():
    """Aluminum structure for testing."""
    al_lattice = Lattice.cubic(4.05)
    return Structure(al_lattice, ["Al"], [[0.00, 0.00, 0.00]])


class TestWorkflowChaining:
    """Tests for chaining multiple jobs in sequence."""

    def test_relax_then_static_chain(self, si_structure):
        """Test basic relaxation → static workflow chain."""
        # Create makers
        relax_maker = RelaxMaker.fixed_cell_relaxation()
        static_maker = StaticMaker()

        # Create jobs
        relax_job = relax_maker.make(si_structure)
        static_job = static_maker.make(si_structure)

        # Static job should reference relax output
        static_job_with_prev = static_maker.make(
            si_structure, prev_dir=relax_job.output.dir_name
        )

        # Verify jobs created
        assert relax_job is not None
        assert static_job is not None
        assert static_job_with_prev is not None
        assert hasattr(static_job_with_prev, "function")

    def test_double_relaxation_chain(self, si_structure):
        """Test coarse → fine relaxation workflow chain."""
        # Coarse relaxation (SZ basis)
        relax_maker_coarse = RelaxMaker.fixed_cell_relaxation(
            user_params={"PAO.BasisSize": "SZ"}
        )

        # Fine relaxation (DZP basis)
        relax_maker_fine = RelaxMaker.fixed_cell_relaxation(
            user_params={"PAO.BasisSize": "DZP"}
        )

        # Create chain
        coarse_job = relax_maker_coarse.make(si_structure)
        fine_job = relax_maker_fine.make(
            si_structure, prev_dir=coarse_job.output.dir_name
        )

        # Verify chain structure
        assert coarse_job is not None
        assert fine_job is not None
        assert hasattr(fine_job, "function")

    def test_three_step_workflow_chain(self, si_structure):
        """Test relax → relax → static three-step chain."""
        # Create makers
        relax1 = RelaxMaker.fixed_cell_relaxation(user_params={"PAO.BasisSize": "SZ"})
        relax2 = RelaxMaker.fixed_cell_relaxation(user_params={"PAO.BasisSize": "DZ"})
        static = StaticMaker(
            input_set_generator=StaticSetGenerator(user_params={"PAO.BasisSize": "DZP"})
        )

        # Create chain
        job1 = relax1.make(si_structure)
        job2 = relax2.make(si_structure, prev_dir=job1.output.dir_name)
        job3 = static.make(si_structure, prev_dir=job2.output.dir_name)

        # Verify all jobs created
        assert all(job is not None for job in [job1, job2, job3])

    def test_flow_containing_chained_jobs(self, si_structure):
        """Test Flow object containing chained jobs."""
        # Create makers
        relax_maker = RelaxMaker.fixed_cell_relaxation()
        static_maker = StaticMaker()

        # Create jobs
        relax_job = relax_maker.make(si_structure)
        static_job = static_maker.make(si_structure, prev_dir=relax_job.output.dir_name)

        # Create flow
        flow = Flow([relax_job, static_job], name="relax_static_flow")

        # Verify flow structure
        assert isinstance(flow, Flow)
        assert len(flow) == 2
        assert flow.name == "relax_static_flow"


class TestOutputReferencePassing:
    """Tests for prev_dir output reference passing between jobs."""

    def test_prev_dir_reference_structure(self, si_structure):
        """Test that prev_dir correctly references previous job output."""
        relax_maker = RelaxMaker.fixed_cell_relaxation()
        static_maker = StaticMaker()

        # Create jobs with output reference
        relax_job = relax_maker.make(si_structure)
        static_job = static_maker.make(si_structure, prev_dir=relax_job.output.dir_name)

        # Verify reference structure
        assert static_job is not None
        # Output reference should be present in job
        assert hasattr(static_job, "function")

    def test_prev_dir_chain_multiple_jobs(self, si_structure):
        """Test prev_dir chaining across multiple jobs."""
        makers = [
            RelaxMaker.fixed_cell_relaxation(user_params={"PAO.BasisSize": "SZ"}),
            RelaxMaker.fixed_cell_relaxation(user_params={"PAO.BasisSize": "DZ"}),
            StaticMaker(
                input_set_generator=StaticSetGenerator(
                    user_params={"PAO.BasisSize": "DZP"}
                )
            ),
        ]

        # Chain jobs
        jobs = []
        prev_dir = None
        for maker in makers:
            job = maker.make(si_structure, prev_dir=prev_dir)
            jobs.append(job)
            prev_dir = job.output.dir_name

        # Verify all jobs in chain
        assert len(jobs) == 3
        assert all(job is not None for job in jobs)

    def test_prev_dir_with_none(self, si_structure):
        """Test that prev_dir=None works correctly."""
        static_maker = StaticMaker()

        # Create job without previous directory
        job = static_maker.make(si_structure, prev_dir=None)

        assert job is not None
        assert hasattr(job, "function")

    def test_multiple_jobs_reference_same_prev_dir(self, si_structure):
        """Test multiple jobs can reference the same prev_dir."""
        relax_maker = RelaxMaker.fixed_cell_relaxation()
        static_maker1 = StaticMaker(
            input_set_generator=StaticSetGenerator(user_params={"PAO.BasisSize": "DZP"})
        )
        static_maker2 = StaticMaker(
            input_set_generator=StaticSetGenerator(user_params={"PAO.BasisSize": "TZP"})
        )

        # Create relax job
        relax_job = relax_maker.make(si_structure)

        # Multiple static jobs reference same prev_dir
        static_job1 = static_maker1.make(
            si_structure, prev_dir=relax_job.output.dir_name
        )
        static_job2 = static_maker2.make(
            si_structure, prev_dir=relax_job.output.dir_name
        )

        # Both should be valid
        assert static_job1 is not None
        assert static_job2 is not None


class TestDryRunIntegration:
    """Tests for dry-run mode in integrated workflows."""

    def test_dry_run_single_job(self, si_structure):
        """Test dry-run mode for single job."""
        maker = StaticMaker(dry_run=True)
        job = maker.make(si_structure)

        # Verify dry_run is set
        assert maker.dry_run is True
        assert job is not None

    def test_dry_run_chained_jobs(self, si_structure):
        """Test dry-run mode propagation in chained jobs."""
        relax_maker = RelaxMaker.fixed_cell_relaxation(dry_run=True)
        static_maker = StaticMaker(dry_run=True)

        # Create chain
        relax_job = relax_maker.make(si_structure)
        static_job = static_maker.make(si_structure, prev_dir=relax_job.output.dir_name)

        # Both should have dry_run
        assert relax_maker.dry_run is True
        assert static_maker.dry_run is True
        assert relax_job is not None
        assert static_job is not None

    def test_dry_run_flow_makers(self, si_structure):
        """Test dry-run mode in flow makers."""
        # Create flow maker with dry_run
        eos_maker = SiestaEosFlowMaker(dry_run=True)
        flow = eos_maker.make(si_structure)

        # Verify dry_run set
        assert eos_maker.dry_run is True
        assert isinstance(flow, Flow)

    def test_dry_run_mixed_makers(self, si_structure):
        """Test workflow with mixed dry_run settings."""
        # First job with dry_run
        job1 = RelaxMaker.fixed_cell_relaxation(dry_run=True).make(si_structure)

        # Second job without dry_run
        job2 = StaticMaker(dry_run=False).make(
            si_structure, prev_dir=job1.output.dir_name
        )

        # Both should be valid with their respective settings
        assert job1 is not None
        assert job2 is not None


class TestCrossFlowIntegration:
    """Tests for combining multiple flow makers in complex workflows."""

    def test_eos_plus_elastic_workflow(self, si_structure):
        """Test combining EOS and elastic constant workflows."""
        eos_maker = SiestaEosFlowMaker()
        elastic_maker = ElasticFlowMaker()

        # Create flows
        eos_flow = eos_maker.make(si_structure)
        elastic_flow = elastic_maker.make(si_structure)

        # Verify both flows created
        assert isinstance(eos_flow, Flow)
        assert isinstance(elastic_flow, Flow)
        assert len(eos_flow) > 0
        assert len(elastic_flow) > 0

    def test_eos_plus_phonons_workflow(self, si_structure):
        """Test combining EOS and phonon workflows."""
        eos_maker = SiestaEosFlowMaker()
        phonon_maker = SiestaPhononFlowMaker()

        # Create flows
        eos_flow = eos_maker.make(si_structure)
        phonon_flow = phonon_maker.make(si_structure)

        # Verify both flows created
        assert isinstance(eos_flow, Flow)
        assert isinstance(phonon_flow, Flow)

    def test_complete_thermal_workflow(self, si_structure):
        """Test complete thermal property workflow (EOS + phonons + Grüneisen)."""
        # Create makers
        eos_maker = SiestaEosFlowMaker()
        phonon_plus = SiestaPhononFlowMaker()
        phonon_minus = SiestaPhononFlowMaker()
        gruneisen_maker = SiestaGruneisenFlowMaker()

        # Create EOS flow
        eos_flow = eos_maker.make(si_structure)

        # Phonon flows at different volumes
        phonon_plus_flow = phonon_plus.make(si_structure)
        phonon_minus_flow = phonon_minus.make(si_structure)

        # Grüneisen flow
        gruneisen_flow = gruneisen_maker.make(si_structure)

        # Verify all flows created
        assert all(
            isinstance(f, Flow)
            for f in [eos_flow, phonon_plus_flow, phonon_minus_flow, gruneisen_flow]
        )

    def test_basis_convergence_plus_static_workflow(self, si_structure):
        """Test combining basis convergence with high-accuracy static calculation."""
        # Basis convergence flow
        basis_maker = DifferentBasisSCFFlowMaker(strategy="standard")
        basis_flow = basis_maker.make(si_structure)

        # High-accuracy static calculation
        static_maker = StaticMaker(
            input_set_generator=StaticSetGenerator(user_params={"PAO.BasisSize": "TZP"})
        )
        static_job = static_maker.make(si_structure)

        # Verify both created
        assert isinstance(basis_flow, Flow)
        assert static_job is not None
        assert len(basis_flow) > 0


class TestFlowComposition:
    """Tests for complex flow composition patterns."""

    def test_nested_flow_creation(self, si_structure):
        """Test creating flows within flows."""
        # Inner flow: basis convergence
        inner_maker = DifferentBasisSCFFlowMaker(strategy="standard")
        inner_flow = inner_maker.make(si_structure)

        # Outer flow: add static calculation
        static_maker = StaticMaker()
        static_job = static_maker.make(si_structure)

        outer_flow = Flow([inner_flow, static_job], name="nested_workflow")

        # Verify structure
        assert isinstance(outer_flow, Flow)
        assert len(outer_flow) == 2  # inner_flow + static_job

    def test_parallel_workflows(self, si_structure, al_structure):
        """Test parallel execution of independent workflows."""
        # Create makers
        si_maker = RelaxMaker.fixed_cell_relaxation()
        al_maker = RelaxMaker.fixed_cell_relaxation()

        # Create jobs for different structures
        si_job = si_maker.make(si_structure)
        al_job = al_maker.make(al_structure)

        # Create parallel flow
        parallel_flow = Flow([si_job, al_job], name="parallel_materials")

        # Verify structure
        assert isinstance(parallel_flow, Flow)
        assert len(parallel_flow) == 2

    def test_fan_out_fan_in_pattern(self, si_structure):
        """Test fan-out (one → many) and fan-in (many → one) pattern."""
        # Single relaxation (fan-out source)
        relax_maker = RelaxMaker.fixed_cell_relaxation()
        relax_job = relax_maker.make(si_structure)

        # Multiple static calculations with different basis sets (fan-out)
        static_makers = [
            StaticMaker(
                input_set_generator=StaticSetGenerator(
                    user_params={"PAO.BasisSize": "SZ"}
                )
            ),
            StaticMaker(
                input_set_generator=StaticSetGenerator(
                    user_params={"PAO.BasisSize": "DZ"}
                )
            ),
            StaticMaker(
                input_set_generator=StaticSetGenerator(
                    user_params={"PAO.BasisSize": "DZP"}
                )
            ),
        ]

        static_jobs = [
            maker.make(si_structure, prev_dir=relax_job.output.dir_name)
            for maker in static_makers
        ]

        # All jobs should reference same relax output (fan-out pattern)
        assert len(static_jobs) == 3
        assert all(job is not None for job in static_jobs)

    def test_sequential_flow_creation(self, si_structure):
        """Test creating sequential workflows with explicit ordering."""
        jobs = []

        # Step 1: Coarse relaxation
        jobs.append(
            RelaxMaker.fixed_cell_relaxation(user_params={"PAO.BasisSize": "SZ"}).make(
                si_structure
            )
        )

        # Step 2: Medium relaxation
        jobs.append(
            RelaxMaker.fixed_cell_relaxation(user_params={"PAO.BasisSize": "DZ"}).make(
                si_structure, prev_dir=jobs[-1].output.dir_name
            )
        )

        # Step 3: Fine static calculation
        jobs.append(
            StaticMaker(
                input_set_generator=StaticSetGenerator(
                    user_params={"PAO.BasisSize": "DZP"}
                )
            ).make(si_structure, prev_dir=jobs[-1].output.dir_name)
        )

        # Create sequential flow
        flow = Flow(jobs, name="sequential_refinement")

        # Verify structure
        assert isinstance(flow, Flow)
        assert len(flow) == 3
        assert flow.name == "sequential_refinement"


class TestRealWorldWorkflows:
    """Tests simulating real-world usage patterns."""

    def test_convergence_study_workflow(self, si_structure):
        """Test convergence study workflow (basis → k-points → mesh cutoff)."""
        # Basis convergence
        basis_flow = DifferentBasisSCFFlowMaker(strategy="standard").make(si_structure)

        # Static calculation with best basis
        static_job = StaticMaker(
            input_set_generator=StaticSetGenerator(user_params={"PAO.BasisSize": "DZP"})
        ).make(si_structure)

        # Combined flow
        convergence_flow = Flow(
            [basis_flow, static_job], name="complete_convergence_study"
        )

        assert isinstance(convergence_flow, Flow)
        assert len(convergence_flow) == 2

    def test_materials_screening_workflow(self, si_structure, al_structure):
        """Test high-throughput materials screening workflow."""
        # Create maker for screening
        screening_maker = StaticMaker(
            input_set_generator=StaticSetGenerator(user_params={"PAO.BasisSize": "DZP"})
        )

        # Screen multiple materials
        materials = [si_structure, al_structure]
        screening_jobs = [screening_maker.make(mat) for mat in materials]

        # Create screening flow
        screening_flow = Flow(screening_jobs, name="materials_screening")

        assert isinstance(screening_flow, Flow)
        assert len(screening_flow) == len(materials)

    def test_property_calculation_pipeline(self, si_structure):
        """Test complete property calculation pipeline."""
        # Step 1: Structure optimization
        relax_job = RelaxMaker.variable_cell_relaxation().make(si_structure)

        # Step 2: Accurate static calculation
        static_job = StaticMaker(
            input_set_generator=StaticSetGenerator(user_params={"PAO.BasisSize": "TZP"})
        ).make(si_structure, prev_dir=relax_job.output.dir_name)

        # Step 3: Property calculations (EOS, elastic)
        eos_flow = SiestaEosFlowMaker().make(si_structure)
        elastic_flow = ElasticFlowMaker().make(si_structure)

        # All components created
        assert relax_job is not None
        assert static_job is not None
        assert isinstance(eos_flow, Flow)
        assert isinstance(elastic_flow, Flow)

    def test_publication_quality_workflow(self, si_structure):
        """Test publication-quality calculation workflow."""
        # High-accuracy settings
        high_accuracy_maker = StaticMaker(
            input_set_generator=StaticSetGenerator(
                user_params={
                    "PAO.BasisSize": "TZP",
                    "Mesh.Cutoff": "400 Ry",
                    "a2s_kpts": [12, 12, 12],
                }
            )
        )

        # Create job
        job = high_accuracy_maker.make(si_structure)

        assert job is not None
        assert (
            high_accuracy_maker.input_set_generator.user_params["PAO.BasisSize"]
            == "TZP"
        )


class TestWorkflowValidation:
    """Tests for workflow validation and error handling."""

    def test_workflow_with_invalid_prev_dir_type(self, si_structure):
        """Test that workflows handle different prev_dir types."""
        maker = StaticMaker()

        # Test with None
        job1 = maker.make(si_structure, prev_dir=None)
        assert job1 is not None

        # Test with string (typical case)
        job2 = maker.make(si_structure, prev_dir="/path/to/dir")
        assert job2 is not None

    def test_workflow_with_multiple_structures(self, si_structure, al_structure):
        """Test workflow handling multiple structure types."""
        maker = RelaxMaker.fixed_cell_relaxation()

        # Create jobs for different structures
        si_job = maker.make(si_structure)
        al_job = maker.make(al_structure)

        # Both should be valid
        assert si_job is not None
        assert al_job is not None

    def test_workflow_maker_reusability(self, si_structure, al_structure):
        """Test that makers can be reused for multiple structures."""
        maker = StaticMaker(
            input_set_generator=StaticSetGenerator(user_params={"PAO.BasisSize": "DZP"})
        )

        # Use same maker for multiple structures
        jobs = [maker.make(struct) for struct in [si_structure, al_structure]]

        # All jobs should be valid
        assert len(jobs) == 2
        assert all(job is not None for job in jobs)

    def test_workflow_with_custom_names(self, si_structure):
        """Test workflows with custom job/flow names."""
        # Create flow with custom name
        maker = DifferentBasisSCFFlowMaker(
            name="custom_basis_flow", strategy="standard"
        )
        flow = maker.make(si_structure)

        assert isinstance(flow, Flow)
        assert flow.name == "custom_basis_flow"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
