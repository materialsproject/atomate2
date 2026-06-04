"""
Tests for defect workflows and ghost atom functionality.

These tests validate:
- Ghost atom creation for vacancies
- DefectFlowMaker with ghost atoms
- FDF generation with ghost atoms (ChemicalSpeciesLabel block)
- Integration with species variants system
"""

import pytest
from pymatgen.core import Lattice, Structure

from atomate2.siesta.flows.defects import (
    DefectFlowMaker,
    create_vacancy_with_ghost,
)
from atomate2.siesta.flows.defects.generation.vacancy import (
    create_vacancy_with_ghost_from_site,
)


@pytest.fixture
def mgo_structure():
    """
    MgO rock-salt structure for defect testing.

    Returns
    -------
    Structure
        MgO unit cell with 8 atoms (4 Mg + 4 O)
    """
    lattice = Lattice.cubic(4.212)
    return Structure(
        lattice,
        ["Mg", "Mg", "Mg", "Mg", "O", "O", "O", "O"],
        [
            [0.0, 0.0, 0.0],  # Mg
            [0.0, 0.5, 0.5],  # Mg
            [0.5, 0.0, 0.5],  # Mg
            [0.5, 0.5, 0.0],  # Mg
            [0.5, 0.5, 0.5],  # O
            [0.5, 0.0, 0.0],  # O
            [0.0, 0.5, 0.0],  # O
            [0.0, 0.0, 0.5],  # O
        ],
    )


@pytest.fixture
def mgo_unitcell(mgo_structure):
    """
    Alias for mgo_structure (for clarity in SiestaVacancyGenerator tests).

    Returns
    -------
    Structure
        MgO unit cell with 8 atoms (4 Mg + 4 O)
    """
    return mgo_structure


@pytest.fixture
def mgo_supercell(mgo_structure):
    """
    MgO 2×2×2 supercell for defect calculations.

    Returns
    -------
    Structure
        MgO supercell with 64 atoms (32 Mg + 32 O)
    """
    return mgo_structure.make_supercell([2, 2, 2])


class TestCreateVacancyWithGhost:
    """Tests for create_vacancy_with_ghost() function."""

    def test_create_vacancy_with_ghost_default(self, mgo_supercell):
        """Test creating vacancy with ghost atom (default behavior)."""
        # Find O atom to remove
        o_indices = [
            i for i, site in enumerate(mgo_supercell) if site.specie.symbol == "O"
        ]
        site_index = o_indices[0]

        # Create vacancy with ghost atom
        vacancy = create_vacancy_with_ghost(mgo_supercell, site_index, use_ghost=True)

        # Check that structure has same number of atoms
        assert len(vacancy) == len(mgo_supercell)

        # Check that ghost_tags site property exists
        assert "ghost_tags" in vacancy.site_properties
        ghost_tags = vacancy.site_properties["ghost_tags"]

        # Check that exactly one ghost atom was created
        assert sum(ghost_tags) == 1

        # Check that the ghost atom is at the correct position
        ghost_index = ghost_tags.index(True)
        assert ghost_index == site_index

        # Check that species_label site property exists
        assert "species_label" in vacancy.site_properties
        species_labels = vacancy.site_properties["species_label"]

        # Check that ghost atom has correct species label
        assert species_labels[ghost_index] == "O_ghost"

        # Check composition (should still have 32 O atoms but one is ghost)
        assert vacancy.composition["O"] == 32
        assert vacancy.composition["Mg"] == 32

    def test_create_vacancy_without_ghost(self, mgo_supercell):
        """Test creating vacancy without ghost atom (complete removal)."""
        # Find O atom to remove
        o_indices = [
            i for i, site in enumerate(mgo_supercell) if site.specie.symbol == "O"
        ]
        site_index = o_indices[0]

        # Create vacancy without ghost atom
        vacancy = create_vacancy_with_ghost(mgo_supercell, site_index, use_ghost=False)

        # Check that structure has one fewer atom
        assert len(vacancy) == len(mgo_supercell) - 1

        # Check composition
        assert vacancy.composition["O"] == 31  # One O removed
        assert vacancy.composition["Mg"] == 32

    def test_create_vacancy_preserves_coordinates(self, mgo_supercell):
        """Test that ghost atom preserves original coordinates."""
        # Find O atom to remove
        o_indices = [
            i for i, site in enumerate(mgo_supercell) if site.specie.symbol == "O"
        ]
        site_index = o_indices[5]  # Use different index

        # Get original coordinates
        original_coords = mgo_supercell[site_index].coords
        original_frac_coords = mgo_supercell[site_index].frac_coords

        # Create vacancy with ghost atom
        vacancy = create_vacancy_with_ghost(mgo_supercell, site_index, use_ghost=True)

        # Check that ghost atom has same coordinates
        ghost_coords = vacancy[site_index].coords
        ghost_frac_coords = vacancy[site_index].frac_coords

        assert pytest.approx(original_coords, abs=1e-6) == ghost_coords
        assert pytest.approx(original_frac_coords, abs=1e-6) == ghost_frac_coords

    def test_create_vacancy_mg_atom(self, mgo_supercell):
        """Test creating Mg vacancy with ghost atom."""
        # Find Mg atom to remove
        mg_indices = [
            i for i, site in enumerate(mgo_supercell) if site.specie.symbol == "Mg"
        ]
        site_index = mg_indices[0]

        # Create vacancy with ghost atom
        vacancy = create_vacancy_with_ghost(mgo_supercell, site_index, use_ghost=True)

        # Check ghost atom properties
        assert "ghost_tags" in vacancy.site_properties
        ghost_tags = vacancy.site_properties["ghost_tags"]
        assert sum(ghost_tags) == 1

        # Check species label
        ghost_index = ghost_tags.index(True)
        species_labels = vacancy.site_properties["species_label"]
        assert species_labels[ghost_index] == "Mg_ghost"

    def test_create_vacancy_from_site_fractional(self, mgo_supercell):
        """Test creating vacancy from fractional coordinates."""
        # Use fractional coordinates of an O atom
        target_frac = [0.5, 0.5, 0.5]

        vacancy, site_index = create_vacancy_with_ghost_from_site(
            mgo_supercell, target_frac, tolerance=0.01, use_ghost=True
        )

        # Check that vacancy was created
        assert len(vacancy) == len(mgo_supercell)
        assert "ghost_tags" in vacancy.site_properties

        # Check that correct site was found and converted to ghost
        ghost_tags = vacancy.site_properties["ghost_tags"]
        assert sum(ghost_tags) == 1

    def test_create_vacancy_from_site_not_found(self, mgo_supercell):
        """Test that ValueError is raised when no site found near target."""
        # Use coordinates far from any atom
        target_frac = [0.1234, 0.5678, 0.9012]

        with pytest.raises(ValueError, match="No site found within"):
            create_vacancy_with_ghost_from_site(
                mgo_supercell, target_frac, tolerance=0.01, use_ghost=True
            )

    def test_create_multiple_vacancies(self, mgo_supercell):
        """Test creating multiple vacancies sequentially."""
        # Find O atoms to remove
        o_indices = [
            i for i, site in enumerate(mgo_supercell) if site.specie.symbol == "O"
        ]

        # Create first vacancy
        vacancy1 = create_vacancy_with_ghost(
            mgo_supercell, o_indices[0], use_ghost=True
        )

        # Create second vacancy from first vacancy
        # Need to update indices after first removal (but we used ghost, so indices unchanged)
        vacancy2 = create_vacancy_with_ghost(vacancy1, o_indices[5], use_ghost=True)

        # Check that two ghost atoms were created
        ghost_tags = vacancy2.site_properties["ghost_tags"]
        assert sum(ghost_tags) == 2


class TestDefectFlowMaker:
    """Tests for DefectFlowMaker with ghost atoms."""

    def test_defect_flow_maker_default_parameters(self):
        """Test default parameters of DefectFlowMaker."""
        maker = DefectFlowMaker()

        assert maker.name == "Defect Calculation"
        assert maker.epsilon_static == 10.0
        assert maker.defect_type == "vacancy"
        assert maker.charge_state == 0
        assert maker.use_ghost_atoms is True  # Should default to True

    def test_defect_flow_maker_custom_parameters(self):
        """Test DefectFlowMaker with custom parameters."""
        maker = DefectFlowMaker(
            name="O vacancy in MgO",
            epsilon_static=9.8,
            defect_type="vacancy",
            charge_state=2,
            use_ghost_atoms=False,
        )

        assert maker.name == "O vacancy in MgO"
        assert maker.epsilon_static == 9.8
        assert maker.defect_type == "vacancy"
        assert maker.charge_state == 2
        assert maker.use_ghost_atoms is False

    def test_defect_flow_maker_creates_flow(self, mgo_supercell):
        """Test that DefectFlowMaker creates a valid flow."""
        # Create vacancy with ghost atom
        o_indices = [
            i for i, site in enumerate(mgo_supercell) if site.specie.symbol == "O"
        ]
        defect_structure = create_vacancy_with_ghost(
            mgo_supercell, o_indices[0], use_ghost=True
        )

        # Create flow
        maker = DefectFlowMaker(
            epsilon_static=9.8,
            defect_type="vacancy",
            chemical_potentials={"O": -5.12, "Mg": -1.51},
        )
        flow = maker.make(
            defect_structure=defect_structure,
            host_structure=mgo_supercell,
            defect_site=[0.5, 0.5, 0.5],
            defect_species="O",
        )

        # Check flow structure
        assert flow.name == "Defect Calculation"
        assert len(flow.jobs) == 3  # defect_relax, host_static, finalize

        # Check job names
        job_names = [job.name for job in flow.jobs]
        assert "Defect_Calculation_defect_relax" in job_names
        assert "Defect_Calculation_host_static" in job_names
        assert "Defect_Calculation_finalize" in job_names

    def test_defect_flow_maker_validates_ghost_atoms(self, mgo_supercell, caplog):
        """Test that DefectFlowMaker validates ghost atoms for vacancies."""
        import logging

        # Create vacancy WITHOUT ghost atom (incorrect for SIESTA)
        o_indices = [
            i for i, site in enumerate(mgo_supercell) if site.specie.symbol == "O"
        ]
        defect_structure = create_vacancy_with_ghost(
            mgo_supercell,
            o_indices[0],
            use_ghost=False,  # No ghost atom!
        )

        # Create flow with use_ghost_atoms=True
        maker = DefectFlowMaker(
            epsilon_static=9.8,
            defect_type="vacancy",
            use_ghost_atoms=True,
            chemical_potentials={"O": -5.12, "Mg": -1.51},
        )

        # Capture logs
        with caplog.at_level(logging.WARNING):
            flow = maker.make(
                defect_structure=defect_structure,
                host_structure=mgo_supercell,
                defect_site=[0.5, 0.5, 0.5],
                defect_species="O",
            )

            # Verify flow was created
            assert flow is not None

        # Check that warning was logged
        assert any(
            "use_ghost_atoms=True but defect_structure does not have" in record.message
            for record in caplog.records
        )

    def test_defect_flow_maker_charged_vacancy(self, mgo_supercell):
        """Test DefectFlowMaker with charged vacancy."""
        # Create vacancy with ghost atom
        o_indices = [
            i for i, site in enumerate(mgo_supercell) if site.specie.symbol == "O"
        ]
        defect_structure = create_vacancy_with_ghost(
            mgo_supercell, o_indices[0], use_ghost=True
        )

        # Create flow for charged defect (V_O^2+)
        maker = DefectFlowMaker(
            name="V_O +2 in MgO",
            epsilon_static=9.8,
            defect_type="vacancy",
            charge_state=2,  # Charged vacancy
            chemical_potentials={"O": -5.12, "Mg": -1.51},
        )

        flow = maker.make(
            defect_structure=defect_structure,
            host_structure=mgo_supercell,
            defect_site=[0.5, 0.5, 0.5],
            defect_species="O",
        )

        # Flow should still work with charged defects. Charged defects also add
        # diagnostic plot jobs (rho_plot, radial_plot) after the core pipeline
        # (defect_relax, host_static, finalize).
        assert flow.name == "V_O +2 in MgO"
        assert len(flow.jobs) == 5
        core_job_names = [job.name for job in flow.jobs[:3]]
        assert any("defect_relax" in name for name in core_job_names)
        assert any("host_static" in name for name in core_job_names)
        assert any("finalize" in name for name in core_job_names)

    def test_defect_flow_maker_substitution_no_ghost(self, mgo_supercell):
        """Test that substitutional defects don't require ghost atoms."""
        # For substitutional defects, ghost atoms are not needed
        # Create a Ca_Mg substitutional defect manually
        subst_structure = mgo_supercell.copy()
        mg_indices = [
            i for i, site in enumerate(subst_structure) if site.specie.symbol == "Mg"
        ]
        subst_structure.replace(mg_indices[0], "Ca")

        # Create flow for substitutional defect
        maker = DefectFlowMaker(
            defect_type="substitutional",
            use_ghost_atoms=True,  # This should not trigger warnings for non-vacancies
            chemical_potentials={"Ca": -2.0, "Mg": -1.51},
        )

        flow = maker.make(
            defect_structure=subst_structure,
            host_structure=mgo_supercell,
            defect_site=[0.0, 0.0, 0.0],
            defect_species="Ca",
        )

        # Should create flow without warnings
        assert len(flow.jobs) == 3


class TestGhostAtomsASEConversion:
    """Tests for ghost atom handling in pymatgen→ASE conversion."""

    def test_ghost_atoms_in_ase_atoms(self, mgo_supercell):
        """Test that ghost atoms are correctly converted to negative Z in ASE."""
        from atomate2.siesta.sets.utils.core import pymatgen_to_ase

        # Create vacancy with ghost atom
        o_indices = [
            i for i, site in enumerate(mgo_supercell) if site.specie.symbol == "O"
        ]
        vacancy = create_vacancy_with_ghost(mgo_supercell, o_indices[0], use_ghost=True)

        # Convert to ASE
        ase_atoms = pymatgen_to_ase(vacancy)

        # Check that ghost atom has negative Z
        atomic_numbers = ase_atoms.get_atomic_numbers()

        # Find ghost atom index
        ghost_tags = vacancy.site_properties["ghost_tags"]
        ghost_index = ghost_tags.index(True)

        # Ghost atom should have negative atomic number
        assert atomic_numbers[ghost_index] < 0
        assert atomic_numbers[ghost_index] == -8  # O has Z=8, ghost should be Z=-8

    def test_species_labels_in_ase_atoms(self, mgo_supercell):
        """Test that species_label site property is preserved in ASE."""
        from atomate2.siesta.sets.utils.core import pymatgen_to_ase

        # Create vacancy with ghost atom
        o_indices = [
            i for i, site in enumerate(mgo_supercell) if site.specie.symbol == "O"
        ]
        vacancy = create_vacancy_with_ghost(mgo_supercell, o_indices[0], use_ghost=True)

        # Convert to ASE
        ase_atoms = pymatgen_to_ase(vacancy)

        # Check that species_labels info is in ASE atoms
        assert "species_labels" in ase_atoms.info
        assert "species_dict" in ase_atoms.info
        assert "species_Z_dict" in ase_atoms.info

        # Check that ghost species is in the dictionary
        species_dict = ase_atoms.info["species_dict"]
        species_Z_dict = ase_atoms.info["species_Z_dict"]

        # Should have O_ghost in species labels
        assert any("ghost" in str(label).lower() for label in species_dict.values())

        # Ghost species should have negative Z
        for species_num, z_value in species_Z_dict.items():
            label = species_dict[species_num]
            if "ghost" in str(label).lower():
                assert z_value < 0


class TestGhostAtomsIntegration:
    """Integration tests for ghost atoms end-to-end."""

    def test_vacancy_workflow_with_ghost_dry_run(self, mgo_supercell):
        """Test complete vacancy workflow with ghost atoms in dry-run mode."""
        from jobflow import run_locally

        # Create vacancy with ghost atom
        o_indices = [
            i for i, site in enumerate(mgo_supercell) if site.specie.symbol == "O"
        ]
        defect_structure = create_vacancy_with_ghost(
            mgo_supercell, o_indices[0], use_ghost=True
        )

        # Create flow
        maker = DefectFlowMaker(
            epsilon_static=9.8,
            defect_type="vacancy",
            charge_state=0,
            chemical_potentials={"O": -5.12, "Mg": -1.51},
        )

        # Enable dry-run mode
        maker.defect_relax_maker.dry_run = True
        maker.host_static_maker.dry_run = True

        flow = maker.make(
            defect_structure=defect_structure,
            host_structure=mgo_supercell,
            defect_site=[0.5, 0.5, 0.5],
            defect_species="O",
        )

        # Run workflow
        results = run_locally(flow, create_folders=True, ensure_success=True)

        # Check that workflow completed
        assert results is not None

        # Get DefectDocument from results
        finalize_job_uuid = next(
            job for job in flow.jobs if job.name.endswith("_finalize")
        ).uuid
        defect_doc = results[finalize_job_uuid][1].output

        # Check DefectDocument fields
        assert defect_doc.defect_type == "vacancy"
        assert defect_doc.defect_species == "O"
        assert defect_doc.charge_state == 0
        assert defect_doc.correction_scheme == "none"  # Neutral defect, no correction
        assert defect_doc.correction_energy == 0.0

    def test_charged_vacancy_workflow_with_ghost_dry_run(self, mgo_supercell):
        """Test charged vacancy workflow with ghost atoms and corrections."""
        from jobflow import run_locally

        # Create vacancy with ghost atom
        o_indices = [
            i for i, site in enumerate(mgo_supercell) if site.specie.symbol == "O"
        ]
        defect_structure = create_vacancy_with_ghost(
            mgo_supercell, o_indices[0], use_ghost=True
        )

        # Create flow for charged defect
        maker = DefectFlowMaker(
            name="V_O +2 in MgO",
            epsilon_static=9.8,
            defect_type="vacancy",
            charge_state=2,
            chemical_potentials={"O": -5.12, "Mg": -1.51},
        )

        # Enable dry-run mode
        maker.defect_relax_maker.dry_run = True
        maker.host_static_maker.dry_run = True

        flow = maker.make(
            defect_structure=defect_structure,
            host_structure=mgo_supercell,
            defect_site=[0.5, 0.5, 0.5],
            defect_species="O",
        )

        # Run workflow
        results = run_locally(flow, create_folders=True, ensure_success=True)

        # Get DefectDocument
        finalize_job_uuid = next(
            job for job in flow.jobs if job.name.endswith("_finalize")
        ).uuid
        defect_doc = results[finalize_job_uuid][1].output

        # Check that correction was applied
        assert defect_doc.charge_state == 2
        assert defect_doc.correction_scheme.lower() == "lany-zunger"
        assert defect_doc.correction_energy != 0.0  # Should have correction
        assert "epsilon_static" in defect_doc.correction_metadata
        assert defect_doc.correction_metadata["epsilon_static"] == 9.8


class TestCorrectionComparisonFlowMaker:
    """Test CorrectionComparisonFlowMaker - unique killer feature."""

    def test_correction_comparison_flow_structure(self, mgo_supercell):
        """Test that comparison flow has correct structure."""
        from atomate2.siesta.flows.defects import (
            CorrectionComparisonFlowMaker,
            create_vacancy_with_ghost,
        )

        # Create vacancy with ghost atom
        o_indices = [
            i for i, site in enumerate(mgo_supercell) if site.specie.symbol == "O"
        ]
        defect_structure = create_vacancy_with_ghost(
            structure=mgo_supercell, site_index=o_indices[0], use_ghost=True
        )

        # Create comparison flow maker
        flow_maker = CorrectionComparisonFlowMaker(
            epsilon_static=9.8,
            defect_type="vacancy",
            charge_state=2,
            correction_schemes=["lany-zunger"],
        )
        flow_maker.defect_relax_maker.dry_run = True
        flow_maker.host_static_maker.dry_run = True

        # Create flow
        flow = flow_maker.make(
            defect_structure=defect_structure,
            host_structure=mgo_supercell,
            defect_site=mgo_supercell[o_indices[0]].frac_coords.tolist(),
            defect_species="O",
        )

        # Check flow structure. For charged defects the flow appends
        # diagnostic plot jobs (rho_plot, radial_plot) after the summary job,
        # so the core pipeline is the first four jobs.
        assert flow.name == "Correction Comparison"
        assert (
            len(flow.jobs) == 6
        )  # defect_relax, host_static, compare, summary, + 2 plots
        assert "defect_relax" in flow.jobs[0].name
        assert "host_static" in flow.jobs[1].name
        assert "compare_corrections" in flow.jobs[2].name
        assert "summary" in flow.jobs[3].name

    def test_correction_comparison_neutral_defect(self, mgo_supercell):
        """Test comparison flow for neutral defect (q=0)."""
        from atomate2.siesta.flows.defects import (
            CorrectionComparisonFlowMaker,
            create_vacancy_with_ghost,
        )
        from jobflow import run_locally

        # Create neutral vacancy
        o_indices = [
            i for i, site in enumerate(mgo_supercell) if site.specie.symbol == "O"
        ]
        defect_structure = create_vacancy_with_ghost(
            structure=mgo_supercell, site_index=o_indices[0], use_ghost=True
        )

        # Create comparison flow maker for neutral defect
        flow_maker = CorrectionComparisonFlowMaker(
            epsilon_static=9.8,
            defect_type="vacancy",
            charge_state=0,  # Neutral
            correction_schemes=["lany-zunger"],
        )
        flow_maker.defect_relax_maker.dry_run = True
        flow_maker.host_static_maker.dry_run = True

        # Create and run flow
        flow = flow_maker.make(
            defect_structure=defect_structure,
            host_structure=mgo_supercell,
            defect_site=mgo_supercell[o_indices[0]].frac_coords.tolist(),
            defect_species="O",
        )

        results = run_locally(flow, create_folders=True)

        # Get summary from last job
        summary_job_uuid = next(
            job for job in flow.jobs if job.name.endswith("_summary")
        ).uuid
        summary = results[summary_job_uuid][1].output

        # Check that neutral defect returns zero corrections
        assert summary["recommendation"] == "none"
        comparison_results = summary["comparison_results"]
        assert comparison_results["charge_state"] == 0
        assert comparison_results["schemes"] == ["none"]
        assert comparison_results["correction_energies"] == [0.0]
        assert comparison_results["statistics"]["mean_correction"] == 0.0
        assert comparison_results["statistics"]["std_correction"] == 0.0

    def test_correction_comparison_charged_defect(self, mgo_supercell):
        """Test comparison flow for charged defect (q=+2)."""
        from atomate2.siesta.flows.defects import (
            CorrectionComparisonFlowMaker,
            create_vacancy_with_ghost,
        )
        from jobflow import run_locally

        # Create charged vacancy
        o_indices = [
            i for i, site in enumerate(mgo_supercell) if site.specie.symbol == "O"
        ]
        defect_structure = create_vacancy_with_ghost(
            structure=mgo_supercell, site_index=o_indices[0], use_ghost=True
        )

        # Create comparison flow maker for charged defect
        flow_maker = CorrectionComparisonFlowMaker(
            epsilon_static=9.8,
            defect_type="vacancy",
            charge_state=2,  # +2 charge
            correction_schemes=["lany-zunger"],
        )
        flow_maker.defect_relax_maker.dry_run = True
        flow_maker.host_static_maker.dry_run = True

        # Create and run flow
        flow = flow_maker.make(
            defect_structure=defect_structure,
            host_structure=mgo_supercell,
            defect_site=mgo_supercell[o_indices[0]].frac_coords.tolist(),
            defect_species="O",
        )

        results = run_locally(flow, create_folders=True)

        # Get summary from last job
        summary_job_uuid = next(
            job for job in flow.jobs if job.name.endswith("_summary")
        ).uuid
        summary = results[summary_job_uuid][1].output

        # Check comparison results
        comparison_results = summary["comparison_results"]
        assert comparison_results["charge_state"] == 2
        assert comparison_results["defect_type"] == "vacancy"
        assert comparison_results["defect_species"] == "O"
        assert comparison_results["epsilon_static"] == 9.8
        assert "lany-zunger" in comparison_results["schemes"][0].lower()

        # Check that corrections were calculated
        assert len(comparison_results["correction_energies"]) == 1
        assert comparison_results["correction_energies"][0] != 0.0  # Should be non-zero

        # Check statistics
        stats = comparison_results["statistics"]
        assert "mean_correction" in stats
        assert "std_correction" in stats
        assert "range_correction" in stats
        assert "mean_formation_energy" in stats

        # Check summary text
        assert "summary_text" in summary
        assert "CORRECTION SCHEME COMPARISON SUMMARY" in summary["summary_text"]
        assert "lany-zunger" in summary["summary_text"].lower()

    def test_correction_comparison_multiple_schemes(self, mgo_supercell):
        """Test comparison flow with multiple correction schemes."""
        from atomate2.siesta.flows.defects import (
            CorrectionComparisonFlowMaker,
            create_vacancy_with_ghost,
        )
        from jobflow import run_locally

        # Create charged vacancy
        o_indices = [
            i for i, site in enumerate(mgo_supercell) if site.specie.symbol == "O"
        ]
        defect_structure = create_vacancy_with_ghost(
            structure=mgo_supercell, site_index=o_indices[0], use_ghost=True
        )

        # Create comparison flow maker with multiple schemes
        flow_maker = CorrectionComparisonFlowMaker(
            epsilon_static=9.8,
            defect_type="vacancy",
            charge_state=2,
            correction_schemes=["lany-zunger", "makov-payne"],
        )
        flow_maker.defect_relax_maker.dry_run = True
        flow_maker.host_static_maker.dry_run = True

        # Create and run flow
        flow = flow_maker.make(
            defect_structure=defect_structure,
            host_structure=mgo_supercell,
            defect_site=mgo_supercell[o_indices[0]].frac_coords.tolist(),
            defect_species="O",
        )

        results = run_locally(flow, create_folders=True)

        # Get summary
        summary_job_uuid = next(
            job for job in flow.jobs if job.name.endswith("_summary")
        ).uuid
        summary = results[summary_job_uuid][1].output

        # Check that both schemes were applied
        comparison_results = summary["comparison_results"]
        assert len(comparison_results["schemes"]) == 2
        assert len(comparison_results["correction_energies"]) == 2
        assert len(comparison_results["corrected_formation_energies"]) == 2
        assert len(comparison_results["metadata"]) == 2

        # Check that both schemes are present
        schemes_lower = [s.lower() for s in comparison_results["schemes"]]
        assert "lany-zunger" in schemes_lower
        assert "makov-payne" in schemes_lower

        # Check recommendation exists
        assert "recommendation" in summary

    def test_correction_comparison_with_metadata(self, mgo_supercell):
        """Test that comparison flow includes detailed metadata."""
        from atomate2.siesta.flows.defects import (
            CorrectionComparisonFlowMaker,
            create_vacancy_with_ghost,
        )
        from jobflow import run_locally

        # Create charged vacancy
        o_indices = [
            i for i, site in enumerate(mgo_supercell) if site.specie.symbol == "O"
        ]
        defect_structure = create_vacancy_with_ghost(
            structure=mgo_supercell, site_index=o_indices[0], use_ghost=True
        )

        # Create comparison flow maker
        flow_maker = CorrectionComparisonFlowMaker(
            epsilon_static=9.8,
            defect_type="vacancy",
            charge_state=2,
            correction_schemes=["lany-zunger"],
        )
        flow_maker.defect_relax_maker.dry_run = True
        flow_maker.host_static_maker.dry_run = True

        # Create and run flow
        flow = flow_maker.make(
            defect_structure=defect_structure,
            host_structure=mgo_supercell,
            defect_site=mgo_supercell[o_indices[0]].frac_coords.tolist(),
            defect_species="O",
        )

        results = run_locally(flow, create_folders=True)

        # Get comparison results
        summary_job_uuid = next(
            job for job in flow.jobs if job.name.endswith("_summary")
        ).uuid
        summary = results[summary_job_uuid][1].output
        comparison_results = summary["comparison_results"]

        # Check metadata structure
        assert len(comparison_results["metadata"]) == 1
        metadata = comparison_results["metadata"][0]

        # Lany-Zunger metadata should include these fields
        expected_fields = [
            "madelung_constant",
            "epsilon_static",
            "characteristic_length_angstrom",
            "charge_state",
        ]
        for field in expected_fields:
            assert field in metadata, f"Missing field: {field}"


class TestSiestaVacancyGenerator:
    """Test automated vacancy generation with SiestaVacancyGenerator."""

    def test_generator_initialization(self, mgo_unitcell):
        """Test generator initialization and symmetry analysis."""
        from atomate2.siesta.flows.defects import SiestaVacancyGenerator

        generator = SiestaVacancyGenerator(mgo_unitcell, use_ghost_atoms=True)

        # Check that symmetry analysis was performed
        assert generator.structure == mgo_unitcell
        assert generator.use_ghost_atoms is True
        assert generator.sga is not None
        assert generator.symmetrized_structure is not None

    def test_get_unique_sites_all_species(self, mgo_unitcell):
        """Test finding all unique sites."""
        from atomate2.siesta.flows.defects import SiestaVacancyGenerator

        generator = SiestaVacancyGenerator(mgo_unitcell)
        unique_sites = generator.get_unique_sites()

        # MgO has 2 unique Wyckoff positions (Mg and O)
        assert len(unique_sites) == 2

        # Check that we have one Mg and one O
        species = [site.species for site in unique_sites]
        assert "Mg" in species
        assert "O" in species

        # Check site information
        for site in unique_sites:
            assert site.wyckoff in ["a", "b"]  # Fm-3m space group
            assert site.multiplicity == 4  # FCC lattice
            assert site.site_index >= 0
            assert len(site.frac_coords) == 3

    def test_get_unique_sites_filtered_by_species(self, mgo_unitcell):
        """Test filtering by species."""
        from atomate2.siesta.flows.defects import SiestaVacancyGenerator

        generator = SiestaVacancyGenerator(mgo_unitcell)

        # Get only O sites
        o_sites = generator.get_unique_sites(species="O")
        assert len(o_sites) == 1
        assert o_sites[0].species == "O"

        # Get only Mg sites
        mg_sites = generator.get_unique_sites(species="Mg")
        assert len(mg_sites) == 1
        assert mg_sites[0].species == "Mg"

        # Get both with list
        all_sites = generator.get_unique_sites(species=["O", "Mg"])
        assert len(all_sites) == 2

    def test_generate_defects_unit_cell(self, mgo_unitcell):
        """Test generating defects in unit cell."""
        from atomate2.siesta.flows.defects import SiestaVacancyGenerator

        generator = SiestaVacancyGenerator(mgo_unitcell, use_ghost_atoms=True)
        defects = generator.generate_defects()

        # Should generate 2 defects (V_Mg and V_O)
        assert len(defects) == 2

        for defect in defects:
            # Check required fields
            assert "structure" in defect
            assert "host_structure" in defect
            assert "species" in defect
            assert "wyckoff" in defect
            assert "charge_state" in defect
            assert "defect_type" in defect

            # Check defect properties
            assert defect["defect_type"] == "vacancy"
            assert defect["use_ghost"] is True
            assert defect["charge_state"] == 0  # Default neutral

            # Check that ghost atom was created
            assert "ghost_tags" in defect["structure"].site_properties
            assert sum(defect["structure"].site_properties["ghost_tags"]) == 1

            # Check that structure has same number of atoms as host
            assert len(defect["structure"]) == len(defect["host_structure"])

    def test_generate_defects_filtered(self, mgo_unitcell):
        """Test generating defects filtered by species."""
        from atomate2.siesta.flows.defects import SiestaVacancyGenerator

        generator = SiestaVacancyGenerator(mgo_unitcell)

        # Generate only O vacancies
        o_defects = generator.generate_defects(species="O")
        assert len(o_defects) == 1
        assert o_defects[0]["species"] == "O"

        # Generate only Mg vacancies
        mg_defects = generator.generate_defects(species="Mg")
        assert len(mg_defects) == 1
        assert mg_defects[0]["species"] == "Mg"

    def test_generate_defects_with_supercell(self, mgo_unitcell):
        """Test generating defects with supercell."""
        from atomate2.siesta.flows.defects import SiestaVacancyGenerator

        generator = SiestaVacancyGenerator(mgo_unitcell)

        # Generate with 2×2×2 supercell
        defects = generator.generate_defects(
            species="O",
            supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
        )

        assert len(defects) == 1
        defect = defects[0]

        # Check that host is 2×2×2 supercell
        assert len(defect["host_structure"]) == 64  # 8 atoms × 8
        assert defect["supercell_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]

        # Check that defect structure also has 64 atoms (with ghost)
        assert len(defect["structure"]) == 64

    def test_generate_defects_multiple_charge_states(self, mgo_unitcell):
        """Test generating defects with multiple charge states."""
        from atomate2.siesta.flows.defects import SiestaVacancyGenerator

        generator = SiestaVacancyGenerator(mgo_unitcell)

        # Generate O vacancies with 3 charge states
        defects = generator.generate_defects(species="O", charge_states=[0, +1, +2])

        # Should have 1 unique site × 3 charge states = 3 defects
        assert len(defects) == 3

        # Check charge states
        charge_states = [d["charge_state"] for d in defects]
        assert 0 in charge_states
        assert 1 in charge_states
        assert 2 in charge_states

        # All should be O vacancies
        for defect in defects:
            assert defect["species"] == "O"

    def test_generate_defects_without_ghost(self, mgo_unitcell):
        """Test generating defects without ghost atoms."""
        from atomate2.siesta.flows.defects import SiestaVacancyGenerator

        generator = SiestaVacancyGenerator(mgo_unitcell, use_ghost_atoms=False)
        defects = generator.generate_defects(species="O")

        assert len(defects) == 1
        defect = defects[0]

        # Check that no ghost atom was used
        assert defect["use_ghost"] is False

        # Structure should have one fewer atom than host
        assert len(defect["structure"]) == len(defect["host_structure"]) - 1

        # Should not have ghost_tags property
        if "ghost_tags" in defect["structure"].site_properties:
            assert sum(defect["structure"].site_properties["ghost_tags"]) == 0


class TestNetChargeInjection:
    """Test that NetCharge parameter is correctly injected for charged defects."""

    def test_netcharge_in_fdf_for_charged_defect(self, mgo_supercell):
        """Test that NetCharge parameter appears in FDF file for charged defects."""
        from jobflow import run_locally
        import glob
        import os

        # Create vacancy with ghost atom
        o_indices = [
            i for i, site in enumerate(mgo_supercell) if site.specie.symbol == "O"
        ]
        defect_structure = create_vacancy_with_ghost(
            mgo_supercell, o_indices[0], use_ghost=True
        )

        # Create DefectFlowMaker with charge_state=+2
        maker = DefectFlowMaker(
            name="NetCharge_Test",
            epsilon_static=9.8,
            defect_type="vacancy",
            chemical_potentials={"O": -5.12, "Mg": -1.51},
            charge_state=2,  # +2 charge - should trigger NetCharge injection
        )

        # Enable dry-run mode
        maker.defect_relax_maker.dry_run = True
        maker.host_static_maker.dry_run = True

        # Verify NetCharge was injected into user_params before make()
        flow = maker.make(
            defect_structure=defect_structure,
            host_structure=mgo_supercell,
            defect_site=[0.5, 0.5, 0.5],
            defect_species="O",
        )

        # Verify NetCharge is in user_params after make()
        assert "NetCharge" in maker.defect_relax_maker.input_set_generator.user_params
        assert (
            maker.defect_relax_maker.input_set_generator.user_params["NetCharge"] == 2
        )

        # Run workflow
        results = run_locally(flow, create_folders=True, ensure_success=True)
        assert results is not None

        # Find the FDF file for defect_relax job
        fdf_pattern = "./job_*/dry_run_output/NetCharge_Test_defect_relax_*/siesta.fdf"
        fdf_files = glob.glob(fdf_pattern)

        assert len(fdf_files) > 0, "No FDF files found for defect_relax job"

        # Get most recent FDF file
        fdf_file = max(fdf_files, key=os.path.getmtime)

        # Read FDF file content
        with open(fdf_file, "r") as f:
            fdf_content = f.read()

        # Verify NetCharge parameter is present
        assert (
            "NetCharge" in fdf_content
        ), "NetCharge parameter not found in FDF file for charged defect!"

        # Verify the value is correct
        netcharge_lines = [
            line for line in fdf_content.split("\n") if "NetCharge" in line
        ]
        assert len(netcharge_lines) > 0, "NetCharge line not found"

        # Check that value is 2.0 or 2
        netcharge_line = netcharge_lines[0]
        assert "2" in netcharge_line, f"NetCharge value incorrect: {netcharge_line}"

    def test_no_netcharge_for_neutral_defect(self, mgo_supercell):
        """Test that NetCharge is NOT set for neutral defects (q=0)."""
        from jobflow import run_locally
        import glob
        import os

        # Create vacancy with ghost atom
        o_indices = [
            i for i, site in enumerate(mgo_supercell) if site.specie.symbol == "O"
        ]
        defect_structure = create_vacancy_with_ghost(
            mgo_supercell, o_indices[0], use_ghost=True
        )

        # Create DefectFlowMaker with charge_state=0 (neutral)
        maker = DefectFlowMaker(
            name="Neutral_Test",
            epsilon_static=9.8,
            defect_type="vacancy",
            chemical_potentials={"O": -5.12, "Mg": -1.51},
            charge_state=0,  # Neutral - should NOT have NetCharge
        )

        # Enable dry-run mode
        maker.defect_relax_maker.dry_run = True
        maker.host_static_maker.dry_run = True

        flow = maker.make(
            defect_structure=defect_structure,
            host_structure=mgo_supercell,
            defect_site=[0.5, 0.5, 0.5],
            defect_species="O",
        )

        # Verify NetCharge is NOT in user_params for neutral defect
        assert (
            "NetCharge" not in maker.defect_relax_maker.input_set_generator.user_params
        )

        # Run workflow
        results = run_locally(flow, create_folders=True, ensure_success=True)
        assert results is not None

        # Find the FDF file for defect_relax job
        fdf_pattern = "./job_*/dry_run_output/Neutral_Test_defect_relax_*/siesta.fdf"
        fdf_files = glob.glob(fdf_pattern)

        if len(fdf_files) > 0:
            fdf_file = max(fdf_files, key=os.path.getmtime)

            # Read FDF file content
            with open(fdf_file, "r") as f:
                fdf_content = f.read()

            # Verify NetCharge is NOT present (or is 0 if present)
            if "NetCharge" in fdf_content:
                netcharge_lines = [
                    line for line in fdf_content.split("\n") if "NetCharge" in line
                ]
                # If NetCharge appears, it should be 0 or 0.0
                for line in netcharge_lines:
                    assert (
                        "0" in line
                    ), f"NetCharge should be 0 for neutral defect: {line}"


class TestMakovPayneCorrection:
    """Tests for Makov-Payne correction scheme."""

    def test_makov_payne_correction_basic(self, mgo_supercell):
        """Test basic Makov-Payne correction calculation."""
        from atomate2.siesta.flows.defects.corrections import MakovPayneCorrection

        # Create Makov-Payne correction scheme
        correction = MakovPayneCorrection(
            epsilon_static=9.8,
            madelung_constant=2.8373,  # Default for cubic
            quadrupole_moment=None,  # Will use Q=0 (conservative)
        )

        # Test properties
        assert correction.name == "Makov-Payne"
        assert correction.charge_model == "point+quadrupole"
        assert correction.requires_dielectric is True
        assert correction.supports_anisotropic is False
        assert correction.requires_potential_data is False

        # Calculate correction for charged defect
        result = correction.calculate_correction(
            defect_structure=mgo_supercell,
            host_structure=mgo_supercell,
            charge_state=2,
            defect_energy=-100.0,
            host_energy=-50.0,
        )

        # Check that correction was calculated
        assert result.correction_energy != 0.0
        assert result.scheme_name == "Makov-Payne"
        assert result.charge_model == "point+quadrupole"
        assert result.converged is True

        # Check metadata includes both monopole and quadrupole terms
        assert "monopole_term_eV" in result.metadata
        assert "quadrupole_term_eV" in result.metadata
        assert "quadrupole_moment_eA2" in result.metadata
        assert result.metadata["quadrupole_moment_eA2"] == 0.0  # Default

        # Check warning for Q=0 assumption
        assert len(result.warnings) > 0
        assert "Quadrupole moment Q=0 assumed" in result.warnings[0]

    def test_makov_payne_with_quadrupole_moment(self, mgo_supercell):
        """Test Makov-Payne with explicit quadrupole moment."""
        from atomate2.siesta.flows.defects.corrections import MakovPayneCorrection

        # Create correction with explicit quadrupole moment
        Q = 5.0  # eÅ²
        correction = MakovPayneCorrection(
            epsilon_static=9.8,
            quadrupole_moment=Q,
        )

        result = correction.calculate_correction(
            defect_structure=mgo_supercell,
            host_structure=mgo_supercell,
            charge_state=2,
            defect_energy=-100.0,
            host_energy=-50.0,
        )

        # Check that quadrupole moment was used
        assert result.metadata["quadrupole_moment_eA2"] == Q
        assert result.metadata["quadrupole_term_eV"] != 0.0

        # Should not have warning about Q=0
        assert len(result.warnings) == 0

    def test_makov_payne_vs_lany_zunger(self, mgo_supercell):
        """Test that Makov-Payne with Q=0 equals Lany-Zunger."""
        from atomate2.siesta.flows.defects.corrections import (
            LanyZungerCorrection,
            MakovPayneCorrection,
        )

        # Create both corrections with same parameters
        epsilon = 9.8
        charge = 2

        lz_correction = LanyZungerCorrection(epsilon_static=epsilon)
        mp_correction = MakovPayneCorrection(
            epsilon_static=epsilon,
            quadrupole_moment=0.0,  # Q=0 should match LZ
        )

        # Calculate both corrections
        lz_result = lz_correction.calculate_correction(
            defect_structure=mgo_supercell,
            host_structure=mgo_supercell,
            charge_state=charge,
            defect_energy=-100.0,
            host_energy=-50.0,
        )

        mp_result = mp_correction.calculate_correction(
            defect_structure=mgo_supercell,
            host_structure=mgo_supercell,
            charge_state=charge,
            defect_energy=-100.0,
            host_energy=-50.0,
        )

        # With Q=0, monopole term should dominate and match LZ
        # Allow small numerical differences
        assert abs(lz_result.correction_energy - mp_result.correction_energy) < 1e-6

    def test_makov_payne_in_comparison_flow(self, mgo_supercell):
        """Test Makov-Payne correction in CorrectionComparisonFlowMaker."""
        from atomate2.siesta.flows.defects import (
            CorrectionComparisonFlowMaker,
            create_vacancy_with_ghost,
        )
        from jobflow import run_locally

        # Create charged vacancy
        o_indices = [
            i for i, site in enumerate(mgo_supercell) if site.specie.symbol == "O"
        ]
        defect_structure = create_vacancy_with_ghost(
            structure=mgo_supercell, site_index=o_indices[0], use_ghost=True
        )

        # Create comparison flow with Makov-Payne
        flow_maker = CorrectionComparisonFlowMaker(
            epsilon_static=9.8,
            defect_type="vacancy",
            charge_state=2,
            correction_schemes=["makov-payne"],
        )
        flow_maker.defect_relax_maker.dry_run = True
        flow_maker.host_static_maker.dry_run = True

        flow = flow_maker.make(
            defect_structure=defect_structure,
            host_structure=mgo_supercell,
            defect_site=mgo_supercell[o_indices[0]].frac_coords.tolist(),
            defect_species="O",
        )

        results = run_locally(flow, create_folders=True)

        # Get summary
        summary_job_uuid = next(
            job for job in flow.jobs if job.name.endswith("_summary")
        ).uuid
        summary = results[summary_job_uuid][1].output

        # Check that Makov-Payne was applied
        comparison_results = summary["comparison_results"]
        assert len(comparison_results["schemes"]) == 1
        assert "makov-payne" in comparison_results["schemes"][0].lower()
        assert comparison_results["correction_energies"][0] != 0.0

        # Check metadata includes Makov-Payne specific fields
        metadata = comparison_results["metadata"][0]
        assert "monopole_term_eV" in metadata
        assert "quadrupole_term_eV" in metadata
        assert "quadrupole_moment_eA2" in metadata


class TestSiestaSubstitutionGenerator:
    """Tests for SiestaSubstitutionGenerator."""

    def test_substitution_generator_init(self, mgo_unitcell):
        """Test initialization of SiestaSubstitutionGenerator."""
        from atomate2.siesta.flows.defects.generation import (
            SiestaSubstitutionGenerator,
        )

        generator = SiestaSubstitutionGenerator(mgo_unitcell)
        assert generator.structure == mgo_unitcell
        assert generator.symprec == 0.1
        assert generator.sga is not None

    def test_get_unique_sites_substitution(self, mgo_unitcell):
        """Test getting unique substitution sites."""
        from atomate2.siesta.flows.defects.generation import (
            SiestaSubstitutionGenerator,
        )

        generator = SiestaSubstitutionGenerator(mgo_unitcell)

        # Get all unique sites
        all_sites = generator.get_unique_sites()
        assert len(all_sites) > 0

        # Get only Mg sites
        mg_sites = generator.get_unique_sites(species="Mg")
        assert len(mg_sites) >= 1
        assert all(site.species == "Mg" for site in mg_sites)

        # Get only O sites
        o_sites = generator.get_unique_sites(species="O")
        assert len(o_sites) >= 1
        assert all(site.species == "O" for site in o_sites)

    def test_generate_substitution_defects(self, mgo_unitcell):
        """Test generating substitution defects."""
        from atomate2.siesta.flows.defects.generation import (
            SiestaSubstitutionGenerator,
        )

        generator = SiestaSubstitutionGenerator(mgo_unitcell)

        # Generate Li_Mg defects
        defects = generator.generate_defects(
            species="Mg",
            dopants="Li",
            charge_states=[0, -1],
        )

        assert len(defects) > 0
        for defect in defects:
            assert defect["defect_type"] == "substitution"
            assert defect["original_species"] == "Mg"
            assert defect["dopant_species"] == "Li"
            assert defect["charge_state"] in [0, -1]
            assert "structure" in defect
            assert "host_structure" in defect

    def test_generate_antisites(self, mgo_unitcell):
        """Test generating antisite defects."""
        from atomate2.siesta.flows.defects.generation import (
            SiestaSubstitutionGenerator,
        )

        generator = SiestaSubstitutionGenerator(mgo_unitcell)

        # Generate antisites
        antisites = generator.generate_antisites(charge_states=[0])

        # Should have both Mg_O and O_Mg
        assert len(antisites) >= 2
        species_pairs = [
            (d["original_species"], d["dopant_species"]) for d in antisites
        ]
        assert ("Mg", "O") in species_pairs or ("O", "Mg") in species_pairs


class TestSiestaInterstitialGenerator:
    """Tests for SiestaInterstitialGenerator."""

    def test_interstitial_generator_init(self, mgo_unitcell):
        """Test initialization of SiestaInterstitialGenerator."""
        from atomate2.siesta.flows.defects.generation import (
            SiestaInterstitialGenerator,
        )

        generator = SiestaInterstitialGenerator(mgo_unitcell, min_dist=1.5)
        assert generator.structure == mgo_unitcell
        assert generator.min_dist == 1.5
        assert generator.symprec == 0.1

    def test_get_interstitial_sites(self, mgo_unitcell):
        """Test finding interstitial sites."""
        from atomate2.siesta.flows.defects.generation import (
            SiestaInterstitialGenerator,
        )

        generator = SiestaInterstitialGenerator(mgo_unitcell, min_dist=1.5)
        sites = generator.get_interstitial_sites()

        # MgO should have some interstitial sites
        # (octahedral and tetrahedral sites in rocksalt structure)
        # Note: Might be zero if min_dist is too large
        assert isinstance(sites, list)

    def test_generate_interstitial_defects(self, mgo_unitcell):
        """Test generating interstitial defects."""
        from atomate2.siesta.flows.defects.generation import (
            SiestaInterstitialGenerator,
        )

        generator = SiestaInterstitialGenerator(mgo_unitcell, min_dist=1.0)

        # Generate Li interstitials
        defects = generator.generate_defects(
            species="Li",
            charge_states=[0, +1],
        )

        # Should generate some interstitials
        if len(defects) > 0:  # Only test if interstitials were found
            for defect in defects:
                assert defect["defect_type"] == "interstitial"
                assert defect["species"] == "Li"
                assert defect["charge_state"] in [0, +1]
                assert "structure" in defect
                assert "host_structure" in defect

                # Interstitial structure should have one more atom
                assert len(defect["structure"]) == len(defect["host_structure"]) + 1


class TestDefectFlowMakerFromPristineStructure:
    """Tests for DefectFlowMaker.from_pristine_structure() classmethod."""

    def test_from_pristine_vacancies(self, mgo_unitcell):
        """Test generating vacancy flows from pristine structure."""
        from jobflow import Flow

        from atomate2.siesta.flows.defects import DefectFlowMaker

        # from_pristine_structure now returns a single parent Flow that wraps
        # the individual defect sub-flows (plus a shared host job and a combined
        # summary job).
        parent_flow = DefectFlowMaker.from_pristine_structure(
            mgo_unitcell,
            defect_type="vacancy",
            charge_states=[0],
            epsilon_static=9.8,
            chemical_potentials={"O": -5.12, "Mg": -1.51},
            dry_run=True,
        )

        # Should generate sub-flows for both Mg and O vacancies
        defect_subflows = [job for job in parent_flow.jobs if isinstance(job, Flow)]
        assert len(defect_subflows) >= 2
        assert all(hasattr(flow, "name") for flow in defect_subflows)
        assert all(hasattr(flow, "jobs") for flow in defect_subflows)

    def test_from_pristine_substitutions(self, mgo_unitcell):
        """Test generating substitution flows from pristine structure."""
        from jobflow import Flow

        from atomate2.siesta.flows.defects import DefectFlowMaker

        parent_flow = DefectFlowMaker.from_pristine_structure(
            mgo_unitcell,
            defect_type="substitution",
            species="Mg",
            dopants="Li",
            charge_states=[0],
            epsilon_static=9.8,
            chemical_potentials={"O": -5.12, "Mg": -1.51, "Li": -1.9},
            dry_run=True,
        )

        # Should generate Li_Mg sub-flow(s)
        defect_subflows = [job for job in parent_flow.jobs if isinstance(job, Flow)]
        assert len(defect_subflows) >= 1
        assert all(hasattr(flow, "name") for flow in defect_subflows)

    def test_from_pristine_antisites(self, mgo_unitcell):
        """Test generating antisite flows from pristine structure."""
        from jobflow import Flow

        from atomate2.siesta.flows.defects import DefectFlowMaker

        parent_flow = DefectFlowMaker.from_pristine_structure(
            mgo_unitcell,
            defect_type="substitution",
            dopants=None,  # Triggers antisite generation
            charge_states=[0],
            epsilon_static=9.8,
            chemical_potentials={"O": -5.12, "Mg": -1.51},
            dry_run=True,
        )

        # Should generate Mg_O and O_Mg sub-flows
        defect_subflows = [job for job in parent_flow.jobs if isinstance(job, Flow)]
        assert len(defect_subflows) >= 2

    def test_from_pristine_interstitials(self, mgo_unitcell):
        """Test generating interstitial flows from pristine structure."""
        from jobflow import Flow

        from atomate2.siesta.flows.defects import DefectFlowMaker

        result = DefectFlowMaker.from_pristine_structure(
            mgo_unitcell,
            defect_type="interstitial",
            species="Li",
            charge_states=[0],
            epsilon_static=9.8,
            chemical_potentials={"O": -5.12, "Mg": -1.51, "Li": -1.9},
            dry_run=True,
        )

        # Returns a parent Flow when interstitial sites are found, otherwise an
        # empty list when no sites are generated.
        assert isinstance(result, (Flow, list))

    def test_from_pristine_with_supercell(self, mgo_unitcell):
        """Test from_pristine_structure with supercell."""
        from jobflow import Flow

        from atomate2.siesta.flows.defects import DefectFlowMaker

        parent_flow = DefectFlowMaker.from_pristine_structure(
            mgo_unitcell,
            defect_type="vacancy",
            species="O",
            supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            charge_states=[0, +2],
            epsilon_static=9.8,
            chemical_potentials={"O": -5.12, "Mg": -1.51},
            dry_run=True,
        )

        # Should generate sub-flows for O vacancy with 2 charge states
        defect_subflows = [job for job in parent_flow.jobs if isinstance(job, Flow)]
        assert len(defect_subflows) >= 2

        # Verify structures are supercells. The defect structure is now the
        # first positional argument of the leading job in each sub-flow.
        for flow in defect_subflows:
            defect_structure = flow.jobs[0].function_args[0]
            # Should be 2×2×2 supercell (8× unit cell atoms)
            assert len(defect_structure) >= len(mgo_unitcell) * 8 - 1  # -1 for vacancy

    def test_from_pristine_invalid_type(self, mgo_unitcell):
        """Test from_pristine_structure with invalid defect type."""
        from atomate2.siesta.flows.defects import DefectFlowMaker
        import pytest

        with pytest.raises(ValueError, match="Unknown defect_type"):
            DefectFlowMaker.from_pristine_structure(
                mgo_unitcell,
                defect_type="invalid_type",
            )

    def test_from_pristine_missing_species_for_interstitial(self, mgo_unitcell):
        """Test from_pristine_structure errors when species not provided for interstitials."""
        from atomate2.siesta.flows.defects import DefectFlowMaker
        import pytest

        with pytest.raises(
            ValueError, match="For interstitial defects, 'species' .* must be specified"
        ):
            DefectFlowMaker.from_pristine_structure(
                mgo_unitcell,
                defect_type="interstitial",
                species=None,  # Should raise error
            )


class TestKumagaiCorrection:
    """Tests for Kumagai-Oba correction scheme."""

    def test_kumagai_instantiation(self):
        """Test that Kumagai correction can be instantiated with basic parameters."""
        from atomate2.siesta.flows.defects.corrections import KumagaiCorrection

        correction = KumagaiCorrection(epsilon_static=10.0)

        assert correction.name == "Kumagai-Oba"
        assert correction.charge_model == "point+atomic_sampling"
        assert correction.epsilon_static == 10.0
        assert correction.requires_dielectric
        assert correction.requires_potential_data
        assert not correction.supports_anisotropic  # Kumagai is isotropic

    def test_kumagai_correction_dry_run(self, mgo_supercell):
        """
        Test Kumagai correction in dry-run mode (no potential data).

        Should return lattice term only with a warning.
        """
        from atomate2.siesta.flows.defects.corrections import KumagaiCorrection

        # Create defect and host structures
        defect_structure = mgo_supercell.copy()
        host_structure = mgo_supercell.copy()

        # Initialize correction
        correction = KumagaiCorrection(epsilon_static=9.8)

        # Calculate correction without potential data (dry-run mode)
        result = correction.calculate_correction(
            defect_structure=defect_structure,
            host_structure=host_structure,
            charge_state=2,
            defect_energy=-1000.0,
            host_energy=-1020.0,
            defect_site=[0.25, 0.25, 0.25],
            potential_data=None,  # No potential data = dry-run
        )

        # Verify basic properties
        assert result.scheme_name == "Kumagai-Oba"
        assert result.correction_energy > 0  # Positive correction for charged defect
        assert result.charge_model == "point+atomic_sampling"
        assert result.converged

        # Verify warning about missing potential data
        assert len(result.warnings) > 0
        assert any("no .VT data" in w for w in result.warnings)

        # Verify metadata
        assert "lattice_term_eV" in result.metadata
        assert "alignment_energy_eV" in result.metadata
        assert result.metadata["alignment_energy_eV"] == 0.0  # No alignment in dry-run
        assert result.metadata["alignment_available"] is False

    def test_kumagai_correction_parameters(self):
        """Test Kumagai correction with custom parameters."""
        from atomate2.siesta.flows.defects.corrections import KumagaiCorrection

        correction = KumagaiCorrection(
            epsilon_static=12.0,
            madelung_constant=2.5,
            sampling_cutoff_fraction=0.75,
            min_sampling_atoms=15,
            outlier_threshold=2.5,
        )

        assert correction.madelung_constant == 2.5
        assert correction.sampling_cutoff_fraction == 0.75
        assert correction.min_sampling_atoms == 15
        assert correction.outlier_threshold == 2.5

    def test_kumagai_in_comparison_flow(self):
        """Test that Kumagai correction can be used in CorrectionComparisonFlowMaker."""
        from atomate2.siesta.flows.defects import CorrectionComparisonFlowMaker

        # Should not raise error
        maker = CorrectionComparisonFlowMaker(
            epsilon_static=9.8,
            defect_type="vacancy",
            charge_state=2,
            correction_schemes=["kumagai", "lany-zunger", "makov-payne", "freysoldt"],
            dry_run=True,
        )

        assert "kumagai" in maker.correction_schemes
        assert maker.epsilon_static == 9.8


# =============================================================================
# SRH Recombination Analysis Tests
# =============================================================================


class TestCaptureParameters:
    """Tests for CaptureParameters class."""

    def test_from_defaults(self):
        """Test creating default capture parameters."""
        from atomate2.siesta.flows.defects.analysis import CaptureParameters

        params = CaptureParameters.from_defaults(temperature=300.0)

        assert params.sigma_n == 1e-15
        assert params.sigma_p == 1e-15
        assert params.method == "default"
        assert params.temperature == 300.0

    def test_from_defaults_custom_temperature(self):
        """Test default parameters with custom temperature."""
        from atomate2.siesta.flows.defects.analysis import CaptureParameters

        params = CaptureParameters.from_defaults(temperature=500.0)

        assert params.temperature == 500.0
        assert params.sigma_n == 1e-15  # Still default values

    def test_from_empirical_fallback(self):
        """Test empirical parameters (currently falls back to defaults)."""
        from atomate2.siesta.flows.defects.analysis import CaptureParameters

        params = CaptureParameters.from_empirical("Si", "vacancy", temperature=300.0)

        # Should fall back to defaults for now
        assert params.method == "default"
        assert params.sigma_n == 1e-15
        assert params.sigma_p == 1e-15

    def test_custom_parameters(self):
        """Test creating custom capture parameters."""
        from atomate2.siesta.flows.defects.analysis import CaptureParameters

        params = CaptureParameters(
            sigma_n=5e-16,
            sigma_p=2e-15,
            method="calculated",
            temperature=350.0,
        )

        assert params.sigma_n == 5e-16
        assert params.sigma_p == 2e-15
        assert params.method == "calculated"
        assert params.temperature == 350.0


class TestSRHAnalyzer:
    """Tests for SRHAnalyzer class."""

    @pytest.fixture
    def simple_formation_diagram(self):
        """Create a simple formation energy diagram for testing."""
        from atomate2.siesta.flows.defects.analysis import (
            FormationEnergyDiagram,
        )
        from atomate2.siesta.flows.defects.analysis.formation_energy import (
            DefectFormationEnergyData,
        )

        defects = [
            DefectFormationEnergyData(
                defect_name="V_O",
                defect_type="vacancy",
                charge_states=[0, +1, +2],
                formation_energies=[3.0, 2.5, 2.8],
                corrected=True,
            ),
        ]

        return FormationEnergyDiagram(
            defects=defects,
            bandgap=5.0,
            vbm_energy=0.0,
        )

    @pytest.fixture
    def srh_analyzer(self, simple_formation_diagram):
        """Create SRHAnalyzer for testing."""
        from atomate2.siesta.flows.defects.analysis import SRHAnalyzer

        return SRHAnalyzer(
            formation_diagram=simple_formation_diagram,
            bandgap=5.0,
            effective_mass_electron=1.0,
            effective_mass_hole=1.0,
        )

    def test_analyzer_initialization(self, srh_analyzer):
        """Test SRHAnalyzer initialization."""
        assert srh_analyzer.bandgap == 5.0
        assert srh_analyzer.m_e_eff == 1.0
        assert srh_analyzer.m_h_eff == 1.0

    def test_calculate_intrinsic_carrier_concentration(self, srh_analyzer):
        """Test intrinsic carrier concentration calculation."""
        n_i = srh_analyzer.calculate_intrinsic_carrier_concentration(temperature=300.0)

        # Should be positive and reasonable for 5 eV bandgap
        assert n_i > 0
        assert n_i < 1e10  # Very small for large bandgap

    def test_intrinsic_concentration_temperature_dependence(self, srh_analyzer):
        """Test that n_i increases with temperature."""
        n_i_300 = srh_analyzer.calculate_intrinsic_carrier_concentration(300.0)
        n_i_600 = srh_analyzer.calculate_intrinsic_carrier_concentration(600.0)

        # Higher temperature should give higher concentration
        assert n_i_600 > n_i_300

    def test_calculate_thermal_velocity(self, srh_analyzer):
        """Test thermal velocity calculation."""
        v_th = srh_analyzer.calculate_thermal_velocity(
            temperature=300.0, mass_ratio=1.0
        )

        # Should be on order of 10^7 cm/s at room temperature
        assert v_th > 1e6
        assert v_th < 1e8

    def test_thermal_velocity_temperature_dependence(self, srh_analyzer):
        """Test that thermal velocity increases with temperature."""
        v_300 = srh_analyzer.calculate_thermal_velocity(300.0)
        v_600 = srh_analyzer.calculate_thermal_velocity(600.0)

        # v_th ∝ sqrt(T), so doubling T should increase by ~sqrt(2)
        assert v_600 / v_300 == pytest.approx(pytest.approx(2**0.5, rel=0.1))

    def test_calculate_lifetimes(self, srh_analyzer):
        """Test SRH lifetime calculation."""
        from atomate2.siesta.flows.defects.analysis import CaptureParameters

        capture_params = CaptureParameters.from_defaults(temperature=300.0)
        defect_concentration = 1e16  # cm^-3

        lifetimes = srh_analyzer.calculate_lifetimes(
            defect_concentration=defect_concentration,
            capture_params=capture_params,
            defect_name="V_O",
        )

        assert lifetimes.defect_name == "V_O"
        assert lifetimes.defect_concentration == defect_concentration
        assert lifetimes.tau_n > 0
        assert lifetimes.tau_p > 0
        assert lifetimes.tau_eff > 0
        assert lifetimes.tau_eff < min(lifetimes.tau_n, lifetimes.tau_p)

    def test_lifetime_concentration_dependence(self, srh_analyzer):
        """Test that lifetimes decrease with defect concentration."""
        from atomate2.siesta.flows.defects.analysis import CaptureParameters

        capture_params = CaptureParameters.from_defaults(temperature=300.0)

        lifetimes_low = srh_analyzer.calculate_lifetimes(
            defect_concentration=1e15,
            capture_params=capture_params,
        )

        lifetimes_high = srh_analyzer.calculate_lifetimes(
            defect_concentration=1e17,
            capture_params=capture_params,
        )

        # Higher defect concentration → shorter lifetimes
        assert lifetimes_high.tau_n < lifetimes_low.tau_n
        assert lifetimes_high.tau_p < lifetimes_low.tau_p

    def test_minority_carrier_lifetime(self, srh_analyzer):
        """Test minority carrier lifetime property."""
        from atomate2.siesta.flows.defects.analysis import CaptureParameters

        # Create asymmetric capture parameters
        capture_params = CaptureParameters(
            sigma_n=1e-15,
            sigma_p=1e-14,  # 10x larger
            method="custom",
            temperature=300.0,
        )

        lifetimes = srh_analyzer.calculate_lifetimes(
            defect_concentration=1e16,
            capture_params=capture_params,
        )

        # Minority carrier lifetime should be the minimum
        assert lifetimes.minority_carrier_lifetime == min(
            lifetimes.tau_n, lifetimes.tau_p
        )
        # Since σ_p is larger, τ_p should be smaller
        assert lifetimes.tau_p < lifetimes.tau_n

    def test_calculate_srh_rate(self, srh_analyzer):
        """Test SRH recombination rate calculation."""
        from atomate2.siesta.flows.defects.analysis import CaptureParameters

        capture_params = CaptureParameters.from_defaults(temperature=300.0)
        lifetimes = srh_analyzer.calculate_lifetimes(
            defect_concentration=1e16,
            capture_params=capture_params,
        )

        # Mid-gap defect
        defect_level = 2.5  # eV (mid-gap for 5 eV bandgap)

        rec_rate, gen_rate = srh_analyzer.calculate_srh_rate(
            defect_level=defect_level,
            electron_concentration=1e17,
            hole_concentration=1e16,
            lifetimes=lifetimes,
            temperature=300.0,
        )

        # Rates should be positive
        assert gen_rate > 0
        # Recombination rate can be positive or negative depending on n*p vs n_i^2
        assert isinstance(rec_rate, float)


@pytest.fixture
def full_defect_data():
    """Create full defect analysis data (shared across SRH test classes)."""
    from atomate2.siesta.flows.defects.analysis import (
        FormationEnergyDiagram,
        ConcentrationResult,
        DefectConcentration,
    )
    from atomate2.siesta.flows.defects.analysis.formation_energy import (
        DefectFormationEnergyData,
        ChargeTransitionLevel,
    )

    # Formation energy diagram
    defects = [
        DefectFormationEnergyData(
            defect_name="V_O",
            defect_type="vacancy",
            charge_states=[0, +1, +2],
            formation_energies=[3.0, 2.5, 2.8],
            corrected=True,
        ),
    ]

    diagram = FormationEnergyDiagram(
        defects=defects,
        bandgap=5.0,
        vbm_energy=0.0,
    )

    # CTLs
    ctls = [
        ChargeTransitionLevel(
            defect_name="V_O",
            q1=0,
            q2=+1,
            fermi_level=2.0,
            formation_energy=2.5,
        ),
    ]

    # Concentration result
    defect_concentrations = [
        DefectConcentration(
            defect_name="V_O",
            charge_state=0,
            concentration=1e16,
            formation_energy=3.0,
        ),
        DefectConcentration(
            defect_name="V_O",
            charge_state=+1,
            concentration=5e15,
            formation_energy=2.5,
        ),
    ]

    concentration_result = ConcentrationResult(
        temperature=300.0,
        fermi_level=2.5,
        electron_concentration=1e17,
        hole_concentration=1e14,
        defect_concentrations=defect_concentrations,
        charge_neutrality_error=0.0,
    )

    return diagram, ctls, concentration_result


class TestSRHAnalysisIntegration:
    """Integration tests for full SRH analysis workflow."""

    def test_full_srh_analysis(self, full_defect_data):
        """Test complete SRH analysis workflow."""
        from atomate2.siesta.flows.defects.analysis import SRHAnalyzer

        diagram, ctls, concentration_result = full_defect_data

        analyzer = SRHAnalyzer(
            formation_diagram=diagram,
            bandgap=5.0,
            effective_mass_electron=1.0,
            effective_mass_hole=1.0,
        )

        result = analyzer.analyze_from_concentration_result(
            concentration_result=concentration_result,
            ctls=ctls,
            capture_params=None,  # Use defaults
        )

        # Verify basic results
        assert result.temperature == 300.0
        assert result.bandgap == 5.0
        assert result.intrinsic_carrier_concentration > 0
        assert len(result.defect_results) == 2  # Two charge states
        assert result.total_recombination_rate != 0
        assert result.dominant_defect != ""

    def test_srh_with_custom_capture_params(self, full_defect_data):
        """Test SRH analysis with custom capture parameters."""
        from atomate2.siesta.flows.defects.analysis import (
            SRHAnalyzer,
            CaptureParameters,
        )

        diagram, ctls, concentration_result = full_defect_data

        # Custom capture params for V_O
        capture_params = {
            "V_O": CaptureParameters(
                sigma_n=5e-16,
                sigma_p=2e-15,
                method="custom",
                temperature=300.0,
            ),
        }

        analyzer = SRHAnalyzer(
            formation_diagram=diagram,
            bandgap=5.0,
        )

        result = analyzer.analyze_from_concentration_result(
            concentration_result=concentration_result,
            ctls=ctls,
            capture_params=capture_params,
        )

        # Verify that custom parameters were used
        for dr in result.defect_results:
            assert dr.lifetimes.capture_params.method == "custom"
            assert dr.lifetimes.capture_params.sigma_n == 5e-16
            assert dr.lifetimes.capture_params.sigma_p == 2e-15

    def test_srh_dominant_defect_identification(self, full_defect_data):
        """Test that dominant defect is correctly identified."""
        from atomate2.siesta.flows.defects.analysis import SRHAnalyzer

        diagram, ctls, concentration_result = full_defect_data

        analyzer = SRHAnalyzer(
            formation_diagram=diagram,
            bandgap=5.0,
        )

        result = analyzer.analyze_from_concentration_result(
            concentration_result=concentration_result,
            ctls=ctls,
        )

        # Find the defect with maximum recombination rate
        max_rate = max(abs(d.recombination_rate) for d in result.defect_results)
        expected_dominant = [
            d.defect_name
            for d in result.defect_results
            if abs(d.recombination_rate) == max_rate
        ][0]

        assert result.dominant_defect == expected_dominant


class TestSRHJobFunctions:
    """Tests for SRH job functions."""

    def test_calculate_srh_analysis_job(self, full_defect_data):
        """Test SRH analysis job function."""
        from atomate2.siesta.flows.defects.analysis import calculate_srh_analysis_job

        diagram, ctls, concentration_result = full_defect_data

        # Create job (don't run it, just verify it can be created)
        job_obj = calculate_srh_analysis_job(
            formation_diagram=diagram,
            concentration_result=concentration_result,
            ctls=ctls,
            bandgap=5.0,
        )

        assert job_obj.function.__name__ == "calculate_srh_analysis_job"

    def test_write_srh_summary_job(self, full_defect_data, tmp_path):
        """Test SRH summary writing job."""
        from atomate2.siesta.flows.defects.analysis import (
            SRHAnalyzer,
            write_srh_summary_job,
        )

        diagram, ctls, concentration_result = full_defect_data

        # Generate SRH result
        analyzer = SRHAnalyzer(diagram, bandgap=5.0)
        srh_result = analyzer.analyze_from_concentration_result(
            concentration_result, ctls
        )

        # Create job
        job_obj = write_srh_summary_job(
            srh_result=srh_result,
            directory=str(tmp_path),
        )

        assert job_obj.function.__name__ == "write_srh_summary_job"


class TestSRHPlotting:
    """Tests for SRH plotting functions."""

    def test_plot_srh_lifetimes(self, full_defect_data, tmp_path):
        """Test SRH lifetime plotting."""
        from atomate2.siesta.flows.defects.analysis import (
            SRHAnalyzer,
            plot_srh_lifetimes,
        )

        diagram, ctls, concentration_result = full_defect_data

        # Generate SRH result
        analyzer = SRHAnalyzer(diagram, bandgap=5.0)
        srh_result = analyzer.analyze_from_concentration_result(
            concentration_result, ctls
        )

        # Plot lifetimes
        output_file = tmp_path / "srh_lifetimes.png"
        plot_srh_lifetimes(srh_result, filename=output_file)

        # Verify file was created
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_plot_srh_recombination_rates(self, full_defect_data, tmp_path):
        """Test SRH recombination rate plotting."""
        from atomate2.siesta.flows.defects.analysis import (
            SRHAnalyzer,
            plot_srh_recombination_rates,
        )

        diagram, ctls, concentration_result = full_defect_data

        # Generate SRH result
        analyzer = SRHAnalyzer(diagram, bandgap=5.0)
        srh_result = analyzer.analyze_from_concentration_result(
            concentration_result, ctls
        )

        # Plot recombination rates
        output_file = tmp_path / "srh_rates.png"
        plot_srh_recombination_rates(srh_result, filename=output_file)

        # Verify file was created
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_write_srh_summary(self, full_defect_data, tmp_path):
        """Test writing SRH summary to text file."""
        from atomate2.siesta.flows.defects.analysis import (
            SRHAnalyzer,
            write_srh_summary,
        )

        diagram, ctls, concentration_result = full_defect_data

        # Generate SRH result
        analyzer = SRHAnalyzer(diagram, bandgap=5.0)
        srh_result = analyzer.analyze_from_concentration_result(
            concentration_result, ctls
        )

        # Write summary
        output_file = tmp_path / "srh_summary.txt"
        write_srh_summary(srh_result, filename=output_file)

        # Verify file was created and has content
        assert output_file.exists()
        content = output_file.read_text()
        assert "SRH RECOMBINATION ANALYSIS SUMMARY" in content
        assert "Total recombination rate:" in content
        assert "Dominant defect:" in content
        assert "Carrier lifetimes:" in content


# ============================================================================
# Surface-Aware Defect Generation Tests
# ============================================================================


@pytest.fixture
def mos2_slab():
    """
    MoS₂ slab structure for surface defect testing.

    Returns
    -------
    Structure
        MoS₂ slab with 3 layers (9 atoms total), vacuum in z-direction
    """
    lattice = Lattice.from_parameters(
        a=3.16, b=3.16, c=20.0, alpha=90, beta=90, gamma=120
    )
    # Each MoS2 trilayer is kept within the generator's default
    # layer_tolerance (0.7 Å) in Cartesian z so that one trilayer is grouped
    # as a single layer (1 Mo + 2 S). Trilayer centres are 3 Å apart (well
    # beyond the tolerance) so the three layers stay distinct.
    return Structure(
        lattice,
        ["Mo", "S", "S", "Mo", "S", "S", "Mo", "S", "S"],
        [
            # Bottom layer (centre z = 6.0 Å)
            [0.333, 0.667, 0.300],  # Mo
            [0.667, 0.333, 0.285],  # S
            [0.667, 0.333, 0.315],  # S
            # Middle layer (centre z = 9.0 Å)
            [0.000, 0.000, 0.450],  # Mo
            [0.333, 0.667, 0.435],  # S
            [0.333, 0.667, 0.465],  # S
            # Top layer / surface (centre z = 12.0 Å)
            [0.333, 0.667, 0.600],  # Mo
            [0.667, 0.333, 0.585],  # S (lower)
            [0.667, 0.333, 0.615],  # S (upper) - Top surface!
        ],
    )


class TestSurfaceVacancyGenerator:
    """Tests for SurfaceVacancyGenerator class."""

    def test_init(self, mos2_slab):
        """Test SurfaceVacancyGenerator initialization."""
        from atomate2.siesta.flows.defects.generation import SurfaceVacancyGenerator

        generator = SurfaceVacancyGenerator(
            slab_structure=mos2_slab,
            surface_layers=1,
            surface_side="top",
            layer_tolerance=0.5,
            use_ghost_atoms=True,
            use_in_plane_symmetry=True,
        )

        assert generator.slab_structure == mos2_slab
        assert generator.surface_layers == 1
        assert generator.surface_side == "top"
        assert generator.layer_tolerance == 0.5
        assert generator.use_ghost_atoms is True
        assert generator.use_in_plane_symmetry is True

        # Check that layers were identified
        assert len(generator.layers) > 0

    def test_layer_identification(self, mos2_slab):
        """Test automatic layer identification by z-coordinate."""
        from atomate2.siesta.flows.defects.generation import SurfaceVacancyGenerator

        generator = SurfaceVacancyGenerator(
            slab_structure=mos2_slab,
            surface_layers=1,
            surface_side="top",
            layer_tolerance=0.7,
        )

        # Should identify 3 layers (bottom, middle, top)
        # Each layer has 3 atoms (1 Mo + 2 S)
        assert len(generator.layers) == 3

        # Verify layers are sorted by z-coordinate (bottom to top)
        for i in range(len(generator.layers) - 1):
            assert generator.layers[i].z_position < generator.layers[i + 1].z_position

        # Check atom counts per layer
        for layer in generator.layers:
            assert len(layer.atom_indices) == 3  # 1 Mo + 2 S

    def test_surface_site_identification_top(self, mos2_slab):
        """Test surface site identification (top surface only)."""
        from atomate2.siesta.flows.defects.generation import SurfaceVacancyGenerator

        generator = SurfaceVacancyGenerator(
            slab_structure=mos2_slab,
            surface_layers=1,
            surface_side="top",
            use_in_plane_symmetry=False,  # Get all sites, no reduction
        )

        # Get S sites on top surface
        surface_sites = generator.get_surface_sites(species="S")

        # Top layer has 2 S atoms
        assert len(surface_sites) == 2

        # All should be marked as top surface
        for site in surface_sites:
            assert site["is_top_surface"] is True
            assert site["is_bottom_surface"] is False

    def test_surface_site_identification_bottom(self, mos2_slab):
        """Test surface site identification (bottom surface only)."""
        from atomate2.siesta.flows.defects.generation import SurfaceVacancyGenerator

        generator = SurfaceVacancyGenerator(
            slab_structure=mos2_slab,
            surface_layers=1,
            surface_side="bottom",
            use_in_plane_symmetry=False,
        )

        # Get S sites on bottom surface
        surface_sites = generator.get_surface_sites(species="S")

        # Bottom layer has 2 S atoms
        assert len(surface_sites) == 2

        # All should be marked as bottom surface
        for site in surface_sites:
            assert site["is_top_surface"] is False
            assert site["is_bottom_surface"] is True

    def test_surface_site_identification_both(self, mos2_slab):
        """Test surface site identification (both surfaces)."""
        from atomate2.siesta.flows.defects.generation import SurfaceVacancyGenerator

        generator = SurfaceVacancyGenerator(
            slab_structure=mos2_slab,
            surface_layers=1,
            surface_side="both",
            use_in_plane_symmetry=False,
        )

        # Get S sites on both surfaces
        surface_sites = generator.get_surface_sites(species="S")

        # Both top and bottom layers: 2 S atoms each = 4 total
        assert len(surface_sites) == 4

        # Should have both top and bottom sites
        top_sites = [s for s in surface_sites if s["is_top_surface"]]
        bottom_sites = [s for s in surface_sites if s["is_bottom_surface"]]
        assert len(top_sites) == 2
        assert len(bottom_sites) == 2

    def test_multiple_surface_layers(self, mos2_slab):
        """Test subsurface defect generation (top 2 layers)."""
        from atomate2.siesta.flows.defects.generation import SurfaceVacancyGenerator

        generator = SurfaceVacancyGenerator(
            slab_structure=mos2_slab,
            surface_layers=2,  # Top 2 layers
            surface_side="top",
            use_in_plane_symmetry=False,
        )

        # Get S sites in top 2 layers
        surface_sites = generator.get_surface_sites(species="S")

        # Top 2 layers: 2 S atoms each = 4 total
        assert len(surface_sites) == 4

    def test_generate_defects_basic(self, mos2_slab):
        """Test basic defect generation."""
        from atomate2.siesta.flows.defects.generation import SurfaceVacancyGenerator

        generator = SurfaceVacancyGenerator(
            slab_structure=mos2_slab,
            surface_layers=1,
            surface_side="top",
            use_ghost_atoms=True,
            use_in_plane_symmetry=False,
        )

        # Generate S vacancies on top surface
        defects = generator.generate_defects(species="S")

        # Should have 2 defects (2 S atoms in top layer)
        assert len(defects) == 2

        # Check defect properties
        for defect in defects:
            assert defect["species"] == "S"
            assert defect["defect_type"] == "surface_vacancy"
            assert defect["use_ghost"] is True
            assert defect["charge_state"] == 0  # Default
            assert "structure" in defect
            assert "host_structure" in defect

            # Verify structure has ghost atom
            assert len(defect["structure"]) == len(mos2_slab)
            assert "ghost_tags" in defect["structure"].site_properties

    def test_generate_defects_multiple_charge_states(self, mos2_slab):
        """Test defect generation with multiple charge states."""
        from atomate2.siesta.flows.defects.generation import SurfaceVacancyGenerator

        generator = SurfaceVacancyGenerator(
            slab_structure=mos2_slab,
            surface_layers=1,
            surface_side="top",
            use_ghost_atoms=True,
            use_in_plane_symmetry=False,
        )

        # Generate S vacancies with multiple charge states
        charge_states = [0, +1, +2]
        defects = generator.generate_defects(species="S", charge_states=charge_states)

        # Should have 2 sites × 3 charge states = 6 defects
        assert len(defects) == 6

        # Check that all charge states are present
        charges_found = set(d["charge_state"] for d in defects)
        assert charges_found == set(charge_states)

    def test_generate_defects_with_supercell(self, mos2_slab):
        """Test defect generation with supercell (in-plane expansion)."""
        from atomate2.siesta.flows.defects.generation import SurfaceVacancyGenerator

        generator = SurfaceVacancyGenerator(
            slab_structure=mos2_slab,
            surface_layers=1,
            surface_side="top",
            use_ghost_atoms=True,
            use_in_plane_symmetry=False,
        )

        # Generate with 2×2×1 supercell (in-plane expansion only)
        supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 1]]
        defects = generator.generate_defects(
            species="S", supercell_matrix=supercell_matrix
        )

        # Should still have 2 defects (1 per unique site, before supercell)
        # Actually, with no symmetry, we get all sites
        # After supercell, we find the corresponding site
        assert len(defects) == 2

        # Check that host structure is a supercell
        for defect in defects:
            # Supercell should have 4× atoms (2×2 in-plane)
            assert len(defect["host_structure"]) == len(mos2_slab) * 4

    def test_invalid_surface_side(self, mos2_slab):
        """Test that invalid surface_side raises error."""
        from atomate2.siesta.flows.defects.generation import SurfaceVacancyGenerator

        with pytest.raises(ValueError, match="surface_side must be"):
            SurfaceVacancyGenerator(
                slab_structure=mos2_slab,
                surface_side="invalid",  # Invalid value
            )

    def test_invalid_surface_layers(self, mos2_slab):
        """Test that invalid surface_layers raises error."""
        from atomate2.siesta.flows.defects.generation import SurfaceVacancyGenerator

        with pytest.raises(ValueError, match="surface_layers must be"):
            SurfaceVacancyGenerator(
                slab_structure=mos2_slab,
                surface_layers=0,  # Must be >= 1
            )

    def test_ghost_atom_vs_complete_removal(self, mos2_slab):
        """Test difference between ghost atom and complete removal."""
        from atomate2.siesta.flows.defects.generation import SurfaceVacancyGenerator

        # With ghost atoms
        generator_ghost = SurfaceVacancyGenerator(
            slab_structure=mos2_slab,
            surface_layers=1,
            surface_side="top",
            use_ghost_atoms=True,
            use_in_plane_symmetry=False,
        )

        defects_ghost = generator_ghost.generate_defects(species="S")

        # Without ghost atoms
        generator_no_ghost = SurfaceVacancyGenerator(
            slab_structure=mos2_slab,
            surface_layers=1,
            surface_side="top",
            use_ghost_atoms=False,
            use_in_plane_symmetry=False,
        )

        defects_no_ghost = generator_no_ghost.generate_defects(species="S")

        # Same number of defects
        assert len(defects_ghost) == len(defects_no_ghost)

        # With ghost: structure has same number of atoms
        for defect in defects_ghost:
            assert len(defect["structure"]) == len(mos2_slab)
            assert "ghost_tags" in defect["structure"].site_properties

        # Without ghost: structure has one fewer atom
        for defect in defects_no_ghost:
            assert len(defect["structure"]) == len(mos2_slab) - 1
            # No ghost_tags property
            assert "ghost_tags" not in defect["structure"].site_properties

    def test_layer_info_attributes(self, mos2_slab):
        """Test LayerInfo attributes."""
        from atomate2.siesta.flows.defects.generation import SurfaceVacancyGenerator

        generator = SurfaceVacancyGenerator(
            slab_structure=mos2_slab,
            surface_layers=1,
            surface_side="top",
        )

        # Check top layer attributes
        top_layer = generator.layers[-1]  # Last layer = top
        assert hasattr(top_layer, "layer_index")
        assert hasattr(top_layer, "z_position")
        assert hasattr(top_layer, "atom_indices")
        assert hasattr(top_layer, "species_count")

        # Check species count
        assert "Mo" in top_layer.species_count
        assert "S" in top_layer.species_count
        assert top_layer.species_count["Mo"] == 1
        assert top_layer.species_count["S"] == 2


class TestSurfaceVacancyGeneratorIntegration:
    """Integration tests for SurfaceVacancyGenerator with DefectFlowMaker."""

    def test_integration_with_defect_flow_maker(self, mos2_slab):
        """Test integration of SurfaceVacancyGenerator with DefectFlowMaker."""
        from atomate2.siesta.flows.defects import (
            DefectFlowMaker,
            SurfaceVacancyGenerator,
        )
        from jobflow import run_locally

        # Generate surface defect
        generator = SurfaceVacancyGenerator(
            slab_structure=mos2_slab,
            surface_layers=1,
            surface_side="top",
            use_ghost_atoms=True,
            use_in_plane_symmetry=False,
        )

        defects = generator.generate_defects(species="S", charge_states=[0])

        # Take first defect
        defect_info = defects[0]

        # Create DefectFlowMaker with 2D slab correction
        flow_maker = DefectFlowMaker(
            epsilon_static=15.0,
            epsilon_parallel=15.0,
            epsilon_perpendicular=7.0,
            correction_scheme="slab2d",
            defect_type="vacancy",
            charge_state=0,
            use_ghost_atoms=True,
            dry_run=True,
            chemical_potentials={"S": -2.5},
        )

        # Create workflow
        flow = flow_maker.make(
            defect_structure=defect_info["structure"],
            host_structure=defect_info["host_structure"],
            defect_site=defect_info["frac_coords"],
            defect_species="S",
        )

        # Verify flow was created
        assert flow is not None
        assert len(flow.jobs) > 0

        # Run dry-run
        results = run_locally(flow, create_folders=True, ensure_success=True)
        assert results is not None
