"""
Tests for specialized SIESTA schema modules (phonon, surface, adsorption).

These tests validate:
- PhononDocument and ThermalProperties schemas
- SurfaceEnergyDocument and TerminationData schemas
- AdsorptionScanDocument and AdsorptionSiteResult schemas
- Pydantic model creation, validation, and serialization
- Field presence and types
- Edge cases and error handling
"""

import pytest
from pymatgen.core import Lattice, Structure


class TestPhononSchemas:
    """Tests for phonon calculation schemas."""

    def test_thermal_properties_creation(self):
        """Test ThermalProperties model creation."""
        from atomate2.siesta.schemas.phonon import ThermalProperties

        thermal = ThermalProperties(
            temperatures=[0, 100, 200, 300],
            free_energy=[-10.0, -10.5, -11.0, -11.5],
            entropy=[0.0, 0.5, 1.0, 1.5],
            heat_capacity=[0.0, 5.0, 10.0, 15.0],
        )

        assert len(thermal.temperatures) == 4
        assert thermal.temperatures[0] == 0
        assert thermal.free_energy[3] == -11.5
        assert thermal.entropy[2] == 1.0
        assert thermal.heat_capacity[1] == 5.0

    def test_thermal_properties_serialization(self):
        """Test ThermalProperties serialization."""
        from atomate2.siesta.schemas.phonon import ThermalProperties

        thermal = ThermalProperties(
            temperatures=[300, 400],
            free_energy=[-11.5, -12.0],
            entropy=[1.5, 2.0],
            heat_capacity=[15.0, 18.0],
        )

        # Serialize to dict
        thermal_dict = thermal.model_dump()
        assert isinstance(thermal_dict, dict)
        assert "temperatures" in thermal_dict
        assert thermal_dict["temperatures"] == [300, 400]

        # Round-trip
        thermal_restored = ThermalProperties(**thermal_dict)
        assert thermal_restored.temperatures == thermal.temperatures

    def test_phonon_document_creation_minimal(self):
        """Test PhononDocument creation with minimal fields."""
        from atomate2.siesta.schemas.phonon import PhononDocument

        si_lattice = Lattice.cubic(5.43)
        structure = Structure(si_lattice, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])

        phonon_doc = PhononDocument(
            structure=structure,
            supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            displacement=0.01,
            symprec=1e-5,
            n_displacements=24,
            force_constants=[[1.0, 0.0], [0.0, 1.0]],
            has_imaginary_frequencies=False,
            min_frequency=-0.1,
            max_frequency=20.5,
        )

        assert phonon_doc.displacement == 0.01
        assert phonon_doc.n_displacements == 24
        assert phonon_doc.has_imaginary_frequencies is False
        assert phonon_doc.thermal_properties is None

    def test_phonon_document_with_thermal_properties(self):
        """Test PhononDocument with thermal properties."""
        from atomate2.siesta.schemas.phonon import PhononDocument, ThermalProperties

        si_lattice = Lattice.cubic(5.43)
        structure = Structure(si_lattice, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])

        thermal = ThermalProperties(
            temperatures=[300],
            free_energy=[-11.5],
            entropy=[1.5],
            heat_capacity=[15.0],
        )

        phonon_doc = PhononDocument(
            structure=structure,
            supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            displacement=0.01,
            symprec=1e-5,
            n_displacements=24,
            force_constants=[[1.0, 0.0], [0.0, 1.0]],
            has_imaginary_frequencies=False,
            min_frequency=-0.1,
            max_frequency=20.5,
            thermal_properties=thermal,
        )

        assert phonon_doc.thermal_properties is not None
        assert phonon_doc.thermal_properties.temperatures == [300]

    def test_phonon_document_imaginary_frequencies(self):
        """Test PhononDocument with imaginary frequencies flag."""
        from atomate2.siesta.schemas.phonon import PhononDocument

        si_lattice = Lattice.cubic(5.43)
        structure = Structure(si_lattice, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])

        phonon_doc = PhononDocument(
            structure=structure,
            supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            displacement=0.01,
            symprec=1e-5,
            n_displacements=24,
            force_constants=[[1.0]],
            has_imaginary_frequencies=True,  # Structural instability
            min_frequency=-2.5,  # Negative frequency
            max_frequency=18.0,
        )

        assert phonon_doc.has_imaginary_frequencies is True
        assert phonon_doc.min_frequency < 0

    def test_phonon_document_serialization(self):
        """Test PhononDocument serialization."""
        from atomate2.siesta.schemas.phonon import PhononDocument

        si_lattice = Lattice.cubic(5.43)
        structure = Structure(si_lattice, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])

        phonon_doc = PhononDocument(
            structure=structure,
            supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            displacement=0.01,
            symprec=1e-5,
            n_displacements=24,
            force_constants=[[1.0]],
            has_imaginary_frequencies=False,
            min_frequency=0.0,
            max_frequency=20.0,
        )

        # Serialize
        doc_dict = phonon_doc.model_dump()
        assert isinstance(doc_dict, dict)
        assert "structure" in doc_dict
        assert doc_dict["displacement"] == 0.01


class TestSurfaceSchemas:
    """Tests for surface energy calculation schemas."""

    def test_termination_data_creation(self):
        """Test TerminationData model creation."""
        from atomate2.siesta.schemas.surface import TerminationData

        term = TerminationData(
            termination="O",
            surface_energy=0.05,
            surface_energy_Jm2=0.8,
            relative_energy=0.0,
            is_lowest=True,
            slab_energy=-100.5,
            n_formula_units=2.0,
            surface_area=50.0,
            n_atoms=20,
            composition={"Mg": 10, "O": 10},
        )

        assert term.termination == "O"
        assert term.is_lowest is True
        assert term.surface_energy == 0.05
        assert term.n_atoms == 20

    def test_termination_data_with_optional_fields(self):
        """Test TerminationData with optional fields."""
        from atomate2.siesta.schemas.surface import TerminationData

        term = TerminationData(
            termination="Mg",
            surface_energy=0.10,
            surface_energy_Jm2=1.6,
            relative_energy=0.05,
            is_lowest=False,
            slab_energy=-99.5,
            n_formula_units=2.0,
            surface_area=50.0,
            n_atoms=20,
            composition={"Mg": 10, "O": 10},
            thickness=15.0,
            is_symmetric=True,
            z_position=7.5,
            top_composition={"Mg": 5},
            bottom_composition={"Mg": 5},
        )

        assert term.thickness == 15.0
        assert term.is_symmetric is True
        assert term.z_position == 7.5
        assert term.top_composition == {"Mg": 5}

    def test_surface_energy_document_creation(self):
        """Test SurfaceEnergyDocument creation."""
        from atomate2.siesta.schemas.surface import (
            SurfaceEnergyDocument,
            TerminationData,
        )

        term1 = TerminationData(
            termination="O",
            surface_energy=0.05,
            surface_energy_Jm2=0.8,
            relative_energy=0.0,
            is_lowest=True,
            slab_energy=-100.5,
            n_formula_units=2.0,
            surface_area=50.0,
            n_atoms=20,
            composition={"Mg": 10, "O": 10},
        )

        term2 = TerminationData(
            termination="Mg",
            surface_energy=0.10,
            surface_energy_Jm2=1.6,
            relative_energy=0.05,
            is_lowest=False,
            slab_energy=-99.5,
            n_formula_units=2.0,
            surface_area=50.0,
            n_atoms=20,
            composition={"Mg": 10, "O": 10},
        )

        surf_doc = SurfaceEnergyDocument(
            miller_indices=(0, 0, 1),
            formula_units_per_cell=1,
            bulk_energy=-50.0,
            terminations=[term1, term2],
            lowest_termination="O",
        )

        assert surf_doc.miller_indices == (0, 0, 1)
        assert surf_doc.bulk_energy == -50.0
        assert surf_doc.lowest_termination == "O"
        assert len(surf_doc.terminations) == 2

    def test_surface_energy_document_with_statistics(self):
        """Test SurfaceEnergyDocument with statistics fields."""
        from atomate2.siesta.schemas.surface import (
            SurfaceEnergyDocument,
            TerminationData,
        )

        term = TerminationData(
            termination="O",
            surface_energy=0.05,
            surface_energy_Jm2=0.8,
            relative_energy=0.0,
            is_lowest=True,
            slab_energy=-100.5,
            n_formula_units=2.0,
            surface_area=50.0,
            n_atoms=20,
            composition={"Mg": 10, "O": 10},
        )

        surf_doc = SurfaceEnergyDocument(
            miller_indices=(1, 1, 0),
            formula_units_per_cell=1,
            bulk_energy=-50.0,
            bulk_energy_per_atom=-2.5,
            terminations=[term],
            lowest_termination="O",
            n_terminations=1,
            energy_spread=0.0,
            slab_directory="/path/to/slabs",
            calculation_method="SIESTA",
        )

        assert surf_doc.n_terminations == 1
        assert surf_doc.energy_spread == 0.0
        assert surf_doc.slab_directory == "/path/to/slabs"
        assert surf_doc.calculation_method == "SIESTA"

    def test_surface_energy_document_serialization(self):
        """Test SurfaceEnergyDocument serialization."""
        from atomate2.siesta.schemas.surface import (
            SurfaceEnergyDocument,
            TerminationData,
        )

        term = TerminationData(
            termination="O",
            surface_energy=0.05,
            surface_energy_Jm2=0.8,
            relative_energy=0.0,
            is_lowest=True,
            slab_energy=-100.5,
            n_formula_units=2.0,
            surface_area=50.0,
            n_atoms=20,
            composition={"Mg": 10, "O": 10},
        )

        surf_doc = SurfaceEnergyDocument(
            miller_indices=(0, 0, 1),
            formula_units_per_cell=1,
            bulk_energy=-50.0,
            terminations=[term],
            lowest_termination="O",
        )

        # Serialize
        doc_dict = surf_doc.model_dump()
        assert isinstance(doc_dict, dict)
        assert "miller_indices" in doc_dict
        assert doc_dict["miller_indices"] == (0, 0, 1)
        assert "terminations" in doc_dict
        assert len(doc_dict["terminations"]) == 1

        # Round-trip
        surf_restored = SurfaceEnergyDocument(**doc_dict)
        assert surf_restored.miller_indices == surf_doc.miller_indices


class TestAdsorptionSchemas:
    """Tests for adsorption calculation schemas."""

    def test_adsorption_site_result_creation(self):
        """Test AdsorptionSiteResult model creation."""
        from atomate2.siesta.schemas.adsorption import AdsorptionSiteResult

        site = AdsorptionSiteResult(
            site_x=0.5,
            site_y=0.5,
            site_x_cart=2.5,
            site_y_cart=2.5,
            adsorption_energy=-1.5,
            adsorption_energy_per_area=-0.03,
            total_energy=-150.0,
            slab_energy=-140.0,
            adsorbate_energy=-8.5,
            surface_area=50.0,
            height=2.0,
            n_atoms=30,
            n_slab_atoms=28,
            n_adsorbate_atoms=2,
        )

        assert site.site_x == 0.5
        assert site.adsorption_energy == -1.5
        assert site.n_atoms == 30
        assert site.height == 2.0

    def test_adsorption_site_result_serialization(self):
        """Test AdsorptionSiteResult serialization."""
        from atomate2.siesta.schemas.adsorption import AdsorptionSiteResult

        site = AdsorptionSiteResult(
            site_x=0.25,
            site_y=0.75,
            site_x_cart=1.25,
            site_y_cart=3.75,
            adsorption_energy=-2.0,
            adsorption_energy_per_area=-0.04,
            total_energy=-151.0,
            slab_energy=-140.0,
            adsorbate_energy=-9.0,
            surface_area=50.0,
            height=2.5,
            n_atoms=32,
            n_slab_atoms=28,
            n_adsorbate_atoms=4,
        )

        # Serialize
        site_dict = site.model_dump()
        assert isinstance(site_dict, dict)
        assert site_dict["site_x"] == 0.25
        assert site_dict["adsorption_energy"] == -2.0

        # Round-trip
        site_restored = AdsorptionSiteResult(**site_dict)
        assert site_restored.site_x == site.site_x

    def test_adsorption_scan_document_creation(self):
        """Test AdsorptionScanDocument creation."""
        from atomate2.siesta.schemas.adsorption import (
            AdsorptionScanDocument,
            AdsorptionSiteResult,
        )

        site1 = AdsorptionSiteResult(
            site_x=0.0,
            site_y=0.0,
            site_x_cart=0.0,
            site_y_cart=0.0,
            adsorption_energy=-1.5,
            adsorption_energy_per_area=-0.03,
            total_energy=-150.0,
            slab_energy=-140.0,
            adsorbate_energy=-8.5,
            surface_area=50.0,
            height=2.0,
            n_atoms=30,
            n_slab_atoms=28,
            n_adsorbate_atoms=2,
        )

        site2 = AdsorptionSiteResult(
            site_x=0.5,
            site_y=0.5,
            site_x_cart=2.5,
            site_y_cart=2.5,
            adsorption_energy=-2.0,
            adsorption_energy_per_area=-0.04,
            total_energy=-151.0,
            slab_energy=-140.0,
            adsorbate_energy=-9.0,
            surface_area=50.0,
            height=2.0,
            n_atoms=30,
            n_slab_atoms=28,
            n_adsorbate_atoms=2,
        )

        scan_doc = AdsorptionScanDocument(
            slab_formula="MgO",
            adsorbate_formula="CO",
            miller_indices=(0, 0, 1),
            grid_size=(2, 2),
            initial_height=2.0,
            surface_area=50.0,
            slab_thickness=15.0,
            total_sites_scanned=4,
            slab_energy=-140.0,
            adsorbate_energy=-8.5,
            best_site_position=(0.5, 0.5),
            best_adsorption_energy=-2.0,
            best_energy_per_area=-0.04,
            mean_adsorption_energy=-1.75,
            std_adsorption_energy=0.25,
            energy_range=0.5,
            site_results=[site1, site2],
        )

        assert scan_doc.slab_formula == "MgO"
        assert scan_doc.adsorbate_formula == "CO"
        assert scan_doc.grid_size == (2, 2)
        assert scan_doc.total_sites_scanned == 4
        assert scan_doc.best_site_position == (0.5, 0.5)
        assert len(scan_doc.site_results) == 2

    def test_adsorption_scan_document_statistics(self):
        """Test AdsorptionScanDocument energy statistics."""
        from atomate2.siesta.schemas.adsorption import (
            AdsorptionScanDocument,
            AdsorptionSiteResult,
        )

        site = AdsorptionSiteResult(
            site_x=0.0,
            site_y=0.0,
            site_x_cart=0.0,
            site_y_cart=0.0,
            adsorption_energy=-1.5,
            adsorption_energy_per_area=-0.03,
            total_energy=-150.0,
            slab_energy=-140.0,
            adsorbate_energy=-8.5,
            surface_area=50.0,
            height=2.0,
            n_atoms=30,
            n_slab_atoms=28,
            n_adsorbate_atoms=2,
        )

        scan_doc = AdsorptionScanDocument(
            slab_formula="Pt",
            adsorbate_formula="H2",
            grid_size=(10, 10),
            initial_height=2.5,
            surface_area=100.0,
            slab_thickness=20.0,
            total_sites_scanned=100,
            slab_energy=-140.0,
            adsorbate_energy=-8.5,
            best_site_position=(0.25, 0.75),
            best_adsorption_energy=-3.5,
            best_energy_per_area=-0.035,
            mean_adsorption_energy=-2.0,
            std_adsorption_energy=0.8,
            energy_range=2.5,
            site_results=[site],
        )

        # Validate statistics
        assert scan_doc.mean_adsorption_energy == -2.0
        assert scan_doc.std_adsorption_energy == 0.8
        assert scan_doc.energy_range == 2.5
        assert scan_doc.best_adsorption_energy < scan_doc.mean_adsorption_energy

    def test_adsorption_scan_document_serialization(self):
        """Test AdsorptionScanDocument serialization."""
        from atomate2.siesta.schemas.adsorption import (
            AdsorptionScanDocument,
            AdsorptionSiteResult,
        )

        site = AdsorptionSiteResult(
            site_x=0.5,
            site_y=0.5,
            site_x_cart=2.5,
            site_y_cart=2.5,
            adsorption_energy=-1.5,
            adsorption_energy_per_area=-0.03,
            total_energy=-150.0,
            slab_energy=-140.0,
            adsorbate_energy=-8.5,
            surface_area=50.0,
            height=2.0,
            n_atoms=30,
            n_slab_atoms=28,
            n_adsorbate_atoms=2,
        )

        scan_doc = AdsorptionScanDocument(
            slab_formula="MgO",
            adsorbate_formula="CO",
            grid_size=(3, 3),
            initial_height=2.0,
            surface_area=50.0,
            slab_thickness=15.0,
            total_sites_scanned=9,
            slab_energy=-140.0,
            adsorbate_energy=-8.5,
            best_site_position=(0.5, 0.5),
            best_adsorption_energy=-2.0,
            best_energy_per_area=-0.04,
            mean_adsorption_energy=-1.5,
            std_adsorption_energy=0.3,
            energy_range=1.0,
            site_results=[site],
        )

        # Serialize
        doc_dict = scan_doc.model_dump()
        assert isinstance(doc_dict, dict)
        assert doc_dict["slab_formula"] == "MgO"
        assert doc_dict["grid_size"] == (3, 3)
        assert len(doc_dict["site_results"]) == 1

        # Round-trip
        scan_restored = AdsorptionScanDocument(**doc_dict)
        assert scan_restored.slab_formula == scan_doc.slab_formula
        assert scan_restored.grid_size == scan_doc.grid_size


class TestSchemaEdgeCases:
    """Test edge cases for specialized schemas."""

    def test_thermal_properties_empty_lists(self):
        """Test ThermalProperties with empty lists."""
        from atomate2.siesta.schemas.phonon import ThermalProperties

        thermal = ThermalProperties(
            temperatures=[],
            free_energy=[],
            entropy=[],
            heat_capacity=[],
        )

        assert len(thermal.temperatures) == 0

    def test_surface_document_single_termination(self):
        """Test SurfaceEnergyDocument with single termination."""
        from atomate2.siesta.schemas.surface import (
            SurfaceEnergyDocument,
            TerminationData,
        )

        term = TerminationData(
            termination="symmetric",
            surface_energy=0.05,
            surface_energy_Jm2=0.8,
            relative_energy=0.0,
            is_lowest=True,
            slab_energy=-100.5,
            n_formula_units=2.0,
            surface_area=50.0,
            n_atoms=20,
            composition={"Si": 20},
            is_symmetric=True,
        )

        surf_doc = SurfaceEnergyDocument(
            miller_indices=(1, 1, 1),
            formula_units_per_cell=1,
            bulk_energy=-50.0,
            terminations=[term],
            lowest_termination="symmetric",
            n_terminations=1,
            energy_spread=0.0,
        )

        assert surf_doc.n_terminations == 1
        assert surf_doc.energy_spread == 0.0
        assert surf_doc.terminations[0].is_symmetric is True

    def test_adsorption_scan_minimal_grid(self):
        """Test AdsorptionScanDocument with minimal 1×1 grid."""
        from atomate2.siesta.schemas.adsorption import (
            AdsorptionScanDocument,
            AdsorptionSiteResult,
        )

        site = AdsorptionSiteResult(
            site_x=0.0,
            site_y=0.0,
            site_x_cart=0.0,
            site_y_cart=0.0,
            adsorption_energy=-1.5,
            adsorption_energy_per_area=-0.03,
            total_energy=-150.0,
            slab_energy=-140.0,
            adsorbate_energy=-8.5,
            surface_area=50.0,
            height=2.0,
            n_atoms=30,
            n_slab_atoms=28,
            n_adsorbate_atoms=2,
        )

        scan_doc = AdsorptionScanDocument(
            slab_formula="Au",
            adsorbate_formula="O",
            grid_size=(1, 1),
            initial_height=2.0,
            surface_area=25.0,
            slab_thickness=10.0,
            total_sites_scanned=1,
            slab_energy=-140.0,
            adsorbate_energy=-8.5,
            best_site_position=(0.0, 0.0),
            best_adsorption_energy=-1.5,
            best_energy_per_area=-0.03,
            mean_adsorption_energy=-1.5,
            std_adsorption_energy=0.0,
            energy_range=0.0,
            site_results=[site],
        )

        assert scan_doc.grid_size == (1, 1)
        assert scan_doc.total_sites_scanned == 1
        assert scan_doc.std_adsorption_energy == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
