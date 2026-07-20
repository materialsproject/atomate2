"""
Tests for adsorption jobs.

These tests validate:
- Adsorbate placement on slabs (top/bottom)
- Grid site generation
- Energy analysis and statistics
- Plotting functionality
- Summary writing
"""

from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Lattice, Molecule, Structure

from atomate2.siesta.jobs.surface.adsorption import (
    add_adsorbate_to_slab,
    analyze_adsorption_scan,
    generate_adsorption_sites,
    plot_adsorption_sites,
    write_adsorption_summary,
)
from atomate2.siesta.schemas.adsorption import (
    AdsorptionScanDocument,
    AdsorptionSiteResult,
)


@pytest.fixture
def simple_slab():
    """Create a simple cubic slab for testing."""
    lattice = Lattice.cubic(5.0)
    return Structure(
        lattice,
        ["Al", "Al", "Al", "Al"],
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 0.2],
            [0.5, 0.5, 0.2],
        ],
    )


@pytest.fixture
def co_molecule():
    """Create CO molecule for testing."""
    return Molecule(["C", "O"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.13]])


@pytest.fixture
def h2_molecule():
    """Create H2 molecule for testing."""
    return Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])


@pytest.fixture
def mock_scan_document():
    """Create a mock AdsorptionScanDocument for testing plotting/summary."""
    # Create some mock site results
    site_results = []
    for i, x in enumerate([0.25, 0.75]):
        for j, y in enumerate([0.25, 0.75]):
            # Create some variation in energies
            base_energy = -2.5
            variation = 0.1 * (i + j)
            energy = base_energy + variation

            site_result = AdsorptionSiteResult(
                site_x=x,
                site_y=y,
                site_x_cart=x * 5.0,
                site_y_cart=y * 5.0,
                adsorption_energy=energy,
                adsorption_energy_per_area=energy / 25.0,
                total_energy=-100.0 + energy,
                slab_energy=-95.0,
                adsorbate_energy=-3.0,
                surface_area=25.0,
                height=2.0,
                n_atoms=6,
                n_slab_atoms=4,
                n_adsorbate_atoms=2,
            )
            site_results.append(site_result)

    # Sort by energy to get best site
    site_results.sort(key=lambda x: x.adsorption_energy)

    return AdsorptionScanDocument(
        slab_formula="Al",
        adsorbate_formula="CO",
        miller_indices=(1, 0, 0),
        grid_size=(2, 2),
        initial_height=2.0,
        surface_area=25.0,
        slab_thickness=1.0,
        total_sites_scanned=4,
        slab_energy=-95.0,
        adsorbate_energy=-3.0,
        best_site_position=(site_results[0].site_x, site_results[0].site_y),
        best_adsorption_energy=site_results[0].adsorption_energy,
        best_energy_per_area=site_results[0].adsorption_energy_per_area,
        mean_adsorption_energy=float(
            np.mean([s.adsorption_energy for s in site_results])
        ),
        std_adsorption_energy=float(
            np.std([s.adsorption_energy for s in site_results])
        ),
        energy_range=max(s.adsorption_energy for s in site_results)
        - min(s.adsorption_energy for s in site_results),
        site_results=site_results,
    )


class TestAddAdsorbateToSlab:
    """Tests for add_adsorbate_to_slab function."""

    def test_add_adsorbate_top_placement(self, simple_slab, co_molecule):
        """Test adding adsorbate on top of slab."""
        position = (0.5, 0.5)  # Center of cell
        height = 2.0

        result = add_adsorbate_to_slab(
            simple_slab, co_molecule, position, height, placement="top"
        )

        # Check structure has more atoms
        assert len(result) == len(simple_slab) + len(co_molecule)

        # Check lattice is preserved
        assert result.lattice == simple_slab.lattice

        # Check adsorbate atoms are at top
        slab_atoms = result[: len(simple_slab)]
        ads_atoms = result[len(simple_slab) :]
        assert len(ads_atoms) == 2  # CO molecule

        # Adsorbate should be above slab atoms
        max_slab_z = max(site.coords[2] for site in slab_atoms)
        min_ads_z = min(site.coords[2] for site in ads_atoms)
        assert min_ads_z > max_slab_z

    def test_add_adsorbate_bottom_placement(self, simple_slab, co_molecule):
        """Test adding adsorbate on bottom of slab."""
        position = (0.5, 0.5)
        height = 2.0

        result = add_adsorbate_to_slab(
            simple_slab, co_molecule, position, height, placement="bottom"
        )

        # Check structure has more atoms
        assert len(result) == len(simple_slab) + len(co_molecule)

        # Adsorbate should be below slab atoms
        slab_atoms = result[: len(simple_slab)]
        ads_atoms = result[len(simple_slab) :]

        min_slab_z = min(site.coords[2] for site in slab_atoms)
        max_ads_z = max(site.coords[2] for site in ads_atoms)
        assert max_ads_z < min_slab_z

    def test_add_adsorbate_different_positions(self, simple_slab, co_molecule):
        """Test adding adsorbate at different positions."""
        positions = [(0.25, 0.25), (0.75, 0.75), (0.0, 0.5)]

        for pos in positions:
            result = add_adsorbate_to_slab(simple_slab, co_molecule, pos, height=2.0)
            assert len(result) == len(simple_slab) + len(co_molecule)

    def test_add_adsorbate_invalid_placement(self, simple_slab, co_molecule):
        """Test that invalid placement raises error."""
        with pytest.raises(ValueError, match="Invalid placement"):
            add_adsorbate_to_slab(
                simple_slab, co_molecule, (0.5, 0.5), 2.0, placement="middle"
            )

    def test_add_adsorbate_with_structure(self, simple_slab):
        """Test adding adsorbate that is a Structure (not Molecule)."""
        # Create a small structure as adsorbate
        ads_lattice = Lattice.cubic(3.0)
        ads_struct = Structure(ads_lattice, ["H"], [[0.0, 0.0, 0.0]])

        result = add_adsorbate_to_slab(simple_slab, ads_struct, (0.5, 0.5), 2.0)

        assert len(result) == len(simple_slab) + len(ads_struct)


class TestGenerateAdsorptionSites:
    """Tests for generate_adsorption_sites job."""

    def test_generate_sites_2x2_grid(self):
        """Test generating 2x2 grid of sites."""
        sites = generate_adsorption_sites.original(grid_size=(2, 2))

        assert len(sites) == 4
        assert all(isinstance(site, tuple) for site in sites)
        assert all(len(site) == 2 for site in sites)

        # Check all coordinates are in [0, 1)
        for site in sites:
            x, y = site
            assert 0.0 <= x < 1.0
            assert 0.0 <= y < 1.0

    def test_generate_sites_4x4_grid(self):
        """Test generating larger 4x4 grid."""
        sites = generate_adsorption_sites.original(grid_size=(4, 4))

        assert len(sites) == 16

        # Check grid spacing
        x_coords = sorted(set(site[0] for site in sites))
        assert len(x_coords) == 4

        # Spacing should be approximately 1/4
        spacings = [x_coords[i + 1] - x_coords[i] for i in range(len(x_coords) - 1)]
        assert all(abs(s - 0.25) < 0.01 for s in spacings)

    def test_generate_sites_non_square_grid(self):
        """Test generating non-square grid."""
        sites = generate_adsorption_sites.original(grid_size=(3, 5))

        assert len(sites) == 15

        # Check x has 3 unique values, y has 5
        x_coords = set(site[0] for site in sites)
        y_coords = set(site[1] for site in sites)
        assert len(x_coords) == 3
        assert len(y_coords) == 5

    def test_generate_sites_single_point(self):
        """Test generating single site (1x1 grid)."""
        sites = generate_adsorption_sites.original(grid_size=(1, 1))

        assert len(sites) == 1
        x, y = sites[0]
        # Should be at center (0.5, 0.5)
        assert abs(x - 0.5) < 0.01
        assert abs(y - 0.5) < 0.01

    def test_generate_sites_centered(self):
        """Test that sites are centered in grid cells."""
        sites = generate_adsorption_sites.original(grid_size=(2, 2))

        # For 2x2 grid, sites should be at 0.25 and 0.75
        expected_coords = [0.25, 0.75]

        x_coords = sorted(set(site[0] for site in sites))
        y_coords = sorted(set(site[1] for site in sites))

        assert len(x_coords) == 2
        assert len(y_coords) == 2
        assert all(abs(x - exp) < 0.01 for x, exp in zip(x_coords, expected_coords))
        assert all(abs(y - exp) < 0.01 for y, exp in zip(y_coords, expected_coords))


class TestAnalyzeAdsorptionScan:
    """Tests for analyze_adsorption_scan job."""

    def test_analyze_basic(self, simple_slab, co_molecule):
        """Test basic analysis with 2x2 grid."""
        site_energies = [
            {"site": (0.25, 0.25), "height": 2.0, "total_energy": -100.0},
            {"site": (0.75, 0.25), "height": 2.0, "total_energy": -100.2},
            {"site": (0.25, 0.75), "height": 2.0, "total_energy": -100.1},
            {"site": (0.75, 0.75), "height": 2.0, "total_energy": -99.8},
        ]
        slab_energy = -95.0
        adsorbate_energy = -3.0

        result = analyze_adsorption_scan.original(
            slab=simple_slab,
            adsorbate=co_molecule,
            site_energies=site_energies,
            slab_energy=slab_energy,
            adsorbate_energy=adsorbate_energy,
            grid_size=(2, 2),
            heights=[2.0],
            miller_indices=(1, 0, 0),
        )

        # Check document structure
        assert isinstance(result, AdsorptionScanDocument)
        assert result.total_sites_scanned == 4
        assert result.grid_size == (2, 2)
        assert result.initial_height == 2.0
        assert result.miller_indices == (1, 0, 0)

        # Check best site (should be site 2 with E=-100.2)
        assert result.best_adsorption_energy == pytest.approx(
            -100.2 - slab_energy - adsorbate_energy
        )
        assert result.best_site_position == (0.75, 0.25)

        # Check statistics
        assert result.mean_adsorption_energy is not None
        assert result.std_adsorption_energy >= 0
        assert result.energy_range > 0

    def test_analyze_formulas(self, simple_slab, co_molecule):
        """Test that formulas are correctly extracted."""
        site_energies = [{"site": (0.5, 0.5), "height": 2.0, "total_energy": -100.0}]

        result = analyze_adsorption_scan.original(
            slab=simple_slab,
            adsorbate=co_molecule,
            site_energies=site_energies,
            slab_energy=-95.0,
            adsorbate_energy=-3.0,
            grid_size=(1, 1),
            heights=[2.0],
        )

        assert result.slab_formula == "Al"
        # Molecule formulas now use Hill notation (space-separated)
        assert result.adsorbate_formula == "C O"

    def test_analyze_surface_area(self, simple_slab, co_molecule):
        """Test surface area calculation."""
        site_energies = [{"site": (0.5, 0.5), "height": 2.0, "total_energy": -100.0}]

        result = analyze_adsorption_scan.original(
            slab=simple_slab,
            adsorbate=co_molecule,
            site_energies=site_energies,
            slab_energy=-95.0,
            adsorbate_energy=-3.0,
            grid_size=(1, 1),
            heights=[2.0],
        )

        # For 5x5 cubic lattice, surface area should be 25.0
        assert result.surface_area == pytest.approx(25.0, rel=0.01)

    def test_analyze_energy_per_area(self, simple_slab, co_molecule):
        """Test energy per area calculation."""
        site_energies = [{"site": (0.5, 0.5), "height": 2.0, "total_energy": -100.0}]
        slab_energy = -95.0
        adsorbate_energy = -3.0

        result = analyze_adsorption_scan.original(
            slab=simple_slab,
            adsorbate=co_molecule,
            site_energies=site_energies,
            slab_energy=slab_energy,
            adsorbate_energy=adsorbate_energy,
            grid_size=(1, 1),
            heights=[2.0],
        )

        ads_energy = -100.0 - slab_energy - adsorbate_energy
        expected_per_area = ads_energy / 25.0

        assert result.best_energy_per_area == pytest.approx(expected_per_area)

    def test_analyze_top_5_sites(self, simple_slab, co_molecule):
        """Test that top 5 sites are correctly identified."""
        # Create more than 5 sites
        site_energies = [
            {
                "site": (x * 0.2, y * 0.2),
                "height": 2.0,
                "total_energy": -100.0 - x * 0.1 - y * 0.1,
            }
            for x in range(4)
            for y in range(4)
        ]  # 16 sites

        result = analyze_adsorption_scan.original(
            slab=simple_slab,
            adsorbate=co_molecule,
            site_energies=site_energies,
            slab_energy=-95.0,
            adsorbate_energy=-3.0,
            grid_size=(4, 4),
            heights=[2.0],
        )

        # Check top 5
        top_5 = result.top_5_sites
        assert len(top_5) == 5

        # Top 5 should be sorted by energy (ascending)
        energies = [site.adsorption_energy for site in top_5]
        assert energies == sorted(energies)


@pytest.mark.matplotlib
class TestPlotAdsorptionSites:
    """Tests for plot_adsorption_sites job."""

    def test_plot_basic(self, mock_scan_document, tmp_path):
        """Test basic plotting functionality."""
        result = plot_adsorption_sites.original(
            scan_doc=mock_scan_document,
            output_dir=str(tmp_path),
            filename="test_plot.png",
        )

        # Check return value
        assert "plot_file" in result
        assert result["plot_file"] == str(tmp_path / "test_plot.png")

        # Check file was created
        plot_file = Path(result["plot_file"])
        assert plot_file.exists()
        assert plot_file.suffix == ".png"

    def test_plot_custom_filename(self, mock_scan_document, tmp_path):
        """Test plotting with custom filename."""
        custom_name = "my_adsorption_map.png"
        result = plot_adsorption_sites.original(
            scan_doc=mock_scan_document, output_dir=str(tmp_path), filename=custom_name
        )

        assert str(tmp_path / custom_name) in result["plot_file"]
        assert Path(result["plot_file"]).exists()

    def test_plot_creates_directory(self, mock_scan_document, tmp_path):
        """Test that plotting creates output directory if needed."""
        nested_dir = tmp_path / "nested" / "output"

        result = plot_adsorption_sites.original(
            scan_doc=mock_scan_document, output_dir=str(nested_dir), filename="plot.png"
        )

        assert Path(result["plot_file"]).exists()
        assert nested_dir.exists()


class TestWriteAdsorptionSummary:
    """Tests for write_adsorption_summary job."""

    def test_write_summary_basic(self, mock_scan_document, tmp_path):
        """Test basic summary writing."""
        result = write_adsorption_summary.original(
            scan_doc=mock_scan_document,
            output_dir=str(tmp_path),
            filename="test_summary.txt",
        )

        # Check return value
        assert "summary_file" in result
        assert result["summary_file"] == str(tmp_path / "test_summary.txt")

        # Check file was created
        summary_file = Path(result["summary_file"])
        assert summary_file.exists()
        assert summary_file.suffix == ".txt"

    def test_write_summary_content(self, mock_scan_document, tmp_path):
        """Test that summary contains expected content."""
        result = write_adsorption_summary.original(
            scan_doc=mock_scan_document,
            output_dir=str(tmp_path),
            filename="summary.txt",
        )

        summary_file = Path(result["summary_file"])
        content = summary_file.read_text()

        # Check for key sections
        assert "ADSORPTION SITE SCAN SUMMARY" in content
        assert "SYSTEM INFORMATION" in content
        assert "ENERGY STATISTICS" in content
        assert "BEST ADSORPTION SITE" in content
        assert "TOP 5 ADSORPTION SITES" in content

        # Check for specific values
        assert mock_scan_document.slab_formula in content
        assert mock_scan_document.adsorbate_formula in content
        assert str(mock_scan_document.best_adsorption_energy) in content
        assert "(1 0 0)" in content  # Miller indices

    def test_write_summary_custom_filename(self, mock_scan_document, tmp_path):
        """Test writing summary with custom filename."""
        custom_name = "my_results.txt"
        result = write_adsorption_summary.original(
            scan_doc=mock_scan_document, output_dir=str(tmp_path), filename=custom_name
        )

        assert str(tmp_path / custom_name) in result["summary_file"]
        assert Path(result["summary_file"]).exists()

    def test_write_summary_formatting(self, mock_scan_document, tmp_path):
        """Test that summary is well-formatted."""
        result = write_adsorption_summary.original(
            scan_doc=mock_scan_document,
            output_dir=str(tmp_path),
            filename="summary.txt",
        )

        content = Path(result["summary_file"]).read_text()

        # Check for consistent formatting
        # Header should have equals signs
        assert "=" * 80 in content
        # Sections should have dashes
        assert "-" * 80 in content

        # Check table formatting for top 5 sites
        assert "Rank" in content
        assert "Position" in content
        assert "E_ads" in content


class TestAdsorptionJobIntegration:
    """Integration tests for adsorption job workflow."""

    def test_full_workflow_components(self, simple_slab, co_molecule, tmp_path):
        """Test that all components work together."""
        # 1. Generate sites
        sites = generate_adsorption_sites.original(grid_size=(2, 2))
        assert len(sites) == 4

        # 2. Simulate placing adsorbate at each site
        structures = []
        for site in sites:
            struct = add_adsorbate_to_slab(simple_slab, co_molecule, site, height=2.0)
            structures.append(struct)
        assert len(structures) == 4

        # 3. Simulate energy calculations (mock energies)
        site_energies = [
            {"site": site, "height": 2.0, "total_energy": -100.0 - i * 0.1}
            for i, site in enumerate(sites)
        ]

        # 4. Analyze results
        analysis = analyze_adsorption_scan.original(
            slab=simple_slab,
            adsorbate=co_molecule,
            site_energies=site_energies,
            slab_energy=-95.0,
            adsorbate_energy=-3.0,
            grid_size=(2, 2),
            heights=[2.0],
        )
        assert isinstance(analysis, AdsorptionScanDocument)

        # 5. Create plot
        plot_result = plot_adsorption_sites.original(
            scan_doc=analysis,
            output_dir=str(tmp_path),
        )
        assert Path(plot_result["plot_file"]).exists()

        # 6. Write summary
        summary_result = write_adsorption_summary.original(
            scan_doc=analysis,
            output_dir=str(tmp_path),
        )
        assert Path(summary_result["summary_file"]).exists()

    def test_workflow_with_different_grid_sizes(self, simple_slab, co_molecule):
        """Test workflow with various grid sizes."""
        grid_sizes = [(2, 2), (3, 3), (4, 4), (2, 4)]

        for grid_size in grid_sizes:
            sites = generate_adsorption_sites.original(grid_size=grid_size)
            expected_sites = grid_size[0] * grid_size[1]
            assert len(sites) == expected_sites

            # Create mock energies
            site_energies = [
                {"site": site, "height": 2.0, "total_energy": -100.0} for site in sites
            ]

            # Analyze
            analysis = analyze_adsorption_scan.original(
                slab=simple_slab,
                adsorbate=co_molecule,
                site_energies=site_energies,
                slab_energy=-95.0,
                adsorbate_energy=-3.0,
                grid_size=grid_size,
                heights=[2.0],
            )

            assert analysis.total_sites_scanned == expected_sites
            assert analysis.grid_size == grid_size
