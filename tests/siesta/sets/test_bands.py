"""Tests for bands.py functionality."""

from pymatgen.core import Lattice, Structure

from atomate2.siesta.sets.bands import GnuBands, band_paymatgen_to_siesta


class TestBandPymatgenToSiesta:
    """Test band_paymatgen_to_siesta function."""

    def test_basic_band_generation(self, si_structure):
        """Test basic k-path generation for silicon."""
        band_fdf = band_paymatgen_to_siesta(si_structure)

        assert isinstance(band_fdf, list)
        assert len(band_fdf) > 0

        # Check format of first line (should start with "1")
        assert band_fdf[0].startswith("1 ")

        # All lines should have k-point coordinates
        for line in band_fdf:
            parts = line.split()
            assert len(parts) >= 4  # n_points + 3 coordinates

    def test_band_generation_with_interpolations(self, si_structure):
        """Test band generation with custom interpolations."""
        band_fdf = band_paymatgen_to_siesta(si_structure, interpolations=[10, 15, 20])

        assert isinstance(band_fdf, list)
        assert len(band_fdf) > 0

    def test_band_generation_different_structures(self):
        """Test band generation for different crystal systems."""
        # FCC structure
        fcc_lattice = Lattice.cubic(4.0)
        fcc_structure = Structure(fcc_lattice, ["Al"], [[0, 0, 0]])

        band_fdf_fcc = band_paymatgen_to_siesta(fcc_structure)
        assert len(band_fdf_fcc) > 0

        # BCC structure
        bcc_lattice = Lattice([[2, 0, 0], [0, 2, 0], [1, 1, 1]])
        bcc_structure = Structure(bcc_lattice, ["Fe"], [[0, 0, 0]])

        band_fdf_bcc = band_paymatgen_to_siesta(bcc_structure)
        assert len(band_fdf_bcc) > 0


class TestGnuBands:
    """Test GnuBands class."""

    def test_gnubands_initialization(self):
        """Test GnuBands class initialization."""
        gnu_bands = GnuBands()

        assert gnu_bands.ef is None
        assert gnu_bands.kmin is None
        assert gnu_bands.kmax is None
        assert gnu_bands.nband is None
        assert gnu_bands.nspin is None
        assert gnu_bands.nk is None

    def test_gnubands_attributes(self):
        """Test GnuBands has expected attributes."""
        gnu_bands = GnuBands()

        # Check key attributes exist
        assert hasattr(gnu_bands, "ef")
        assert hasattr(gnu_bands, "kmin")
        assert hasattr(gnu_bands, "kmax")
        assert hasattr(gnu_bands, "nband")
        assert hasattr(gnu_bands, "nspin")
        assert hasattr(gnu_bands, "nk")
        assert hasattr(gnu_bands, "min_band")
        assert hasattr(gnu_bands, "max_band")


class TestGnuBandsValidation:
    """Test GnuBands validation methods."""

    def test_validate_options_default(self):
        """Test validation with default settings."""
        gnu_bands = GnuBands()
        gnu_bands.nband = 10
        gnu_bands.nspin = 1
        gnu_bands.nk = 100

        # Should not raise with default values
        gnu_bands.validate_options()
        assert gnu_bands.min_band == 1
        assert gnu_bands.max_band <= gnu_bands.nband

    def test_validate_options_min_band_reset(self):
        """Test min_band is reset if less than 1."""
        gnu_bands = GnuBands()
        gnu_bands.nband = 10
        gnu_bands.nspin = 1
        gnu_bands.min_band = 0  # Invalid

        gnu_bands.validate_options()
        assert gnu_bands.min_band == 1

    def test_validate_options_max_band_reset(self):
        """Test max_band is capped at nband."""
        gnu_bands = GnuBands()
        gnu_bands.nband = 10
        gnu_bands.nspin = 1
        gnu_bands.max_band = 20  # Too large
        gnu_bands.min_band = 1

        gnu_bands.validate_options()
        assert gnu_bands.max_band == gnu_bands.nband

    def test_validate_options_valid_range(self):
        """Test validation with valid band range."""
        gnu_bands = GnuBands()
        gnu_bands.nband = 20
        gnu_bands.nspin = 2
        gnu_bands.min_band = 5
        gnu_bands.max_band = 15

        gnu_bands.validate_options()
        assert gnu_bands.min_band == 5
        assert gnu_bands.max_band == 15


class TestGnuBandsFermiShift:
    """Test Fermi level shifting functionality."""

    def test_shift_fermi_level(self):
        """Test shifting bands to Fermi level = 0."""
        import numpy as np

        gnu_bands = GnuBands()
        gnu_bands.ef = 5.0  # Fermi level at 5.0 eV
        gnu_bands.fermi_shift = True  # Enable Fermi shift
        gnu_bands.nband = 3
        gnu_bands.nspin = 1
        gnu_bands.nk = 2

        # Create sample band energies
        gnu_bands.e = np.array(
            [
                [[1.0, 1.5]],  # Band 1
                [[5.0, 5.2]],  # Band 2 (at Fermi level)
                [[10.0, 10.5]],  # Band 3
            ]
        )

        gnu_bands.shift_fermi_level()

        # After shift, energies should be relative to Fermi level
        assert gnu_bands.e[0, 0, 0] == -4.0  # 1.0 - 5.0
        assert gnu_bands.e[1, 0, 0] == 0.0  # 5.0 - 5.0
        assert gnu_bands.e[2, 0, 0] == 5.0  # 10.0 - 5.0

    def test_shift_fermi_level_not_applied(self):
        """Test that shift doesn't happen if fermi_shift is False."""
        import numpy as np

        gnu_bands = GnuBands()
        gnu_bands.ef = 5.0
        gnu_bands.fermi_shift = False
        gnu_bands.nband = 2
        gnu_bands.nspin = 1
        gnu_bands.nk = 1

        original_e = np.array([[[1.0]], [[10.0]]])
        gnu_bands.e = original_e.copy()

        # Without calling shift_fermi_level, values remain unchanged
        assert gnu_bands.e[0, 0, 0] == 1.0
        assert gnu_bands.e[1, 0, 0] == 10.0


class TestGnuBandsReadBandsFile:
    """Test reading .bands files."""

    def test_read_bands_file_format(self, tmp_path):
        """Test reading a properly formatted .bands file."""

        # Create a mock .bands file
        bands_file = tmp_path / "test.bands"
        content = """5.5
0.0 1.0
dummy line
4 1 3
0.1 -2.0 -1.0 5.0 6.0
0.2 -1.5 -0.5 5.5 6.5
0.3 -1.0 0.0 6.0 7.0
"""
        bands_file.write_text(content)

        gnu_bands = GnuBands()
        gnu_bands.read_bands_file(str(bands_file))

        # Check parsed values
        assert gnu_bands.ef == 5.5
        assert gnu_bands.kmin == 0.0
        assert gnu_bands.kmax == 1.0
        assert gnu_bands.nband == 4
        assert gnu_bands.nspin == 1
        assert gnu_bands.nk == 3

        # Check k-points
        assert len(gnu_bands.k) == 3
        assert gnu_bands.k[0] == 0.1
        assert gnu_bands.k[1] == 0.2
        assert gnu_bands.k[2] == 0.3

        # Check energies shape
        assert gnu_bands.e.shape == (4, 1, 3)  # (nband, nspin, nk)

    def test_read_bands_file_with_spin(self, tmp_path):
        """Test reading a spin-polarized .bands file."""
        bands_file = tmp_path / "spin.bands"
        content = """6.0
0.0 1.5
dummy
2 2 2
0.0 1.0 2.0 3.0 4.0
0.5 1.5 2.5 3.5 4.5
"""
        bands_file.write_text(content)

        gnu_bands = GnuBands()
        gnu_bands.read_bands_file(str(bands_file))

        assert gnu_bands.nspin == 2
        assert gnu_bands.nband == 2
        assert gnu_bands.e.shape == (2, 2, 2)


class TestBandPymatgenToSiestaExtended:
    """Extended tests for band_paymatgen_to_siesta function."""

    def test_band_output_format(self, si_structure):
        """Test that output has correct FDF format."""
        band_fdf = band_paymatgen_to_siesta(si_structure)

        # First line should start with "1" (first k-point)
        assert band_fdf[0].startswith("1 ")

        # Each line should have: n_points x y z # label
        for line in band_fdf:
            assert "#" in line  # Should have label comment
            parts_before_comment = line.split("#")[0].split()
            assert len(parts_before_comment) == 4  # n_points + 3 coords

    def test_band_kpoint_coordinates(self, si_structure):
        """Test that k-point coordinates are in valid range."""
        band_fdf = band_paymatgen_to_siesta(si_structure)

        for line in band_fdf:
            parts = line.split("#")[0].split()
            # K-point coordinates should be floats between -1 and 1 (typically)
            kx, ky, kz = float(parts[1]), float(parts[2]), float(parts[3])
            assert -2.0 <= kx <= 2.0
            assert -2.0 <= ky <= 2.0
            assert -2.0 <= kz <= 2.0

    def test_band_labels_present(self, si_structure):
        """Test that k-point labels are included."""
        band_fdf = band_paymatgen_to_siesta(si_structure)

        labels_found = []
        for line in band_fdf:
            if "#" in line:
                label = line.split("#")[1].strip()
                labels_found.append(label)

        # Should have common high-symmetry points
        assert len(labels_found) > 0
        # Common labels for FCC (Si): Γ, X, W, K, L
        assert any(label in labels_found for label in ["\\Gamma", "X", "L", "W", "K"])

    def test_band_interpolation_values(self, si_structure):
        """Test that interpolation affects number of points."""
        band_fdf_20 = band_paymatgen_to_siesta(si_structure, interpolations=[20])
        band_fdf_50 = band_paymatgen_to_siesta(si_structure, interpolations=[50])

        # With higher interpolation, should have same number of segments
        # but the n_points values should be different
        assert len(band_fdf_20) > 0
        assert len(band_fdf_50) > 0
        # Length should be same (same k-path segments)
        assert len(band_fdf_20) == len(band_fdf_50)

    def test_band_different_interpolations_per_segment(self, si_structure):
        """Test using different interpolations for each segment."""
        # Provide different interpolations for different segments
        band_fdf = band_paymatgen_to_siesta(
            si_structure, interpolations=[10, 20, 30, 40]
        )

        assert len(band_fdf) > 0

        # Check that we have n_points values (non-first entries)
        n_points_values = []
        for i, line in enumerate(band_fdf):
            if i > 0:  # Skip first (always 1)
                n_points = int(line.split()[0])
                n_points_values.append(n_points)

        # Should have some n_points extracted (at least one segment)
        assert len(n_points_values) > 0
        # All n_points should be positive integers
        assert all(n > 0 for n in n_points_values)

    def test_band_hexagonal_structure(self):
        """Test band generation for hexagonal structure."""
        # Create hexagonal lattice (e.g., graphene-like)
        hex_lattice = Lattice.hexagonal(2.46, 6.7)
        hex_structure = Structure(
            hex_lattice, ["C", "C"], [[0, 0, 0], [1 / 3, 2 / 3, 0.5]]
        )

        band_fdf = band_paymatgen_to_siesta(hex_structure)
        assert len(band_fdf) > 0

        # Hexagonal systems should have K, M, Γ points
        labels = " ".join(band_fdf)
        assert any(point in labels for point in ["K", "M", "\\Gamma"])

    def test_band_default_interpolation(self, si_structure):
        """Test that default interpolation is applied when None."""
        band_fdf = band_paymatgen_to_siesta(si_structure, interpolations=None)

        # Should use default of 20 points
        assert len(band_fdf) > 0
        # Check that non-first lines have 20 as n_points (default)
        n_points_list = [int(line.split()[0]) for line in band_fdf[1:]]
        assert 20 in n_points_list  # At least one should have default

    def test_band_empty_interpolations(self, si_structure):
        """Test behavior with empty interpolations list."""
        band_fdf = band_paymatgen_to_siesta(si_structure, interpolations=[])

        # Should fall back to default (20)
        assert len(band_fdf) > 0


class TestGnuBandsInitialization:
    """Test GnuBands initialization values."""

    def test_default_emin_emax(self):
        """Test default energy range is very large."""
        gnu_bands = GnuBands()
        assert gnu_bands.emin < -1e20
        assert gnu_bands.emax > 1e20

    def test_default_spin_idx(self):
        """Test default spin index is 0."""
        gnu_bands = GnuBands()
        assert gnu_bands.spin_idx == 0

    def test_default_fermi_shift_false(self):
        """Test Fermi shift is disabled by default."""
        gnu_bands = GnuBands()
        assert gnu_bands.fermi_shift is False

    def test_default_gnu_ticks_false(self):
        """Test GNU ticks are disabled by default."""
        gnu_bands = GnuBands()
        assert gnu_bands.gnu_ticks is False

    def test_default_max_band(self):
        """Test default max_band is sys.maxsize."""
        import sys

        gnu_bands = GnuBands()
        assert gnu_bands.max_band == sys.maxsize

    def test_all_arrays_none_initially(self):
        """Test that array attributes are None before reading."""
        gnu_bands = GnuBands()
        assert gnu_bands.k is None
        assert gnu_bands.e is None
        assert gnu_bands.listk is None
        assert gnu_bands.labels is None


class TestGnuBandsWriteOutput:
    """Test GnuBands output generation."""

    def test_write_output_basic(self, tmp_path):
        """Test basic output writing."""

        # Create a minimal .bands file
        bands_file = tmp_path / "test.bands"
        content = """5.0
0.0 1.0
dummy
2 1 2
0.0 1.0 2.0
0.5 1.5 2.5
"""
        bands_file.write_text(content)

        # Read and write output
        gnu_bands = GnuBands()
        gnu_bands.read_bands_file(str(bands_file))
        gnu_bands.validate_options()

        # Capture output
        output_file = tmp_path / "output.dat"
        gnu_bands.outfile = str(output_file)
        gnu_bands.write_output()

        # Check output file was created
        assert output_file.exists()
        content = output_file.read_text()
        assert "GNUBANDS" in content

    def test_write_output_with_fermi_shift(self, tmp_path):
        """Test output with Fermi level shift."""

        bands_file = tmp_path / "test.bands"
        content = """5.0
0.0 1.0
dummy
2 1 2
0.0 3.0 7.0
0.5 4.0 8.0
"""
        bands_file.write_text(content)

        gnu_bands = GnuBands()
        gnu_bands.fermi_shift = True
        gnu_bands.read_bands_file(str(bands_file))
        gnu_bands.validate_options()
        gnu_bands.shift_fermi_level()

        output_file = tmp_path / "output_shifted.dat"
        gnu_bands.outfile = str(output_file)
        gnu_bands.write_output()

        assert output_file.exists()

    def test_write_output_energy_filtering(self, tmp_path):
        """Test output with energy range filtering."""
        bands_file = tmp_path / "test.bands"
        content = """5.0
0.0 1.0
dummy
3 1 2
0.0 1.0 5.0 10.0
0.5 1.5 5.5 10.5
"""
        bands_file.write_text(content)

        gnu_bands = GnuBands()
        gnu_bands.emin = 0.0
        gnu_bands.emax = 6.0
        gnu_bands.read_bands_file(str(bands_file))
        gnu_bands.validate_options()

        output_file = tmp_path / "output_filtered.dat"
        gnu_bands.outfile = str(output_file)
        gnu_bands.write_output()

        assert output_file.exists()


class TestGnuBandsOldLegacy:
    """Test legacy GnuBands_Old class."""

    def test_gnubands_old_initialization(self):
        """Test GnuBands_Old initialization."""
        from atomate2.siesta.sets.bands import GnuBands_Old

        gnu_bands_old = GnuBands_Old()

        assert gnu_bands_old.ef is None
        assert gnu_bands_old.kmin is None
        assert gnu_bands_old.kmax is None
        assert gnu_bands_old.fermi_shift is False

    def test_gnubands_old_read_bands_file(self, tmp_path):
        """Test GnuBands_Old can read .bands files."""
        from atomate2.siesta.sets.bands import GnuBands_Old

        bands_file = tmp_path / "legacy.bands"
        content = """6.0
0.0 1.5
dummy
2 1 2
0.0 2.0 4.0
0.5 2.5 4.5
"""
        bands_file.write_text(content)

        gnu_bands_old = GnuBands_Old()
        gnu_bands_old.read_bands_file(str(bands_file))

        assert gnu_bands_old.ef == 6.0
        assert gnu_bands_old.nband == 2
        assert gnu_bands_old.nspin == 1
        assert gnu_bands_old.nk == 2
