"""SIESTA band-structure k-path generation and .bands file utilities."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import TYPE_CHECKING

import click
import numpy as np
from pymatgen.symmetry.bandstructure import HighSymmKpath

if TYPE_CHECKING:
    from pymatgen.core import Structure

logger = logging.getLogger(__name__)


def band_paymatgen_to_siesta(
    structure: Structure, interpolations: list[int] | None = None
) -> list[str]:
    """Read a pymatgen structure and return the SIESTA band k-path."""
    logger.info("band_paymatgen_to_siesta()")

    # Generate the k-path using symmetry of the structure
    kpath = HighSymmKpath(structure)

    # Access the k-points and the path
    kpoints = kpath.kpath["kpoints"]
    path = kpath.kpath["path"]

    # Interpolation settings: define the number of points between high-symmetry k-points
    if interpolations is None:
        interpolations = [20]  # Customize as needed

    # if wave_func_k_point_scale is None:
    #    wave_func_k_point_scale = "WaveFuncKPointsScale ReciprocalLatticeVectors"
    band_fdf_arguments = []

    counter = 0
    for i, segment in enumerate(path):
        for kp in segment:
            n_points = (
                interpolations[i] if i < len(interpolations) else 20
            )  # Default to 20 if not specified
            # Write the start point (always start with 1)
            if counter == 0:
                band_fdf_arguments.append(
                    f"{1} {kpoints[kp][0]:.6f} {kpoints[kp][1]:.6f} "
                    f"{kpoints[kp][2]:.6f} # {kp}"
                )
            else:
                band_fdf_arguments.append(
                    f"{n_points} {kpoints[kp][0]:.6f} {kpoints[kp][1]:.6f} "
                    f"{kpoints[kp][2]:.6f} # {kp}"
                )
            counter = +1

    logger.info("K-path generated for siesta fdf")
    return band_fdf_arguments  # tuple(band_fdf_arguments)


class GnuBands_Old:  # noqa: N801  public legacy class name referenced by tests
    """
    Legacy band structure parser and plotter for SIESTA .bands files.

    Reads SIESTA band structure output and prepares data for plotting with GNUplot.
    This is a legacy implementation - consider using modern plotting tools instead.
    """

    def __init__(self) -> None:
        """Initialize GnuBands_Old with default band structure parameters."""
        self.ef = None
        self.kmin = None
        self.kmax = None
        self.emin = -1e30
        self.emax = 1e30
        self.nband = None
        self.nspin = None
        self.nk = None
        self.min_band = 1
        self.max_band = sys.maxsize
        self.spin_idx = 0
        self.fermi_shift = False
        self.gnu_ticks = False
        self.outfile = None
        self.bands_u = None
        self.k = None
        self.e = None
        self.listk = None
        self.labels = None
        logger.info("GnuBands_Old.__init__()")

    def read_bands_file(self, bandfile: str) -> None:
        """
        Read band structure data from a SIESTA .bands file.

        Parameters
        ----------
        bandfile : str
            Path to the SIESTA .bands output file
        """
        logger.info("GnuBands_Old.read_bands_file()")
        with open(bandfile) as f:
            self.ef = float(f.readline())
            self.kmin, self.kmax = map(float, f.readline().split())
            f.readline()  # skip dummy line
            self.nband, self.nspin, self.nk = map(int, f.readline().split())

            self.k = np.zeros(self.nk)
            self.e = np.zeros((self.nband, self.nspin, self.nk))

            for ik in range(self.nk):
                line = f.readline().split()
                self.k[ik] = float(line[0])
                for ispin in range(self.nspin):
                    for iband in range(self.nband):
                        self.e[iband, ispin, ik] = float(line[1 + iband])

    def process_options(self, args: list[str]) -> None:
        """
        Process command-line arguments for band structure plotting options.

        Parameters
        ----------
        args : list
            Command-line arguments list
        """
        logger.info("GnuBands_Old.process_options()")
        parser = argparse.ArgumentParser(description="Process options for GnuBands.")
        parser.add_argument(
            "-G", action="store_true", help="Print GNUplot commands for correct labels"
        )
        parser.add_argument("-s", type=int, help="Only plot selected spin bands")
        parser.add_argument(
            "-F", action="store_true", help="Shift energy to Fermi-level"
        )
        parser.add_argument("-e", type=float, help="Minimum energy to write")
        parser.add_argument("-E", type=float, help="Maximum energy to write")
        parser.add_argument("-b", type=int, help="First band to write")
        parser.add_argument("-B", type=int, help="Last band to write")
        parser.add_argument("-o", type=str, help="Specify output file")

        opts = parser.parse_args(args)

        self.gnu_ticks = opts.G
        self.spin_idx = opts.s or 0
        self.fermi_shift = opts.F
        if opts.e:
            self.emin = opts.e
        if opts.E:
            self.emax = opts.E
        if opts.b:
            self.min_band = opts.b
        if opts.B:
            self.max_band = opts.B
        if opts.o:
            self.outfile = opts.o

    def validate_options(self) -> None:
        """
        Validate band structure plotting options and ranges.

        Checks that selected spin channel, band indices, and energy ranges
        are within valid limits for the loaded band structure data.

        Raises
        ------
        SystemExit
            If validation fails (invalid spin index or band range)
        """
        logger.info("GnuBands_Old.validate_options()")
        if self.spin_idx > 0 and (self.spin_idx < 1 or self.spin_idx > self.nspin):
            logger.warning("Selected spin does not exist")
            sys.exit(1)

        if self.min_band < 1:
            logger.warning("Min_band implicitly reset to 1")
            self.min_band = 1
        if self.min_band > self.nband:
            logger.error(
                f"Min_band is too large (min_band, nband): "
                f"{self.min_band}, {self.nband}"
            )
            sys.exit(1)
        if self.max_band > self.nband:
            logger.warning(
                f"Max_band is too large (max_band, nband): "
                f"{self.max_band}, {self.nband}"
            )
            logger.warning(
                "Max_band will be effectively reset to its maximum allowed value"
            )
            self.max_band = self.nband
        if self.max_band < self.min_band:
            logger.error(
                f"Max_band is less than min_band: (max_band, eff min_band): "
                f"{self.max_band}, {self.min_band}"
            )
            sys.exit(1)

    def shift_fermi_level(self) -> None:
        """Shift band energies so that the Fermi level is at zero energy."""
        logger.info("GnuBands_Old.shift_fermi_level()")
        if self.fermi_shift:
            self.e -= self.ef

    def write_output(self) -> None:
        """
        Write formatted band structure data for plotting with GNUplot.

        Generates output containing band energies vs k-points in a format
        suitable for GNUplot. Includes metadata header and filters data
        by selected bands, spins, and energy range.

        Output is written to self.outfile if specified, otherwise to stdout.
        """
        logger.info("GnuBands_Old.write_output()")
        output = []
        output.append(
            "# GNUBANDS: Utility for SIESTA to transform bands output "
            "into Gnuplot format"
        )
        output.append(f"# E_F = {self.ef}")
        output.append(f"# k_min, k_max = {self.kmin}, {self.kmax}")
        output.append(f"# E_min, E_max = {self.emin}, {self.emax}")
        output.append(f"# Nbands, Nspin, Nk = {self.nband}, {self.nspin}, {self.nk}")
        output.append(f"# Using min_band, max_band = {self.min_band}, {self.max_band}")
        output.append(f"# Total number of bands = {self.max_band - self.min_band + 1}")
        output.append("# k            E[eV]")

        delta = 1e-5
        for ispin in range(self.spin_idx if self.spin_idx > 0 else 1, self.nspin + 1):
            for iband in range(self.min_band, self.max_band + 1):
                for ik in range(self.nk):
                    if (
                        self.emin - delta
                        <= self.e[iband - 1, ispin - 1, ik]
                        <= self.emax + delta
                    ):
                        output.append(  # noqa: PERF401  nested conditional loop
                            f"{self.k[ik]:14.6f} "
                            f"{self.e[iband - 1, ispin - 1, ik]:14.6f} {ispin:3d}"
                        )

        if self.outfile:
            with open(self.outfile, "w") as f:
                f.write("\n".join(output))
        else:
            print("\n".join(output))  # noqa: T201  CLI writes result to stdout

    def run(self, args: list[str], bandfile: str) -> None:
        """
        Execute complete workflow for processing SIESTA band structure file.

        This is the main entry point that coordinates reading the band structure,
        processing command-line options, validating inputs, optionally shifting
        Fermi level, and writing the formatted output.

        Parameters
        ----------
        args : list
            Command-line arguments to parse
        bandfile : str
            Path to SIESTA .bands output file
        """
        logger.info("GnuBands_Old.run()")
        self.process_options(args)
        self.read_bands_file(bandfile)
        self.validate_options()
        self.shift_fermi_level()
        self.write_output()


# if __name__ == "__main__":
#    bandfile = 'SystemLabel.bands'  # Example filename; replace with actual file
#    gnubands = GnuBands()
#    gnubands.run(sys.argv[1:], bandfile)


class GnuBands:
    """
    Modern band structure parser and plotter for SIESTA .bands files using Click CLI.

    Reads SIESTA band structure output and prepares data for plotting with GNUplot.
    This is the modern implementation using Click for command-line argument parsing.
    Provides filtering options for spins, bands, and energy ranges.
    """

    def __init__(self) -> None:
        """Initialize GnuBands with default band structure parameters."""
        logger.info("GnuBands.run()")
        self.ef = None
        self.kmin = None
        self.kmax = None
        self.emin = -1e30
        self.emax = 1e30
        self.nband = None
        self.nspin = None
        self.nk = None
        self.min_band = 1
        self.max_band = sys.maxsize
        self.spin_idx = 0
        self.fermi_shift = False
        self.gnu_ticks = False
        self.outfile = None
        self.bands_u = None
        self.k = None
        self.e = None
        self.listk = None
        self.labels = None

    def read_bands_file(self, bandfile: str) -> None:
        """
        Read band structure data from a SIESTA .bands file.

        Parameters
        ----------
        bandfile : str
            Path to the SIESTA .bands output file
        """
        logger.info("GnuBands.read_bands_file()")
        with open(bandfile) as f:
            self.ef = float(f.readline())
            self.kmin, self.kmax = map(float, f.readline().split())
            f.readline()  # skip dummy line
            self.nband, self.nspin, self.nk = map(int, f.readline().split())

            self.k = np.zeros(self.nk)
            self.e = np.zeros((self.nband, self.nspin, self.nk))

            for ik in range(self.nk):
                line = f.readline().split()
                self.k[ik] = float(line[0])
                for ispin in range(self.nspin):
                    for iband in range(self.nband):
                        self.e[iband, ispin, ik] = float(line[1 + iband])

    def validate_options(self) -> None:
        """
        Validate band structure plotting options and ranges.

        Checks that selected spin channel, band indices, and energy ranges
        are within valid limits for the loaded band structure data.

        Raises
        ------
        SystemExit
            If validation fails (invalid spin index or band range)
        """
        logger.info("GnuBands.validate_options()")
        if self.spin_idx > 0 and (self.spin_idx < 1 or self.spin_idx > self.nspin):
            logger.warning("Selected spin does not exist")
            sys.exit(1)

        if self.min_band < 1:
            logger.warning("Min_band implicitly reset to 1")
            self.min_band = 1
        if self.min_band > self.nband:
            logger.error(
                f"Min_band is too large (min_band, nband): "
                f"{self.min_band}, {self.nband}"
            )
            sys.exit(1)
        if self.max_band > self.nband:
            logger.warning(
                f"Max_band is too large (max_band, nband): "
                f"{self.max_band}, {self.nband}"
            )
            logger.warning(
                "Max_band will be effectively reset to its maximum allowed value"
            )
            self.max_band = self.nband
        if self.max_band < self.min_band:
            logger.error(
                f"Max_band is less than min_band: (max_band, eff min_band): "
                f"{self.max_band}, {self.min_band}"
            )
            sys.exit(1)

    def shift_fermi_level(self) -> None:
        """Shift band energies so that the Fermi level is at zero energy."""
        logger.info("GnuBands.shift_fermi_level()")
        if self.fermi_shift:
            self.e -= self.ef

    def write_output(self) -> None:
        """
        Write formatted band structure data for plotting with GNUplot.

        Generates output containing band energies vs k-points in a format
        suitable for GNUplot. Includes metadata header and filters data
        by selected bands, spins, and energy range.

        Output is written to self.outfile if specified, otherwise to stdout.
        """
        logger.info("GnuBands.write_output()")
        output = []
        output.append(
            "# GNUBANDS: Utility for SIESTA to transform bands output "
            "into Gnuplot format"
        )
        output.append(f"# E_F = {self.ef}")
        output.append(f"# k_min, k_max = {self.kmin}, {self.kmax}")
        output.append(f"# E_min, E_max = {self.emin}, {self.emax}")
        output.append(f"# Nbands, Nspin, Nk = {self.nband}, {self.nspin}, {self.nk}")
        output.append(f"# Using min_band, max_band = {self.min_band}, {self.max_band}")
        output.append(f"# Total number of bands = {self.max_band - self.min_band + 1}")
        output.append("# k            E[eV]")

        delta = 1e-5
        for ispin in range(self.spin_idx if self.spin_idx > 0 else 1, self.nspin + 1):
            for iband in range(self.min_band, self.max_band + 1):
                for ik in range(self.nk):
                    if (
                        self.emin - delta
                        <= self.e[iband - 1, ispin - 1, ik]
                        <= self.emax + delta
                    ):
                        output.append(  # noqa: PERF401  nested conditional loop
                            f"{self.k[ik]:14.6f} "
                            f"{self.e[iband - 1, ispin - 1, ik]:14.6f} {ispin:3d}"
                        )

        if self.outfile:
            with open(self.outfile, "w") as f:
                f.write("\n".join(output))
        else:
            print("\n".join(output))  # noqa: T201  CLI writes result to stdout

    def run(
        self,
        bandfile: str,
        spin_idx: int,
        fermi_shift: bool,
        emin: float | None,
        emax: float | None,
        min_band: int | None,
        max_band: int | None,
        gnu_ticks: bool,
        outfile: str | None,
    ) -> None:
        """
        Execute complete workflow for processing SIESTA band structure file.

        This is the main entry point that coordinates reading the band structure,
        applying plot options, validating inputs, optionally shifting Fermi level,
        and writing the formatted output.

        Parameters
        ----------
        bandfile : str
            Path to SIESTA .bands output file
        spin_idx : int
            Selected spin channel (0 for all, 1-nspin for specific)
        fermi_shift : bool
            If True, shift energies so Fermi level is at zero
        emin : float or None
            Minimum energy to include in output
        emax : float or None
            Maximum energy to include in output
        min_band : int or None
            First band index to include
        max_band : int or None
            Last band index to include
        gnu_ticks : bool
            If True, print GNUplot tick commands
        outfile : str or None
            Output file path (None for stdout)
        """
        logger.info("GnuBands.run()")
        self.spin_idx = spin_idx
        self.fermi_shift = fermi_shift
        self.emin = emin or self.emin
        self.emax = emax or self.emax
        self.min_band = min_band or self.min_band
        self.max_band = max_band or self.max_band
        self.gnu_ticks = gnu_ticks
        self.outfile = outfile

        self.read_bands_file(bandfile)
        self.validate_options()
        self.shift_fermi_level()
        self.write_output()


@click.command()
@click.argument("bandfile", type=click.Path(exists=True))
@click.option(
    "-s", "--spin-idx", default=0, help="Only plot selected spin bands [1, nspin]"
)
@click.option("-F", "--fermi-shift", is_flag=True, help="Shift energy to Fermi-level")
@click.option("-e", "--emin", type=float, help="Minimum energy to write")
@click.option("-E", "--emax", type=float, help="Maximum energy to write")
@click.option("-b", "--min-band", type=int, help="First band to write")
@click.option("-B", "--max-band", type=int, help="Last band to write")
@click.option(
    "-G", "--gnu-ticks", is_flag=True, help="Print GNUplot commands for correct labels"
)
@click.option("-o", "--outfile", type=click.Path(), help="Specify output file")
def cli(
    bandfile: str,
    spin_idx: int,
    fermi_shift: bool,
    emin: float | None,
    emax: float | None,
    min_band: int | None,
    max_band: int | None,
    gnu_ticks: bool,
    outfile: str | None,
) -> None:
    """Process bands file and generate Gnuplot-ready output."""
    gnubands = GnuBands()
    gnubands.run(
        bandfile,
        spin_idx,
        fermi_shift,
        emin,
        emax,
        min_band,
        max_band,
        gnu_ticks,
        outfile,
    )


if __name__ == "__main__":
    cli()
