"""Common helper functions for NEB workflows."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from ase.io import write
from ase.mep import NEB
from jobflow import job
from pymatgen.io.ase import AseAtomsAdaptor

if TYPE_CHECKING:
    from pymatgen.core import Structure

from atomate2.siesta import SETTINGS

logger = logging.getLogger(__name__)


@job
def generate_neb_band(
    number_of_images: int,
    initial: Structure,
    final: Structure,
    interpolation_method: str = "idpp",
) -> str:
    """
    Generate a NEB band of images using ASE interpolation and create neb.lua script.

    This function uses ASE's NEB (Nudged Elastic Band) implementation to
    create intermediate images between initial and final structures, and also
    generates a properly configured neb.lua script for SIESTA.

    Parameters
    ----------
    number_of_images : int
        Number of intermediate images to generate (not including endpoints).
    initial : Structure
        The initial structure (starting point).
    final : Structure
        The final structure (end point).
    interpolation_method : str
        ASE NEB interpolation method. Options: "idpp" (image-dependent pair
        potential, recommended), "linear" (simple linear interpolation).
        Default is "idpp".

    Returns
    -------
    str
        Current working directory path containing the generated XYZ files
        (siesta.0.xyz, siesta.1.xyz, ..., siesta.N.xyz) and neb.lua script.
    """
    logger.info(
        f"generate_neb_band() with interpolation method: {interpolation_method}"
    )
    initial_ase = AseAtomsAdaptor.get_atoms(initial)
    final_ase = AseAtomsAdaptor.get_atoms(final)

    images = [initial_ase]
    images += [initial_ase.copy() for _ in range(number_of_images)]
    images += [final_ase]

    # With no intermediate images there is nothing to interpolate (and ASE's
    # NEB.interpolate() raises on a 2-image band), so only do it when needed.
    if number_of_images > 0:
        neb = NEB(images)
        neb.interpolate(interpolation_method)  # Interpolate using specified method

    # Save each NEB image as .xyz
    neb_image_files = []
    for i, image in enumerate(images):
        image_file = f"siesta.{i}.xyz"
        write(image_file, image)
        neb_image_files.append(image_file)

    # Create neb.lua script with correct n_images
    # Total images = initial + intermediate + final = number_of_images + 2
    _create_neb_lua_script(number_of_images)

    logger.info(f"Generated {len(images)} NEB images and neb.lua script")
    return os.getcwd()


def _create_neb_lua_script(n_intermediate_images: int) -> None:
    """
    Create a neb.lua script with the correct number of images.

    Copies the neb.lua template from FLOS_PATH/examples/ and modifies
    the n_images parameter to match the number of intermediate images.

    Parameters
    ----------
    n_intermediate_images : int
        Number of intermediate images (excluding initial and final).
    """
    print(  # noqa: T201
        "DEBUG: _create_neb_lua_script called with "
        f"n_intermediate_images={n_intermediate_images}"
    )
    # Get FLOS path from settings
    flos_path = SETTINGS.FLOS_PATH
    if flos_path is None:
        raise ValueError(
            "FLOS_PATH not set. Please set it in ~/.atomate2siesta.yaml or as "
            "an environment variable."
        )

    flos_dir = Path(flos_path)
    # Older flos releases ship the template as examples/neb.lua; current flos
    # renamed it to examples/neb_simple.lua. Accept either.
    template_candidates = [
        flos_dir / "examples" / "neb.lua",
        flos_dir / "examples" / "neb_simple.lua",
    ]
    neb_template = next((path for path in template_candidates if path.exists()), None)

    if neb_template is None:
        candidates = " or ".join(str(path) for path in template_candidates)
        raise FileNotFoundError(
            f"NEB template not found at {candidates}. "
            f"Please ensure FLOS is properly installed at {flos_dir}"
        )

    # Read the template
    with open(neb_template) as f:
        lua_content = f.read()

    # Replace n_images with the correct value
    # The template has: local n_images = 6
    # We need to replace 6 with our value

    # First, let's check if the pattern exists
    if "local n_images" not in lua_content:
        raise ValueError(
            "Template neb.lua does not contain 'local n_images' definition"
        )

    # Debug: Log the original line
    original_match = re.search(r"local n_images\s*=\s*\d+", lua_content)
    if original_match:
        logger.info(f"Found original line: '{original_match.group()}'")
    else:
        logger.warning("Pattern not found - this is unexpected!")

    # Replace the value - be more specific with the pattern
    lua_content_modified = re.sub(
        r"local n_images\s*=\s*\d+",
        f"local n_images = {n_intermediate_images}",
        lua_content,
        count=1,  # Only replace the first occurrence
    )

    # Debug: Verify the replacement worked
    new_match = re.search(r"local n_images\s*=\s*\d+", lua_content_modified)
    if new_match:
        print(f"DEBUG: After replacement: '{new_match.group()}'")  # noqa: T201
        logger.info(f"After replacement: '{new_match.group()}'")
        if str(n_intermediate_images) not in new_match.group():
            print(  # noqa: T201
                f"DEBUG: Replacement FAILED! Expected {n_intermediate_images}, "
                f"got: {new_match.group()}"
            )
            logger.error(
                f"Replacement FAILED! Expected {n_intermediate_images}, "
                f"got: {new_match.group()}"
            )
    else:
        print("DEBUG: No match found after replacement!")  # noqa: T201

    lua_content = lua_content_modified

    # Also add error handling for file opening in the read_geom function
    # Replace: local file = io.open(filename, "r")
    # With proper error checking
    lua_content = lua_content.replace(
        'local file = io.open(filename, "r")',
        """local file = io.open(filename, "r")
   if not file then
      error("Cannot open file: " .. filename)
   end""",
    )

    # Write the modified script
    with open("neb.lua", "w") as f:
        f.write(lua_content)

    # Verify what was written
    with open("neb.lua") as f:
        verify_content = f.read()
        verify_match = re.search(r"local n_images\s*=\s*\d+", verify_content)
        if verify_match:
            print(f"DEBUG: Verified file contents: '{verify_match.group()}'")  # noqa: T201
        else:
            print("DEBUG: WARNING - Could not find n_images in written file!")  # noqa: T201

    logger.info(
        f"Created neb.lua from FLOS template with n_images={n_intermediate_images}"
    )
