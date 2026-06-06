#!/usr/bin/env python
"""Example 1: Structure Analysis - See what Recipe Book will do."""

from pymatgen.core import Lattice, Structure
from atomate2.siesta.recipes import RecipeBook

# Create example structure (Silicon)
silicon = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])

# Analyze structure and print recommendations
RecipeBook.print_analysis(silicon)
