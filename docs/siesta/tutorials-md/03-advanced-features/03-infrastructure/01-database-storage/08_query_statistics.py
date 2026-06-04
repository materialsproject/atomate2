#!/usr/bin/env python
"""Database statistics and summary."""

from maggma.stores import MongoStore
from collections import Counter

# Connect to database
store = MongoStore(
    database="atomate2siesta", collection_name="tasks", host="localhost", port=27017
)

store.connect()

print("Database Statistics")
print("=" * 70)

# Get all documents
docs = list(store.query())

print(f"\nTotal calculations: {len(docs)}")

# Count by calculation type
calc_types = [doc.get("name", "Unknown") for doc in docs]
type_counts = Counter(calc_types)

print("\nCalculations by type:")
for calc_type, count in type_counts.most_common():
    print(f"  {calc_type:50s}: {count:4d}")

# Count by formula (handle both dict and list outputs)
formulas = []
for doc in docs:
    output = doc.get("output", {})
    if isinstance(output, dict):
        formula = output.get("formula_pretty", "Unknown")
    else:
        formula = "Unknown"
    formulas.append(formula)

formula_counts = Counter(formulas)

print("\nMaterials studied (top 10):")
for formula, count in formula_counts.most_common(10):
    formula_str = formula if formula else "Unknown"
    print(f"  {formula_str:10s}: {count:4d} calculations")

# Energy statistics
energies = []
bandgaps = []

for doc in docs:
    output = doc.get("output", {})
    if isinstance(output, dict):
        calc_output = output.get("output")
        if calc_output and isinstance(calc_output, dict):
            energy = calc_output.get("energy")
            bandgap = calc_output.get("bandgap")

            if energy is not None:
                energies.append(energy)
            if bandgap is not None:
                bandgaps.append(bandgap)

if energies:
    print("\nEnergy statistics:")
    print(f"  Min:     {min(energies):10.3f} eV")
    print(f"  Max:     {max(energies):10.3f} eV")
    print(f"  Average: {sum(energies)/len(energies):10.3f} eV")

if bandgaps:
    print("\nBandgap statistics:")
    print(f"  Min:     {min(bandgaps):10.3f} eV")
    print(f"  Max:     {max(bandgaps):10.3f} eV")
    print(f"  Average: {sum(bandgaps)/len(bandgaps):10.3f} eV")

store.close()
