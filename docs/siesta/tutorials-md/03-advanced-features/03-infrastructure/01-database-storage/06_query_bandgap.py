#!/usr/bin/env python
"""Find materials by bandgap (semiconductors, metals, insulators)."""

from maggma.stores import MongoStore

# Connect to database
store = MongoStore(
    database="atomate2siesta", collection_name="tasks", host="localhost", port=27017
)

store.connect()

print("Materials Classification by Bandgap")
print("=" * 60)

# Get all documents with bandgap data
docs = list(store.query(criteria={"output.output.bandgap": {"$exists": True}}))

# Classify materials
metals = []
semiconductors = []
insulators = []

for doc in docs:
    output = doc.get("output", {})

    # Handle both dict and list outputs
    if not isinstance(output, dict):
        continue

    calc_output = output.get("output", {})
    bandgap = calc_output.get("bandgap")
    formula = output.get("formula_pretty", "Unknown")
    energy = calc_output.get("energy", 0)

    # Skip if bandgap is None
    if bandgap is None:
        continue

    material = {"formula": formula, "bandgap": bandgap, "energy": energy}

    if bandgap < 0.1:
        metals.append(material)
    elif bandgap < 3.0:
        semiconductors.append(material)
    else:
        insulators.append(material)

# Print results
print(f"\nMetals (Eg < 0.1 eV): {len(metals)}")
for m in metals[:5]:
    print(f"  {m['formula']:10s}  Eg = {m['bandgap']:.3f} eV  E = {m['energy']:.2f} eV")

print(f"\nSemiconductors (0.1 ≤ Eg < 3.0 eV): {len(semiconductors)}")
for m in semiconductors[:5]:
    print(f"  {m['formula']:10s}  Eg = {m['bandgap']:.3f} eV  E = {m['energy']:.2f} eV")

print(f"\nInsulators (Eg ≥ 3.0 eV): {len(insulators)}")
for m in insulators[:5]:
    print(f"  {m['formula']:10s}  Eg = {m['bandgap']:.3f} eV  E = {m['energy']:.2f} eV")

store.close()
