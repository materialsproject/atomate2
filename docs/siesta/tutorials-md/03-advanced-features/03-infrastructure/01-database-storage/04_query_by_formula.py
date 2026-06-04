#!/usr/bin/env python
"""Query calculations by chemical formula."""

from maggma.stores import MongoStore

# Connect to database
store = MongoStore(
    database="atomate2siesta", collection_name="tasks", host="localhost", port=27017
)

store.connect()

# Query by formula
formula = "Si"  # Change this to search for different materials
print(f"Searching for: {formula}")
print("-" * 50)

# Get all documents with this formula
docs = list(store.query(criteria={"output.formula_pretty": formula}))

print(f"Found {len(docs)} calculations for {formula}\n")

# Show details for each
for i, doc in enumerate(docs[:5], 1):  # Show first 5
    output = doc.get("output", {})

    # Handle both dict and list outputs
    if isinstance(output, dict):
        calc_output = output.get("output", {})
        energy = calc_output.get("energy", "N/A")
        bandgap = calc_output.get("bandgap", "N/A")
    else:
        energy = "N/A"
        bandgap = "N/A"

    print(f"{i}. {doc.get('name', 'N/A')}")
    print(f"   Energy:  {energy} eV")
    print(f"   Bandgap: {bandgap} eV")
    print(f"   Date:    {doc.get('completed_at', 'N/A')}")
    print()

if len(docs) > 5:
    print(f"... and {len(docs) - 5} more")

store.close()
