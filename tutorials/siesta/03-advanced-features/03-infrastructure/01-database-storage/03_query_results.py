#!/usr/bin/env python
"""Query results from MongoDB - simple example."""

from maggma.stores import MongoStore

# Connect to database
store = MongoStore(
    database="atomate2siesta", collection_name="tasks", host="localhost", port=27017
)

store.connect()

# Count documents
print(f"Total documents: {store.count()}")

# Get one document
doc = store.query_one()

if doc:
    # Extract data (handle nested structure)
    output = doc.get("output", {})
    calc_output = output.get("output", {})

    # Print results
    print(f"Formula:      {output.get('formula_pretty', 'N/A')}")
    print(f"Energy:       {calc_output.get('energy', 'N/A')} eV")
    print(f"Bandgap:      {calc_output.get('bandgap', 'N/A')} eV")
    print(f"Calculation:  {doc.get('name', 'N/A')}")
else:
    print("No documents found")

store.close()
