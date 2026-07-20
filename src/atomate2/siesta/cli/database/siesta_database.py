"""CLI for testing and managing atomate2siesta database connections."""

from __future__ import annotations

import click
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Initialize rich console
console = Console()


def test_mongodb_connection(
    host: str, port: int, database: str, collection_name: str
) -> tuple:
    """Test connection to MongoDB database.

    Parameters
    ----------
    host : str
        MongoDB host address
    port : int
        MongoDB port number
    database : str
        Database name
    collection_name : str
        Collection name

    Returns
    -------
    tuple
        (success: bool, client, db, collection, error_msg: str)
    """
    try:
        from pymongo import MongoClient
        from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

        # Create connection with timeout
        client = MongoClient(host=host, port=port, serverSelectionTimeoutMS=5000)

        # Test connection
        client.admin.command("ping")

        # Get database and collection
        db = client[database]
        collection = db[collection_name]

        return True, client, db, collection, None  # noqa: TRY300

    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        return False, None, None, None, f"Connection failed: {e}"
    except ImportError:
        return (
            False,
            None,
            None,
            None,
            "pymongo not installed. Run: pip install pymongo",
        )
    except Exception as e:  # noqa: BLE001 friendly CLI error handler
        return False, None, None, None, f"Unexpected error: {e}"


def test_maggma_store(
    host: str, port: int, database: str, collection_name: str
) -> tuple:
    """Test Maggma MongoStore connection.

    Parameters
    ----------
    host : str
        MongoDB host address
    port : int
        MongoDB port number
    database : str
        Database name
    collection_name : str
        Collection name

    Returns
    -------
    tuple
        (success: bool, store, error_msg: str)
    """
    try:
        from maggma.stores import MongoStore

        # Create store
        store = MongoStore(
            database=database, collection_name=collection_name, host=host, port=port
        )

        # Test connection
        with store:
            store.connect()

        return True, store, None  # noqa: TRY300

    except ImportError:
        return False, None, "maggma not installed. Run: pip install maggma"
    except Exception as e:  # noqa: BLE001 friendly CLI error handler
        return False, None, f"Store connection failed: {e}"


@click.group()
@click.version_option("0.1.0")
def cli() -> None:
    """Command-line interface for atomate2siesta database testing."""


@cli.command()
@click.option(
    "--host", default="localhost", help="MongoDB host address (default: localhost)"
)
@click.option("--port", default=27017, type=int, help="MongoDB port (default: 27017)")
@click.option(
    "--database",
    default="atomate2siesta",
    help="Database name (default: atomate2siesta)",
)
@click.option("--collection", default="tasks", help="Collection name (default: tasks)")
def test(host: str, port: int, database: str, collection: str) -> None:
    """Test MongoDB connection and display database statistics."""
    console.print(
        Panel(
            f"[bold cyan]Testing MongoDB Connection[/bold cyan]\n"
            f"Host: {host}:{port}\n"
            f"Database: {database}\n"
            f"Collection: {collection}",
            style="blue",
        )
    )

    # Test pymongo connection
    console.print("\n[yellow]Testing PyMongo connection...[/yellow]")
    success, client, db, coll, error = test_mongodb_connection(
        host, port, database, collection
    )

    if not success:
        console.print(f"[red]✗ PyMongo connection failed:[/red] {error}")
        return

    console.print("[green]✓ PyMongo connection successful[/green]")

    # Display database statistics
    try:
        doc_count = coll.count_documents({})

        # Get database stats
        db_stats = db.command("dbStats")
        num_collections = db_stats.get("collections", 0)

        # Check if database is actually set up with data
        if doc_count == 0 and num_collections == 0:
            console.print(
                Panel(
                    f"[bold yellow]⚠ Database '{database}' is NOT set up"
                    f"[/bold yellow]\n\n"
                    f"The database '{database}' does not exist or has no "
                    f"collections.\n\n"
                    f"[bold cyan]To create the database structure:[/bold cyan]\n"
                    f"  → Run: [white]atomate2siesta-database create --create-indexes"
                    f"[/white]\n\n"
                    f"[bold cyan]Alternative - Run calculations directly:[/bold cyan]\n"
                    f"  1. Configure database storage (see config command)\n"
                    f"  2. Run SIESTA calculations with database store\n"
                    f"  3. Database and collection will be created automatically\n\n"
                    f"[bold cyan]For help:[/bold cyan]\n"
                    f"  → [white]atomate2siesta-database config[/white]  "
                    f"(show setup examples)\n"
                    f"  → [white]atomate2siesta-database create --help[/white]  "
                    f"(create options)",
                    style="yellow",
                    title="Database Not Found",
                )
            )
        elif doc_count == 0:
            console.print(
                Panel(
                    f"[bold green]✓ Database '{database}' is set up correctly!"
                    f"[/bold green]\n\n"
                    f"[white]Database has {num_collections} collection(s) with "
                    f"indexes ready.[/white]\n"
                    f"[white]Collection '{collection}' exists but has no "
                    f"calculation results yet.[/white]\n\n"
                    f"[bold cyan]Everything is ready! Next steps:[/bold cyan]\n"
                    f"  1. Run SIESTA calculations with database storage\n"
                    f"  2. Results will be automatically stored in '{collection}'\n"
                    f"  3. Use 'list' and 'query' commands to view results\n\n"
                    f"[bold cyan]Need help getting started?[/bold cyan]\n"
                    f"  → [white]atomate2siesta-database config[/white]  "
                    f"(setup examples)\n"
                    f"  → See tutorial: [white]tutorials/13-database-storage/[/white]",
                    style="green",
                    title="Database Ready",
                )
            )
        else:
            console.print(
                Panel(
                    f"[bold green]✓ Database '{database}' is set up and populated"
                    f"[/bold green]\n\n"
                    f"Collection '{collection}' has {doc_count} documents",
                    style="green",
                    title="Database Status",
                )
            )

        # Create stats table
        table = Table(title="Database Statistics", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Database", database)
        table.add_row("Collections", str(num_collections))
        table.add_row("Documents", str(doc_count))
        table.add_row(
            "Data Size", f"{db_stats.get('dataSize', 0) / 1024 / 1024:.2f} MB"
        )
        table.add_row(
            "Storage Size", f"{db_stats.get('storageSize', 0) / 1024 / 1024:.2f} MB"
        )
        table.add_row("Indexes", str(db_stats.get("indexes", 0)))

        console.print(table)

        # Test Maggma store
        console.print("\n[yellow]Testing Maggma store...[/yellow]")
        success, _store, error = test_maggma_store(host, port, database, collection)

        if success:
            console.print("[green]✓ Maggma store connection successful[/green]")
        else:
            console.print(f"[red]✗ Maggma store failed:[/red] {error}")

    except Exception as e:  # noqa: BLE001 friendly CLI error handler
        console.print(f"[red]Error getting database stats:[/red] {e}")
    finally:
        if client:
            client.close()


@cli.command()
@click.option(
    "--host", default="localhost", help="MongoDB host address (default: localhost)"
)
@click.option("--port", default=27017, type=int, help="MongoDB port (default: 27017)")
@click.option(
    "--database",
    default="atomate2siesta",
    help="Database name (default: atomate2siesta)",
)
@click.option("--collection", default="tasks", help="Collection name (default: tasks)")
@click.option(
    "--limit",
    default=10,
    type=int,
    help="Maximum number of documents to display (default: 10)",
)
def list(  # noqa: A001 Click command name must stay `list`
    host: str, port: int, database: str, collection: str, limit: int
) -> None:
    """List recent documents in the database."""
    success, client, _db, coll, error = test_mongodb_connection(
        host, port, database, collection
    )

    if not success:
        console.print(f"[red]Connection failed:[/red] {error}")
        return

    try:
        # Show database connection info
        console.print(
            Panel(
                f"[bold cyan]Database Connection[/bold cyan]\n"
                f"Host: {host}:{port}\n"
                f"Database: {database}\n"
                f"Collection: {collection}",
                style="blue",
            )
        )

        doc_count = coll.count_documents({})
        console.print(f"\n[cyan]Found {doc_count} documents in '{collection}'[/cyan]")

        if doc_count == 0:
            console.print("[yellow]No documents found in collection[/yellow]")
            return

        # Get recent documents
        documents = [doc for doc in coll.find().sort("_id", -1).limit(limit)]

        # Create table
        table = Table(
            title=f"Recent Documents (showing "
            f"{min(limit, len(documents))} of {doc_count})",
            box=box.ROUNDED,
        )
        table.add_column("UUID", style="cyan", overflow="fold")
        table.add_column("Formula", style="green")
        table.add_column("State", style="yellow")
        table.add_column("Energy (eV)", style="magenta")
        table.add_column("Calculation Type", style="blue")

        for doc in documents:
            uuid_str = str(doc.get("uuid", "N/A"))[:36]

            # Fields are nested inside 'output' (SiestaTaskDoc structure)
            output = doc.get("output", {})
            # Handle case where output is a string instead of dict
            if not isinstance(output, dict):
                output = {}
            formula = output.get("formula_pretty", output.get("formula", "N/A"))
            state = output.get("state", "N/A")

            # Energy is in nested output.output dict
            nested_output = output.get("output", {})
            if not isinstance(nested_output, dict):
                nested_output = {}
            energy = nested_output.get("energy", "N/A")
            if isinstance(energy, (int, float)):
                energy = f"{energy:.4f}"

            # Get calculation type from top-level name or nested task_label
            calc_type = doc.get("name", output.get("task_label", "N/A"))

            table.add_row(uuid_str, formula, state, str(energy), calc_type)

        console.print(table)

    except Exception as e:  # noqa: BLE001 friendly CLI error handler
        console.print(f"[red]Error querying database:[/red] {e}")
    finally:
        if client:
            client.close()


@cli.command()
@click.option(
    "--host", default="localhost", help="MongoDB host address (default: localhost)"
)
@click.option("--port", default=27017, type=int, help="MongoDB port (default: 27017)")
@click.option(
    "--database",
    default="atomate2siesta",
    help="Database name (default: atomate2siesta)",
)
@click.option("--collection", default="tasks", help="Collection name (default: tasks)")
@click.option("--formula", help="Query by chemical formula")
@click.option("--state", help="Query by calculation state (successful, failed, etc.)")
@click.option("--calc-type", help="Query by calculation type")
@click.option("--energy-min", type=float, help="Minimum energy (eV)")
@click.option("--energy-max", type=float, help="Maximum energy (eV)")
@click.option(
    "--export",
    type=click.Choice(["csv", "json"]),
    help="Export results to file (csv or json)",
)
@click.option(
    "--output",
    "output_file",
    default="query_results",
    help="Output filename (without extension)",
)
@click.option(
    "--latest",
    type=int,
    help="Show N most recent calculations (sorted by completion time)",
)
def query(
    host: str,
    port: int,
    database: str,
    collection: str,
    formula: str | None,
    state: str | None,
    calc_type: str | None,
    energy_min: float | None,
    energy_max: float | None,
    export: str | None,
    output_file: str,
    latest: int | None,
) -> None:
    """Query documents with various filters and export options."""
    # Check if at least one query option is provided (unless --latest is used)
    if not latest and not any([formula, state, calc_type, energy_min, energy_max]):
        console.print(
            "[red]Error: At least one query option must be provided[/red]\n"
            "Use --formula, --state, --calc-type, --energy-min, "
            "--energy-max, or --latest"
        )
        return

    success, client, _db, coll, error = test_mongodb_connection(
        host, port, database, collection
    )

    if not success:
        console.print(f"[red]Connection failed:[/red] {error}")
        return

    try:
        # Build query dictionary based on provided options
        query_dict = {}
        query_parts = []

        if formula:
            query_dict["$or"] = [
                {"output.formula": formula},
                {"output.formula_pretty": formula},
            ]
            query_parts.append(f"Formula: [green]{formula}[/green]")

        if state:
            query_dict["output.state"] = state
            query_parts.append(f"State: [yellow]{state}[/yellow]")

        if calc_type:
            query_dict["name"] = {"$regex": calc_type, "$options": "i"}
            query_parts.append(f"Type: [blue]{calc_type}[/blue]")

        if energy_min is not None or energy_max is not None:
            energy_query = {}
            if energy_min is not None:
                energy_query["$gte"] = energy_min
                query_parts.append(f"Energy ≥ [magenta]{energy_min}[/magenta] eV")
            if energy_max is not None:
                energy_query["$lte"] = energy_max
                query_parts.append(f"Energy ≤ [magenta]{energy_max}[/magenta] eV")
            query_dict["output.output.energy"] = energy_query

        # Add latest filter info
        if latest:
            query_parts.append(f"Latest: [cyan]{latest}[/cyan] calculations")

        # Show database connection info
        query_info = "\n".join(query_parts) if query_parts else "All documents"
        console.print(
            Panel(
                f"[bold cyan]Database Query[/bold cyan]\n"
                f"Host: {host}:{port}\n"
                f"Database: {database}\n"
                f"Collection: {collection}\n\n"
                f"[bold]Filters:[/bold]\n{query_info}",
                style="blue",
            )
        )

        # Execute query with sorting and limit
        if latest:
            # Sort by completion time (descending) and limit
            cursor = coll.find(query_dict).sort("completed_at", -1).limit(latest)
            documents = [doc for doc in cursor]
        else:
            documents = [doc for doc in coll.find(query_dict)]

        console.print(f"\n[cyan]Found {len(documents)} matching documents[/cyan]")

        if len(documents) == 0:
            console.print("[yellow]No matching documents found[/yellow]")
            return

        # Limit display to reasonable number (but export all)
        # If --latest is used, show all requested documents
        DISPLAY_LIMIT = latest or 50  # noqa: N806 local display-limit constant
        display_docs = documents[:DISPLAY_LIMIT]

        if len(documents) > DISPLAY_LIMIT and not latest:
            console.print(
                f"[yellow]Displaying first {DISPLAY_LIMIT} of "
                f"{len(documents)} documents[/yellow]"
            )
            console.print("[dim]Use --export to save all results to file[/dim]\n")

        # Create detailed table and collect data for export
        table_title = (
            f"Query Results (showing {min(len(documents), DISPLAY_LIMIT)} "
            f"of {len(documents)})"
        )
        table = Table(title=table_title, box=box.ROUNDED)
        table.add_column("UUID", style="cyan", overflow="fold", width=12)
        table.add_column("Formula", style="green", width=10)
        table.add_column("State", style="yellow", width=10)
        table.add_column("Energy (eV)", style="magenta", justify="right", width=12)
        table.add_column("Type", style="blue", width=15)
        table.add_column("K-points", style="white", width=10)
        table.add_column("Basis", style="cyan", width=8)
        table.add_column("Mesh Cutoff", style="dim", width=12)
        table.add_column("Completed", style="dim", width=16)

        # Collect data for export
        export_data = []

        # Display limited rows in table
        for doc in display_docs:
            uuid_str = str(doc.get("uuid", "N/A"))[:12]

            # Fields are nested inside 'output' (SiestaTaskDoc structure)
            output = doc.get("output", {})
            # Handle case where output is a string instead of dict
            if not isinstance(output, dict):
                output = {}
            state = output.get("state", "N/A")

            # Energy is in nested output.output dict
            nested_output = output.get("output", {})
            if not isinstance(nested_output, dict):
                nested_output = {}
            energy = nested_output.get("energy", "N/A")
            if isinstance(energy, (int, float)):
                energy = f"{energy:.4f}"

            # Get calculation type
            calc_type = doc.get("name", output.get("task_label", "N/A"))

            # Get k-points, basis, mesh cutoff from input
            input_data = output.get("input", {})
            if not isinstance(input_data, dict):
                input_data = {}

            # Extract k-points (now stored directly in input)
            kpts = input_data.get("kpts", "N/A")
            # Use type() instead of isinstance to avoid shadowing from
            # function name 'list'
            if type(kpts).__name__ == "list" and len(kpts) == 3:
                kpts = f"{kpts[0]}×{kpts[1]}×{kpts[2]}"  # noqa: RUF001 k-point separator

            # Get basis size (now stored directly in input)
            basis = input_data.get("basis_size", "N/A")

            # Get mesh cutoff (now stored directly in input)
            mesh_cutoff = input_data.get("mesh_cutoff", "N/A")
            if mesh_cutoff != "N/A" and mesh_cutoff is not None:
                # Format mesh cutoff
                try:
                    mesh_cutoff = f"{float(mesh_cutoff):.0f} Ry"
                except (ValueError, TypeError):
                    mesh_cutoff = "N/A"

            # Get completion timestamp
            completed_at = doc.get("completed_at", output.get("completed_at", "N/A"))
            if completed_at != "N/A":
                # Format datetime if it's a datetime object
                try:
                    if hasattr(completed_at, "strftime"):
                        completed_at = completed_at.strftime("%Y-%m-%d %H:%M")
                    else:
                        # Try to parse string datetime
                        from datetime import datetime

                        dt = datetime.fromisoformat(
                            str(completed_at).replace("Z", "+00:00")
                        )
                        completed_at = dt.strftime("%Y-%m-%d %H:%M")
                except (ValueError, TypeError, AttributeError):
                    completed_at = str(completed_at)[:16]

            # Get formula
            formula_val = output.get("formula_pretty", output.get("formula", "N/A"))

            table.add_row(
                uuid_str,
                formula_val,
                state,
                str(energy),
                calc_type,
                str(kpts),
                str(basis),
                str(mesh_cutoff),
                completed_at,
            )

        console.print(table)

        # Collect ALL data for export (not just displayed rows)
        if export:
            console.print("\n[cyan]Preparing export data...[/cyan]")
            for doc in documents:
                # Extract same fields as table
                output = doc.get("output", {})
                if not isinstance(output, dict):
                    output = {}

                formula_val = output.get("formula_pretty", output.get("formula", "N/A"))
                state = output.get("state", "N/A")

                nested_output = output.get("output", {})
                if not isinstance(nested_output, dict):
                    nested_output = {}
                energy = nested_output.get("energy", "N/A")

                calc_type = doc.get("name", output.get("task_label", "N/A"))

                input_data = output.get("input", {})
                if not isinstance(input_data, dict):
                    input_data = {}

                # Extract k-points (now stored directly in input)
                kpts = input_data.get("kpts", "N/A")
                # Use type() instead of isinstance to avoid shadowing from
                # function name 'list'
                if type(kpts).__name__ == "list" and len(kpts) == 3:
                    kpts = f"{kpts[0]}×{kpts[1]}×{kpts[2]}"  # noqa: RUF001 k-point separator

                # Get basis size (now stored directly in input)
                basis = input_data.get("basis_size", "N/A")

                # Get mesh cutoff (now stored directly in input)
                mesh_cutoff = input_data.get("mesh_cutoff", "N/A")
                if mesh_cutoff != "N/A" and mesh_cutoff is not None:
                    try:
                        mesh_cutoff = f"{float(mesh_cutoff):.0f} Ry"
                    except (ValueError, TypeError):
                        mesh_cutoff = "N/A"

                completed_at = doc.get(
                    "completed_at", output.get("completed_at", "N/A")
                )

                export_data.append(
                    {
                        "UUID": str(doc.get("uuid", "N/A")),
                        "Formula": formula_val,
                        "State": state,
                        "Energy_eV": energy,
                        "Calculation_Type": calc_type,
                        "K_points": str(kpts),
                        "Basis": str(basis),
                        "Mesh_Cutoff": str(mesh_cutoff),
                        "Completed": str(completed_at),
                    }
                )

        # Export data if requested
        if export and export_data:
            try:
                if export == "csv":
                    import csv

                    filename = f"{output_file}.csv"
                    with open(filename, "w", newline="") as csvfile:
                        fieldnames = export_data[0].keys()
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(export_data)
                    console.print(
                        f"\n[green]✓ Exported {len(export_data)} documents "
                        f"to {filename}[/green]"
                    )

                elif export == "json":
                    import json

                    filename = f"{output_file}.json"
                    with open(filename, "w") as jsonfile:
                        json.dump(export_data, jsonfile, indent=2)
                    console.print(
                        f"\n[green]✓ Exported {len(export_data)} documents "
                        f"to {filename}[/green]"
                    )

            except Exception as e:  # noqa: BLE001 friendly CLI error handler
                console.print(f"[red]Error exporting data:[/red] {e}")

    except Exception as e:  # noqa: BLE001 friendly CLI error handler
        console.print(f"[red]Error querying database:[/red] {e}")
    finally:
        if client:
            client.close()


@cli.command()
@click.option(
    "--host", default="localhost", help="MongoDB host address (default: localhost)"
)
@click.option("--port", default=27017, type=int, help="MongoDB port (default: 27017)")
@click.option(
    "--database",
    default="atomate2siesta",
    help="Database name (default: atomate2siesta)",
)
def stats(host: str, port: int, database: str) -> None:
    """Display comprehensive database statistics."""
    success, client, db, _, error = test_mongodb_connection(
        host, port, database, "tasks"
    )

    if not success:
        console.print(f"[red]Connection failed:[/red] {error}")
        return

    try:
        # Get all collections
        collections = db.list_collection_names()

        console.print(
            Panel(
                f"[bold cyan]Database: {database}[/bold cyan]\n"
                f"Host: {host}:{port}\n"
                f"Collections: {len(collections)}",
                style="blue",
            )
        )

        # Create collections table
        table = Table(title="Collection Statistics", box=box.ROUNDED)
        table.add_column("Collection", style="cyan")
        table.add_column("Documents", style="green", justify="right")
        table.add_column("Size (MB)", style="magenta", justify="right")
        table.add_column("Avg Doc Size (KB)", style="yellow", justify="right")

        total_docs = 0
        total_size = 0

        for coll_name in collections:
            stats = db.command("collStats", coll_name)

            doc_count = stats.get("count", 0)
            size_mb = stats.get("size", 0) / 1024 / 1024
            avg_size_kb = stats.get("avgObjSize", 0) / 1024 if doc_count > 0 else 0

            table.add_row(
                coll_name, str(doc_count), f"{size_mb:.2f}", f"{avg_size_kb:.2f}"
            )

            total_docs += doc_count
            total_size += stats.get("size", 0)

        # Add total row
        table.add_row(
            "[bold]TOTAL[/bold]",
            f"[bold]{total_docs}[/bold]",
            f"[bold]{total_size / 1024 / 1024:.2f}[/bold]",
            "",
        )

        console.print(table)

        # Get database-level stats
        db_stats = db.command("dbStats")

        info_table = Table(title="Database Info", box=box.ROUNDED)
        info_table.add_column("Metric", style="cyan")
        info_table.add_column("Value", style="green")

        info_table.add_row(
            "Data Size", f"{db_stats.get('dataSize', 0) / 1024 / 1024:.2f} MB"
        )
        info_table.add_row(
            "Storage Size", f"{db_stats.get('storageSize', 0) / 1024 / 1024:.2f} MB"
        )
        info_table.add_row("Total Indexes", str(db_stats.get("indexes", 0)))
        info_table.add_row(
            "Index Size", f"{db_stats.get('indexSize', 0) / 1024 / 1024:.2f} MB"
        )

        console.print(info_table)

    except Exception as e:  # noqa: BLE001 friendly CLI error handler
        console.print(f"[red]Error getting statistics:[/red] {e}")
    finally:
        if client:
            client.close()


@cli.command()
@click.option(
    "--host", default="localhost", help="MongoDB host address (default: localhost)"
)
@click.option("--port", default=27017, type=int, help="MongoDB port (default: 27017)")
@click.option(
    "--database",
    default="atomate2siesta",
    help="Database name (default: atomate2siesta)",
)
@click.option("--collection", default="tasks", help="Collection name (default: tasks)")
@click.option("--force", is_flag=True, help="Skip confirmation prompt")
@click.option(
    "--drop-collection",
    is_flag=True,
    help="Drop the entire collection (not just documents)",
)
@click.option(
    "--drop-database",
    is_flag=True,
    help="Drop the entire database (EXTREMELY DANGEROUS)",
)
def clear(
    host: str,
    port: int,
    database: str,
    collection: str,
    force: bool,
    drop_collection: bool,
    drop_database: bool,
) -> None:
    """Clear documents, drop collection, or drop database (USE WITH CAUTION)."""
    # Check for conflicting options
    if drop_database and drop_collection:
        console.print(
            "[red]Error: Cannot use both --drop-collection and --drop-database[/red]"
        )
        return

    success, client, db, coll, error = test_mongodb_connection(
        host, port, database, collection
    )

    if not success:
        console.print(f"[red]Connection failed:[/red] {error}")
        return

    try:
        # Option 1: Drop entire database
        if drop_database:
            # Get database stats for warning
            db_stats = db.command("dbStats")
            collections = db.list_collection_names()
            num_collections = len(collections)

            console.print(
                Panel(
                    f"[bold red]DANGER: This will DELETE THE ENTIRE DATABASE!"
                    f"[/bold red]\n\n"
                    f"Database: {database}\n"
                    f"Collections: {num_collections}\n"
                    f"Total Size: "
                    f"{db_stats.get('dataSize', 0) / 1024 / 1024:.2f} MB\n\n"
                    f"[yellow]All collections and all data will be permanently "
                    f"lost![/yellow]",
                    style="red",
                    title="⚠️  EXTREME CAUTION  ⚠️",
                )
            )

            if not force:
                # Extra confirmation for database deletion
                console.print(
                    f"\n[red]Type the database name '{database}' to confirm:[/red]"
                )
                confirmation = click.prompt("Database name", type=str)

                if confirmation != database:
                    console.print(
                        "[yellow]Database name doesn't match. "
                        "Operation cancelled.[/yellow]"
                    )
                    return

                if not click.confirm(
                    f"Are you ABSOLUTELY SURE you want to delete "
                    f"database '{database}'?",
                    abort=True,
                ):
                    return

            client.drop_database(database)
            console.print(
                f"\n[green]✓ Dropped database '{database}' and all its "
                f"collections[/green]"
            )

        # Option 2: Drop collection (including indexes and all documents)
        elif drop_collection:
            # Check if collection exists
            if collection not in db.list_collection_names():
                console.print(
                    f"[yellow]Collection '{collection}' does not exist[/yellow]"
                )
                return

            doc_count = coll.count_documents({})
            num_indexes = len([idx for idx in coll.list_indexes()])

            console.print(
                Panel(
                    f"[bold red]WARNING: This will DROP THE ENTIRE COLLECTION!"
                    f"[/bold red]\n\n"
                    f"Database: {database}\n"
                    f"Collection: {collection}\n"
                    f"Documents: {doc_count}\n"
                    f"Indexes: {num_indexes}\n\n"
                    f"[yellow]The collection and all its indexes will be "
                    f"deleted![/yellow]",
                    style="red",
                    title="Collection Deletion",
                )
            )

            if not force and not click.confirm(
                f"Are you sure you want to drop collection '{collection}'?",
                abort=True,
            ):
                return

            coll.drop()
            console.print(
                f"[green]✓ Dropped collection '{collection}' "
                f"(including all documents and indexes)[/green]"
            )

        # Option 3: Clear documents only (default behavior)
        else:
            doc_count = coll.count_documents({})

            console.print(
                Panel(
                    f"[bold red]WARNING: This will delete all {doc_count} "
                    f"documents![/bold red]\n"
                    f"Database: {database}\n"
                    f"Collection: {collection}\n\n"
                    f"[yellow]Indexes will be preserved.[/yellow]",
                    style="red",
                )
            )

            if not force and not click.confirm(
                "Are you sure you want to delete all documents?", abort=True
            ):
                return

            result = coll.delete_many({})
            console.print(
                f"[green]✓ Deleted {result.deleted_count} documents "
                f"from '{collection}'[/green]"
            )
            console.print("[cyan]Collection and indexes are preserved[/cyan]")

    except click.exceptions.Abort:
        console.print("[yellow]Operation cancelled by user[/yellow]")
    except Exception as e:  # noqa: BLE001 friendly CLI error handler
        console.print(f"[red]Error during operation:[/red] {e}")
        import traceback

        console.print(f"[red]{traceback.format_exc()}[/red]")
    finally:
        if client:
            client.close()


@cli.command()
@click.option(
    "--host", default="localhost", help="MongoDB host address (default: localhost)"
)
@click.option("--port", default=27017, type=int, help="MongoDB port (default: 27017)")
@click.option(
    "--database",
    default="atomate2siesta",
    help="Database name (default: atomate2siesta)",
)
@click.option("--collection", default="tasks", help="Collection name (default: tasks)")
@click.option(
    "--create-indexes",
    is_flag=True,
    help="Create recommended indexes for better query performance",
)
def create(
    host: str, port: int, database: str, collection: str, create_indexes: bool
) -> None:
    """Create database and collection with optional indexes."""
    console.print(
        Panel(
            f"[bold cyan]Creating Database Structure[/bold cyan]\n"
            f"Host: {host}:{port}\n"
            f"Database: {database}\n"
            f"Collection: {collection}",
            style="blue",
        )
    )

    # Test connection first
    console.print("\n[yellow]Testing MongoDB connection...[/yellow]")
    success, client, db, coll, error = test_mongodb_connection(
        host, port, database, collection
    )

    if not success:
        console.print(f"[red]✗ Connection failed:[/red] {error}")
        console.print("\n[yellow]Make sure MongoDB is running:[/yellow]")
        console.print("  macOS: [cyan]brew services start mongodb-community[/cyan]")
        console.print("  Linux: [cyan]sudo systemctl start mongodb[/cyan]")
        return

    console.print("[green]✓ MongoDB connection successful[/green]")

    try:
        # Check if collection already exists
        existing_collections = db.list_collection_names()

        if collection in existing_collections:
            doc_count = coll.count_documents({})
            console.print(
                f"\n[yellow]⚠ Collection '{collection}' already exists with "
                f"{doc_count} documents[/yellow]"
            )

            if not click.confirm(
                "Do you want to continue (will not delete existing data)?", default=True
            ):
                console.print("[yellow]Operation cancelled[/yellow]")
                return
        else:
            # Create collection by inserting and removing a dummy document
            # This ensures the collection is created
            dummy_id = coll.insert_one({"_dummy": True}).inserted_id
            coll.delete_one({"_id": dummy_id})
            console.print(f"[green]✓ Created collection '{collection}'[/green]")

        # Create recommended indexes if requested
        if create_indexes:
            console.print("\n[yellow]Creating recommended indexes...[/yellow]")

            indexes_created = []

            # Index on uuid (unique)
            try:
                coll.create_index("uuid", unique=True)
                indexes_created.append("uuid (unique)")
                console.print("[green]  ✓ Created index on 'uuid' (unique)[/green]")
            except Exception as e:  # noqa: BLE001 friendly CLI error handler
                console.print(
                    f"[yellow]  ⚠ Index on 'uuid' may already exist: {e}[/yellow]"
                )

            # Index on formula for faster querying
            try:
                coll.create_index("formula")
                indexes_created.append("formula")
                console.print("[green]  ✓ Created index on 'formula'[/green]")
            except Exception as e:  # noqa: BLE001 friendly CLI error handler
                console.print(
                    f"[yellow]  ⚠ Index on 'formula' may already exist: {e}[/yellow]"
                )

            # Index on formula_pretty
            try:
                coll.create_index("formula_pretty")
                indexes_created.append("formula_pretty")
                console.print("[green]  ✓ Created index on 'formula_pretty'[/green]")
            except Exception as e:  # noqa: BLE001 friendly CLI error handler
                console.print(
                    f"[yellow]  ⚠ Index on 'formula_pretty' may already "
                    f"exist: {e}[/yellow]"
                )

            # Index on state for filtering successful/failed calculations
            try:
                coll.create_index("state")
                indexes_created.append("state")
                console.print("[green]  ✓ Created index on 'state'[/green]")
            except Exception as e:  # noqa: BLE001 friendly CLI error handler
                console.print(
                    f"[yellow]  ⚠ Index on 'state' may already exist: {e}[/yellow]"
                )

            # Index on output.energy for energy range queries
            try:
                coll.create_index("output.energy")
                indexes_created.append("output.energy")
                console.print("[green]  ✓ Created index on 'output.energy'[/green]")
            except Exception as e:  # noqa: BLE001 friendly CLI error handler
                console.print(
                    f"[yellow]  ⚠ Index on 'output.energy' may already "
                    f"exist: {e}[/yellow]"
                )

            # Index on completed_at for time-based queries
            try:
                coll.create_index("completed_at")
                indexes_created.append("completed_at")
                console.print("[green]  ✓ Created index on 'completed_at'[/green]")
            except Exception as e:  # noqa: BLE001 friendly CLI error handler
                console.print(
                    f"[yellow]  ⚠ Index on 'completed_at' may already "
                    f"exist: {e}[/yellow]"
                )

        # Get final statistics
        coll_stats = db.command("collStats", collection)
        # Convert to list to get length
        num_indexes = len([idx for idx in coll.list_indexes()])

        # Display summary
        console.print(
            Panel(
                f"[bold green]✓ Database setup complete![/bold green]\n\n"
                f"Database: {database}\n"
                f"Collection: {collection}\n"
                f"Documents: {coll_stats.get('count', 0)}\n"
                f"Indexes: {num_indexes}",
                style="green",
                title="Setup Summary",
            )
        )

        # Show index information
        if create_indexes:
            console.print("\n[cyan]Created Indexes:[/cyan]")
            for idx in coll.list_indexes():
                idx_name = idx.get("name", "unknown")
                idx_keys = idx.get("key", {})
                unique = " (unique)" if idx.get("unique", False) else ""
                console.print(f"  • {idx_name}: {dict(idx_keys)}{unique}")

        # Show next steps
        console.print(
            Panel(
                "[bold cyan]Next Steps:[/bold cyan]\n\n"
                "1. Run calculations with database storage:\n"
                "   [white]See: atomate2siesta-database config[/white]\n\n"
                "2. Verify setup:\n"
                "   [white]atomate2siesta-database test[/white]\n\n"
                "3. Monitor your database:\n"
                "   [white]atomate2siesta-database stats[/white]",
                style="blue",
            )
        )

    except Exception as e:  # noqa: BLE001 friendly CLI error handler
        console.print(f"[red]Error creating database structure:[/red] {e}")
        import traceback

        console.print(f"[red]{traceback.format_exc()}[/red]")
    finally:
        if client:
            client.close()


@cli.command()
@click.option(
    "--generate",
    is_flag=True,
    help="Generate ~/.jobflow.yaml file with recommended settings",
)
@click.option(
    "--host", default="localhost", help="MongoDB host address (default: localhost)"
)
@click.option("--port", default=27017, type=int, help="MongoDB port (default: 27017)")
@click.option(
    "--database",
    default="atomate2siesta",
    help="Database name (default: atomate2siesta)",
)
@click.option("--force", is_flag=True, help="Overwrite existing ~/.jobflow.yaml file")
def config(generate: bool, host: str, port: int, database: str, force: bool) -> None:
    """Show example database configuration files or generate ~/.jobflow.yaml."""
    # If --generate flag is used, create the file
    if generate:
        from pathlib import Path

        jobflow_path = Path.home() / ".jobflow.yaml"

        # Check if file exists
        if jobflow_path.exists() and not force:
            console.print(
                Panel(
                    f"[bold yellow]⚠ File already exists:[/bold yellow]\n\n"
                    f"[white]{jobflow_path}[/white]\n\n"
                    f"Use [cyan]--force[/cyan] to overwrite existing file.",
                    style="yellow",
                    title="File Exists",
                )
            )

            if not click.confirm("Do you want to overwrite it?", default=False):
                console.print("[yellow]Operation cancelled[/yellow]")
                return

        # Generate the configuration content
        config_content = f"""# Jobflow configuration for atomate2siesta
# This file is automatically loaded by jobflow

JOB_STORE:
  docs_store:
    type: MongoStore
    database: {database}
    collection_name: tasks
    host: {host}
    port: {port}

  additional_stores:
    data:
      type: GridFSStore
      database: {database}
      collection_name: task_data
      host: {host}
      port: {port}
"""

        # Write the file
        try:
            with open(jobflow_path, "w") as f:
                f.write(config_content)

            console.print(
                Panel(
                    f"[bold green]✓ Successfully created jobflow configuration!"
                    f"[/bold green]\n\n"
                    f"[white]File location:[/white] [cyan]{jobflow_path}[/cyan]\n"
                    f"[white]Database:[/white] {database}\n"
                    f"[white]Host:[/white] {host}:{port}\n\n"
                    f"[bold cyan]Next steps:[/bold cyan]\n"
                    f"1. Verify database is running:\n"
                    f"   [white]atomate2siesta-database test[/white]\n\n"
                    f"2. Create database structure (if needed):\n"
                    f"   [white]atomate2siesta-database create --create-indexes"
                    f"[/white]\n\n"
                    f"3. Run calculations:\n"
                    f"   Results will be automatically stored in MongoDB!\n\n"
                    f"[yellow]Note:[/yellow] Jobflow automatically loads this "
                    f"file on import.",
                    style="green",
                    title="Configuration Created",
                )
            )

            # Show the file contents
            console.print(
                Panel(
                    f"[bold cyan]File contents:[/bold cyan]\n\n"
                    f"[white]{config_content}[/white]",
                    style="blue",
                    title="~/.jobflow.yaml",
                )
            )

            return  # noqa: TRY300

        except Exception as e:  # noqa: BLE001 friendly CLI error handler
            console.print(f"[red]Error creating file:[/red] {e}")
            import traceback

            console.print(f"[red]{traceback.format_exc()}[/red]")
            return

    # Otherwise, show examples (existing behavior)
    jobflow_config = """# ~/.jobflow.yaml
JOB_STORE:
  docs_store:
    type: MongoStore
    database: atomate2siesta
    collection_name: tasks
    host: localhost
    port: 27017

  additional_stores:
    data:
      type: GridFSStore
      database: atomate2siesta
      collection_name: task_data
      host: localhost
      port: 27017
"""

    python_method1 = """# Method 1: Using ~/.jobflow.yaml (Automatic)
from jobflow import run_locally
from atomate2.siesta.jobs.core import RelaxMaker
from pymatgen.core import Structure

# Load structure
structure = Structure.from_file("structure.cif")

# Create job
job = RelaxMaker.fixed_cell_relaxation().make(structure)

# Run - jobflow automatically reads ~/.jobflow.yaml
# Results are stored in MongoDB as configured in the YAML file
results = run_locally(job, create_folders=True)
"""

    python_method2 = """# Method 2: Explicit store parameter (Manual)
from maggma.stores import MongoStore
from jobflow import run_locally
from atomate2.siesta.jobs.core import RelaxMaker
from pymatgen.core import Structure

# Create store explicitly
store = MongoStore(
    database="atomate2siesta",
    collection_name="tasks",
    host="localhost",
    port=27017
)

# Load structure and create job
structure = Structure.from_file("structure.cif")
job = RelaxMaker.fixed_cell_relaxation().make(structure)

# Run with explicit store parameter
results = run_locally(job, create_folders=True, store=store)
"""

    console.print(
        Panel(
            "[bold yellow]📋 Two Ways to Configure Database Storage[/bold yellow]\n\n"
            "[bold cyan]Option 1: Jobflow Config File (Recommended)[/bold cyan]\n"
            "Jobflow automatically reads [white]~/.jobflow.yaml[/white]\n"
            "This is the preferred method for production workflows.\n\n"
            "[bold cyan]Option 2: Python Store Parameter[/bold cyan]\n"
            "Pass [white]store=[/white] parameter to [white]run_locally()[/white]\n"
            "Useful for testing or one-off calculations.",
            style="blue",
            title="Configuration Methods",
        )
    )

    console.print(
        Panel(
            "[bold cyan]Jobflow Configuration (~/.jobflow.yaml)[/bold cyan]\n\n"
            f"[white]{jobflow_config}[/white]\n"
            "[yellow]How it works:[/yellow]\n"
            "• Jobflow automatically loads this file on import\n"
            "• All run_locally() calls use this configuration\n"
            "• No need to pass store parameter in Python code\n"
            "• Best for production workflows and HPC",
            style="blue",
        )
    )

    console.print(
        Panel(
            "[bold cyan]Method 1: Using ~/.jobflow.yaml (Recommended)[/bold cyan]\n\n"
            f"[white]{python_method1}[/white]",
            style="green",
        )
    )

    console.print(
        Panel(
            "[bold cyan]Method 2: Explicit Store Parameter[/bold cyan]\n\n"
            f"[white]{python_method2}[/white]",
            style="green",
        )
    )

    console.print(
        Panel(
            "[bold yellow]Setup Steps:[/bold yellow]\n\n"
            "1. [cyan]Start MongoDB[/cyan]\n"
            "   macOS: [white]brew services start mongodb-community[/white]\n"
            "   Linux: [white]sudo systemctl start mongodb[/white]\n\n"
            "2. [cyan]Install packages[/cyan]\n"
            "   [white]pip install pymongo maggma[/white]\n\n"
            "3. [cyan]Create database structure[/cyan]\n"
            "   [white]atomate2siesta-database create --create-indexes[/white]\n\n"
            "4. [cyan]Test connection[/cyan]\n"
            "   [white]atomate2siesta-database test[/white]\n\n"
            "5. [cyan]Choose configuration method[/cyan]\n"
            "   • Create [white]~/.jobflow.yaml[/white] (recommended), OR\n"
            "   • Pass [white]store=[/white] parameter in Python\n\n"
            "6. [cyan]Run calculations[/cyan]\n"
            "   Results automatically stored in MongoDB!",
            style="blue",
            title="Quick Start Guide",
        )
    )


@cli.command()
@click.option(
    "--check-only",
    is_flag=True,
    help="Only check if MongoDB is installed (don't install)",
)
@click.option(
    "--start-service", is_flag=True, help="Start MongoDB service after installation"
)
@click.option(
    "--stop-service", is_flag=True, help="Stop MongoDB service (existing installation)"
)
@click.option(
    "--local",
    is_flag=True,
    help="Install MongoDB in user mode (no root/sudo required)",
)
@click.option(
    "--install-dir",
    default=None,
    help="Custom installation directory for local install (default: ~/.local/mongodb)",
)
def setup(
    check_only: bool,
    start_service: bool,
    stop_service: bool,
    local: bool,
    install_dir: str | None,
) -> None:
    """Install and configure MongoDB on local machine.

    This command helps you install MongoDB Community Edition on your local
    computer for storing atomate2siesta calculation results.

    Supports:
        • macOS (via Homebrew or local user install)
        • Ubuntu/Debian (via apt or local user install)
        • RedHat/CentOS/Fedora (via yum/dnf or local user install)
        • Any Linux (local user install without root)

    Examples
    --------
        # Check if MongoDB is installed
        atomate2siesta-database setup --check-only

        # Install MongoDB system-wide (requires root/sudo)
        atomate2siesta-database setup

        # Install MongoDB in user mode (no root required)
        atomate2siesta-database setup --local

        # Install in custom directory
        atomate2siesta-database setup --local --install-dir ~/my-mongodb

        # Install system-wide and start service
        atomate2siesta-database setup --start-service

        # Stop MongoDB service (works for both local and system installs)
        atomate2siesta-database setup --stop-service
    """
    import platform
    import subprocess
    import sys
    from pathlib import Path

    console.print()
    header = Panel(
        Text("MongoDB Setup for atomate2siesta", style="bold cyan", justify="center"),
        style="cyan",
    )
    console.print(header)

    # Detect operating system
    os_type = platform.system().lower()

    console.print(
        f"\n[cyan]Detected OS:[/cyan] {platform.system()} {platform.release()}"
    )

    # Handle --stop-service flag
    if stop_service:
        console.print("\n[yellow]Stopping MongoDB service...[/yellow]")

        # Check for local installation
        local_mongo = Path.home() / ".local" / "mongodb"
        stop_script = local_mongo / "stop-mongodb.sh"

        if stop_script.exists():
            # Local installation - use stop script
            try:
                result = subprocess.run(
                    [str(stop_script)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0:
                    console.print(
                        "[green]✓ MongoDB stopped (local installation)[/green]"
                    )
                else:
                    console.print(f"[yellow]{result.stderr}[/yellow]")
                    # Try manual stop
                    bin_dir = local_mongo / "bin"
                    config_file = local_mongo / "etc" / "mongod.conf"
                    if bin_dir.exists() and config_file.exists():
                        result = subprocess.run(
                            [
                                str(bin_dir / "mongod"),
                                "--shutdown",
                                "--config",
                                str(config_file),
                            ],
                            capture_output=True,
                            text=True,
                            check=False,
                        )
                        if result.returncode == 0:
                            console.print("[green]✓ MongoDB stopped[/green]")
                        else:
                            console.print(
                                f"[red]✗ Failed to stop: {result.stderr}[/red]"
                            )
            except Exception as e:  # noqa: BLE001 friendly CLI error handler
                console.print(f"[red]✗ Error: {e}[/red]")

        elif os_type == "darwin":
            # macOS - use brew services
            try:
                result = subprocess.run(
                    ["brew", "services", "stop", "mongodb-community"],  # noqa: S607
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0:
                    console.print("[green]✓ MongoDB stopped (Homebrew)[/green]")
                else:
                    console.print(f"[red]✗ Failed: {result.stderr}[/red]")
            except Exception as e:  # noqa: BLE001 friendly CLI error handler
                console.print(f"[red]✗ Error: {e}[/red]")

        elif os_type == "linux":
            # Linux - use systemctl
            try:
                result = subprocess.run(
                    ["sudo", "systemctl", "stop", "mongod"],  # noqa: S607
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0:
                    console.print("[green]✓ MongoDB stopped (systemd)[/green]")
                else:
                    console.print(f"[yellow]{result.stderr}[/yellow]")
                    console.print(
                        "[dim]Note: You may need sudo privileges to stop "
                        "system MongoDB[/dim]"
                    )
            except Exception as e:  # noqa: BLE001 friendly CLI error handler
                console.print(f"[red]✗ Error: {e}[/red]")

        else:
            console.print(
                f"[yellow]Unsupported OS for --stop-service: {os_type}[/yellow]"
            )

        # Show status after stopping
        console.print("\n[yellow]Checking MongoDB status...[/yellow]")
        subprocess.run(
            ["atomate2siesta-database", "status"],  # noqa: S607
            check=False,
        )

        return

    # Check if MongoDB is already installed
    console.print("\n[yellow]Checking for existing MongoDB installation...[/yellow]")

    try:
        result = subprocess.run(
            ["mongod", "--version"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

        if result.returncode == 0:
            # Extract version from output
            version_line = result.stdout.split("\n")[0]
            console.print(
                f"[green]✓ MongoDB is already installed:[/green] {version_line}"
            )

            if check_only:
                console.print("\n[cyan]MongoDB is ready to use![/cyan]")

                # Show service status
                console.print("\n[yellow]To start MongoDB:[/yellow]")
                if os_type == "darwin":
                    console.print(
                        "  [cyan]brew services start mongodb-community[/cyan]"
                    )
                elif os_type == "linux":
                    console.print("  [cyan]sudo systemctl start mongod[/cyan]")

                return

            # Ask if user wants to proceed anyway
            if not click.confirm(
                "\nMongoDB is already installed. Continue with setup anyway?",
                default=False,
            ):
                console.print("[yellow]Setup cancelled[/yellow]")
                return

    except (FileNotFoundError, subprocess.TimeoutExpired):
        console.print("[yellow]MongoDB not found[/yellow]")

        if check_only:
            console.print("\n[red]✗ MongoDB is not installed[/red]")
            console.print("\nRun without [cyan]--check-only[/cyan] to install MongoDB")
            sys.exit(1)

    # Local installation (user mode, no root required)
    if local:
        import shutil
        import tarfile
        import urllib.request
        from pathlib import Path

        console.print(
            "\n[bold cyan]Installing MongoDB in User Mode "
            "(No Root Required)[/bold cyan]\n"
        )

        console.print(
            Panel(
                "[bold]Two installation methods:[/bold]\n\n"
                "[cyan]1. Automatic (recommended):[/cyan]\n"
                "   Downloads MongoDB binaries automatically\n"
                "   Uses MongoDB Community 7.0.5 from fastdl.mongodb.org\n\n"
                "[cyan]2. Manual:[/cyan]\n"
                "   Download from: "
                "[blue]https://www.mongodb.com/try/download/community[/blue]\n"
                "   Extract to custom location with --install-dir\n\n"
                "[dim]This installer uses automatic method[/dim]",
                style="blue",
                title="Installation Methods",
            )
        )

        # Set installation directory
        if install_dir:
            mongo_dir = Path(install_dir).expanduser()
        else:
            mongo_dir = Path.home() / ".local" / "mongodb"

        console.print(f"\n[cyan]Installation directory:[/cyan] {mongo_dir}")

        # Create directories
        mongo_dir.mkdir(parents=True, exist_ok=True)
        data_dir = mongo_dir / "data"
        log_dir = mongo_dir / "log"
        config_dir = mongo_dir / "etc"

        data_dir.mkdir(exist_ok=True)
        log_dir.mkdir(exist_ok=True)
        config_dir.mkdir(exist_ok=True)

        # Determine MongoDB download URL based on OS
        if os_type == "darwin":
            # macOS
            arch = platform.machine()
            if arch == "arm64":
                mongo_url = (
                    "https://fastdl.mongodb.org/osx/mongodb-macos-arm64-7.0.5.tgz"
                )
                mongo_file = "mongodb-macos-arm64-7.0.5.tgz"
            else:
                mongo_url = (
                    "https://fastdl.mongodb.org/osx/mongodb-macos-x86_64-7.0.5.tgz"
                )
                mongo_file = "mongodb-macos-x86_64-7.0.5.tgz"
        elif os_type == "linux":
            # Linux
            mongo_url = (
                "https://fastdl.mongodb.org/linux/"
                "mongodb-linux-x86_64-ubuntu2204-7.0.5.tgz"
            )
            mongo_file = "mongodb-linux-x86_64-ubuntu2204-7.0.5.tgz"
        else:
            console.print(
                f"[red]✗ Unsupported OS for local installation: {os_type}[/red]"
            )
            console.print("\nLocal installation supports macOS and Linux only.")
            sys.exit(1)

        # Download MongoDB
        console.print(f"\n[yellow]Downloading MongoDB from {mongo_url}...[/yellow]")
        console.print("[dim]This may take several minutes (~150 MB)...[/dim]\n")

        download_path = mongo_dir / mongo_file

        try:
            # Download with progress
            def reporthook(blocknum: int, blocksize: int, totalsize: int) -> None:
                readsofar = blocknum * blocksize
                if totalsize > 0:
                    percent = readsofar * 100 / totalsize
                    s = (
                        f"\r[cyan]Progress:[/cyan] {percent:.1f}% "
                        f"({readsofar // (1024 * 1024)} MB / "
                        f"{totalsize // (1024 * 1024)} MB)"
                    )
                    console.print(s, end="")
                    if readsofar >= totalsize:
                        console.print()

            urllib.request.urlretrieve(mongo_url, download_path, reporthook)  # noqa: S310
            console.print("[green]✓ Download complete[/green]")

        except Exception as e:  # noqa: BLE001 friendly CLI error handler
            console.print(f"\n[red]✗ Download failed: {e}[/red]")
            sys.exit(1)

        # Extract MongoDB
        console.print("\n[yellow]Extracting MongoDB binaries...[/yellow]")
        try:
            with tarfile.open(download_path, "r:gz") as tar:
                tar.extractall(mongo_dir)  # noqa: S202

            # Find extracted directory (e.g., mongodb-macos-arm64-7.0.5)
            extracted_dirs = [
                d
                for d in mongo_dir.iterdir()
                if d.is_dir() and d.name.startswith("mongodb-")
            ]

            if not extracted_dirs:
                console.print("[red]✗ Failed to find extracted MongoDB directory[/red]")
                sys.exit(1)

            extracted_dir = extracted_dirs[0]

            # Move binaries to mongo_dir/bin
            bin_dir = mongo_dir / "bin"
            if bin_dir.exists():
                shutil.rmtree(bin_dir)
            shutil.move(str(extracted_dir / "bin"), str(bin_dir))

            # Clean up
            download_path.unlink()
            shutil.rmtree(extracted_dir)

            console.print("[green]✓ MongoDB binaries extracted[/green]")

        except Exception as e:  # noqa: BLE001 friendly CLI error handler
            console.print(f"[red]✗ Extraction failed: {e}[/red]")
            sys.exit(1)

        # Create MongoDB configuration file
        config_file = config_dir / "mongod.conf"
        config_content = f"""# MongoDB configuration file (user mode)
systemLog:
  destination: file
  path: {log_dir}/mongod.log
  logAppend: true

storage:
  dbPath: {data_dir}

net:
  port: 27017
  bindIp: 127.0.0.1

processManagement:
  fork: false
"""

        with open(config_file, "w") as f:
            f.write(config_content)

        console.print(f"[green]✓ Configuration file created:[/green] {config_file}")

        # Create start/stop scripts
        start_script = mongo_dir / "start-mongodb.sh"
        stop_script = mongo_dir / "stop-mongodb.sh"
        status_script = mongo_dir / "status-mongodb.sh"

        start_content = f"""#!/bin/bash
# Start MongoDB in user mode (background)
echo "Starting MongoDB..."
{bin_dir}/mongod --config {config_file} --fork
echo "MongoDB started. Check status with: ps aux | grep mongod"
"""

        stop_content = f"""#!/bin/bash
# Stop MongoDB gracefully
echo "Stopping MongoDB..."
{bin_dir}/mongod --shutdown --config {config_file}
echo "MongoDB stopped."
"""

        status_content = f"""#!/bin/bash
# Check MongoDB status
echo "Checking MongoDB process..."
ps aux | grep mongod | grep -v grep

echo ""
echo "Checking port 27017..."
lsof -i :27017 2>/dev/null || echo "Port 27017 not in use"

echo ""
echo "Recent log entries:"
tail -n 5 {log_dir}/mongod.log 2>/dev/null || echo "No log file found"
"""

        with open(start_script, "w") as f:
            f.write(start_content)
        start_script.chmod(0o755)

        with open(stop_script, "w") as f:
            f.write(stop_content)
        stop_script.chmod(0o755)

        with open(status_script, "w") as f:
            f.write(status_content)
        status_script.chmod(0o755)

        console.print("[green]✓ Start/stop/status scripts created[/green]")

        # Show installation summary
        console.print(
            Panel(
                f"[bold green]✓ MongoDB installed successfully in user mode!"
                f"[/bold green]\n\n"
                f"[bold cyan]Installation Directory:[/bold cyan]\n"
                f"  MongoDB: [white]{mongo_dir}[/white]\n"
                f"  Binaries: [white]{bin_dir}[/white]\n"
                f"  Data: [white]{data_dir}[/white]\n"
                f"  Logs: [white]{log_dir}[/white]\n"
                f"  Config: [white]{config_file}[/white]\n\n"
                f"[bold cyan]Start MongoDB:[/bold cyan]\n"
                f"  [white]{start_script}[/white]  [dim](runs in background)[/dim]\n"
                f"  OR\n"
                f"  [white]{bin_dir}/mongod --config {config_file} --fork[/white]\n\n"
                f"[bold cyan]Check Status:[/bold cyan]\n"
                f"  [white]{status_script}[/white]  "
                f"[dim](shows process, port, and logs)[/dim]\n"
                f"  OR\n"
                f"  [white]ps aux | grep mongod[/white]\n"
                f"  OR (check port):\n"
                f"  [white]lsof -i :27017[/white]\n\n"
                f"[bold cyan]Stop MongoDB:[/bold cyan]\n"
                f"  [white]{stop_script}[/white]\n"
                f"  OR\n"
                f"  [white]{bin_dir}/mongod --shutdown --config {config_file}[/white]\n"
                f"  OR (if unresponsive):\n"
                f"  [white]pkill mongod[/white]\n\n"
                f"[bold cyan]View Logs:[/bold cyan]\n"
                f"  [white]tail -f {log_dir}/mongod.log[/white]\n\n"
                f"[bold cyan]Add to PATH (recommended):[/bold cyan]\n"
                f"  Add this to your ~/.bashrc or ~/.zshrc:\n"
                f'  [white]export PATH="{bin_dir}:$PATH"[/white]\n'
                f"  Then reload: [white]source ~/.bashrc[/white]\n\n"
                f"[bold cyan]Next Steps:[/bold cyan]\n"
                f"1. Start MongoDB:\n"
                f"   [white]{start_script} &[/white]\n"
                f"2. Verify it's running:\n"
                f"   [white]ps aux | grep mongod[/white]\n"
                f"3. Test connection:\n"
                f"   [white]atomate2siesta-database test[/white]\n"
                f"4. Create database structure:\n"
                f"   [white]atomate2siesta-database create --create-indexes[/white]\n"
                f"5. Generate configuration:\n"
                f"   [white]atomate2siesta-database config --generate[/white]",
                style="green",
                title="Local Installation Complete",
            )
        )

        return

    # Installation based on OS (system-wide with root)
    if os_type == "darwin":
        # macOS installation via Homebrew
        console.print(
            "\n[bold cyan]Installing MongoDB on macOS via Homebrew[/bold cyan]\n"
        )

        # Check if Homebrew is installed
        try:
            subprocess.run(
                ["brew", "--version"],  # noqa: S607
                capture_output=True,
                check=True,
            )
            console.print("[green]✓ Homebrew is installed[/green]")
        except (FileNotFoundError, subprocess.CalledProcessError):
            console.print(
                Panel(
                    "[bold red]✗ Homebrew is not installed[/bold red]\n\n"
                    "Homebrew is required to install MongoDB on macOS.\n\n"
                    "[bold cyan]To install Homebrew:[/bold cyan]\n"
                    '[white]/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"[/white]\n\n'
                    "Then run this command again.",
                    style="red",
                    title="Homebrew Required",
                )
            )
            sys.exit(1)

        # Add MongoDB tap
        console.print("\n[yellow]Adding MongoDB Homebrew tap...[/yellow]")
        try:
            subprocess.run(
                ["brew", "tap", "mongodb/brew"],  # noqa: S607
                capture_output=True,
                check=True,
            )
            console.print("[green]✓ MongoDB tap added[/green]")
        except subprocess.CalledProcessError as e:
            console.print(f"[yellow]⚠ Tap may already exist: {e}[/yellow]")

        # Install MongoDB
        console.print("\n[yellow]Installing MongoDB Community Edition...[/yellow]")
        console.print("[dim]This may take several minutes...[/dim]\n")

        try:
            result = subprocess.run(
                ["brew", "install", "mongodb-community"],  # noqa: S607
                capture_output=False,  # Show output to user
                text=True,
                check=False,
            )

            if result.returncode == 0:
                console.print("\n[green]✓ MongoDB installed successfully![/green]")
            else:
                console.print("\n[red]✗ Installation failed[/red]")
                sys.exit(1)

        except Exception as e:  # noqa: BLE001 friendly CLI error handler
            console.print(f"\n[red]✗ Installation error: {e}[/red]")
            sys.exit(1)

        # Start service if requested
        if start_service:
            console.print("\n[yellow]Starting MongoDB service...[/yellow]")
            try:
                subprocess.run(
                    ["brew", "services", "start", "mongodb-community"],  # noqa: S607
                    capture_output=True,
                    check=True,
                )
                console.print("[green]✓ MongoDB service started[/green]")
            except subprocess.CalledProcessError as e:
                console.print(f"[red]✗ Failed to start service: {e}[/red]")

        # Show next steps
        console.print(
            Panel(
                "[bold green]✓ MongoDB installation complete![/bold green]\n\n"
                "[bold cyan]Service Management:[/bold cyan]\n"
                "  Start:   [white]brew services start mongodb-community[/white]\n"
                "  Stop:    [white]brew services stop mongodb-community[/white]\n"
                "  Restart: [white]brew services restart mongodb-community[/white]\n"
                "  Status:  [white]brew services list[/white]\n\n"
                "[bold cyan]Configuration:[/bold cyan]\n"
                "  Config:  [white]/opt/homebrew/etc/mongod.conf[/white]\n"
                "  Data:    [white]/opt/homebrew/var/mongodb[/white]\n"
                "  Logs:    [white]/opt/homebrew/var/log/mongodb[/white]\n\n"
                "[bold cyan]Next Steps:[/bold cyan]\n"
                "1. Start MongoDB service (if not already running)\n"
                "2. Test connection:\n"
                "   [white]atomate2siesta-database test[/white]\n"
                "3. Create database structure:\n"
                "   [white]atomate2siesta-database create --create-indexes[/white]\n"
                "4. Generate configuration:\n"
                "   [white]atomate2siesta-database config --generate[/white]",
                style="green",
                title="Installation Complete",
            )
        )

    elif os_type == "linux":
        # Linux installation
        console.print("\n[bold cyan]Installing MongoDB on Linux[/bold cyan]\n")

        # Detect Linux distribution
        try:
            with open("/etc/os-release") as f:
                os_info = f.read().lower()

            if "ubuntu" in os_info or "debian" in os_info:
                distro = "ubuntu"
                console.print("[cyan]Detected distribution:[/cyan] Ubuntu/Debian")
            elif "rhel" in os_info or "centos" in os_info or "fedora" in os_info:
                distro = "rhel"
                console.print(
                    "[cyan]Detected distribution:[/cyan] RedHat/CentOS/Fedora"
                )
            else:
                console.print(
                    Panel(
                        "[bold yellow]⚠ Unsupported Linux distribution"
                        "[/bold yellow]\n\n"
                        "This tool supports Ubuntu/Debian and RedHat/CentOS/Fedora.\n\n"
                        "[bold cyan]Manual installation:[/bold cyan]\n"
                        "Visit: [blue]https://www.mongodb.com/docs/manual/installation/[/blue]",
                        style="yellow",
                    )
                )
                sys.exit(1)
        except FileNotFoundError:
            console.print("[red]Could not detect Linux distribution[/red]")
            sys.exit(1)

        # Ubuntu/Debian installation
        if distro == "ubuntu":
            console.print(
                Panel(
                    "[bold yellow]MongoDB Installation on Ubuntu/Debian"
                    "[/bold yellow]\n\n"
                    "This requires root/sudo access. The following commands "
                    "will be executed:\n\n"
                    "1. Import MongoDB GPG key\n"
                    "2. Add MongoDB repository\n"
                    "3. Update package database\n"
                    "4. Install MongoDB Community Edition\n\n"
                    "[bold cyan]Manual installation steps:[/bold cyan]\n"
                    "[white]# Import GPG key\n"
                    "wget -qO - https://www.mongodb.org/static/pgp/server-7.0.asc "
                    "| sudo apt-key add -\n\n"
                    "# Add repository\n"
                    'echo "deb [ arch=amd64,arm64 ] '
                    "https://repo.mongodb.org/apt/ubuntu "
                    '$(lsb_release -sc)/mongodb-org/7.0 multiverse" | sudo tee '
                    "/etc/apt/sources.list.d/mongodb-org-7.0.list\n\n"
                    "# Install\n"
                    "sudo apt-get update\n"
                    "sudo apt-get install -y mongodb-org[/white]",
                    style="blue",
                )
            )

            if not click.confirm(
                "\nProceed with automated installation?", default=True
            ):
                console.print("[yellow]Installation cancelled[/yellow]")
                console.print("\nFor manual installation, visit:")
                console.print(
                    "[blue]https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/[/blue]"
                )
                return

            console.print("\n[yellow]Installing MongoDB (requires sudo)...[/yellow]\n")

            commands = [
                (
                    "wget -qO - https://www.mongodb.org/static/pgp/server-7.0.asc "
                    "| sudo apt-key add -",
                    "Importing GPG key",
                ),
                (
                    'echo "deb [ arch=amd64,arm64 ] '
                    "https://repo.mongodb.org/apt/ubuntu "
                    '$(lsb_release -sc)/mongodb-org/7.0 multiverse" | sudo tee '
                    "/etc/apt/sources.list.d/mongodb-org-7.0.list",
                    "Adding repository",
                ),
                ("sudo apt-get update", "Updating package database"),
                ("sudo apt-get install -y mongodb-org", "Installing MongoDB"),
            ]

            for cmd, description in commands:
                console.print(f"[cyan]{description}...[/cyan]")
                result = subprocess.run(  # noqa: S602 shell pipeline for apt install
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode != 0:
                    console.print(f"[red]✗ Failed: {result.stderr}[/red]")
                    sys.exit(1)
                console.print(f"[green]✓ {description} complete[/green]")

            console.print("\n[green]✓ MongoDB installed successfully![/green]")

            # Start service if requested
            if start_service:
                console.print("\n[yellow]Starting MongoDB service...[/yellow]")
                result = subprocess.run(
                    ["sudo", "systemctl", "start", "mongod"],  # noqa: S607
                    capture_output=True,
                    check=False,
                )
                if result.returncode == 0:
                    console.print("[green]✓ MongoDB service started[/green]")

                    # Enable on boot
                    subprocess.run(
                        ["sudo", "systemctl", "enable", "mongod"],  # noqa: S607
                        capture_output=True,
                        check=False,
                    )
                    console.print("[green]✓ MongoDB enabled on boot[/green]")
                else:
                    console.print("[red]✗ Failed to start service[/red]")

            # Show next steps
            console.print(
                Panel(
                    "[bold green]✓ MongoDB installation complete![/bold green]\n\n"
                    "[bold cyan]Service Management:[/bold cyan]\n"
                    "  Start:   [white]sudo systemctl start mongod[/white]\n"
                    "  Stop:    [white]sudo systemctl stop mongod[/white]\n"
                    "  Restart: [white]sudo systemctl restart mongod[/white]\n"
                    "  Status:  [white]sudo systemctl status mongod[/white]\n"
                    "  Enable:  [white]sudo systemctl enable mongod[/white]\n\n"
                    "[bold cyan]Configuration:[/bold cyan]\n"
                    "  Config: [white]/etc/mongod.conf[/white]\n"
                    "  Data:   [white]/var/lib/mongodb[/white]\n"
                    "  Logs:   [white]/var/log/mongodb[/white]\n\n"
                    "[bold cyan]Next Steps:[/bold cyan]\n"
                    "1. Start MongoDB service:\n"
                    "   [white]sudo systemctl start mongod[/white]\n"
                    "2. Test connection:\n"
                    "   [white]atomate2siesta-database test[/white]\n"
                    "3. Create database structure:\n"
                    "   [white]atomate2siesta-database create --create-indexes[/white]",
                    style="green",
                    title="Installation Complete",
                )
            )

        # RedHat/CentOS/Fedora installation
        elif distro == "rhel":
            console.print(
                Panel(
                    "[bold yellow]MongoDB Installation on RedHat/CentOS/Fedora"
                    "[/bold yellow]\n\n"
                    "[bold cyan]Manual installation recommended:[/bold cyan]\n\n"
                    "Visit: [blue]https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-red-hat/[/blue]\n\n"
                    "[bold]Quick steps:[/bold]\n"
                    "1. Create /etc/yum.repos.d/mongodb-org-7.0.repo\n"
                    "2. sudo yum install -y mongodb-org\n"
                    "3. sudo systemctl start mongod",
                    style="yellow",
                )
            )

    else:
        # Windows or other OS
        console.print(
            Panel(
                f"[bold yellow]⚠ Unsupported operating system: {os_type}"
                f"[/bold yellow]\n\n"
                "This tool supports macOS and Linux.\n\n"
                "[bold cyan]For Windows:[/bold cyan]\n"
                "1. Download MongoDB Community Edition:\n"
                "   [blue]https://www.mongodb.com/try/download/community[/blue]\n"
                "2. Run the installer\n"
                "3. MongoDB will be available as a Windows service\n\n"
                "[bold cyan]For other systems:[/bold cyan]\n"
                "Visit: [blue]https://www.mongodb.com/docs/manual/installation/[/blue]",
                style="yellow",
                title="Manual Installation Required",
            )
        )


@cli.command()
@click.option(
    "--port", default=27017, type=int, help="MongoDB port to check (default: 27017)"
)
def status(port: int) -> None:
    """Check if MongoDB is running and show process information.

    This command checks:
        • MongoDB process status (ps aux)
        • Port availability (lsof)
        • Connection test

    Examples
    --------
        # Check default MongoDB instance
        atomate2siesta-database status

        # Check custom port
        atomate2siesta-database status --port 27018
    """
    import subprocess

    console.print(
        Panel(
            f"[bold cyan]MongoDB Status Check[/bold cyan]\nPort: {port}",
            style="blue",
        )
    )

    # Check for mongod process
    console.print("\n[yellow]Checking for MongoDB process...[/yellow]")
    try:
        result = subprocess.run(
            ["ps", "aux"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        mongod_lines = [
            line
            for line in result.stdout.split("\n")
            if "mongod" in line and "grep" not in line
        ]

        if mongod_lines:
            console.print("[green]✓ MongoDB process found:[/green]\n")
            # Create table for process info
            table = Table(box=box.ROUNDED)
            table.add_column("PID", style="cyan")
            table.add_column("CPU%", style="yellow")
            table.add_column("MEM%", style="magenta")
            table.add_column("Command", style="white", overflow="fold")

            for line in mongod_lines:
                parts = line.split()
                if len(parts) >= 11:
                    pid = parts[1]
                    cpu = parts[2]
                    mem = parts[3]
                    cmd = " ".join(parts[10:])[:80]
                    table.add_row(pid, cpu, mem, cmd)

            console.print(table)
        else:
            console.print("[red]✗ No MongoDB process found[/red]")
            console.print("\n[yellow]To start MongoDB:[/yellow]")
            console.print(
                "  Local install: [cyan]~/.local/mongodb/start-mongodb.sh[/cyan]"
            )
            console.print(
                "  System install (macOS): "
                "[cyan]brew services start mongodb-community[/cyan]"
            )
            console.print(
                "  System install (Linux): [cyan]sudo systemctl start mongod[/cyan]"
            )

    except Exception as e:  # noqa: BLE001 friendly CLI error handler
        console.print(f"[red]Error checking process: {e}[/red]")

    # Check port usage
    console.print(f"\n[yellow]Checking port {port}...[/yellow]")
    try:
        result = subprocess.run(
            ["lsof", "-i", f":{port}"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

        if result.returncode == 0 and result.stdout.strip():
            console.print(f"[green]✓ Port {port} is in use:[/green]\n")
            lines = result.stdout.strip().split("\n")
            if len(lines) > 1:
                console.print(f"[dim]{lines[0]}[/dim]")
                for line in lines[1:]:
                    console.print(f"  {line}")
        else:
            console.print(f"[red]✗ Port {port} is not in use[/red]")
            console.print(f"[yellow]MongoDB is not listening on port {port}[/yellow]")

    except FileNotFoundError:
        console.print(
            "[yellow]'lsof' command not found (install lsof package)[/yellow]"
        )
    except Exception as e:  # noqa: BLE001 friendly CLI error handler
        console.print(f"[red]Error checking port: {e}[/red]")

    # Test actual connection
    console.print(
        f"\n[yellow]Testing MongoDB connection on localhost:{port}...[/yellow]"
    )
    success, client, _db, _coll, error = test_mongodb_connection(
        "localhost", port, "admin", "test"
    )

    if success:
        console.print("[green]✓ MongoDB connection successful![/green]")

        # Get server info
        try:
            server_info = client.server_info()
            console.print(
                f"\n[cyan]MongoDB Version:[/cyan] "
                f"{server_info.get('version', 'Unknown')}"
            )

            # Get database list
            db_names = client.list_database_names()
            console.print(f"[cyan]Databases:[/cyan] {', '.join(db_names)}")

        except Exception as e:  # noqa: BLE001 friendly CLI error handler
            console.print(f"[yellow]Could not retrieve server info: {e}[/yellow]")
        finally:
            client.close()

        console.print(
            Panel(
                "[bold green]✓ MongoDB is running and accessible![/bold green]\n\n"
                "[bold cyan]Next steps:[/bold cyan]\n"
                "  • Test database: [white]atomate2siesta-database test[/white]\n"
                "  • Create database: "
                "[white]atomate2siesta-database create --create-indexes[/white]\n"
                "  • View data: [white]atomate2siesta-database list[/white]",
                style="green",
                title="MongoDB Active",
            )
        )
    else:
        console.print(f"[red]✗ MongoDB connection failed:[/red] {error}")

        console.print(
            Panel(
                "[bold yellow]MongoDB may not be running or accessible"
                "[/bold yellow]\n\n"
                "[bold cyan]Troubleshooting:[/bold cyan]\n"
                "1. Check if MongoDB is running:\n"
                "   [white]ps aux | grep mongod[/white]\n\n"
                "2. Start MongoDB:\n"
                "   Local: [white]~/.local/mongodb/start-mongodb.sh[/white]\n"
                "   macOS: [white]brew services start mongodb-community[/white]\n"
                "   Linux: [white]sudo systemctl start mongod[/white]\n\n"
                "3. Check logs:\n"
                "   Local: [white]tail -f ~/.local/mongodb/log/mongod.log[/white]\n"
                "   macOS: "
                "[white]tail -f /opt/homebrew/var/log/mongodb/mongod.log[/white]\n"
                "   Linux: [white]tail -f /var/log/mongodb/mongod.log[/white]\n\n"
                "4. Verify port is correct (default: 27017)",
                style="yellow",
                title="Connection Failed",
            )
        )


@cli.command()
def info() -> None:
    """Show information about database CLI commands and usage.

    This command displays comprehensive information about all available
    database commands, examples, and common workflows.
    """
    console.print()

    # Header
    header = Panel(
        Text("atomate2siesta Database CLI", style="bold cyan", justify="center"),
        style="cyan",
    )
    console.print(header)

    console.print("\n[bold]Overview:[/bold]\n")
    console.print(
        "This tool helps you manage MongoDB databases for atomate2siesta by:\n"
        "  • Testing database connections\n"
        "  • Creating database structure and indexes\n"
        "  • Querying calculation results\n"
        "  • Monitoring database statistics\n"
        "  • Managing configuration files"
    )

    console.print("\n[bold]Commands:[/bold]\n")

    commands_table = Table(box=None)
    commands_table.add_column("Command", style="cyan", no_wrap=True)
    commands_table.add_column("Description")

    commands_table.add_row("setup", "Install MongoDB (system-wide or local)")
    commands_table.add_row("status", "Check if MongoDB is running")
    commands_table.add_row("test", "Test MongoDB connection and show database status")
    commands_table.add_row("create", "Create database with indexes")
    commands_table.add_row("list", "List recent calculation results")
    commands_table.add_row("query", "Query by chemical formula")
    commands_table.add_row("stats", "Show comprehensive statistics")
    commands_table.add_row("clear", "Delete documents or collections")
    commands_table.add_row("config", "Show/generate configuration files")
    commands_table.add_row("info", "Show this information")

    console.print(commands_table)

    console.print("\n[bold]Quick Start Examples:[/bold]\n")

    examples_panel = Panel(
        "# Install MongoDB system-wide (requires root)\n"
        "[cyan]atomate2siesta-database setup --start-service[/cyan]\n\n"
        "# Install MongoDB in user mode (no root required)\n"
        "[cyan]atomate2siesta-database setup --local[/cyan]\n\n"
        "# Check if MongoDB is installed\n"
        "[cyan]atomate2siesta-database setup --check-only[/cyan]\n\n"
        "# Check if MongoDB is running\n"
        "[cyan]atomate2siesta-database status[/cyan]\n\n"
        "# Test MongoDB connection\n"
        "[cyan]atomate2siesta-database test[/cyan]\n\n"
        "# Create database with indexes\n"
        "[cyan]atomate2siesta-database create --create-indexes[/cyan]\n\n"
        "# Generate ~/.jobflow.yaml configuration\n"
        "[cyan]atomate2siesta-database config --generate[/cyan]\n\n"
        "# List recent calculations\n"
        "[cyan]atomate2siesta-database list --limit 20[/cyan]\n\n"
        "# Query specific material\n"
        "[cyan]atomate2siesta-database query Si[/cyan]\n\n"
        "# Show database statistics\n"
        "[cyan]atomate2siesta-database stats[/cyan]\n\n"
        "# Remote database\n"
        "[cyan]atomate2siesta-database test --host server.com --port 27017[/cyan]",
        style="green",
    )
    console.print(examples_panel)

    console.print("\n[bold]Configuration Methods:[/bold]\n")

    config_table = Table(box=None)
    config_table.add_column("Method", style="cyan")
    config_table.add_column("Description")

    config_table.add_row(
        "~/.jobflow.yaml", "Automatic config (recommended for production)"
    )
    config_table.add_row("store parameter", "Manual config in Python code")
    config_table.add_row("Environment variables", "MONGO_HOST, MONGO_PORT, etc.")

    console.print(config_table)

    console.print("\n[bold]Common Workflows:[/bold]\n")

    workflow1 = Panel(
        "[bold]1. Initial Setup (Fresh Install)[/bold]\n\n"
        "# Install MongoDB in user mode (no root required)\n"
        "[cyan]atomate2siesta-database setup --local[/cyan]\n\n"
        "# Start MongoDB\n"
        "[cyan]~/.local/mongodb/start-mongodb.sh[/cyan]\n\n"
        "# Check status\n"
        "[cyan]atomate2siesta-database status[/cyan]\n\n"
        "# Test connection\n"
        "[cyan]atomate2siesta-database test[/cyan]\n\n"
        "# Create structure with indexes\n"
        "[cyan]atomate2siesta-database create --create-indexes[/cyan]\n\n"
        "# Generate configuration\n"
        "[cyan]atomate2siesta-database config --generate[/cyan]",
        style="blue",
        title="Workflow 1: First Time Setup",
    )
    console.print(workflow1)

    workflow2 = Panel(
        "[bold]2. Running Calculations[/bold]\n\n"
        "# Verify database is ready\n"
        "[cyan]atomate2siesta-database test[/cyan]\n\n"
        "# Run your calculation (Python)\n"
        "[dim]from jobflow import run_locally\n"
        "from atomate2.siesta.jobs.core import RelaxMaker\n"
        "job = RelaxMaker.fixed_cell_relaxation().make(structure)\n"
        "results = run_locally(job, create_folders=True)[/dim]\n\n"
        "# Results are automatically stored in MongoDB!",
        style="blue",
        title="Workflow 2: Running Calculations",
    )
    console.print(workflow2)

    workflow3 = Panel(
        "[bold]3. Monitoring & Analysis[/bold]\n\n"
        "# View recent calculations\n"
        "[cyan]atomate2siesta-database list[/cyan]\n\n"
        "# Query specific material\n"
        "[cyan]atomate2siesta-database query Si[/cyan]\n\n"
        "# Show statistics\n"
        "[cyan]atomate2siesta-database stats[/cyan]\n\n"
        "# Clear test data\n"
        "[cyan]atomate2siesta-database clear[/cyan]",
        style="blue",
        title="Workflow 3: Monitoring Results",
    )
    console.print(workflow3)

    console.print("\n[bold]Remote Database Setup:[/bold]\n")

    remote_panel = Panel(
        "# Test remote connection\n"
        "[cyan]atomate2siesta-database test --host cluster.edu "
        "--port 27017 --database mydb[/cyan]\n\n"
        "# Create on remote server\n"
        "[cyan]atomate2siesta-database create --host cluster.edu "
        "--create-indexes[/cyan]\n\n"
        "# Generate config for remote database\n"
        "[cyan]atomate2siesta-database config --generate --host cluster.edu "
        "--port 27017[/cyan]\n\n"
        "# Query remote database\n"
        "[cyan]atomate2siesta-database list --host cluster.edu --database mydb[/cyan]",
        style="green",
    )
    console.print(remote_panel)

    console.print("\n[bold]Integration with Other Tools:[/bold]\n")

    integration_table = Table(box=None)
    integration_table.add_column("Tool", style="cyan")
    integration_table.add_column("Purpose")

    integration_table.add_row(
        "atomate2siesta-cluster", "Set up HPC clusters for calculations"
    )
    integration_table.add_row(
        "atomate2siesta-jobflow-remote", "Configure remote job submission"
    )
    integration_table.add_row("atomate2siesta-inputs", "Generate SIESTA input files")

    console.print(integration_table)

    console.print("\n[bold]Useful Tips:[/bold]\n")

    tips_panel = Panel(
        "• Use [cyan]--help[/cyan] with any command for detailed options\n"
        "• [cyan]~/.jobflow.yaml[/cyan] is loaded automatically by jobflow\n"
        "• Indexes improve query performance significantly\n"
        "• Use [cyan]stats[/cyan] to monitor database growth\n"
        "• [cyan]config[/cyan] shows both automatic and manual methods\n"
        "• All commands support remote databases with [cyan]--host[/cyan]",
        style="yellow",
    )
    console.print(tips_panel)

    console.print("\n[bold]Documentation:[/bold]\n")
    console.print("  • Full Guide: [blue]docs/CLI_DATABASE.md[/blue]")
    console.print("  • Tutorial: [blue]tutorials/13-database-storage/[/blue]")
    console.print("  • MongoDB: [blue]https://www.mongodb.com/docs/[/blue]")
    console.print("  • Maggma: [blue]https://materialsproject.github.io/maggma/[/blue]")

    console.print()


if __name__ == "__main__":
    cli()
