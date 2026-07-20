"""Display available workflow types."""

from rich import box
from rich.console import Console
from rich.table import Table

console = Console()


def show_workflows() -> None:
    """
    Display all workflow types.

    Automatically discovers FlowMakers and Makers from the codebase.
    """
    from atomate2.siesta.cli.info.workflow_details import discover_flowmakers

    console.print("\n[bold cyan]Available Workflows[/bold cyan]\n")

    # Get all FlowMakers
    flowmakers = discover_flowmakers()

    # Categorize FlowMakers by type
    categories = {
        "Single Jobs": [
            ("RelaxMaker", "Structure relaxation (fixed/variable cell)", "5-15 min"),
            ("StaticMaker", "Single-point energy calculation", "2-5 min"),
            ("BandStructureMaker", "Electronic band structure", "10-20 min"),
            ("DOSMaker", "Density of states (total)", "10-20 min"),
            ("PDOSMaker", "Projected density of states", "10-20 min"),
        ],
        "Vibrational Properties": [
            ("SiestaPhononFlowMaker", None, "30-60 min"),
            ("SiestaGruneisenFlowMaker", None, "2-4 hours"),
            ("SiestaQhaFlowMaker", None, "4-8 hours"),
        ],
        "Mechanical Properties": [
            ("SiestaEosFlowMaker", None, "15-30 min"),
            ("ElasticFlowMaker", None, "30-60 min"),
        ],
        "Surface & Adsorption": [
            ("SurfaceEnergyFlowMaker", None, "30-90 min"),
            ("MultiSurfaceEnergyFlowMaker", None, "1-3 hours"),
            ("AdsorptionScanFlowMaker", None, "1-3 hours"),
            ("AdsorptionOptimizationFlowMaker", None, "30-60 min"),
        ],
        "Transition States": [
            ("NebDirectFlowMaker", None, "2-6 hours"),
            ("AseNebFlowMaker", None, "2-6 hours"),
            ("NebVacancyExchangeFlowMaker", None, "3-8 hours"),
        ],
        "Convergence Testing": [
            ("MeshCutoffConvergenceFlowMaker", None, "1-2 hours"),
            ("KpointsConvergenceFlowMaker", None, "1-2 hours"),
            ("PhononConvergenceFlowMaker", None, "4-8 hours"),
        ],
        "Advanced": [
            ("DifferentBasisFlowMaker", None, "20-40 min"),
            ("DifferentBasisSCFFlowMaker", None, "10-20 min"),
            ("DifferentBasisRelaxFlowMaker", None, "20-40 min"),
            ("EOSFullBasisConvergenceFlowMaker", None, "2-4 hours"),
        ],
    }

    for category, makers in categories.items():
        table = Table(title=f"{category} ({len(makers)})", box=box.ROUNDED)
        table.add_column("Maker/FlowMaker", style="cyan", width=35)
        table.add_column("Description", style="white", width=50)
        table.add_column("Runtime", style="yellow", width=12)

        for maker_name, description, runtime in makers:
            # Get description from FlowMaker if not provided
            desc = description
            if desc is None and maker_name in flowmakers:
                desc = flowmakers[maker_name].get("description", "No description")
                if len(desc) > 50:
                    desc = desc[:47] + "..."

            table.add_row(maker_name, desc or "No description", runtime)

        console.print(table)
        console.print()

    console.print(
        "[dim]Use 'atomate2siesta-info workflows <name>' for detailed information[/dim]"
    )
    console.print(
        "[dim]Use 'atomate2siesta-info workflows --list-all' "
        "to see all FlowMakers[/dim]\n"
    )
