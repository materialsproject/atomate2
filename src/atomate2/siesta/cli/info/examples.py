"""Display quick start examples."""

from rich.console import Console

console = Console()


def show_examples():
    """Show quick start examples."""
    console.print("\n[bold cyan]Quick Start Examples[/bold cyan]\n")

    # Example 1: Workflow generator
    console.print("[bold yellow]1. Generate a workflow script (easiest!)[/bold yellow]")
    console.print("[dim]# From structure to running calculation in 30 seconds[/dim]")
    console.print("[green]$ atomate2siesta-maker relax Si.cif[/green]")
    console.print("[green]$ python relax_Si.py[/green]")
    console.print()

    # Example 2: Interactive mode
    console.print(
        "[bold yellow]2. Interactive workflow generation (zero memorization!)[/bold yellow]"
    )
    console.print("[dim]# Guided step-by-step prompts for any workflow[/dim]")
    console.print("[green]$ atomate2siesta-maker --interactive[/green]")
    console.print(
        "[dim]# Select workflow → structure → preset → execution mode → done![/dim]"
    )
    console.print()

    # Example 3: Recipe book
    console.print("[bold yellow]3. Use Recipe Book (one-liner)[/bold yellow]")
    console.print("[dim]# 50+ lines of code reduced to 1 line[/dim]")
    console.print("[cyan]from atomate2.siesta.recipes import RecipeBook[/cyan]")
    console.print("[cyan]from pymatgen.core import Structure[/cyan]")
    console.print("[cyan]from jobflow import run_locally[/cyan]")
    console.print()
    console.print("[cyan]structure = Structure.from_file('Si.cif')[/cyan]")
    console.print("[cyan]flow = RecipeBook.band_structure_workflow(structure)[/cyan]")
    console.print("[cyan]results = run_locally(flow, create_folders=True)[/cyan]")
    console.print()

    # Example 4: Python API
    console.print("[bold yellow]4. Python API with tier preset[/bold yellow]")
    console.print("[dim]# Full control with best-practice defaults[/dim]")
    console.print("[cyan]from atomate2.siesta.jobs.core import RelaxMaker[/cyan]")
    console.print(
        "[cyan]from atomate2.siesta.sets.tiers import apply_tier_preset[/cyan]"
    )
    console.print("[cyan]from jobflow import run_locally[/cyan]")
    console.print()
    console.print("[cyan]maker = RelaxMaker.fixed_cell_relaxation()[/cyan]")
    console.print("[cyan]maker = apply_tier_preset(maker, 'relax_standard')[/cyan]")
    console.print("[cyan]job = maker.make(structure)[/cyan]")
    console.print("[cyan]results = run_locally(job, create_folders=True)[/cyan]")
    console.print()

    # Example 5: Custom parameters
    console.print("[bold yellow]5. Custom parameters[/bold yellow]")
    console.print("[dim]# Override any default[/dim]")
    console.print("[cyan]maker = RelaxMaker.fixed_cell_relaxation([/cyan]")
    console.print("[cyan]    user_params={[/cyan]")
    console.print("[cyan]        'PAO.BasisSize': 'TZP',[/cyan]")
    console.print("[cyan]        'kpts': [8, 8, 8],[/cyan]")
    console.print("[cyan]        'Mesh.Cutoff': '300 Ry',[/cyan]")
    console.print("[cyan]        'xc_authors': 'PBEsol',  # Default is PBE[/cyan]")
    console.print("[cyan]    }[/cyan]")
    console.print("[cyan])[/cyan]")
    console.print()

    # Example 6: Browse tutorials
    console.print("[bold yellow]6. Browse interactive tutorials[/bold yellow]")
    console.print("[dim]# Tutorials with search and copy functionality[/dim]")
    console.print("[green]$ atomate2siesta-tutorials list[/green]")
    console.print("[green]$ atomate2siesta-tutorials search phonon[/green]")
    console.print(
        "[green]$ atomate2siesta-tutorials show 01-basics/01-relaxation[/green]"
    )
    console.print()

    console.print("[dim]More examples: tutorials/ directory[/dim]\n")
