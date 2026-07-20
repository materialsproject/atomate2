#!/usr/bin/env python
import sys
import os
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
import pyfiglet


def print_fancy_logo():
    """
    Clears the terminal and prints a merged Atomate2-Siesta logo using 'rich'
    and 'pyfiglet' libraries for styling and layout.
    """
    # Initialize a Rich Console
    console = Console()

    # Clear the terminal screen before printing
    os.system("cls" if os.name == "nt" else "clear")

    # --- Create the merged logo using pyfiglet ---
    font_style = "slant"

    # Generate ASCII art for each part
    atomate_art = pyfiglet.figlet_format("Atomate", font=font_style)
    two_art = pyfiglet.figlet_format("2", font=font_style)
    siesta_art = pyfiglet.figlet_format("Siesta", font=font_style)

    # Split the ASCII art into lines
    atomate_lines = atomate_art.splitlines()
    two_lines = two_art.splitlines()
    siesta_lines = siesta_art.splitlines()

    # Find the maximum number of lines to align them vertically
    max_lines = max(len(atomate_lines), len(two_lines), len(siesta_lines))

    # Pad shorter lists with empty strings to match the max length
    # This ensures all parts align correctly if they have different heights
    atomate_lines += [" " * len(atomate_lines[0] if atomate_lines else "")] * (
        max_lines - len(atomate_lines)
    )
    two_lines += [" " * len(two_lines[0] if two_lines else "")] * (
        max_lines - len(two_lines)
    )
    siesta_lines += [" " * len(siesta_lines[0] if siesta_lines else "")] * (
        max_lines - len(siesta_lines)
    )

    # --- Combine the parts into a single Rich Text object with styles ---
    merged_logo_text = Text()
    for i in range(max_lines):
        # Append each part of the line with its specific color
        merged_logo_text.append(atomate_lines[i], style="bold blue")
        merged_logo_text.append(two_lines[i], style="bold cyan")
        # Add a visual separator between the logos
        # merged_logo_text.append("   +   ")
        merged_logo_text.append(siesta_lines[i], style="bold yellow")

        # Add a newline character after each line except the last one
        if i < max_lines - 1:
            merged_logo_text.append("\n")

    # Create a single panel for the merged logo
    merged_panel = Panel(
        Align.center(merged_logo_text, vertical="middle"),
        title="[bold]Wellcome to Atomate2 + Siesta: A Powerful Combination[/bold]",
        border_style="green",
        expand=False,
        padding=(2, 4),
    )

    # --- Print the merged logo ---
    console.print("\n")
    console.print(Align.center(merged_panel))
    console.print("\n" * 2)


if __name__ == "__main__":
    # Before running, ensure 'rich' and 'pyfiglet' are installed:
    # pip install rich pyfiglet
    try:
        print_fancy_logo()
    except KeyboardInterrupt:
        # Handle user interruption (Ctrl+C) gracefully
        print("\n\nLogo display stopped by user.")
        sys.exit(0)
