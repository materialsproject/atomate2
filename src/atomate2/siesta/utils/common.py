"""Tools for logging."""

from __future__ import annotations

import logging

from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

logger = logging.getLogger(__name__)
console = Console()


def print_in_box_rich(settings_dict: dict) -> None:
    """
    Print settings in a visually appealing box using the 'rich' library.

    Parameters
    ----------
    settings_dict : dict
        A dictionary containing the settings to display.
    """
    # console = Console()

    # Format the settings into a Rich Text object
    settings_text = Text()
    num_items = len(settings_dict)
    for i, (key, value) in enumerate(settings_dict.items()):
        settings_text.append(f"{key}: ", style="bold magenta")
        settings_text.append(str(value), style="white")
        # Only add a newline if it's not the last item
        if i < num_items - 1:
            settings_text.append("\n")

    # Create a panel to display the settings
    settings_panel = Panel(
        settings_text,
        title="[bold]Configuration Settings[/bold]",
        border_style="cyan",
        expand=False,
        padding=(1, 2),
    )

    console.print(Align.center(settings_panel))


def print_in_box(text_lines: list[str]) -> None:
    """
    Print text lines inside an ASCII box border.

    Parameters
    ----------
    text_lines : list of str
        Lines of text to print within the box
    """
    logger.info("print_in_box()")
    # Find the length of the longest line for proper box sizing
    max_length = max(len(line) for line in text_lines)

    # Print top border
    print("+" + "-" * (max_length + 2) + "+")  # noqa: T201

    # Print each line with side borders
    for line in text_lines:
        print(f"| {line.ljust(max_length)} |")  # noqa: T201

    # Print bottom border
    print("+" + "-" * (max_length + 2) + "+")  # noqa: T201


def print_docstring_in_box(docstring: str, title: str = "Class Description") -> None:
    """
    Print a docstring in a visually appealing box using the 'rich' library.

    This function takes a docstring and a title, then formats them into a
    centered, styled panel for clear and attractive display in the console.

    The display can be disabled by setting SIESTA_SHOW_DOCSTRINGS=False in
    ~/.atomate2siesta.yaml or by setting the atomate2_SIESTA_SHOW_DOCSTRINGS
    environment variable to False.

    Parameters
    ----------
    docstring : str
        The docstring or any text string to display inside the panel.
    title : str, optional
        The title to be displayed on the panel's border.
        Defaults to "Class Description".

    # --- Example Usage ---
    if __name__ == "__main__":
        # Get the docstring from the class
        doc_to_print = DifferentBasisSCFAdvance.__doc__

        # Get the class name to use as a title
        class_name = DifferentBasisSCFAdvance.__name__

        # Call the function to print it
        print_docstring_in_box(doc_to_print, title=class_name)

    """
    # Check if docstring display is enabled
    from atomate2.siesta import SETTINGS

    if not SETTINGS.SIESTA_SHOW_DOCSTRINGS:
        return

    # Initialize the console
    console = Console()

    # Create a panel to display the docstring
    # The docstring is passed directly as the renderable content.
    # The strip() method is used to remove any leading/trailing whitespace.
    doc_panel = Panel(
        docstring.strip(),
        title=f"[bold cyan]{title}[/bold cyan]",
        border_style="magenta",
        expand=False,  # The panel will not expand to the full width of the terminal
        padding=(1, 2),  # (vertical, horizontal) padding
    )

    # Print the panel, centered in the terminal
    console.print(Align.center(doc_panel))
