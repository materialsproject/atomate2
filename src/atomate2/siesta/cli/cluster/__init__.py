"""CLI for remote cluster setup with conda environment for jobflow-remote.

This module provides command-line tools to help users set up remote HPC clusters
for atomate2siesta calculations by creating conda environments and installing
necessary packages via SSH.

The CLI has been refactored into separate modules for better organization:
- ssh_utils.py: SSH connection and tunnel management
- proxy_utils.py: Squid and remote proxy configuration
- commands.py: CLI command implementations
"""

from __future__ import annotations

# Import the CLI group from commands module
from .commands import cli

# Export the CLI group
__all__ = ["cli"]
