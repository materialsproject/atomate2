"""Utility functions for jobflow-remote CLI.

This module provides helper functions for managing jobflow-remote
configurations, YAML files, and other common operations.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import yaml
from rich.console import Console

if TYPE_CHECKING:
    from pathlib import Path
    from typing import TextIO

console = Console()


def _backup_config(config_path: Path) -> Path:
    """Create a backup of the configuration file.

    Args:
        config_path: Path to the configuration file to backup

    Returns
    -------
        Path to the backup file
    """
    # local-time timestamp intended for the backup filename suffix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005
    backup_path = (
        config_path.parent
        / f"{config_path.stem}.backup_{timestamp}{config_path.suffix}"
    )
    backup_path.write_text(config_path.read_text())
    return backup_path


def _update_nested_dict(base_dict: dict, update_dict: dict) -> dict:
    """Recursively update a nested dictionary.

    Args:
        base_dict: The base dictionary to update
        update_dict: The dictionary with updates

    Returns
    -------
        Updated dictionary
    """
    for key, value in update_dict.items():
        if (
            isinstance(value, dict)
            and key in base_dict
            and isinstance(base_dict[key], dict)
        ):
            base_dict[key] = _update_nested_dict(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def _load_yaml_config(config_path: Path) -> dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file

    Returns
    -------
        Dictionary with configuration data
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def _save_yaml_config(
    config_path: Path, config_data: dict, add_comments: bool = False
) -> None:
    """Save dictionary to YAML configuration file.

    Args:
        config_path: Path to save the YAML file
        config_data: Dictionary with configuration data
        add_comments: If True, add descriptive comments to the YAML file
    """
    with open(config_path, "w") as f:
        if add_comments:
            # Write YAML with inline comments
            _write_yaml_with_comments(f, config_data)
        else:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)


def _write_yaml_with_comments(file_handle: TextIO, config_data: dict) -> None:
    """Write YAML configuration with inline comments.

    Args:
        file_handle: File handle to write to
        config_data: Dictionary with configuration data
    """
    # Comments for different configuration sections
    # Use "*" as wildcard to match any worker name
    comments = {
        "name": "Project name for jobflow-remote",
        "workers": {
            "_section": "Worker configurations for job execution",
            "*": {  # Matches any worker name
                "type": "Worker type: 'local' for testing, 'remote' for HPC",
                "scheduler_type": "Scheduler: 'shell', 'slurm', 'pbs', 'sge', 'lsf'",
                "work_dir": "Directory where jobs will run on the worker",
                "pre_run": (
                    "Commands to run before job execution (e.g., activate environment)"
                ),
                "timeout_execute": "Timeout in seconds for job execution",
                "host": "Remote host address (for remote workers)",
                "user": "Username for remote connection",
            },
        },
        "queue": {
            "_section": "Queue store configuration (MongoDB)",
            "store": {
                "_section": "MongoDB connection settings for job queue",
                "type": "Store type (typically MongoStore)",
                "host": "MongoDB server hostname",
                "database": "Database name for job queue",
                "username": "MongoDB username (optional)",
                "password": "MongoDB password (optional)",
                "collection_name": "Collection name for jobs",
                "port": "MongoDB port (default: 27017)",
            },
        },
        "exec_config": "Execution configuration (typically empty)",
        "jobstore": {
            "_section": "Job output storage configuration",
            "docs_store": {
                "_section": "MongoDB store for job output documents",
                "type": "Store type (typically MongoStore)",
                "database": "Database name for job outputs",
                "host": "MongoDB server hostname",
                "port": "MongoDB port",
                "username": "MongoDB username (optional)",
                "password": "MongoDB password (optional)",
                "collection_name": "Collection name for outputs",
            },
            "additional_stores": {
                "_section": "Additional storage for large data (e.g., GridFS)",
                "data": {
                    "_section": "GridFS store for large binary data",
                    "type": "Store type (typically GridFSStore)",
                    "database": "Database name",
                    "host": "MongoDB server hostname",
                    "port": "MongoDB port",
                    "username": "MongoDB username (optional)",
                    "password": "MongoDB password (optional)",
                    "collection_name": "Collection name for binary blobs",
                },
            },
        },
    }

    def write_dict(
        data: dict, indent: int = 0, comment_path: list | None = None
    ) -> None:
        """Recursively write dictionary with comments."""
        if comment_path is None:
            comment_path = []

        for key, value in data.items():
            current_path = [*comment_path, key]

            # Get comment for this key
            comment = _get_comment(comments, current_path)

            # Write section comment if available
            if isinstance(value, dict) and "_section" in _get_comment_dict(
                comments, current_path
            ):
                section_comment = _get_comment_dict(comments, current_path).get(
                    "_section"
                )
                if section_comment:
                    file_handle.write(f"\n{'  ' * indent}# {section_comment}\n")

            # Write key-value pair
            if isinstance(value, dict):
                if comment:
                    file_handle.write(f"{'  ' * indent}{key}:  # {comment}\n")
                else:
                    file_handle.write(f"{'  ' * indent}{key}:\n")
                write_dict(value, indent + 1, current_path)
            elif isinstance(value, list):
                if comment:
                    file_handle.write(f"{'  ' * indent}{key}:  # {comment}\n")
                else:
                    file_handle.write(f"{'  ' * indent}{key}:\n")
                file_handle.writelines(
                    f"{'  ' * (indent + 1)}- {item}\n" for item in value
                )
            elif comment:
                file_handle.write(f"{'  ' * indent}{key}: {value}  # {comment}\n")
            else:
                file_handle.write(f"{'  ' * indent}{key}: {value}\n")

    # Write header comment
    file_handle.write("# Jobflow Remote Configuration File\n")
    file_handle.write("# Generated by atomate2siesta-jobflow-remote\n")
    file_handle.write("# Documentation: https://matgenix.github.io/jobflow-remote/\n\n")

    write_dict(config_data)


def _get_comment_dict(comments: dict, path: list) -> dict:
    """Navigate to a specific path in the comments dictionary.

    Supports wildcard "*" matching for dynamic keys (e.g., worker names).
    """
    current = comments
    for key in path:
        if isinstance(current, dict):
            # Try exact match first
            if key in current:
                current = current[key]
            # Try wildcard match
            elif "*" in current:
                current = current["*"]
            else:
                return {}
        else:
            return {}
    return current if isinstance(current, dict) else {}


def _get_comment(comments: dict, path: list) -> str:
    """Get comment for a specific configuration path.

    Supports wildcard "*" matching for dynamic keys (e.g., worker names).
    """
    current = comments
    for key in path:
        if isinstance(current, dict):
            # Try exact match first
            if key in current:
                if isinstance(current[key], dict):
                    current = current[key]
                else:
                    return current[key]
            # Try wildcard match
            elif "*" in current:
                if isinstance(current["*"], dict):
                    current = current["*"]
                else:
                    return current["*"]
            else:
                return ""
        else:
            return ""
    return ""
