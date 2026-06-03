"""SSH connection and tunnel management utilities for cluster setup.

This module provides functions for SSH command execution and tunnel management,
including dynamic port forwarding (SOCKS proxy) and reverse port forwarding.
"""

from __future__ import annotations

import subprocess
from typing import Optional

from rich.console import Console

console = Console()


def show_verbose_output(stdout: str, stderr: str, verbose: bool) -> None:
    """Show command output if verbose mode is enabled.

    Parameters
    ----------
    stdout : str
        Standard output from command
    stderr : str
        Standard error from command
    verbose : bool
        If True, display output
    """
    if not verbose:
        return

    if stdout and stdout.strip():
        console.print("[dim]  stdout:[/dim]")
        for line in stdout.strip().split("\n"):
            console.print(f"[dim]    {line}[/dim]")

    if stderr and stderr.strip():
        console.print("[dim]  stderr:[/dim]")
        for line in stderr.strip().split("\n"):
            console.print(f"[dim]    {line}[/dim]")


def run_ssh_command(
    host: str,
    user: Optional[str],
    command: str,
    password: Optional[str] = None,
    identity_file: Optional[str] = None,
    use_ssh_config: bool = False,
    timeout: int = 300,
) -> tuple[int, str, str]:
    """Run a command on remote host via SSH.

    Parameters
    ----------
    host : str
        Remote host address or SSH config alias
    user : str, optional
        Username for SSH connection (ignored if use_ssh_config is True)
    command : str
        Command to execute on remote host
    password : str, optional
        Password for SSH authentication (uses sshpass)
    identity_file : str, optional
        Path to SSH private key file
    use_ssh_config : bool, optional
        If True, use SSH config alias without user@ prefix
    timeout : int, optional
        Command timeout in seconds (default: 300)

    Returns
    -------
    tuple[int, str, str]
        Return code, stdout, and stderr
    """
    # Build SSH command
    ssh_cmd = ["ssh"]

    if identity_file:
        ssh_cmd.extend(["-i", identity_file])

    # Add common SSH options
    ssh_cmd.extend(
        [
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "LogLevel=ERROR",
        ]
    )

    # Construct the SSH destination
    if use_ssh_config:
        # Use SSH config alias directly (host is the alias)
        ssh_destination = host
    elif user:
        # Explicit user@host format
        ssh_destination = f"{user}@{host}"
    else:
        # Just hostname (SSH will use default user or config)
        ssh_destination = host

    ssh_cmd.extend([ssh_destination, command])

    # Use sshpass if password is provided
    if password:
        ssh_cmd = ["sshpass", "-p", password] + ssh_cmd

    try:
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", f"Command timed out after {timeout} seconds"
    except FileNotFoundError as e:
        if "sshpass" in str(e) and password:
            return 1, "", "sshpass not found. Install it or use SSH key authentication."
        return 1, "", str(e)


def run_ssh_command_with_tunnel(
    host: str,
    user: Optional[str],
    command: str,
    proxy_port: int,
    password: Optional[str] = None,
    identity_file: Optional[str] = None,
    use_ssh_config: bool = False,
    timeout: int = 600,
) -> tuple[int, str, str]:
    """Run a command on remote host via SSH with persistent reverse tunnel.

    This function creates a reverse SSH tunnel and runs the command in the same
    SSH session, ensuring the tunnel stays alive during command execution.

    Parameters
    ----------
    host : str
        Remote host address or SSH config alias
    user : str, optional
        Username for SSH connection
    command : str
        Command to execute on remote host
    proxy_port : int
        Local proxy port to tunnel (e.g., 8080 for SquidMan)
    password : str, optional
        Password for SSH authentication
    identity_file : str, optional
        Path to SSH private key file
    use_ssh_config : bool, optional
        If True, use SSH config alias
    timeout : int, optional
        Command timeout in seconds (default: 600 for long conda operations)

    Returns
    -------
    tuple[int, str, str]
        Return code, stdout, and stderr
    """
    # Build SSH command with reverse tunnel
    ssh_cmd = ["ssh"]

    if identity_file:
        ssh_cmd.extend(["-i", identity_file])

    # Add reverse tunnel option
    ssh_cmd.extend(["-R", f"{proxy_port}:localhost:{proxy_port}"])

    # Add common SSH options
    ssh_cmd.extend(
        [
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "LogLevel=ERROR",
            "-o",
            "ExitOnForwardFailure=yes",
        ]
    )

    # Construct the SSH destination
    if use_ssh_config:
        ssh_destination = host
    elif user:
        ssh_destination = f"{user}@{host}"
    else:
        ssh_destination = host

    ssh_cmd.extend([ssh_destination, command])

    # Use sshpass if password is provided
    if password:
        ssh_cmd = ["sshpass", "-p", password] + ssh_cmd

    try:
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", f"Command timed out after {timeout} seconds"
    except FileNotFoundError as e:
        if "sshpass" in str(e) and password:
            return 1, "", "sshpass not found. Install it or use SSH key authentication."
        return 1, "", str(e)


def create_ssh_tunnel(
    host: str,
    user: Optional[str],
    port: int,
    identity_file: Optional[str] = None,
    use_ssh_config: bool = False,
) -> Optional[bool]:
    """Create SSH tunnel with dynamic port forwarding (SOCKS proxy).

    This creates a local SOCKS proxy that tunnels through the SSH connection,
    allowing the remote cluster to access the internet via your local machine.

    Parameters
    ----------
    host : str
        Remote host address or SSH config alias
    user : str, optional
        Username for SSH connection
    port : int
        Local port for SOCKS proxy
    identity_file : str, optional
        Path to SSH private key file
    use_ssh_config : bool, optional
        If True, use SSH config alias

    Returns
    -------
    subprocess.Popen, optional
        The tunnel process, or None if failed
    """
    # First, kill any existing tunnels on this port
    # This prevents "Address already in use" errors from zombie SSH processes
    cleanup_ssh_tunnel(port)

    # Build SSH command for tunnel
    ssh_cmd = ["ssh", "-D", str(port), "-N", "-f"]

    if identity_file:
        ssh_cmd.extend(["-i", identity_file])

    # Add common SSH options
    ssh_cmd.extend(
        [
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "LogLevel=ERROR",
            "-o",
            "ExitOnForwardFailure=yes",
        ]
    )

    # Construct the SSH destination
    if use_ssh_config:
        ssh_cmd.append(host)
    else:
        if user:
            ssh_cmd.append(f"{user}@{host}")
        else:
            ssh_cmd.append(host)

    # Start the tunnel
    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            console.print(f"[red]Failed to create SSH tunnel: {result.stderr}[/red]")
            return None

        console.print(f"[green]✓ SSH tunnel created on port {port}[/green]")
        return True
    except subprocess.TimeoutExpired:
        console.print("[red]SSH tunnel creation timed out[/red]")
        return None
    except Exception as e:
        console.print(f"[red]Error creating SSH tunnel: {e}[/red]")
        return None


def cleanup_ssh_tunnel(port: int) -> None:
    """Clean up SSH tunnel by killing the process using the port.

    Parameters
    ----------
    port : int
        Port number of the tunnel to cleanup
    """
    try:
        # Find and kill SSH process using the port (both -D and -R tunnels)
        for pattern in [f"ssh.*-D.*{port}", f"ssh.*-R.*{port}"]:
            result = subprocess.run(
                ["pgrep", "-f", pattern], capture_output=True, text=True
            )
            if result.stdout:
                pids = result.stdout.strip().split("\n")
                for pid in pids:
                    if pid:
                        subprocess.run(["kill", pid], capture_output=True)
        console.print(f"[green]✓ SSH tunnel cleaned up (port {port})[/green]")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not cleanup tunnel: {e}[/yellow]")


def create_ssh_reverse_tunnel(
    host: str,
    user: Optional[str],
    port: int,
    identity_file: Optional[str],
    use_ssh_config: bool,
) -> Optional[bool]:
    """Create SSH reverse port forwarding tunnel.

    This creates a tunnel so that the remote host's localhost:port
    forwards to the local machine's localhost:port.

    WARNING: This creates a background tunnel that may not persist during
    long operations. For conda/pip operations, use run_ssh_command_with_tunnel()
    instead to ensure the tunnel stays alive.

    Parameters
    ----------
    host : str
        Remote host address or SSH config alias
    user : str, optional
        Username for SSH connection
    port : int
        Port number for the tunnel
    identity_file : str, optional
        Path to SSH private key file
    use_ssh_config : bool
        If True, use SSH config alias

    Returns
    -------
    bool or None
        True if successful, None if failed
    """
    # First, kill any existing tunnels on this port
    # This prevents accumulation of zombie SSH processes from previous failed attempts
    cleanup_ssh_tunnel(port)

    # Build SSH command for reverse tunnel
    ssh_cmd = ["ssh", "-R", f"{port}:localhost:{port}", "-N", "-f"]

    if identity_file:
        ssh_cmd.extend(["-i", identity_file])

    # Add common SSH options
    ssh_cmd.extend(
        [
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "LogLevel=ERROR",
            "-o",
            "ExitOnForwardFailure=yes",
        ]
    )

    # Construct the SSH destination
    if use_ssh_config:
        ssh_cmd.append(host)
    else:
        if user:
            ssh_cmd.append(f"{user}@{host}")
        else:
            ssh_cmd.append(host)

    # Start the reverse tunnel
    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            console.print(
                f"[red]Failed to create reverse tunnel: {result.stderr}[/red]"
            )
            # Clean up any zombie SSH processes that might have forked with -f
            cleanup_ssh_tunnel(port)
            return None

        console.print(
            f"[green]✓ Reverse SSH tunnel created (remote port {port} → local port {port})[/green]"
        )
        return True
    except subprocess.TimeoutExpired:
        console.print("[red]Reverse tunnel creation timed out[/red]")
        cleanup_ssh_tunnel(port)
        return None
    except Exception as e:
        console.print(f"[red]Error creating reverse tunnel: {e}[/red]")
        cleanup_ssh_tunnel(port)
        return None
