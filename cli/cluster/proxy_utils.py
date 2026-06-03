"""Squid proxy and remote proxy configuration utilities.

This module provides functions for managing Squid HTTP proxy on the local machine
and configuring proxy settings on remote clusters.
"""

from __future__ import annotations

import os
import subprocess
import time
from typing import Optional

from rich.console import Console

from .ssh_utils import run_ssh_command, show_verbose_output

console = Console()


def is_squid_installed() -> bool:
    """Check if squid is installed (system or locally compiled).

    Returns
    -------
    bool
        True if squid is installed, False otherwise
    """
    from pathlib import Path

    # Check for locally compiled squid first
    local_squid_binary = Path.home() / ".local" / "squid" / "sbin" / "squid"
    if local_squid_binary.exists():
        return True

    # Check for system squid
    try:
        result = subprocess.run(
            ["which", "squid"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def is_squid_running(port: int = 9999) -> bool:
    """Check if squid is running on the specified port.

    Parameters
    ----------
    port : int, optional
        Port to check (default: 9999)

    Returns
    -------
    bool
        True if squid is running, False otherwise
    """
    try:
        result = subprocess.run(
            ["lsof", "-i", f":{port}"], capture_output=True, text=True
        )
        return "squid" in result.stdout.lower()
    except Exception:
        return False


def is_proxy_running(port: int = 8080) -> bool:
    """Check if any HTTP proxy is running on the specified port.

    This checks for common proxy applications including SquidMan, squid,
    and other proxy servers.

    Parameters
    ----------
    port : int, optional
        Port to check (default: 8080 for SquidMan)

    Returns
    -------
    bool
        True if a proxy is running on the port, False otherwise
    """
    try:
        result = subprocess.run(
            ["lsof", "-i", f":{port}"], capture_output=True, text=True
        )
        # Check for common proxy applications
        proxy_apps = ["squid", "proxy", "privoxy", "tinyproxy"]
        stdout_lower = result.stdout.lower()
        return any(app in stdout_lower for app in proxy_apps) or result.returncode == 0
    except Exception:
        return False


def start_squid(port: int = 9999, remove_old_config: bool = False) -> bool:
    """Start squid HTTP proxy.

    Parameters
    ----------
    port : int, optional
        Port for squid to listen on (default: 9999)
    remove_old_config : bool, optional
        Remove old configuration file before starting (default: False)

    Returns
    -------
    bool
        True if squid started successfully, False otherwise
    """
    # Check if already running
    if is_squid_running(port):
        console.print(f"[yellow]Squid is already running on port {port}[/yellow]")
        return True

    # Check if squid is installed
    if not is_squid_installed():
        console.print("[red]✗ Squid is not installed![/red]")
        console.print("\n[yellow]Install squid:[/yellow]")
        console.print("  macOS:  [cyan]brew install squid[/cyan]")
        console.print("  Ubuntu: [cyan]sudo apt-get install squid[/cyan]")
        console.print("  CentOS: [cyan]sudo yum install squid[/cyan]")
        return False

    # Check if port is already in use
    try:
        port_check = subprocess.run(
            ["lsof", "-i", f":{port}"], capture_output=True, text=True
        )
        if port_check.returncode == 0:
            # Port is in use
            console.print(f"[yellow]Warning: Port {port} is already in use[/yellow]")
            console.print(f"[dim]{port_check.stdout.strip()}[/dim]")
            console.print("\n[yellow]Options:[/yellow]")
            console.print("  1. Kill the process using the port")
            console.print("  2. Use a different port: [cyan]--port <number>[/cyan]")

            # Check if it's an SSH tunnel
            if "ssh" in port_check.stdout.lower():
                console.print("\n[cyan]Looks like an SSH tunnel. To remove it:[/cyan]")
                console.print(f"  [cyan]kill $(lsof -t -i:{port})[/cyan]")
            return False
    except Exception:
        pass  # lsof might not be available on all systems

    # Create simple HTTP proxy config (no SSL/TLS needed)
    config_content = f"""# Squid configuration for atomate2siesta-cluster
# Auto-generated - do not edit manually
# Simple HTTP proxy without SSL - traffic is already encrypted via SSH tunnel

# Listen on localhost only for security
http_port 127.0.0.1:{port}

# Disable caching (we just need a proxy)
cache deny all

# Minimal logging
access_log none
cache_log /dev/null

# Allow all localhost connections
acl localnet src 127.0.0.1
http_access allow localnet
http_access deny all
"""

    # Save config to a known location
    config_dir = os.path.expanduser("~/.atomate2siesta-cluster")
    os.makedirs(config_dir, exist_ok=True)
    config_file = os.path.join(config_dir, "squid.conf")

    # Handle config file creation/removal
    config_exists = os.path.exists(config_file)

    if remove_old_config and config_exists:
        # Remove old config and create fresh one
        try:
            os.remove(config_file)
            console.print("[dim]Removed old squid configuration[/dim]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not remove old config: {e}[/yellow]")
        # Create new config
        with open(config_file, "w") as f:
            f.write(config_content)
        console.print("[dim]Created fresh squid configuration[/dim]")
    elif not config_exists:
        # Create config for the first time
        with open(config_file, "w") as f:
            f.write(config_content)
        console.print("[dim]Created squid configuration[/dim]")
    else:
        # Check if existing config has the same port
        existing_port = None
        try:
            with open(config_file, "r") as f:
                for line in f:
                    if line.strip().startswith("http_port"):
                        # Extract port from line like "http_port 127.0.0.1:3129"
                        parts = line.strip().split(":")
                        if len(parts) >= 2:
                            existing_port = int(parts[-1])
                        break
        except Exception:
            pass

        if existing_port and existing_port != port:
            # Port mismatch - warn and suggest --remove
            console.print(
                f"[yellow]⚠ Config file exists with different port ({existing_port})[/yellow]"
            )
            console.print(
                f"[yellow]  You requested port {port}, but config has port {existing_port}[/yellow]"
            )
            console.print(f"[yellow]  Location: {config_file}[/yellow]")
            console.print("\n[yellow]Options:[/yellow]")
            console.print(
                f"  1. Use existing config:  [cyan]atomate2siesta-cluster squid start --port {existing_port}[/cyan]"
            )
            console.print(
                f"  2. Recreate with new port: [cyan]atomate2siesta-cluster squid start --port {port} --remove[/cyan]"
            )
            return False
        else:
            # Use existing config (port matches or couldn't detect)
            console.print("[dim]Using existing squid configuration[/dim]")
            console.print(f"[dim]  Location: {config_file}[/dim]")
            if existing_port:
                console.print(f"[dim]  Port: {existing_port}[/dim]")
            console.print("[dim]  Tip: Use --remove to recreate config[/dim]")

    # Start squid
    try:
        from pathlib import Path

        # Check for locally compiled squid first
        local_squid_binary = Path.home() / ".local" / "squid" / "sbin" / "squid"
        if local_squid_binary.exists():
            squid_cmd = str(local_squid_binary)
            installation_type = "local (compiled)"
        else:
            squid_cmd = "squid"
            installation_type = "system"

        result = subprocess.run(
            [squid_cmd, "-f", config_file], capture_output=True, text=True
        )
        if result.returncode != 0:
            console.print(f"[red]✗ Failed to start squid: {result.stderr}[/red]")
            return False

        # Wait a moment for squid to start
        time.sleep(2)

        # Verify it's running
        if is_squid_running(port):
            console.print(
                f"[green]✓ Squid started on port {port} ({installation_type})[/green]"
            )
            console.print(f"[dim]Config file: {config_file}[/dim]")
            return True
        else:
            console.print("[red]✗ Squid started but not responding[/red]")
            return False

    except Exception as e:
        console.print(f"[red]✗ Error starting squid: {e}[/red]")
        return False


def stop_squid() -> bool:
    """Stop squid HTTP proxy.

    Returns
    -------
    bool
        True if squid stopped successfully, False otherwise
    """
    try:
        from pathlib import Path

        # Check for locally compiled squid first
        local_squid_binary = Path.home() / ".local" / "squid" / "sbin" / "squid"
        if local_squid_binary.exists():
            squid_cmd = str(local_squid_binary)
        else:
            squid_cmd = "squid"

        # Try graceful shutdown first
        result = subprocess.run(
            [squid_cmd, "-k", "shutdown"], capture_output=True, text=True
        )

        if result.returncode == 0:
            console.print("[green]✓ Squid stopped[/green]")
            return True

        # If that doesn't work, kill it
        result = subprocess.run(["killall", "squid"], capture_output=True, text=True)

        if result.returncode == 0:
            console.print("[green]✓ Squid killed[/green]")
            return True
        else:
            console.print("[yellow]Squid may not be running[/yellow]")
            return False

    except Exception as e:
        console.print(f"[red]✗ Error stopping squid: {e}[/red]")
        return False


def detect_running_squid_port() -> Optional[int]:
    """Detect which port squid is actually running on.

    Returns
    -------
    int or None
        Port number if squid is running, None otherwise
    """
    try:
        # Use lsof to find squid process and its listening port
        result = subprocess.run(
            ["lsof", "-i", "-P", "-n", "-a", "-c", "squid"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0 and result.stdout:
            # Parse lsof output to find the port
            for line in result.stdout.split("\n"):
                if "LISTEN" in line and "TCP" in line:
                    # Extract port from lines like: "TCP 127.0.0.1:9999 (LISTEN)"
                    parts = line.split()
                    for part in parts:
                        if ":" in part and "127.0.0.1" in part:
                            port_str = part.split(":")[-1]
                            # Handle port names like "distinct" (port 9999)
                            if port_str.isdigit():
                                return int(port_str)
                            # Try to resolve port name
                            elif port_str == "distinct":
                                return 9999
                            elif port_str == "squid-http":
                                return 3129
        return None
    except Exception:
        return None


def get_squid_status(port: int = 9999) -> dict:
    """Get squid status information.

    Parameters
    ----------
    port : int, optional
        Port to check (default: 9999). If squid is not running on this port,
        will auto-detect the actual port.

    Returns
    -------
    dict
        Status information including actual_port if detected
    """
    installed = is_squid_installed()
    running_on_requested_port = is_squid_running(port)

    # If not running on requested port, try to detect actual port
    actual_port = None
    if not running_on_requested_port:
        actual_port = detect_running_squid_port()

    status = {
        "installed": installed,
        "running": running_on_requested_port,
        "port": port,
        "actual_port": actual_port,
        "proxy_url": f"http://127.0.0.1:{port}" if running_on_requested_port else None,
    }
    return status


def detect_proxy_on_remote(
    host: str,
    user: Optional[str],
    password: Optional[str],
    identity_file: Optional[str],
    use_ssh_config: bool,
) -> Optional[str]:
    """Try to auto-detect proxy settings on remote cluster.

    Parameters
    ----------
    host : str
        Remote host address or SSH config alias
    user : str, optional
        Username for SSH connection
    password : str, optional
        Password for SSH authentication
    identity_file : str, optional
        Path to SSH private key file
    use_ssh_config : bool
        If True, use SSH config alias

    Returns
    -------
    str or None
        Proxy URL if found, None otherwise
    """
    # Check environment variables for proxy
    proxy_check_cmd = (
        "echo $http_proxy $https_proxy $HTTP_PROXY $HTTPS_PROXY | "
        "tr ' ' '\\n' | grep -i http | head -1"
    )

    returncode, stdout, stderr = run_ssh_command(
        host, user, proxy_check_cmd, password, identity_file, use_ssh_config
    )

    if returncode == 0 and stdout.strip():
        return stdout.strip()

    return None


def configure_proxy_on_remote(
    host: str,
    user: Optional[str],
    proxy_url: str,
    password: Optional[str],
    identity_file: Optional[str],
    use_ssh_config: bool,
    verbose: bool,
) -> bool:
    """Configure proxy settings on remote cluster.

    Creates .condarc and pip.conf with proxy settings.

    Parameters
    ----------
    host : str
        Remote host address or SSH config alias
    user : str, optional
        Username for SSH connection
    proxy_url : str
        Proxy URL (e.g., http://proxy.cluster.edu:8080)
    password : str, optional
        Password for SSH authentication
    identity_file : str, optional
        Path to SSH private key file
    use_ssh_config : bool
        If True, use SSH config alias
    verbose : bool
        If True, show detailed output

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    console.print(f"[cyan]Configuring proxy: {proxy_url}[/cyan]")

    # Create .condarc with proxy settings
    condarc_content = f"""# Proxy configuration for conda
# Generated by atomate2siesta-cluster

proxy_servers:
  http: {proxy_url}
  https: {proxy_url}

ssl_verify: true

channels:
  - conda-forge
  - defaults
"""

    # Create .condarc
    condarc_cmd = f"cat > ~/.condarc << 'CONDARC_EOF'\n{condarc_content}\nCONDARC_EOF"
    returncode, stdout, stderr = run_ssh_command(
        host, user, condarc_cmd, password, identity_file, use_ssh_config
    )

    if returncode != 0:
        console.print("[yellow]Warning: Failed to create .condarc[/yellow]")
        show_verbose_output(stdout, stderr, verbose)
        return False

    # Create pip.conf with proxy settings
    pip_conf_content = f"""# Proxy configuration for pip
# Generated by atomate2siesta-cluster

[global]
proxy = {proxy_url}
trusted-host = pypi.org
               pypi.python.org
               files.pythonhosted.org
"""

    # Create ~/.config/pip directory and pip.conf
    pip_conf_cmd = (
        f"mkdir -p ~/.config/pip && "
        f"cat > ~/.config/pip/pip.conf << 'PIPCONF_EOF'\n{pip_conf_content}\nPIPCONF_EOF"
    )
    returncode, stdout, stderr = run_ssh_command(
        host, user, pip_conf_cmd, password, identity_file, use_ssh_config
    )

    if returncode != 0:
        console.print("[yellow]Warning: Failed to create pip.conf[/yellow]")
        show_verbose_output(stdout, stderr, verbose)
        return False

    console.print("[green]✓ Proxy configuration created[/green]")
    return True


def add_proxy_to_remote_bashrc(
    host: str,
    user: Optional[str],
    proxy_url: str,
    password: Optional[str],
    identity_file: Optional[str],
    use_ssh_config: bool,
    verbose: bool,
) -> bool:
    """Add proxy environment variables to remote ~/.bashrc.

    Parameters
    ----------
    host : str
        Remote host address or SSH config alias
    user : str, optional
        Username for SSH connection
    proxy_url : str
        Proxy URL (e.g., http://127.0.0.1:3129)
    password : str, optional
        Password for SSH authentication
    identity_file : str, optional
        Path to SSH private key file
    use_ssh_config : bool
        If True, use SSH config alias
    verbose : bool
        If True, show detailed output

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    console.print("[cyan]Adding proxy exports to ~/.bashrc...[/cyan]")

    # Check if proxy exports already exist in bashrc
    check_cmd = (
        'grep -q "export http_proxy=" ~/.bashrc && echo "exists" || echo "not_exists"'
    )
    returncode, stdout, stderr = run_ssh_command(
        host, user, check_cmd, password, identity_file, use_ssh_config
    )

    if returncode != 0:
        console.print("[yellow]Warning: Could not check ~/.bashrc[/yellow]")
        show_verbose_output(stdout, stderr, verbose)
        return False

    if stdout.strip() == "exists":
        console.print("[dim]Proxy exports already exist in ~/.bashrc[/dim]")
        return True

    # Add proxy exports to bashrc with comment marker
    proxy_exports = f"""
# Proxy configuration for air-gapped cluster
# Generated by atomate2siesta-cluster
export http_proxy={proxy_url}
export https_proxy={proxy_url}
"""

    # Append to bashrc
    append_cmd = (
        f"cat >> ~/.bashrc << 'PROXY_EXPORTS_EOF'\n{proxy_exports}\nPROXY_EXPORTS_EOF"
    )
    returncode, stdout, stderr = run_ssh_command(
        host, user, append_cmd, password, identity_file, use_ssh_config
    )

    if returncode != 0:
        console.print(
            "[yellow]Warning: Failed to add proxy exports to ~/.bashrc[/yellow]"
        )
        show_verbose_output(stdout, stderr, verbose)
        return False

    console.print("[green]✓ Proxy exports added to ~/.bashrc[/green]")
    console.print(f"[dim]  export http_proxy={proxy_url}[/dim]")
    console.print(f"[dim]  export https_proxy={proxy_url}[/dim]")
    return True


def is_proxy_error(error_output: str) -> bool:
    """Detect if error is related to internet/proxy issues.

    Parameters
    ----------
    error_output : str
        Error output from command

    Returns
    -------
    bool
        True if error appears to be proxy-related
    """
    proxy_error_patterns = [
        "CondaHTTPError",
        "Connection refused",
        "Connection reset by peer",
        "Max retries exceeded",
        "HTTPSConnectionPool",
        "ConnectionResetError(104",
        "HTTP 000 CONNECTION FAILED",
        "HTTP error occurred when trying to retrieve",
        "repo.anaconda.com blocked",
    ]

    return any(pattern in error_output for pattern in proxy_error_patterns)


def show_proxy_error_help(proxy_url: Optional[str]) -> None:
    """Show helpful error message for proxy issues.

    Parameters
    ----------
    proxy_url : str, optional
        Proxy URL that was attempted, if any
    """
    console.print("\n[red]❌ Internet connection failed![/red]")
    console.print(
        "\n[yellow]This cluster appears to require a proxy for internet access.[/yellow]"
    )

    console.print("\n[bold]Solutions:[/bold]")
    console.print("  1. Get proxy URL from your cluster administrator")
    console.print("     Common formats:")
    console.print("       • http://proxy.cluster.edu:8080")
    console.print("       • http://proxy.bsc.es:8080 (for BSC clusters)")
    console.print()
    console.print("  2. Re-run with proxy option:")
    console.print(
        "       atomate2siesta-cluster setup --host <host> --proxy http://proxy:8080"
    )
    console.print()
    console.print("  3. Try auto-detection:")
    console.print("       atomate2siesta-cluster setup --host <host> --auto-proxy")
    console.print()
    console.print("  4. Check cluster documentation for internet access policies")
    console.print()

    if proxy_url:
        console.print(
            f"[yellow]Note: You used proxy {proxy_url}, but it may be incorrect.[/yellow]"
        )
        console.print(
            "[yellow]      Please verify the proxy URL with your administrator.[/yellow]"
        )
