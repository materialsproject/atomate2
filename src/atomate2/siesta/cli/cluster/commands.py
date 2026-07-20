"""CLI commands for remote cluster setup.

This module contains all Click command implementations for the cluster setup tool.
"""

from __future__ import annotations

import getpass
import subprocess
import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm
from rich.table import Table
from rich.text import Text

from .proxy_utils import (
    add_proxy_to_remote_bashrc,
    configure_proxy_on_remote,
    detect_proxy_on_remote,
    get_squid_status,
    is_proxy_error,
    is_squid_installed,
    is_squid_running,
    show_proxy_error_help,
    start_squid,
    stop_squid,
)
from .ssh_utils import (
    cleanup_ssh_tunnel,
    create_ssh_reverse_tunnel,
    run_ssh_command,
    show_verbose_output,
)

console = Console()


@click.group()
def cli():
    """Command-line interface for remote cluster setup."""


@cli.command()
@click.option(
    "--host",
    required=True,
    help="Remote cluster hostname, IP address, or SSH config alias",
)
@click.option(
    "--user",
    help="Username for SSH connection (not needed if using SSH config)",
)
@click.option(
    "--identity-file",
    "-i",
    help="Path to SSH private key file",
)
@click.option(
    "--password",
    is_flag=True,
    help="Prompt for password (if not using SSH key)",
)
@click.option(
    "--ssh-config",
    is_flag=True,
    help="Use SSH config alias (no user@ prefix needed)",
)
@click.option(
    "--env-name",
    default="atomate2siesta",
    help="Name for conda environment (default: atomate2siesta)",
)
@click.option(
    "--python-version",
    default="3.11",
    help="Python version for conda environment (default: 3.11)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed command output (stdout/stderr)",
)
@click.option(
    "--install-siesta",
    is_flag=True,
    help="Install SIESTA from conda-forge (siesta=*=*mpich*)",
)
@click.option(
    "--proxy",
    help="HTTP/HTTPS proxy URL (e.g., http://proxy.cluster.edu:8080)",
)
@click.option(
    "--auto-proxy",
    is_flag=True,
    help="Try to auto-detect proxy from remote cluster",
)
@click.option(
    "--ssh-tunnel",
    is_flag=True,
    help="Create SSH tunnel for internet access on air-gapped clusters (uses local machine as proxy)",
)
@click.option(
    "--tunnel-port",
    default=9999,
    help="Local port for SSH tunnel/squid proxy (default: 9999)",
)
@click.option(
    "--use-squid",
    is_flag=True,
    help="Automatically start/stop squid HTTP proxy (SquidMan replacement)",
)
@click.option(
    "--keep-squid-running",
    is_flag=True,
    help="Keep squid running after setup completes (only with --use-squid)",
)
@click.option(
    "--add-proxy-to-bashrc",
    is_flag=True,
    help="Add proxy environment variables to remote ~/.bashrc (works with --use-squid, --ssh-tunnel, or --proxy)",
)
def setup(
    host: str,
    user: str | None,
    identity_file: str | None,
    password: bool,
    ssh_config: bool,
    env_name: str,
    python_version: str,
    verbose: bool,
    install_siesta: bool,
    proxy: str | None,
    auto_proxy: bool,
    ssh_tunnel: bool,
    tunnel_port: int,
    use_squid: bool,
    keep_squid_running: bool,
    add_proxy_to_bashrc: bool,
):
    """Set up conda environment on remote cluster for jobflow-remote.

    This command SSHs to a remote cluster, creates a conda environment in $HOME,
    and installs both jobflow-remote and atomate2siesta (from GitHub repository).

    Examples
    --------
        # Using an SSH config alias
        atomate2siesta-cluster setup --host mycluster --ssh-config

        # Install with SIESTA included
        atomate2siesta-cluster setup --host mycluster --ssh-config --install-siesta

        # Using SSH key authentication
        atomate2siesta-cluster setup --host cluster.university.edu --user myuser -i ~/.ssh/id_rsa

        # Using password authentication
        atomate2siesta-cluster setup --host cluster.university.edu --user myuser --password

        # Custom environment name and Python version
        atomate2siesta-cluster setup --host mycluster --ssh-config --env-name myenv --python-version 3.11

        # Air-gapped cluster with squid proxy (persistent proxy in bashrc)
        atomate2siesta-cluster setup --host mn5-glogin1 --ssh-config --use-squid --add-proxy-to-bashrc

        # Air-gapped cluster with SSH tunnel
        atomate2siesta-cluster setup --host mycluster --ssh-config --ssh-tunnel

        # Explicit proxy with bashrc configuration
        atomate2siesta-cluster setup --host mycluster --ssh-config --proxy http://proxy.university.edu:3128 --add-proxy-to-bashrc

        # Verbose mode to see detailed output
        atomate2siesta-cluster setup --host mycluster --ssh-config --verbose
    """
    console.print("\n[bold cyan]Remote Cluster Setup for atomate2siesta[/bold cyan]\n")

    # Handle SSH config mode
    if ssh_config:
        console.print(f"[dim]Using SSH config alias: {host}[/dim]")
        if user:
            console.print(
                "[yellow]Note: --user is ignored when using --ssh-config[/yellow]"
            )
        user = None  # SSH config will provide the user
    # Get username if not provided
    elif not user:
        user = getpass.getuser()
        console.print(f"[dim]Using current user: {user}[/dim]")

    # Get password if requested
    ssh_password = None
    if password:
        if ssh_config:
            ssh_password = getpass.getpass(f"Password for {host}: ")
        else:
            ssh_password = getpass.getpass(f"Password for {user}@{host}: ")

    # Validate authentication method
    if not identity_file and not password and not ssh_config:
        console.print("[yellow]Warning: No authentication method specified.[/yellow]")
        console.print(
            "[yellow]Assuming SSH keys are configured (e.g., ssh-agent or ~/.ssh/config).[/yellow]\n"
        )

    # Initialize proxy_url
    proxy_url = proxy

    # Handle Squid HTTP proxy for air-gapped clusters
    squid_started = False
    reverse_tunnel_created = False
    tunnel_created = (
        False  # Track if SSH tunnel was created (in fallback or main handling)
    )
    using_socks_proxy = False  # Flag to indicate if we're using SOCKS proxy (ssh -D)
    if use_squid:
        from pathlib import Path

        console.print(
            f"[cyan]Starting Squid HTTP proxy on port {tunnel_port}...[/cyan]"
        )

        # Check for locally compiled squid first
        local_squid_binary = Path.home() / ".local" / "squid" / "sbin" / "squid"
        if local_squid_binary.exists():
            console.print(
                f"[green]✓ Found locally compiled squid:[/green] {local_squid_binary}"
            )
        elif is_squid_installed():
            console.print("[green]✓ Found system squid[/green]")
        else:
            console.print("[red]✗ Squid is not installed![/red]")
            console.print("\n[yellow]Install squid first:[/yellow]")
            console.print("  [cyan]atomate2siesta-cluster squid install[/cyan]")
            console.print(
                "  [cyan]atomate2siesta-cluster squid install --local --compile[/cyan]  (no sudo)"
            )
            console.print("\nOr manually:")
            console.print("  macOS:  [cyan]brew install squid[/cyan]")
            console.print("  Ubuntu: [cyan]sudo apt-get install squid[/cyan]")
            sys.exit(1)

        # Start squid on local machine
        if start_squid(tunnel_port):
            squid_started = True

            # Create reverse SSH tunnel so remote cluster can access local squid
            console.print(f"[cyan]Creating reverse SSH tunnel to {host}...[/cyan]")
            tunnel_result = create_ssh_reverse_tunnel(
                host, user, tunnel_port, identity_file, ssh_config
            )

            if tunnel_result:
                reverse_tunnel_created = True
                proxy_url = f"http://127.0.0.1:{tunnel_port}"
                console.print(
                    f"[green]✓ Remote cluster can now access squid via {proxy_url}[/green]"
                )
                console.print(
                    "[dim]Note: Traffic flows: Remote → SSH tunnel → Your Mac → Internet[/dim]"
                )
                console.print(
                    "[dim]Using persistent tunnel for all commands (SquidMan fix)[/dim]"
                )
                if keep_squid_running:
                    console.print(
                        "[dim]Squid and tunnel will keep running after setup completes[/dim]\n"
                    )
                else:
                    console.print(
                        "[dim]Squid and tunnel will be stopped after setup completes[/dim]\n"
                    )
            else:
                console.print("[red]✗ Failed to create reverse SSH tunnel![/red]")
                console.print(
                    "[yellow]This cluster's SSH server doesn't allow remote port forwarding (-R)[/yellow]"
                )
                console.print(
                    "[yellow]This is common on HPC clusters for security reasons.[/yellow]\n"
                )

                # Stop squid since we're giving up
                if squid_started:
                    stop_squid()
                    squid_started = False

                console.print(
                    "[bold]This means --use-squid won't work on this cluster.[/bold]\n"
                )
                console.print("[yellow]Your options:[/yellow]")
                console.print(
                    "  1. [bold]Check if cluster has direct internet access[/bold]"
                )
                console.print(
                    f"     Run: [cyan]ssh {host} 'curl -I https://www.google.com'[/cyan]"
                )
                console.print("     If it works, you don't need a proxy!\n")
                console.print(
                    "  2. [bold]Ask cluster admin about proxy configuration[/bold]"
                )
                console.print(
                    "     Many HPC systems provide http_proxy environment variables"
                )
                console.print(
                    "     Check: [cyan]ssh {host} 'echo $http_proxy'[/cyan]\n"
                )
                console.print(
                    "  3. [bold]Use cluster's local conda installation[/bold]"
                )
                console.print(
                    "     If available: [cyan]module load conda[/cyan] or [cyan]module load miniconda[/cyan]\n"
                )
                console.print(
                    "[dim]Note: SSH SOCKS proxy (ssh -D) won't help here because it creates"
                )
                console.print(
                    "the proxy on YOUR machine, not the cluster. The cluster can't access it.[/dim]"
                )
                sys.exit(1)
        else:
            console.print("[red]✗ Failed to start Squid![/red]")
            console.print("\n[yellow]Troubleshooting:[/yellow]")
            console.print("  • Check if port is already in use")
            console.print("  • Try a different port with --tunnel-port <port>")
            console.print(
                "  • Check squid status: [cyan]atomate2siesta-cluster squid status[/cyan]"
            )
            sys.exit(1)

    # Handle SSH tunnel for air-gapped clusters (only if not already created in fallback)
    if ssh_tunnel and not tunnel_created:
        console.print(
            "[yellow]⚠ --ssh-tunnel option is not supported on this cluster[/yellow]\n"
        )
        console.print(
            "[dim]This option requires reverse port forwarding (-R) which this cluster blocks.[/dim]\n"
        )
        console.print("[yellow]Your options:[/yellow]")
        console.print("  1. [bold]Check if cluster has direct internet access[/bold]")
        console.print(
            f"     Run: [cyan]ssh {host} 'curl -I https://www.google.com'[/cyan]\n"
        )
        console.print("  2. [bold]Ask cluster admin about proxy configuration[/bold]")
        console.print("     Check: [cyan]ssh {host} 'echo $http_proxy'[/cyan]\n")
        console.print("  3. [bold]Use cluster's local conda installation[/bold]")
        console.print("     Try: [cyan]ssh {host} 'module load conda'[/cyan]\n")
        sys.exit(1)

    # Handle proxy configuration (if not already set by squid/tunnel)
    if not ssh_tunnel and not use_squid:
        if auto_proxy and not proxy_url:
            console.print("[cyan]Attempting to auto-detect proxy...[/cyan]")
            detected_proxy = detect_proxy_on_remote(
                host, user, ssh_password, identity_file, ssh_config
            )
            if detected_proxy:
                console.print(f"[green]✓ Detected proxy: {detected_proxy}[/green]")
                proxy_url = detected_proxy
            else:
                console.print("[yellow]No proxy detected on remote cluster[/yellow]")

    if proxy_url:
        console.print(f"[cyan]Will use proxy: {proxy_url}[/cyan]\n")

    # Show connection info
    connection_info = Table(show_header=False, box=None)
    connection_info.add_column("Field", style="cyan")
    connection_info.add_column("Value")
    connection_info.add_row("Host", host)
    connection_info.add_row("User", user or "from SSH config")
    connection_info.add_row("Environment Name", env_name)
    connection_info.add_row("Python Version", python_version)

    console.print(connection_info)
    console.print()

    # Confirm before proceeding
    if not Confirm.ask("Proceed with cluster setup?"):
        console.print("[yellow]Setup cancelled.[/yellow]")
        sys.exit(0)

    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Step 1: Test SSH connection
        task = progress.add_task("[cyan]Testing SSH connection...", total=None)
        returncode, stdout, stderr = run_ssh_command(
            host,
            user,
            "echo 'Connection successful'",
            ssh_password,
            identity_file,
            ssh_config,
        )

        if returncode != 0:
            progress.stop()
            console.print("\n[bold red]✗ SSH connection failed![/bold red]")
            console.print(f"[red]Error: {stderr}[/red]")
            sys.exit(1)

        progress.update(task, completed=True)
        console.print("[green]✓ SSH connection successful[/green]")
        show_verbose_output(stdout, stderr, verbose)

        # Step 2: Check if conda is available
        task = progress.add_task("[cyan]Checking for conda installation...", total=None)
        # Use 'conda --version' instead of 'which conda' because conda is often a shell function
        # and 'which' would return the function definition instead of the path
        returncode, stdout, stderr = run_ssh_command(
            host, user, "conda --version 2>&1", ssh_password, identity_file, ssh_config
        )

        if returncode != 0:
            progress.stop()
            console.print("\n[bold red]✗ conda not found on remote host![/bold red]")

            # Check if we have proxy configured (--use-squid or --ssh-tunnel)
            if use_squid or ssh_tunnel:
                console.print(
                    "[yellow]No problem! We'll install it using the configured proxy.[/yellow]"
                )

                # Show correct proxy URL based on type
                if using_socks_proxy:
                    console.print(
                        f"\n[dim]Proxy is available at: socks5://127.0.0.1:{tunnel_port}[/dim]"
                    )
                else:
                    console.print(
                        f"\n[dim]Proxy is available at: http://127.0.0.1:{tunnel_port}[/dim]"
                    )

                # Test if proxy works
                console.print("[dim]Testing proxy connectivity...[/dim]")

                if using_socks_proxy:
                    # For SOCKS proxy, use curl with --socks5 flag
                    proxy_test_cmd = f"""
                    if command -v curl >/dev/null 2>&1; then
                        timeout 10 curl --socks5 127.0.0.1:{tunnel_port} -I --silent --max-time 5 https://www.google.com >/dev/null 2>&1 && echo 'WORKS' || echo 'FAILED'
                    else
                        echo 'NO_CURL'
                    fi
                    """
                else:
                    # For HTTP proxy (squid/reverse tunnel), use http_proxy env var
                    proxy_test_cmd = f"""
                    export http_proxy=http://127.0.0.1:{tunnel_port}
                    export https_proxy=http://127.0.0.1:{tunnel_port}
                    if command -v wget >/dev/null 2>&1; then
                        timeout 10 wget --spider --quiet https://www.google.com 2>/dev/null && echo 'WORKS' || echo 'FAILED'
                    else
                        timeout 10 curl -I --silent --max-time 5 https://www.google.com >/dev/null 2>&1 && echo 'WORKS' || echo 'FAILED'
                    fi
                    """

                returncode_proxy, stdout_proxy, stderr_proxy = run_ssh_command(
                    host, user, proxy_test_cmd, ssh_password, identity_file, ssh_config
                )

                if "WORKS" in stdout_proxy:
                    console.print(
                        "[green]✓ Proxy is working! Installing conda...[/green]\n"
                    )

                    # Install conda using proxy
                    if using_socks_proxy:
                        # For SOCKS proxy, use curl with --socks5
                        install_cmd = f"""
                        cd ~
                        curl --socks5 127.0.0.1:{tunnel_port} -o Miniconda3-latest-Linux-x86_64.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
                        bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
                        source $HOME/miniconda3/bin/activate
                        conda init bash
                        echo "Conda installed successfully"
                        """
                    else:
                        # For HTTP proxy, use http_proxy env vars
                        install_cmd = f"""
                        export http_proxy=http://127.0.0.1:{tunnel_port}
                        export https_proxy=http://127.0.0.1:{tunnel_port}
                        cd ~
                        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
                        bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
                        source $HOME/miniconda3/bin/activate
                        conda init bash
                        echo "Conda installed successfully"
                        """

                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console,
                    ) as progress:
                        task = progress.add_task(
                            "[cyan]Installing Miniconda...", total=None
                        )
                        (
                            returncode_install,
                            stdout_install,
                            stderr_install,
                        ) = run_ssh_command(
                            host,
                            user,
                            install_cmd,
                            ssh_password,
                            identity_file,
                            ssh_config,
                        )
                        progress.update(task, completed=True)

                    if (
                        returncode_install == 0
                        and "Conda installed successfully" in stdout_install
                    ):
                        console.print(
                            "[bold green]✓ Conda installed successfully![/bold green]"
                        )
                        console.print(
                            "\n[yellow]⚠ Important: You need to log out and back in for conda to be available in PATH[/yellow]"
                        )
                        console.print(
                            "[yellow]Then run this command again to continue setup:[/yellow]"
                        )
                        if use_squid:
                            console.print(
                                f"  [cyan]atomate2siesta-cluster setup --host {host} --ssh-config --use-squid[/cyan]"
                            )
                        else:
                            console.print(
                                f"  [cyan]atomate2siesta-cluster setup --host {host} --ssh-config --ssh-tunnel[/cyan]"
                            )
                        sys.exit(0)
                    else:
                        console.print("[red]✗ Failed to install conda[/red]")
                        console.print(f"[dim]Error: {stderr_install}[/dim]")
                        sys.exit(1)
                else:
                    console.print("[red]✗ Proxy is not working![/red]")
                    console.print(
                        "[yellow]The reverse tunnel/proxy setup failed.[/yellow]"
                    )
                    console.print("\n[bold]Please check:[/bold]")
                    console.print(
                        "  1. Squid is running: [cyan]atomate2siesta-cluster squid status[/cyan]"
                    )
                    console.print("  2. Reverse tunnel is active")
                    sys.exit(1)

            # No proxy configured, show manual instructions
            console.print(
                "[yellow]You need to install Miniconda in your HOME directory first.[/yellow]"
            )

            # Check internet connectivity to provide appropriate instructions
            console.print("\n[dim]Checking internet connectivity...[/dim]")
            internet_check_cmd = """
            if timeout 3 bash -c 'echo > /dev/tcp/8.8.8.8/53' 2>/dev/null; then
                echo 'YES'
            else
                if command -v nc >/dev/null 2>&1; then
                    timeout 3 nc -z -w2 8.8.8.8 53 >/dev/null 2>&1 && echo 'YES' || echo 'NO'
                else
                    echo 'NO'
                fi
            fi
            """
            returncode_net, stdout_net, stderr_net = run_ssh_command(
                host, user, internet_check_cmd, ssh_password, identity_file, ssh_config
            )
            has_internet = stdout_net.strip() == "YES"

            if has_internet:
                console.print("\n[green]✓ Cluster has internet access[/green]")
                console.print("\n[bold]Installation instructions:[/bold]")
                console.print(
                    "  [cyan]wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh[/cyan]"
                )
                console.print(
                    "  [cyan]bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3[/cyan]"
                )
                console.print("  [cyan]source $HOME/miniconda3/bin/activate[/cyan]")
                console.print("  [cyan]conda init bash[/cyan]")
                console.print(
                    "\n[dim]Then log out and back in, and run this setup command again.[/dim]"
                )
            else:
                console.print(
                    "\n[red]✗ Cluster is air-gapped (no internet access)[/red]"
                )
                console.print(
                    "\n[bold yellow]⚠ Special installation required for air-gapped clusters[/bold yellow]"
                )
                console.print("\n[bold]Option 1: SSH SOCKS Tunnel (Recommended)[/bold]")
                console.print(
                    "  [dim]Step 1:[/dim] On your [bold]LOCAL machine[/bold], create SSH SOCKS tunnel:"
                )
                console.print(f"    [cyan]ssh -D 9999 -N -f {host}[/cyan]")
                console.print(
                    "\n  [dim]Step 2:[/dim] On the [bold]CLUSTER[/bold] (via another SSH session):"
                )
                console.print(
                    "    [cyan]export http_proxy=http://127.0.0.1:9999[/cyan]"
                )
                console.print(
                    "    [cyan]export https_proxy=http://127.0.0.1:9999[/cyan]"
                )
                console.print(
                    "    [cyan]wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh[/cyan]"
                )
                console.print(
                    "    [cyan]bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3[/cyan]"
                )
                console.print("    [cyan]source $HOME/miniconda3/bin/activate[/cyan]")
                console.print("    [cyan]conda init bash[/cyan]")
                console.print("\n  [dim]Step 3:[/dim] Log out and back in")
                console.print("\n  [dim]Step 4:[/dim] Run setup with SSH tunnel:")
                console.print(
                    f"    [cyan]atomate2siesta-cluster setup --host {host} --ssh-config --ssh-tunnel[/cyan]"
                )
                console.print("\n[bold]Option 2: Squid Proxy (using port 9999)[/bold]")
                console.print(
                    "  [dim]Step 1:[/dim] On your [bold]LOCAL machine[/bold], install and start Squid:"
                )
                console.print("    [cyan]atomate2siesta-cluster squid install[/cyan]")
                console.print(
                    "    [cyan]atomate2siesta-cluster squid start --port 9999[/cyan]"
                )
                console.print(
                    "\n  [dim]Step 2:[/dim] Create SSH reverse tunnel to forward Squid:"
                )
                console.print(
                    f"    [cyan]ssh -R 9999:localhost:9999 -N -f {host}[/cyan]"
                )
                console.print(
                    "    [dim](maps local squid:9999 → cluster port:9999)[/dim]"
                )
                console.print(
                    "\n  [dim]Step 3:[/dim] On the [bold]CLUSTER[/bold] (via another SSH session), install conda using proxy:"
                )
                console.print(
                    "    [cyan]export http_proxy=http://127.0.0.1:9999[/cyan]"
                )
                console.print(
                    "    [cyan]export https_proxy=http://127.0.0.1:9999[/cyan]"
                )
                console.print(
                    "    [cyan]wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh[/cyan]"
                )
                console.print(
                    "    [cyan]bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3[/cyan]"
                )
                console.print("    [cyan]source $HOME/miniconda3/bin/activate[/cyan]")
                console.print("    [cyan]conda init bash[/cyan]")
                console.print(
                    "\n  [dim]Step 4:[/dim] Add proxy to ~/.bashrc (persistent):"
                )
                console.print(
                    "    [cyan]echo 'export http_proxy=http://127.0.0.1:9999' >> ~/.bashrc[/cyan]"
                )
                console.print(
                    "    [cyan]echo 'export https_proxy=http://127.0.0.1:9999' >> ~/.bashrc[/cyan]"
                )
                console.print(
                    "\n  [dim]Step 5:[/dim] Log out and back in, then run setup with Squid:"
                )
                console.print(
                    f"    [cyan]atomate2siesta-cluster setup --host {host} --ssh-config --use-squid[/cyan]"
                )
                console.print("\n[bold]Option 3: Transfer Miniconda Installer[/bold]")
                console.print(
                    "  [dim]Step 1:[/dim] On your [bold]LOCAL machine[/bold], download installer:"
                )
                console.print(
                    "    [cyan]wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh[/cyan]"
                )
                console.print("\n  [dim]Step 2:[/dim] Transfer to cluster:")
                console.print(
                    f"    [cyan]scp Miniconda3-latest-Linux-x86_64.sh {host}:~/[/cyan]"
                )
                console.print(
                    "\n  [dim]Step 3:[/dim] On the [bold]CLUSTER[/bold], install:"
                )
                console.print(
                    "    [cyan]bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3[/cyan]"
                )
                console.print("    [cyan]source $HOME/miniconda3/bin/activate[/cyan]")
                console.print("    [cyan]conda init bash[/cyan]")
                console.print(
                    "\n  [dim]Step 4:[/dim] Log out and back in, then run setup:"
                )
                console.print(
                    f"    [cyan]atomate2siesta-cluster setup --host {host} --ssh-config --ssh-tunnel[/cyan]"
                )
                console.print(
                    "\n[bold]Option 4: Build Offline Environment (Complete Solution)[/bold]"
                )
                console.print(
                    "  [dim]Skip conda installation completely! Build full environment locally:[/dim]"
                )
                console.print(
                    "    [cyan]atomate2siesta-cluster build-offline --install-siesta[/cyan]"
                )
                console.print(f"    [cyan]scp atomate2siesta.tar.gz {host}:~/[/cyan]")
                console.print(
                    "\n  [dim]Then on cluster, unpack and use directly (no setup command needed!):[/dim]"
                )
                console.print(
                    "    [cyan]mkdir -p ~/miniconda3/envs/atomate2siesta[/cyan]"
                )
                console.print(
                    "    [cyan]tar -xzf atomate2siesta.tar.gz -C ~/miniconda3/envs/atomate2siesta[/cyan]"
                )
                console.print(
                    "    [cyan]source ~/miniconda3/envs/atomate2siesta/bin/activate[/cyan]"
                )
                console.print("    [cyan]conda-unpack[/cyan]")

            console.print("\n[bold cyan]Recommendation:[/bold cyan]")
            console.print(
                "  • [bold]Option 1 (SSH Tunnel)[/bold] - Easiest, works for conda + package installation"
            )
            console.print(
                "  • [bold]Option 2 (Squid)[/bold] - More persistent, good for multiple setups"
            )
            console.print(
                "  • [bold]Option 4 (Build Offline)[/bold] - Most reliable, includes everything pre-built"
            )
            console.print(
                f"\n[dim]Run [cyan]atomate2siesta-cluster status --host {host} --ssh-config[/cyan] to verify installation[/dim]"
            )
            sys.exit(1)

        conda_version = stdout.strip()
        progress.update(task, completed=True)
        console.print(f"[green]✓ conda found ({conda_version})[/green]")
        show_verbose_output(stdout, stderr, verbose)

        # Step 3: Check if environment already exists
        task = progress.add_task(
            f"[cyan]Checking if environment '{env_name}' exists...", total=None
        )
        # More robust check: try to activate the environment
        # This will fail if it doesn't exist or is corrupted
        check_env_cmd = (
            f"source $(conda info --base)/etc/profile.d/conda.sh && "
            f"conda activate {env_name} 2>/dev/null && echo 'EXISTS'"
        )
        returncode, stdout, stderr = run_ssh_command(
            host,
            user,
            check_env_cmd,
            ssh_password,
            identity_file,
            ssh_config,
        )

        env_exists = "EXISTS" in stdout
        progress.update(task, completed=True)

        if env_exists:
            console.print(f"[yellow]✓ Environment '{env_name}' already exists[/yellow]")

            # Check Python version in existing environment
            task = progress.add_task("[cyan]Checking Python version...", total=None)

            check_python_cmd = (
                f"source $(conda info --base)/etc/profile.d/conda.sh && "
                f"conda activate {env_name} && "
                f"python -c 'import sys; print(f\"{{sys.version_info.major}}.{{sys.version_info.minor}}\")'"
            )

            returncode, stdout, stderr = run_ssh_command(
                host, user, check_python_cmd, ssh_password, identity_file, ssh_config
            )

            if returncode != 0:
                progress.update(task, completed=True)
                progress.stop()
                console.print(
                    "[red]✗ Failed to check Python version in existing environment[/red]"
                )
                if not Confirm.ask("Remove existing environment and recreate?"):
                    console.print("[yellow]Setup cancelled.[/yellow]")
                    sys.exit(0)
                needs_rebuild = True
            else:
                existing_py_version = stdout.strip()
                progress.update(task, completed=True)
                console.print(f"[green]✓ Python {existing_py_version} found[/green]")

                # Check if Python version matches what we want
                if existing_py_version != python_version:
                    progress.stop()
                    console.print(
                        f"[yellow]⚠ Python version mismatch: found {existing_py_version}, want {python_version}[/yellow]"
                    )
                    if not Confirm.ask(
                        f"Remove existing environment and recreate with Python {python_version}?"
                    ):
                        console.print("[yellow]Setup cancelled.[/yellow]")
                        sys.exit(0)
                    needs_rebuild = True
                else:
                    # Python version is correct, we can just install missing packages
                    needs_rebuild = False
                    progress.stop()
        else:
            console.print("[green]✓ Environment name is available[/green]")
            needs_rebuild = True

    # Continue with setup
    console.print()  # Clear line before starting new progress

    # Configure proxy if provided
    if proxy_url:
        success = configure_proxy_on_remote(
            host,
            user,
            proxy_url,
            ssh_password,
            identity_file,
            ssh_config,
            verbose,
        )
        if not success:
            console.print(
                "[yellow]Warning: Proxy configuration may be incomplete[/yellow]"
            )
            console.print("[yellow]Continuing anyway...[/yellow]\n")

        # Add proxy exports to bashrc if requested
        if add_proxy_to_bashrc and success:
            bashrc_success = add_proxy_to_remote_bashrc(
                host,
                user,
                proxy_url,
                ssh_password,
                identity_file,
                ssh_config,
                verbose,
            )
            if not bashrc_success:
                console.print(
                    "[yellow]Warning: Could not add proxy to ~/.bashrc[/yellow]"
                )
                console.print("[dim]You may need to add these manually:[/dim]")
                console.print(f"[dim]  export http_proxy={proxy_url}[/dim]")
                console.print(f"[dim]  export https_proxy={proxy_url}[/dim]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Remove existing environment if needed rebuild
        if env_exists and needs_rebuild:
            task = progress.add_task(
                "[cyan]Removing existing environment...", total=None
            )

            # Double-check that environment actually exists before removing
            # (in case the previous check was a false positive)
            verify_cmd = (
                f"source $(conda info --base)/etc/profile.d/conda.sh && "
                f"conda activate {env_name} 2>/dev/null && echo 'EXISTS'"
            )
            returncode_verify, stdout_verify, _ = run_ssh_command(
                host, user, verify_cmd, ssh_password, identity_file, ssh_config
            )

            if "EXISTS" in stdout_verify:
                # Environment really exists, remove it
                returncode, stdout, stderr = run_ssh_command(
                    host,
                    user,
                    f"conda env remove -n {env_name} -y",
                    ssh_password,
                    identity_file,
                    ssh_config,
                )

                if returncode != 0:
                    progress.stop()
                    # Filter out module load noise from stderr
                    clean_stderr = "\n".join(
                        line
                        for line in stderr.split("\n")
                        if not line.strip().startswith("load ")
                        and "Set INTEL compilers" not in line
                    )
                    console.print(
                        "\n[bold red]✗ Failed to remove existing environment![/bold red]"
                    )
                    if clean_stderr.strip():
                        console.print(f"[red]Error: {clean_stderr}[/red]")
                    sys.exit(1)

                progress.update(task, completed=True)
                console.print("[green]✓ Removed existing environment[/green]")
                show_verbose_output(stdout, stderr, verbose)
            else:
                # Environment doesn't actually exist, skip removal
                progress.update(task, completed=True)
                console.print("[dim]Environment doesn't exist, skipping removal[/dim]")

        # Step 4: Create conda environment (only if rebuilding)
        if needs_rebuild:
            task = progress.add_task(
                f"[cyan]Creating conda environment '{env_name}'...", total=None
            )
            # Add proxy environment variables if proxy is configured
            if proxy_url:
                create_cmd = (
                    f"http_proxy={proxy_url} https_proxy={proxy_url} "
                    f"HTTP_PROXY={proxy_url} HTTPS_PROXY={proxy_url} "
                    f"conda create -n {env_name} python={python_version} -y"
                )
            else:
                create_cmd = f"conda create -n {env_name} python={python_version} -y"

            # Note: We already have a persistent background tunnel (created with -f flag)
            # Just use regular run_ssh_command - the existing tunnel handles proxy
            returncode, stdout, stderr = run_ssh_command(
                host, user, create_cmd, ssh_password, identity_file, ssh_config
            )

            if returncode != 0:
                progress.stop()
                console.print(
                    "\n[bold red]✗ Failed to create conda environment![/bold red]"
                )
                console.print(f"[red]Error: {stderr}[/red]")

                # Check if it's a proxy error
                if is_proxy_error(stderr):
                    show_proxy_error_help(proxy_url)
                sys.exit(1)

            progress.update(task, completed=True)
            console.print(f"[green]✓ Created conda environment '{env_name}'[/green]")
            show_verbose_output(stdout, stderr, verbose)
        else:
            console.print(
                f"[green]✓ Using existing environment '{env_name}' with Python {python_version}[/green]"
            )

        # Step 5: Install jobflow-remote
        progress.stop()
        console.print("\n[cyan]Installing jobflow-remote...[/cyan]")
        console.print(
            "  [dim]Repository: https://pypi.org/project/jobflow-remote/[/dim]"
        )
        progress.start()
        task = progress.add_task("[cyan]Installing...", total=None)

        # Add proxy environment variables if proxy is configured
        if proxy_url:
            proxy_env = (
                f"http_proxy={proxy_url} https_proxy={proxy_url} "
                f"HTTP_PROXY={proxy_url} HTTPS_PROXY={proxy_url} "
            )
            install_cmd = (
                f"source $(conda info --base)/etc/profile.d/conda.sh && "
                f"conda activate {env_name} && "
                f"{proxy_env}pip install jobflow-remote"
            )
        else:
            install_cmd = (
                f"source $(conda info --base)/etc/profile.d/conda.sh && "
                f"conda activate {env_name} && "
                f"pip install jobflow-remote"
            )

        # Note: We already have a persistent background tunnel (created with -f flag)
        # Just use regular run_ssh_command - the existing tunnel handles proxy
        returncode, stdout, stderr = run_ssh_command(
            host, user, install_cmd, ssh_password, identity_file, ssh_config
        )

        if returncode != 0:
            progress.stop()
            console.print("\n[bold red]✗ Failed to install jobflow-remote![/bold red]")
            console.print(f"[red]Error: {stderr}[/red]")

            # Check if it's a proxy error
            if is_proxy_error(stderr):
                show_proxy_error_help(proxy_url)
            sys.exit(1)

        progress.update(task, completed=True)
        console.print("[green]✓ Installed jobflow-remote[/green]")
        show_verbose_output(stdout, stderr, verbose)

        # Step 6: Install atomate2 with the SIESTA extra
        progress.stop()
        console.print("\n[cyan]Installing atomate2[siesta]...[/cyan]")
        console.print("  [dim]Package: atomate2[siesta] (PyPI)[/dim]")

        progress.start()
        task = progress.add_task("[cyan]Installing...", total=None)

        # Add proxy environment variables if proxy is configured
        if proxy_url:
            proxy_env = (
                f"http_proxy={proxy_url} https_proxy={proxy_url} "
                f"HTTP_PROXY={proxy_url} HTTPS_PROXY={proxy_url} "
            )
            install_atomate2_cmd = (
                f"source $(conda info --base)/etc/profile.d/conda.sh && "
                f"conda activate {env_name} && "
                f"{proxy_env}pip install 'atomate2[siesta]'"
            )
        else:
            install_atomate2_cmd = (
                f"source $(conda info --base)/etc/profile.d/conda.sh && "
                f"conda activate {env_name} && "
                f"pip install 'atomate2[siesta]'"
            )

        # Note: We already have a persistent background tunnel (created with -f flag)
        # Just use regular run_ssh_command - the existing tunnel handles proxy
        returncode, stdout, stderr = run_ssh_command(
            host,
            user,
            install_atomate2_cmd,
            ssh_password,
            identity_file,
            ssh_config,
        )

        if returncode != 0:
            progress.stop()
            # Filter out module load noise from stderr
            clean_stderr = "\n".join(
                line
                for line in stderr.split("\n")
                if not line.strip().startswith("load ")
                and "Set INTEL compilers" not in line
                and "UCX" not in line
                and line.strip()  # Remove empty lines
            )

            console.print("\n[bold red]✗ Failed to install atomate2siesta![/bold red]")
            console.print(f"[red]Error: {clean_stderr}[/red]")

            # Check if it's a proxy error first
            if is_proxy_error(stderr):
                show_proxy_error_help(proxy_url)
            else:
                console.print("\n[yellow]Troubleshooting:[/yellow]")
                console.print(
                    "  • Ensure the cluster can reach PyPI (internet access or a proxy)"
                )
                console.print(
                    "  • Update pip in the environment: pip install --upgrade pip"
                )
            sys.exit(1)

        progress.update(task, completed=True)
        console.print("[green]✓ Installed atomate2[siesta][/green]")
        show_verbose_output(stdout, stderr, verbose)

        # Step 6.5: Install SIESTA (optional)
        if install_siesta:
            task = progress.add_task(
                "[cyan]Installing SIESTA from conda-forge...", total=None
            )

            # Add proxy environment variables if proxy is configured
            if proxy_url:
                proxy_env = (
                    f"http_proxy={proxy_url} https_proxy={proxy_url} "
                    f"HTTP_PROXY={proxy_url} HTTPS_PROXY={proxy_url} "
                )
                install_siesta_cmd = (
                    f"source $(conda info --base)/etc/profile.d/conda.sh && "
                    f"conda activate {env_name} && "
                    f'{proxy_env}conda install -y -c conda-forge "siesta=*=*mpich*"'
                )
            else:
                install_siesta_cmd = (
                    f"source $(conda info --base)/etc/profile.d/conda.sh && "
                    f"conda activate {env_name} && "
                    f'conda install -y -c conda-forge "siesta=*=*mpich*"'
                )

            # Note: We already have a persistent background tunnel (created with -f flag)
            # Just use regular run_ssh_command - the existing tunnel handles proxy
            returncode, stdout, stderr = run_ssh_command(
                host,
                user,
                install_siesta_cmd,
                ssh_password,
                identity_file,
                ssh_config,
            )

            if returncode != 0:
                progress.stop()
                console.print(
                    "\n[bold yellow]⚠ SIESTA installation failed[/bold yellow]"
                )
                console.print(f"[yellow]Error: {stderr}[/yellow]")

                # Check if it's a proxy error
                if is_proxy_error(stderr):
                    show_proxy_error_help(proxy_url)
                else:
                    console.print(
                        "\n[yellow]Note: You can install SIESTA manually later with:[/yellow]"
                    )
                    console.print(f"  conda activate {env_name}")
                    console.print('  conda install -c conda-forge "siesta=*=*mpich*"')
                show_verbose_output(stdout, stderr, verbose)
            else:
                progress.update(task, completed=True)
                console.print("[green]✓ Installed SIESTA from conda-forge[/green]")
                show_verbose_output(stdout, stderr, verbose)

        # Step 7: Verify installation
        task = progress.add_task("[cyan]Verifying installations...", total=None)

        verify_cmd = (
            f"source $(conda info --base)/etc/profile.d/conda.sh && "
            f"conda activate {env_name} && "
            f"python -c 'import jobflow_remote; import atomate2.siesta; "
            f'print(f"jobflow-remote: {{jobflow_remote.__version__}}"); '
            f'print(f"atomate2siesta: {{atomate2.siesta.__version__}}")\''
        )

        returncode, stdout, stderr = run_ssh_command(
            host, user, verify_cmd, ssh_password, identity_file, ssh_config
        )

        if returncode != 0:
            progress.stop()
            console.print(
                "\n[bold yellow]⚠ Installation verification failed[/bold yellow]"
            )
            console.print(f"[yellow]Error: {stderr}[/yellow]")
            show_verbose_output(stdout, stderr, verbose)
        else:
            versions = stdout.strip()
            progress.update(task, completed=True)
            console.print("[green]✓ Installations verified:[/green]")
            for line in versions.split("\n"):
                console.print(f"  [dim]{line}[/dim]")
            show_verbose_output(stdout, stderr, verbose)

        # Step 8: Generate .atomate2.yaml configuration file
        task = progress.add_task(
            "[cyan]Creating .atomate2.yaml configuration...", total=None
        )

        # Create the configuration file content with all available settings
        config_content = """# atomate2siesta configuration file
# Settings loaded in priority order:
#   1. Environment variables (with atomate2_ prefix)
#   2. This configuration file
#   3. Built-in defaults

# ============================================================================
# SIESTA Commands
# ============================================================================

# Command to run SIESTA
SIESTA_CMD: siesta < siesta.fdf > siesta.out

# Command to run Vibra (phonon analysis)
VIBRA_CMD: vibra < siesta.fdf > siesta.vibra.out

# Command to run optical_input (optical properties preprocessing)
OPTICAL_INPUT_CMD: optical_input < siesta.EPSIMG

# Command to run optical (optical properties calculation)
OPTICAL_CMD: optical < siesta.EPSIMG

# ============================================================================
# Paths
# ============================================================================

# Path to pseudopotential files
# Common options:
#   - ONCVPSP-PBEsol-FR-PDv0.4-Standard  (recommended for solids)
#   - ONCVPSP-PBE-SR-PDv0.4-Standard     (recommended for molecules)
#   - ONCVPSP-PBE-FR-PDv0.4-Standard     (fully relativistic)
SIESTA_PP_PATH: '$HOME/.siesta/pseudos/ONCVPSP-PBEsol-FR-PDv0.4-Standard/'

# Path to FLOS library (for variable-cell optimizations)
# Download from: https://github.com/siesta-project/flos
FLOS_PATH: "$HOME/apps/flos"

# ============================================================================
# Display Settings
# ============================================================================

# Show welcome banner on module import (true/false)
SIESTA_SHOW_BANNER: true

# Show FlowMaker docstrings when calling .make() (true/false)
SIESTA_SHOW_DOCSTRINGS: true

# Parameter evolution tracking display level:
#   - "none"    : No parameter tracking display
#   - "user"    : Show only initial user-provided parameters
#   - "diff"    : Show only changes (added/modified by dataclasses and powerups)
#   - "summary" : Show initial + changes summary (default)
#   - "full"    : Show all stages with complete final parameter table
SIESTA_SHOW_PARAMETER_EVOLUTION: summary

# ============================================================================
# File Compression
# ============================================================================

# Compress output files (true/false/"atomate")
#   - true     : Compress all files
#   - "atomate": Compress only simulation-related files (recommended)
#   - false    : No compression
SIESTA_ZIP_FILES: atomate

# ============================================================================
# Symmetry and Analysis Settings
# ============================================================================

# Symmetry precision for spglib symmetry finding (in Angstroms)
SYMPREC: 0.1

# Symmetry precision for phonon calculations (in Angstroms)
# More strict than SYMPREC for accurate force constants
PHONON_SYMPREC: 0.0001

# ============================================================================
# Elastic Constants
# ============================================================================

# Method for fitting elastic tensors
# Options:
#   - "finite_difference" : 2nd or 3rd order finite differences (recommended)
#   - "pseudoinverse"     : Pseudoinverse fitting
#   - "independent"       : Independent component fitting
ELASTIC_FITTING_METHOD: finite_difference
"""

        # Create the config file on remote cluster
        create_config_cmd = f"cat > $HOME/.atomate2.yaml << 'EOF'\n{config_content}EOF"

        returncode, stdout, stderr = run_ssh_command(
            host, user, create_config_cmd, ssh_password, identity_file, ssh_config
        )

        if returncode != 0:
            progress.stop()
            console.print(
                "\n[bold yellow]⚠ Failed to create .atomate2.yaml[/bold yellow]"
            )
            console.print(f"[yellow]Error: {stderr}[/yellow]")
            console.print("\n[yellow]Note: You can create it manually later:[/yellow]")
            console.print("  [cyan]atomate2siesta-config create[/cyan]")
            console.print("\nOr see the template with all available settings:")
            console.print("  [cyan]atomate2siesta-config show[/cyan]")
            show_verbose_output(stdout, stderr, verbose)
        else:
            progress.update(task, completed=True)
            console.print("[green]✓ Created .atomate2.yaml in $HOME[/green]")
            show_verbose_output(stdout, stderr, verbose)

    # Show completion message
    console.print()
    ssh_cmd = f"ssh {host}" if ssh_config or not user else f"ssh {user}@{host}"

    # Build installed packages message
    installed_msg = "with jobflow-remote and atomate2siesta installed"
    if install_siesta:
        installed_msg += " (including SIESTA)"
    installed_msg += "."

    # Cleanup squid and reverse tunnel if they were started (unless user wants to keep them running)
    if squid_started and not keep_squid_running:
        console.print()
        console.print("[cyan]Stopping Squid HTTP proxy and reverse tunnel...[/cyan]")
        stop_squid()
        if reverse_tunnel_created:
            cleanup_ssh_tunnel(tunnel_port)
        console.print("[dim]Squid and tunnel have been stopped[/dim]")
        console.print(
            "[dim]To restart: atomate2siesta-cluster setup ... --use-squid[/dim]"
        )
    elif squid_started and keep_squid_running:
        console.print()
        console.print("[green]✓ Squid and reverse tunnel are still running[/green]")
        console.print(
            f"[dim]Proxy URL (on remote cluster): http://127.0.0.1:{tunnel_port}[/dim]"
        )
        console.print("[dim]To stop squid: atomate2siesta-cluster squid stop[/dim]")
        console.print(
            f"[dim]To stop tunnel: kill $(pgrep -f 'ssh.*-R.*{tunnel_port}')[/dim]"
        )

    # Cleanup SSH tunnel if it was created
    if tunnel_created:
        console.print()
        cleanup_ssh_tunnel(tunnel_port)
        console.print(
            f"[dim]To recreate tunnel: ssh -D {tunnel_port} -N -f {host}[/dim]"
        )

    completion_panel = Panel(
        f"[bold]Setup Complete![/bold]\n\n"
        f"The conda environment '{env_name}' has been created on {host}\n"
        f"{installed_msg}\n\n"
        f"[bold]Installed package:[/bold]\n"
        f"  • atomate2[siesta] (PyPI)\n"
        + (
            "\n[bold]SIESTA:[/bold] Installed from conda-forge\n"
            if install_siesta
            else ""
        )
        + f"\n[bold]Configuration:[/bold]\n"
        f"  • Created ~/.atomate2.yaml with default SIESTA settings\n"
        f"  • Paths use $HOME variable for portability\n"
        f"\n[bold]To configure jobflow-remote:[/bold]\n"
        f"  [cyan]{ssh_cmd}[/cyan]\n"
        f"  [cyan]conda activate {env_name}[/cyan]\n"
        f"  [cyan]jf project generate myproject[/cyan]\n"
        f"  [cyan]jf admin reset[/cyan]\n\n"
        f"[bold]Next steps:[/bold]\n"
        f"1. Configure MongoDB connection in jobflow-remote\n"
        + (
            "2. SIESTA is ready to use!\n"
            if install_siesta
            else "2. Install SIESTA or verify executable path\n"
        )
        + "3. Edit ~/.atomate2.yaml if needed (pseudopotentials, FLOS path)\n"
        "4. Start runner: [cyan]jf runner start -d[/cyan]\n"
        "5. Submit jobs from your local machine",
        title="Success",
        style="green",
    )
    console.print(completion_panel)
    console.print()


@cli.command()
@click.option(
    "--host",
    required=True,
    help="Remote cluster hostname, IP address, or SSH config alias",
)
@click.option(
    "--user",
    help="Username for SSH connection (not needed if using SSH config)",
)
@click.option(
    "--identity-file",
    "-i",
    help="Path to SSH private key file",
)
@click.option(
    "--password",
    is_flag=True,
    help="Prompt for password (if not using SSH key)",
)
@click.option(
    "--ssh-config",
    is_flag=True,
    help="Use SSH config alias (no user@ prefix needed)",
)
@click.option(
    "--env-name",
    default="atomate2siesta",
    help="Name of conda environment to check (default: atomate2siesta)",
)
@click.option(
    "--use-squid",
    is_flag=True,
    help="Test internet connectivity through squid proxy",
)
@click.option(
    "--squid-port",
    default=9999,
    help="Port for squid proxy on remote cluster (default: 9999)",
)
def status(
    host: str,
    user: str | None,
    identity_file: str | None,
    password: bool,
    ssh_config: bool,
    env_name: str,
    use_squid: bool,
    squid_port: int,
):
    """Check status of remote cluster environment.

    This command checks:
      - SSH connectivity
      - Conda installation and version
      - Environment existence
      - Installed packages (jobflow-remote, atomate2siesta)
      - Internet connectivity (direct access, proxy configuration)

    Examples
    --------
        # Using SSH config alias
        atomate2siesta-cluster status --host mycluster --ssh-config

        # Check default environment
        atomate2siesta-cluster status --host cluster.university.edu --user myuser

        # Check custom environment
        atomate2siesta-cluster status --host mycluster --ssh-config --env-name myenv

        # Diagnose connectivity issues on air-gapped clusters
        atomate2siesta-cluster status --host mn5-glogin1 --ssh-config

        # Test if squid proxy is working (requires SSH tunnel + squid running locally)
        atomate2siesta-cluster status --host mn5-glogin1 --ssh-config --use-squid --squid-port 9999
    """
    console.print("\n[bold cyan]Checking Remote Cluster Status[/bold cyan]\n")

    # Handle SSH config mode
    if ssh_config:
        console.print(f"[dim]Using SSH config alias: {host}[/dim]")
        if user:
            console.print(
                "[yellow]Note: --user is ignored when using --ssh-config[/yellow]"
            )
        user = None
    # Get username if not provided
    elif not user:
        user = getpass.getuser()

    # Get password if requested
    ssh_password = None
    if password:
        if ssh_config:
            ssh_password = getpass.getpass(f"Password for {host}: ")
        else:
            ssh_password = getpass.getpass(f"Password for {user}@{host}: ")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Test SSH connection
        task = progress.add_task("[cyan]Connecting to remote host...", total=None)
        returncode, stdout, stderr = run_ssh_command(
            host, user, "echo 'Connected'", ssh_password, identity_file, ssh_config
        )

        if returncode != 0:
            progress.stop()
            console.print("\n[bold red]✗ SSH connection failed![/bold red]")
            console.print(f"[red]Error: {stderr}[/red]")
            sys.exit(1)

        progress.remove_task(task)
        progress.console.print("[green]✓ Connected to remote host[/green]")

        # Check conda with multiple strategies
        task = progress.add_task("[cyan]Checking conda installation...", total=None)

        # Strategy 1: Try direct conda command (NOT module system!)
        # We ONLY want user-owned conda installations, not system modules
        conda_check_cmd = """
        # Try direct conda (if in PATH and not from module)
        if command -v conda >/dev/null 2>&1; then
            # Make sure it's not from /apps (system module)
            conda_path=$(which conda)
            if [[ ! "$conda_path" =~ ^/apps/ ]]; then
                conda --version && echo "METHOD:direct"
                exit 0
            fi
        fi

        # Try common user conda paths (HOME directory only)
        for conda_path in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "$HOME/mambaforge"; do
            if [ -f "$conda_path/etc/profile.d/conda.sh" ]; then
                source "$conda_path/etc/profile.d/conda.sh"
                if command -v conda >/dev/null 2>&1; then
                    conda --version && echo "METHOD:sourced:$conda_path"
                    exit 0
                fi
            fi
            if [ -f "$conda_path/bin/conda" ]; then
                "$conda_path/bin/conda" --version && echo "METHOD:direct:$conda_path"
                exit 0
            fi
        done

        # Check if conda is available via module system (informational only)
        if command -v module >/dev/null 2>&1; then
            for mod in miniconda/24.1.2 miniconda3 miniconda anaconda/2024.02 anaconda/2023.07 anaconda3 anaconda; do
                if module load "$mod" >/dev/null 2>&1; then
                    if command -v conda >/dev/null 2>&1; then
                        echo "CONDA_VIA_MODULE:$mod"
                        exit 2
                    fi
                fi
            done
        fi

        # Not found
        echo "CONDA_NOT_FOUND"
        exit 1
        """

        returncode, stdout, stderr = run_ssh_command(
            host, user, conda_check_cmd, ssh_password, identity_file, ssh_config
        )

        conda_found = False
        conda_stdout = stdout  # Save conda output for parsing
        packages_stdout = ""  # Initialize for package listing later

        # Check if conda is only available via module (not usable for setup)
        if returncode == 2 or "CONDA_VIA_MODULE:" in conda_stdout:
            progress.remove_task(task)
            module_name = ""
            if "CONDA_VIA_MODULE:" in conda_stdout:
                module_name = (
                    conda_stdout.strip().split("CONDA_VIA_MODULE:")[1].split()[0]
                )

            progress.console.print(
                "[yellow]⚠ Conda only available via module system[/yellow]"
            )
            if module_name:
                progress.console.print(f"[dim]  Found: module load {module_name}[/dim]")
            progress.console.print("\n[yellow]  Why this won't work:[/yellow]")
            progress.console.print(
                "    • Module conda is [bold]system-wide[/bold] (read-only, can't install packages)"
            )
            progress.console.print(
                "    • [bold]Not persistent[/bold] - need 'module load' every session"
            )
            progress.console.print(
                "    • [bold]Can't create environments[/bold] in your HOME directory"
            )
            progress.console.print(
                "    • Setup command requires [bold]user-owned conda[/bold] in $HOME"
            )
            progress.console.print(
                "\n[cyan]  Solution: Install your own conda in $HOME/miniconda3[/cyan]"
            )
            conda_found = False
        elif returncode != 0 or "CONDA_NOT_FOUND" in conda_stdout:
            progress.remove_task(task)
            progress.console.print(
                "[yellow]⚠ conda not found in your HOME directory[/yellow]"
            )
            conda_found = False
        else:
            conda_found = True

            # Parse conda version and method (filter out noise from module system)
            lines = [
                line.strip()
                for line in conda_stdout.strip().split("\n")
                if line.strip()
            ]

            # Find conda version line (starts with "conda" or "miniconda" or "anaconda")
            conda_version = ""
            conda_method = "unknown"

            for i, line in enumerate(lines):
                # Look for conda version
                if line.lower().startswith(("conda ", "miniconda ", "anaconda ")):
                    conda_version = line
                    # Check next line for METHOD
                    if i + 1 < len(lines) and lines[i + 1].startswith("METHOD:"):
                        conda_method = lines[i + 1].replace("METHOD:", "")
                    break

            # Fallback if no version found
            if not conda_version:
                conda_version = "version unknown"

            progress.remove_task(task)
            if "direct" in conda_method and ":" not in conda_method:
                progress.console.print(f"[green]✓ {conda_version} (in PATH)[/green]")
            elif "sourced:" in conda_method:
                path = conda_method.split("sourced:")[1]
                progress.console.print(
                    f"[green]✓ {conda_version}[/green] [dim](sourced from {path})[/dim]"
                )
            elif "direct:" in conda_method:
                path = conda_method.split("direct:")[1]
                progress.console.print(
                    f"[green]✓ {conda_version}[/green] [dim](found at {path})[/dim]"
                )
            elif "module:" in conda_method:
                mod = conda_method.split("module:")[1]
                progress.console.print(
                    f"[green]✓ {conda_version}[/green] [dim](via module {mod})[/dim]"
                )
            else:
                progress.console.print(f"[green]✓ {conda_version}[/green]")

        # Only check environment and packages if conda is found
        if conda_found:
            # Check if environment exists
            task = progress.add_task(
                f"[cyan]Checking environment '{env_name}'...", total=None
            )
            returncode, stdout_env, stderr = run_ssh_command(
                host,
                user,
                f"conda env list | grep -w {env_name}",
                ssh_password,
                identity_file,
                ssh_config,
            )

            env_exists = returncode == 0
            progress.remove_task(task)

            if env_exists:
                progress.console.print(
                    f"[green]✓ Environment '{env_name}' exists[/green]"
                )

                # Check installed packages
                task = progress.add_task(
                    "[cyan]Checking installed packages...", total=None
                )
                check_cmd = (
                    f"source $(conda info --base)/etc/profile.d/conda.sh && "
                    f"conda activate {env_name} && "
                    f"pip list | grep -E '(jobflow-remote|atomate2siesta|maggma|monty|pydantic)'"
                )

                returncode, packages_stdout, stderr = run_ssh_command(
                    host, user, check_cmd, ssh_password, identity_file, ssh_config
                )

                progress.remove_task(task)

                # Check if jobflow-remote is installed
                if "jobflow-remote" in packages_stdout:
                    progress.console.print("[green]✓ jobflow-remote installed[/green]")
                else:
                    progress.console.print(
                        "[yellow]⚠ jobflow-remote not installed[/yellow]"
                    )

                # Check if atomate2siesta is installed
                if "atomate2siesta" in packages_stdout:
                    progress.console.print("[green]✓ atomate2siesta installed[/green]")
                else:
                    progress.console.print(
                        "[yellow]⚠ atomate2siesta not installed[/yellow]"
                    )
            else:
                progress.console.print(
                    f"[yellow]⚠ Environment '{env_name}' not found[/yellow]"
                )
                packages_stdout = ""  # No packages to show

        # Check internet connectivity with fast, simple test
        if use_squid:
            task = progress.add_task(
                f"[cyan]Checking internet via squid proxy (port {squid_port})...",
                total=None,
            )
        else:
            task = progress.add_task(
                "[cyan]Checking internet connectivity...", total=None
            )

        # Test direct access (without proxy)
        direct_access_cmd = """
        # Method 1: Try bash built-in TCP test (fastest, no external commands)
        if timeout 3 bash -c 'echo > /dev/tcp/8.8.8.8/53' 2>/dev/null; then
            echo 'YES'
        else
            # Method 2: Try nc (netcat) if available
            if command -v nc >/dev/null 2>&1; then
                timeout 3 nc -z -w2 8.8.8.8 53 >/dev/null 2>&1 && echo 'YES' || echo 'NO'
            else
                # Method 3: Assume blocked (default for air-gapped clusters)
                echo 'NO'
            fi
        fi
        """
        returncode, stdout_direct, stderr = run_ssh_command(
            host, user, direct_access_cmd, ssh_password, identity_file, ssh_config
        )
        has_direct_access = stdout_direct.strip() == "YES"

        # Check proxy configuration (current environment) FIRST
        proxy_cmd = 'echo "http_proxy=$http_proxy https_proxy=$https_proxy"'
        returncode, stdout_proxy, stderr = run_ssh_command(
            host, user, proxy_cmd, ssh_password, identity_file, ssh_config
        )

        # Parse proxy info to see if it's already configured
        proxy_env = stdout_proxy.strip()
        has_proxy_env = (
            "http_proxy=" in proxy_env
            and proxy_env.split("http_proxy=")[1].split()[0] != ""
        )

        # Check if the configured proxy matches the squid port we're testing
        proxy_matches_squid = False
        if has_proxy_env:
            proxy_url = proxy_env.split("http_proxy=")[1].split()[0]
            if (
                f"127.0.0.1:{squid_port}" in proxy_url
                or f"localhost:{squid_port}" in proxy_url
            ):
                proxy_matches_squid = True

        # If --use-squid is set, test internet through squid proxy
        squid_works = False
        squid_vars_set = False
        if use_squid:
            # Check if proxy vars are already set to the expected squid port
            if proxy_matches_squid:
                squid_vars_set = True
                # Test with existing proxy configuration
                squid_test_cmd = """
                # Test with current proxy settings (already configured)
                if command -v wget >/dev/null 2>&1; then
                    if timeout 10 wget --spider --quiet https://www.google.com 2>/dev/null; then
                        echo 'PROXY_WORKS'
                    else
                        echo 'PROXY_FAILED'
                    fi
                elif command -v curl >/dev/null 2>&1; then
                    if timeout 10 curl -I --silent --max-time 5 https://www.google.com >/dev/null 2>&1; then
                        echo 'PROXY_WORKS'
                    else
                        echo 'PROXY_FAILED'
                    fi
                else
                    echo 'NO_TOOLS'
                fi
                """
            else:
                squid_vars_set = False
                # Test with temporary proxy settings
                squid_test_cmd = f"""
                # Test with temporary proxy settings (not configured in environment)
                export http_proxy=http://127.0.0.1:{squid_port}
                export https_proxy=http://127.0.0.1:{squid_port}

                if command -v wget >/dev/null 2>&1; then
                    if timeout 10 wget --spider --quiet https://www.google.com 2>/dev/null; then
                        echo 'PROXY_WORKS'
                    else
                        echo 'PROXY_FAILED'
                    fi
                elif command -v curl >/dev/null 2>&1; then
                    if timeout 10 curl -I --silent --max-time 5 https://www.google.com >/dev/null 2>&1; then
                        echo 'PROXY_WORKS'
                    else
                        echo 'PROXY_FAILED'
                    fi
                else
                    echo 'NO_TOOLS'
                fi
                """

            returncode, stdout_squid, stderr = run_ssh_command(
                host, user, squid_test_cmd, ssh_password, identity_file, ssh_config
            )
            squid_works = "PROXY_WORKS" in stdout_squid

        # Check .condarc for proxy
        condarc_cmd = "test -f ~/.condarc && grep -A2 'proxy_servers' ~/.condarc || echo 'No proxy in .condarc'"
        returncode, stdout_condarc, stderr = run_ssh_command(
            host, user, condarc_cmd, ssh_password, identity_file, ssh_config
        )

        progress.remove_task(task)

    # Show internet connectivity status
    console.print("\n[bold]Internet Connectivity:[/bold]\n")

    connectivity_table = Table(show_header=False, box=None)
    connectivity_table.add_column("Test", style="cyan")
    connectivity_table.add_column("Status")

    if has_direct_access:
        connectivity_table.add_row(
            "Direct Access", "[green]✓ Available[/green] [dim](no proxy needed)[/dim]"
        )
    else:
        connectivity_table.add_row("Direct Access", "[red]✗ Blocked[/red]")

    # Show squid proxy test results if --use-squid was used
    if use_squid:
        if squid_works and squid_vars_set:
            connectivity_table.add_row(
                f"Squid Proxy (:{squid_port})",
                "[green]✓ Working[/green] [dim](proxy vars configured)[/dim]",
            )
        elif squid_works and not squid_vars_set:
            connectivity_table.add_row(
                f"Squid Proxy (:{squid_port})",
                "[yellow]⚠ Works but vars NOT set[/yellow] [dim](need to export)[/dim]",
            )
        else:
            connectivity_table.add_row(
                f"Squid Proxy (:{squid_port})",
                "[red]✗ Not working[/red] [dim](check proxy/tunnel)[/dim]",
            )

    # Show proxy environment variables
    if has_proxy_env:
        proxy_url = proxy_env.split("http_proxy=")[1].split()[0]
        if use_squid and proxy_matches_squid:
            connectivity_table.add_row(
                "Proxy (Environment)", f"[green]✓ Configured[/green]: {proxy_url}"
            )
        else:
            connectivity_table.add_row(
                "Proxy (Environment)", f"[yellow]Configured[/yellow]: {proxy_url}"
            )
    elif use_squid:
        connectivity_table.add_row(
            "Proxy (Environment)",
            "[red]✗ Not configured[/red] [dim](needed for squid!)[/dim]",
        )
    else:
        connectivity_table.add_row("Proxy (Environment)", "[dim]Not configured[/dim]")

    # Check .condarc
    if "proxy_servers" in stdout_condarc:
        connectivity_table.add_row("Proxy (.condarc)", "[yellow]Configured[/yellow]")
    else:
        connectivity_table.add_row("Proxy (.condarc)", "[dim]Not configured[/dim]")

    console.print(connectivity_table)

    # Provide recommendations
    console.print()

    # Special handling for --use-squid flag
    if use_squid:
        console.print(
            f"[dim]Testing with squid proxy on port {squid_port} (default: 9999, use --squid-port to change)[/dim]\n"
        )

        if squid_works and squid_vars_set:
            # Perfect: proxy works AND vars are already configured
            console.print(
                "[bold green]✓ Squid proxy is fully configured and working![/bold green]"
            )
            console.print(
                f"[bold green]Proxy variables are set: http_proxy=http://127.0.0.1:{squid_port}[/bold green]"
            )
            console.print(
                "[bold green]Internet is accessible through proxy[/bold green]"
            )
            if conda_found:
                console.print("\n[bold]Ready to use:[/bold]")
                console.print(
                    "  • Test package installation: [cyan]conda search python[/cyan]"
                )
                console.print(
                    "  • Install packages: [cyan]conda install <package>[/cyan]"
                )
            else:
                console.print("\n[bold]Next step:[/bold]")
                console.print("  • Install conda (proxy will work automatically)")

        elif squid_works and not squid_vars_set:
            # Proxy works in test but vars not configured permanently
            console.print(
                "[yellow]⚠ Squid proxy works, but environment variables NOT configured![/yellow]"
            )
            console.print("[dim]Test succeeded with temporary proxy settings[/dim]")
            console.print(
                "[dim]But http_proxy/https_proxy are not set in your environment[/dim]\n"
            )
            console.print("[bold red]ACTION REQUIRED ON THE CLUSTER:[/bold red]")
            console.print(
                "  You need to configure proxy variables permanently on the cluster.\n"
            )
            console.print("  [bold]Step 1: SSH to the cluster[/bold]")
            console.print(f"    [cyan]ssh {host}[/cyan]\n")
            console.print(
                "  [bold]Step 2: Add to ~/.bashrc (recommended - permanent)[/bold]"
            )
            console.print(
                f"    [cyan]echo 'export http_proxy=http://127.0.0.1:{squid_port}' >> ~/.bashrc[/cyan]"
            )
            console.print(
                f"    [cyan]echo 'export https_proxy=http://127.0.0.1:{squid_port}' >> ~/.bashrc[/cyan]"
            )
            console.print("    [cyan]source ~/.bashrc[/cyan]\n")
            console.print(
                "  [bold]OR Step 2: Export now (temporary - only for current session)[/bold]"
            )
            console.print(
                f"    [cyan]export http_proxy=http://127.0.0.1:{squid_port}[/cyan]"
            )
            console.print(
                f"    [cyan]export https_proxy=http://127.0.0.1:{squid_port}[/cyan]\n"
            )
            console.print("  [bold]Step 3: Verify internet access[/bold]")
            console.print("    [cyan]wget --spider https://www.google.com[/cyan]")
            console.print("    [dim](should succeed without errors)[/dim]")

        else:
            # Proxy not working
            console.print("[red]✗ Squid proxy is not working![/red]")
            if not squid_vars_set:
                console.print(
                    "[yellow]Note: Proxy environment variables are also not configured[/yellow]\n"
                )
            console.print("\n[bold]Troubleshooting steps:[/bold]")
            console.print(
                "  1. Check if squid is running on your [bold]LOCAL[/bold] machine:"
            )
            console.print(
                f"     [cyan]atomate2siesta-cluster squid status --port {squid_port}[/cyan]"
            )
            console.print("     If not running, start it:")
            console.print(
                f"     [cyan]atomate2siesta-cluster squid start --port {squid_port}[/cyan]\n"
            )
            console.print(
                "  2. Check if SSH reverse tunnel is active on your [bold]LOCAL[/bold] machine:"
            )
            console.print(f"     [cyan]lsof -i :{squid_port}[/cyan]")
            console.print(
                "     [dim](should show ssh listening on port {squid_port})[/dim]\n"
            )
            console.print("  3. If tunnel not found, create SSH reverse tunnel:")
            console.print(
                f"     [cyan]ssh -R {squid_port}:localhost:{squid_port} {host}[/cyan]"
            )
            console.print(
                f"     [dim](maps local squid:{squid_port} → remote port:{squid_port})[/dim]"
            )
            console.print("     [dim]Keep this terminal open![/dim]\n")
            console.print("  4. Configure proxy on cluster (in a different terminal):")
            console.print(f"     [cyan]ssh {host}[/cyan]")
            console.print(
                f"     [cyan]export http_proxy=http://127.0.0.1:{squid_port}[/cyan]"
            )
            console.print(
                f"     [cyan]export https_proxy=http://127.0.0.1:{squid_port}[/cyan]"
            )
            console.print("     Or add to ~/.bashrc for persistence:")
            console.print(
                f"     [cyan]echo 'export http_proxy=http://127.0.0.1:{squid_port}' >> ~/.bashrc[/cyan]"
            )
            console.print(
                f"     [cyan]echo 'export https_proxy=http://127.0.0.1:{squid_port}' >> ~/.bashrc[/cyan]"
            )
            console.print("     [cyan]source ~/.bashrc[/cyan]\n")
            console.print("  5. Verify the complete chain:")
            console.print(
                f"     • Local squid: [cyan]atomate2siesta-cluster squid status --port {squid_port}[/cyan]"
            )
            console.print(
                f"     • SSH tunnel: [cyan]lsof -i :{squid_port}[/cyan] (on local machine)"
            )
            console.print(
                f"     • Cluster proxy: [cyan]echo $http_proxy[/cyan] (should show http://127.0.0.1:{squid_port})"
            )
            console.print(
                "     • Internet test: [cyan]wget --spider https://www.google.com[/cyan] (on cluster)"
            )
            console.print(
                "\n  [dim]Tip: Using the same port ({squid_port}) for both local squid and remote access keeps it simple![/dim]"
            )
    elif has_direct_access:
        console.print("[green]✓ Cluster has direct internet access[/green]")
        if conda_found:
            console.print(
                "[dim]Package installation should work without proxy configuration[/dim]"
            )
        else:
            console.print(
                "[yellow]Install conda first, then run atomate2siesta-cluster setup[/yellow]"
            )
    elif has_proxy_env or "proxy_servers" in stdout_condarc:
        console.print("[yellow]⚠ Proxy configured but direct access blocked[/yellow]")
        if conda_found:
            console.print(
                "[dim]Package installation will use proxy - test with: conda search python[/dim]"
            )
        else:
            console.print(
                "[yellow]Install conda first, then run atomate2siesta-cluster setup with proxy flags[/yellow]"
            )
    else:
        console.print("[red]⚠ No internet access detected![/red]")
        console.print("\n[bold]This cluster appears to be air-gapped.[/bold]")

        if not conda_found:
            console.print(
                "\n[bold yellow]⚠ IMPORTANT: You need to install your own conda first![/bold yellow]"
            )
            console.print(
                "  (Module conda is read-only and won't work for package installation)"
            )
            console.print("\n[cyan]Install Miniconda in your HOME directory[/cyan]")
            console.print(
                "  Since the cluster is air-gapped, use one of these methods:"
            )
            console.print("\n  [bold]Option A: SSH SOCKS Proxy (simplest)[/bold]")
            console.print("    # On your LOCAL machine:")
            console.print("    ssh -D 9999 -N -f " + host)
            console.print("\n    # On the CLUSTER (via another SSH session):")
            console.print("    export http_proxy=http://127.0.0.1:9999")
            console.print("    export https_proxy=http://127.0.0.1:9999")
            console.print(
                "    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
            )
            console.print(
                "    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3"
            )
            console.print("    source $HOME/miniconda3/bin/activate")
            console.print("    conda init bash")
            console.print("\n  [bold]Option B: Squid Proxy (persistent)[/bold]")
            console.print("    # On your LOCAL machine:")
            console.print("    atomate2siesta-cluster squid start --port 9999")
            console.print("    ssh -R 9999:localhost:9999 -N -f " + host)
            console.print("\n    # On the CLUSTER (via another SSH session):")
            console.print("    export http_proxy=http://127.0.0.1:9999")
            console.print("    export https_proxy=http://127.0.0.1:9999")
            console.print(
                "    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
            )
            console.print(
                "    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3"
            )
            console.print("    # Add to ~/.bashrc for persistence:")
            console.print(
                "    echo 'export http_proxy=http://127.0.0.1:9999' >> ~/.bashrc"
            )
            console.print(
                "    echo 'export https_proxy=http://127.0.0.1:9999' >> ~/.bashrc"
            )
            console.print(
                "\n  [bold]Option C: Transfer Installer (no tunnel needed)[/bold]"
            )
            console.print("    # Download on your local machine:")
            console.print(
                "    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
            )
            console.print("    # Transfer to cluster:")
            console.print("    scp Miniconda3-latest-Linux-x86_64.sh " + host + ":~/")
            console.print("    # Install on cluster:")
            console.print(
                "    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3"
            )
            console.print("    source $HOME/miniconda3/bin/activate")
            console.print("    conda init bash")
            console.print("\n[cyan]After installing conda, run setup command:[/cyan]")
            console.print()
        console.print("  • With SSH SOCKS tunnel:")
        console.print(
            "    [cyan]atomate2siesta-cluster setup --host "
            + host
            + " --ssh-config --ssh-tunnel[/cyan]"
        )
        console.print("  • With Squid proxy:")
        console.print("    [cyan]atomate2siesta-cluster squid start --port 9999[/cyan]")
        console.print(
            "    [cyan]atomate2siesta-cluster setup --host "
            + host
            + " --ssh-config --use-squid[/cyan]"
        )
        console.print("  3. [cyan]Offline Environment[/cyan]:")
        console.print("     atomate2siesta-cluster build-offline --install-siesta")

    # Show package information
    console.print("\n[bold]Installed Packages:[/bold]\n")

    if packages_stdout:
        packages_table = Table(box=None)
        packages_table.add_column("Package", style="cyan")
        packages_table.add_column("Version", style="green")

        for line in packages_stdout.strip().split("\n"):
            if line.strip():
                parts = line.split()
                if len(parts) >= 2:
                    packages_table.add_row(parts[0], parts[1])

        console.print(packages_table)
    else:
        console.print("[yellow]No relevant packages found[/yellow]")

    console.print()


@cli.command()
def info():
    """Show information about cluster setup.

    This command displays usage information, examples, and common workflows.
    """
    console.print()

    header = Panel(
        Text("Remote Cluster Setup Helper", style="bold cyan", justify="center"),
        style="cyan",
    )
    console.print(header)

    console.print("\n[bold]Overview:[/bold]\n")
    console.print(
        "This tool helps you set up jobflow-remote on remote HPC clusters by:\n"
        "  • SSHing to the cluster\n"
        "  • Creating a conda environment in $HOME\n"
        "  • Installing jobflow-remote for job submission\n\n"
        "[dim]Note: atomate2siesta should be installed on your LOCAL machine,\n"
        "not on the cluster. Jobs are submitted from local to cluster.[/dim]"
    )

    console.print("\n[bold]Commands:[/bold]\n")

    commands_table = Table(box=None)
    commands_table.add_column("Command", style="cyan", no_wrap=True)
    commands_table.add_column("Description")

    commands_table.add_row("ssh-setup", "Manage SSH keys and config (add/status/test)")
    commands_table.add_row("setup", "Set up conda environment on remote cluster")
    commands_table.add_row("status", "Check status of remote environment")
    commands_table.add_row(
        "squid", "Manage Squid HTTP proxy (install/start/stop/status)"
    )
    commands_table.add_row(
        "build-offline", "Build offline environment for air-gapped clusters"
    )
    commands_table.add_row("info", "Show this information")

    console.print(commands_table)

    console.print("\n[bold]Authentication Methods:[/bold]\n")

    auth_table = Table(box=None)
    auth_table.add_column("Method", style="cyan")
    auth_table.add_column("Usage")

    auth_table.add_row("SSH Config", "--ssh-config (uses ~/.ssh/config)")
    auth_table.add_row("SSH Key", "--identity-file ~/.ssh/id_rsa")
    auth_table.add_row("Password", "--password (will prompt)")
    auth_table.add_row("SSH Agent", "No flags (uses configured keys)")

    console.print(auth_table)

    console.print("\n[bold]SSH Setup Commands (NEW!):[/bold]\n")

    ssh_setup_table = Table(box=None)
    ssh_setup_table.add_column("Command", style="cyan")
    ssh_setup_table.add_column("Description")

    ssh_setup_table.add_row(
        "ssh-setup add",
        "Add SSH config entry (generates keys, enables passwordless login)",
    )
    ssh_setup_table.add_row(
        "ssh-setup status", "Show SSH keys, config entries, and agent status"
    )
    ssh_setup_table.add_row("ssh-setup test", "Test connections to configured hosts")

    console.print(ssh_setup_table)
    console.print(
        "[dim]Tip: Use 'ssh-setup add' to avoid manually editing ~/.ssh/config![/dim]"
    )

    console.print("\n[bold]Examples:[/bold]\n")

    examples_panel = Panel(
        "# NEW! Set up SSH access (easiest way):\n"
        "[cyan]atomate2siesta-cluster ssh-setup add --alias mycluster --hostname cluster.edu --user myuser --generate-key --copy-id[/cyan]\n\n"
        "# Check your SSH configuration:\n"
        "[cyan]atomate2siesta-cluster ssh-setup status[/cyan]\n\n"
        "# Test SSH connection:\n"
        "[cyan]atomate2siesta-cluster ssh-setup test mycluster[/cyan]\n\n"
        "# Set up cluster with SSH config (RECOMMENDED):\n"
        "[cyan]atomate2siesta-cluster setup --host mycluster --ssh-config --install-siesta[/cyan]\n\n"
        "# Check status:\n"
        "[cyan]atomate2siesta-cluster status --host mycluster --ssh-config[/cyan]\n\n"
        "# Air-gapped cluster with squid proxy:\n"
        "[cyan]atomate2siesta-cluster squid install[/cyan]\n"
        "[cyan]atomate2siesta-cluster squid start[/cyan]\n"
        "[cyan]atomate2siesta-cluster setup --host mn5 --ssh-config --use-squid[/cyan]",
        style="green",
    )
    console.print(examples_panel)

    console.print("\n[bold]Prerequisites:[/bold]\n")
    console.print("  • SSH access to the cluster")
    console.print("  • Conda/Miniconda installed on the cluster")
    console.print("  • sshpass installed locally (if using password authentication)")

    console.print("\n[bold]Quick Start Workflow:[/bold]\n")

    workflow_panel = Panel(
        "1. Set up SSH access (EASY WAY - NEW!):\n"
        "   [cyan]atomate2siesta-cluster ssh-setup add \\\n"
        "       --alias mycluster \\\n"
        "       --hostname cluster.university.edu \\\n"
        "       --user myuser \\\n"
        "       --generate-key --copy-id[/cyan]\n\n"
        "   [dim]This generates SSH keys, creates config, and enables passwordless login![/dim]\n\n"
        "2. Check your SSH setup:\n"
        "   [cyan]atomate2siesta-cluster ssh-setup status[/cyan]\n"
        "   [cyan]atomate2siesta-cluster ssh-setup test mycluster[/cyan]\n\n"
        "3. Set up cluster environment:\n"
        "   [cyan]atomate2siesta-cluster setup --host mycluster --ssh-config --install-siesta[/cyan]\n\n"
        "4. SSH to cluster and configure jobflow-remote:\n"
        "   [cyan]ssh mycluster[/cyan]\n"
        "   [cyan]conda activate atomate2siesta[/cyan]\n"
        "   [cyan]jf project generate myproject[/cyan]\n"
        "   [cyan]jf admin reset[/cyan]\n\n"
        "5. Start runner on cluster:\n"
        "   [cyan]jf runner start -d[/cyan]\n\n"
        "6. On your LOCAL machine, install atomate2siesta:\n"
        "   [cyan]pip install atomate2siesta[/cyan]\n\n"
        "7. Submit jobs from your local machine:\n"
        "   [cyan]python submit_workflow.py[/cyan]",
        style="green",
    )
    console.print(workflow_panel)

    console.print()


def _build_offline_docker(
    output: str,
    env_name: str,
    python_version: str,
    install_siesta: bool,
):
    """Build offline environment using Docker for Linux compatibility."""
    import os
    import subprocess
    import tempfile

    # Check if Docker is installed
    console.print("[cyan]Checking Docker installation...[/cyan]")
    try:
        result = subprocess.run(
            ["docker", "--version"], capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            console.print("[red]✗ Docker not found![/red]")
            console.print("\n[yellow]Please install Docker:[/yellow]")
            console.print(
                "  macOS:   https://docs.docker.com/desktop/install/mac-install/"
            )
            console.print(
                "  Windows: https://docs.docker.com/desktop/install/windows-install/"
            )
            console.print("  Linux:   https://docs.docker.com/engine/install/")
            sys.exit(1)
        console.print(f"[green]✓ Docker found:[/green] {result.stdout.strip()}\n")
    except Exception as e:
        console.print(f"[red]✗ Docker check failed: {e}[/red]")
        sys.exit(1)

    # Check if Docker daemon is running
    console.print("[cyan]Checking Docker daemon...[/cyan]")
    try:
        result = subprocess.run(
            ["docker", "info"], capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            console.print("[red]✗ Docker daemon is not running![/red]")
            console.print(
                "\n[yellow]Please start Docker Desktop or Docker daemon[/yellow]"
            )
            sys.exit(1)
        console.print("[green]✓ Docker daemon is running[/green]\n")
    except Exception as e:
        console.print(f"[red]✗ Docker daemon check failed: {e}[/red]")
        sys.exit(1)

    # Create temporary directory for build context
    with tempfile.TemporaryDirectory() as tmpdir:
        console.print(f"[cyan]Creating build context in {tmpdir}...[/cyan]\n")

        # Create build script that will run inside Docker
        build_script = f"""#!/bin/bash
set -e

echo "Installing conda-pack in base environment..."
conda install -n base -y -c conda-forge conda-pack

echo "Creating conda environment '{env_name}'..."
conda create -y -n {env_name} python={python_version}

echo "Installing jobflow-remote..."
conda run -n {env_name} pip install jobflow-remote

echo "Installing atomate2[siesta]..."
"""

        # Install atomate2 with the SIESTA extra
        build_script += f"""
conda run -n {env_name} pip install 'atomate2[siesta]'
"""

        # Add SIESTA installation if requested
        if install_siesta:
            build_script += f"""
echo "Installing SIESTA..."
conda run -n {env_name} conda install -y -c conda-forge 'siesta=*=*mpich*' || echo "Warning: SIESTA installation failed"
"""

        # Add conda-pack step
        build_script += f"""
echo "Packing environment..."
conda run -n base conda-pack -n {env_name} -o /output/{output} --compress-level 6

echo "Build complete!"
ls -lh /output/{output}
"""

        # Write build script to temp directory
        script_path = os.path.join(tmpdir, "build.sh")
        with open(script_path, "w") as f:
            f.write(build_script)
        os.chmod(script_path, 0o755)

        # Get absolute path for output directory
        output_dir = os.path.abspath(os.path.dirname(output) or ".")
        output_filename = os.path.basename(output)

        console.print("[bold]Docker Build Configuration:[/bold]")
        console.print("  Base Image:        continuumio/miniconda3:latest")
        console.print(f"  Build Script:      {script_path}")
        console.print(f"  Output Directory:  {output_dir}")
        console.print(f"  Output File:       {output_filename}\n")

        # Run Docker build
        console.print(
            "[cyan]Starting Docker build (this may take 5-10 minutes)...[/cyan]\n"
        )

        docker_cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{tmpdir}:/build",
            "-v",
            f"{output_dir}:/output",
            "--platform",
            "linux/amd64",
            "continuumio/miniconda3:latest",
            "/bin/bash",
            "/build/build.sh",
        ]

        console.print("[dim]Running Docker command:[/dim]")
        console.print(f"[dim]{' '.join(docker_cmd)}[/dim]\n")

        try:
            # Run Docker with real-time output
            process = subprocess.Popen(
                docker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Stream output
            if process.stdout:
                for line in process.stdout:
                    console.print(f"[dim]│[/dim] {line.rstrip()}")

            process.wait()

            if process.returncode != 0:
                console.print("\n[red]✗ Docker build failed![/red]")
                sys.exit(1)

            console.print("\n[green]✓ Docker build completed successfully![/green]\n")

            # Check if output file exists
            output_path = os.path.join(output_dir, output_filename)
            if not os.path.exists(output_path):
                console.print(f"[red]✗ Output file not found: {output_path}[/red]")
                sys.exit(1)

            # Get file size
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            console.print(
                f"[green]✓ Created Linux-compatible tarball: {output_path}[/green]"
            )
            console.print(f"[green]✓ File size: {file_size_mb:.1f} MB[/green]\n")

            # Show next steps
            _show_offline_next_steps(output_filename, env_name)

        except KeyboardInterrupt:
            console.print("\n[yellow]Build interrupted by user[/yellow]")
            sys.exit(1)
        except Exception as e:
            console.print(f"\n[red]✗ Docker build error: {e}[/red]")
            sys.exit(1)


def _show_offline_next_steps(output: str, env_name: str):
    """Show next steps for using the offline environment."""
    next_steps = Panel(
        "[bold]Next Steps:[/bold]\n\n"
        f"1. Transfer to cluster:\n"
        f"   [cyan]scp {output} user@cluster:~/[/cyan]\n\n"
        f"2. SSH to cluster and unpack:\n"
        f"   [cyan]ssh user@cluster[/cyan]\n"
        f"   [cyan]mkdir -p ~/miniconda3/envs/{env_name}[/cyan]\n"
        f"   [cyan]tar -xzf {output} -C ~/miniconda3/envs/{env_name}[/cyan]\n\n"
        f"3. Activate and fix paths:\n"
        f"   [cyan]source ~/miniconda3/envs/{env_name}/bin/activate[/cyan]\n"
        f"   [cyan]conda-unpack[/cyan]\n\n"
        f"4. Verify installation:\n"
        f"   [cyan]python -c \"import jobflow_remote; import atomate2.siesta; print('Success!')\"[/cyan]\n\n"
        f"[bold]Or use automated transfer:[/bold]\n"
        f"   [cyan]atomate2siesta-cluster setup --host cluster --ssh-config --offline {output}[/cyan]",
        title="📦 Offline Environment Ready",
        style="green",
    )
    console.print(next_steps)


@cli.command()
@click.option(
    "--output",
    "-o",
    default="atomate2siesta.tar.gz",
    help="Output filename for packed environment (default: atomate2siesta.tar.gz)",
)
@click.option(
    "--env-name",
    default="atomate2siesta",
    help="Name for conda environment (default: atomate2siesta)",
)
@click.option(
    "--python-version",
    default="3.11",
    help="Python version for conda environment (default: 3.11)",
)
@click.option(
    "--install-siesta",
    is_flag=True,
    help="Include SIESTA in the packed environment (from conda-forge)",
)
@click.option(
    "--use-docker",
    is_flag=True,
    help="Build inside Docker container for Linux compatibility (recommended for macOS/Windows)",
)
def build_offline(
    output: str,
    env_name: str,
    python_version: str,
    install_siesta: bool,
    use_docker: bool,
):
    """Build offline conda environment for air-gapped clusters.

    This command builds a complete conda environment locally, packs it using
    conda-pack, and creates a tarball that can be transferred to clusters with
    no internet access (like MareNostrum 5).

    The packed environment includes:
    - jobflow-remote
    - atomate2siesta
    - SIESTA (optional, if --install-siesta is used)

    Requirements:
    - conda or miniconda installed
    - conda-pack (will be installed automatically in base environment if missing)
    - For Docker mode: Docker Desktop or Docker Engine

    Architecture Requirements:
    - Target cluster: Linux x86_64
    - Your build system: Must match target OR use --use-docker

    Docker Mode (Recommended for macOS/Windows):
    - Builds Linux-compatible environment using Docker container
    - Works on any platform (macOS, Windows, Linux)
    - Requires Docker Desktop or Docker Engine installed
    - conda-pack installed automatically inside container

    Native Mode:
    - Builds directly on your system
    - Only works if building on Linux x86_64
    - Fails on macOS/Windows without --use-docker
    - conda-pack installed automatically in base environment if not present

    Note: conda-pack is conda-only (not available via pip) and will be
    automatically installed to your conda base environment if not found.
    This is normal and required for packing environments.

    Examples
    --------
        # Docker build (works on macOS/Windows/Linux)
        atomate2siesta-cluster build-offline --use-docker --output mn5-env.tar.gz

        # Native build (only on Linux x86_64)
        atomate2siesta-cluster build-offline --output mn5-env.tar.gz --install-siesta

        # Docker build with SIESTA and SSH authentication
        atomate2siesta-cluster build-offline --use-docker --install-siesta
    """
    console.print(
        Panel.fit(
            "[bold]Build Offline Environment for Air-Gapped Clusters[/bold]\n"
            "For clusters like MareNostrum 5 with no internet access",
            style="bold blue",
        )
    )

    console.print("\n[bold]Configuration:[/bold]")
    console.print(f"  Environment Name:  {env_name}")
    console.print(f"  Python Version:    {python_version}")
    console.print(f"  Output File:       {output}")
    console.print(f"  Include SIESTA:    {'Yes' if install_siesta else 'No'}")
    console.print(f"  Use Docker:        {'Yes' if use_docker else 'No'}\n")

    # Check if running on Linux x86_64
    import platform

    current_system = platform.system()
    current_arch = platform.machine()

    # If using Docker, delegate to Docker build
    if use_docker:
        console.print(
            "[cyan]Using Docker to build Linux-compatible environment...[/cyan]\n"
        )
        _build_offline_docker(
            output=output,
            env_name=env_name,
            python_version=python_version,
            install_siesta=install_siesta,
        )
        return

    # Native build - check architecture compatibility
    if current_system != "Linux" or current_arch != "x86_64":
        console.print(
            Panel.fit(
                "[bold red]❌ ARCHITECTURE MISMATCH - BUILD WILL FAIL ON CLUSTER![/bold red]\n\n"
                f"[bold]Your System:[/bold]     {current_system} {current_arch}\n"
                f"[bold]Target Cluster:[/bold]  Linux x86_64\n\n"
                "[bold yellow]The Python binaries you build here CANNOT run on the cluster![/bold yellow]\n"
                "[dim]You will get: 'cannot execute binary file' errors[/dim]\n\n"
                "[bold cyan]Solutions:[/bold cyan]\n"
                "  1. [green]Use Docker (recommended):[/green]\n"
                f"     [cyan]atomate2siesta-cluster build-offline --use-docker -o {output}[/cyan]\n\n"
                "  2. [yellow]Build on a Linux x86_64 machine[/yellow]\n\n"
                "  3. [yellow]Use a cloud Linux instance or VM[/yellow]",
                style="bold red",
                title="⚠️  CRITICAL WARNING",
            )
        )
        console.print()
        if not Confirm.ask(
            "[bold red]Continue anyway?[/bold red] (tarball will NOT work on cluster)"
        ):
            console.print(
                "[red]Aborted. Use --use-docker for Linux compatibility.[/red]"
            )
            sys.exit(1)
        console.print("\n[yellow]⚠️  Proceeding with incompatible build...[/yellow]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Step 1: Check if conda is available
        task = progress.add_task("[cyan]Checking for conda...", total=None)
        result = subprocess.run(["which", "conda"], capture_output=True, text=True)
        if result.returncode != 0:
            progress.stop()
            console.print("[red]✗ conda not found![/red]")
            console.print("\nPlease install conda or miniconda:")
            console.print("  https://docs.conda.io/en/latest/miniconda.html")
            sys.exit(1)
        progress.update(task, completed=True)
        console.print("[green]✓ conda found[/green]")

        # Step 2: Check if conda-pack is installed in base environment
        task = progress.add_task("[cyan]Checking for conda-pack...", total=None)
        result = subprocess.run(
            ["conda", "list", "-n", "base", "conda-pack"],
            capture_output=True,
            text=True,
        )
        if "conda-pack" not in result.stdout:
            progress.stop()
            console.print(
                "[yellow]Installing conda-pack in base environment...[/yellow]"
            )
            console.print(
                "[dim]conda-pack is required for packing environments (conda-only tool)[/dim]"
            )
            console.print(
                "[dim]This is normal and only happens once - it will be installed to your base environment[/dim]\n"
            )
            result = subprocess.run(
                [
                    "conda",
                    "install",
                    "-n",
                    "base",
                    "-c",
                    "conda-forge",
                    "conda-pack",
                    "-y",
                ],
                capture_output=False,  # Show installation progress to user
                text=True,
            )
            if result.returncode != 0:
                console.print("[red]✗ Failed to install conda-pack![/red]")
                console.print(
                    "\n[yellow]Tip: You can pre-install it manually:[/yellow]"
                )
                console.print(
                    "  [cyan]conda install -n base -c conda-forge conda-pack[/cyan]"
                )
                sys.exit(1)
            console.print("\n[green]✓ conda-pack installed successfully[/green]\n")
            # Restart progress after installation
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            )
            progress.start()
            task = progress.add_task("[cyan]Resuming build...", total=None)
        progress.update(task, completed=True)
        console.print("[green]✓ conda-pack available[/green]")

        # Step 3: Create conda environment
        task = progress.add_task(
            f"[cyan]Creating conda environment '{env_name}'...", total=None
        )
        result = subprocess.run(
            ["conda", "create", "-n", env_name, f"python={python_version}", "-y"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            progress.stop()
            console.print(f"[red]✗ Failed to create environment '{env_name}'![/red]")
            console.print(f"[red]{result.stderr}[/red]")
            sys.exit(1)
        progress.update(task, completed=True)
        console.print(f"[green]✓ Created environment '{env_name}'[/green]")

        # Step 4: Install jobflow-remote
        task = progress.add_task("[cyan]Installing jobflow-remote...", total=None)
        console.print(
            "  [dim]Repository: https://pypi.org/project/jobflow-remote/[/dim]"
        )
        result = subprocess.run(
            ["conda", "run", "-n", env_name, "pip", "install", "jobflow-remote"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            progress.stop()
            console.print("[red]✗ Failed to install jobflow-remote![/red]")
            console.print(f"[red]{result.stderr}[/red]")
            sys.exit(1)
        progress.update(task, completed=True)
        console.print("[green]✓ Installed jobflow-remote[/green]")

        # Step 5: Install atomate2 with the SIESTA extra
        task = progress.add_task("[cyan]Installing atomate2[siesta]...", total=None)
        console.print("  [dim]Package: atomate2[siesta] (PyPI)[/dim]")

        result = subprocess.run(
            ["conda", "run", "-n", env_name, "pip", "install", "atomate2[siesta]"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            progress.stop()
            console.print("[red]✗ Failed to install atomate2[siesta]![/red]")
            console.print(f"[red]{result.stderr}[/red]")
            console.print("\n[yellow]Troubleshooting:[/yellow]")
            console.print(
                "  • Ensure the cluster can reach PyPI (internet access or a proxy)"
            )
            sys.exit(1)
        progress.update(task, completed=True)
        console.print("[green]✓ Installed atomate2[siesta][/green]")

        # Step 6: Install SIESTA (optional)
        if install_siesta:
            task = progress.add_task(
                "[cyan]Installing SIESTA from conda-forge...", total=None
            )
            result = subprocess.run(
                [
                    "conda",
                    "run",
                    "-n",
                    env_name,
                    "conda",
                    "install",
                    "-y",
                    "-c",
                    "conda-forge",
                    "siesta=*=*mpich*",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                progress.stop()
                console.print("[yellow]⚠ SIESTA installation failed[/yellow]")
                console.print(f"[yellow]{result.stderr}[/yellow]")
                console.print("\n[yellow]Continuing without SIESTA...[/yellow]")
            else:
                progress.update(task, completed=True)
                console.print("[green]✓ Installed SIESTA[/green]")

        # Step 7: Pack the environment
        task = progress.add_task(
            f"[cyan]Packing environment to {output}...", total=None
        )
        result = subprocess.run(
            [
                "conda",
                "run",
                "-n",
                "base",
                "conda-pack",
                "-n",
                env_name,
                "-o",
                output,
                "--compress-level",
                "6",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            progress.stop()
            console.print("[red]✗ Failed to pack environment![/red]")
            console.print(f"[red]{result.stderr}[/red]")
            sys.exit(1)
        progress.update(task, completed=True)
        console.print(f"[green]✓ Packed environment to {output}[/green]")

        # Step 8: Get file size
        import os

        file_size_mb = os.path.getsize(output) / (1024 * 1024)
        console.print(f"[green]✓ File size: {file_size_mb:.1f} MB[/green]")

    console.print(
        "\n[bold green]✓ Offline environment built successfully![/bold green]\n"
    )

    # Show next steps
    _show_offline_next_steps(output, env_name)

    # Cleanup option
    if Confirm.ask(f"\nRemove local environment '{env_name}'? (keeps {output})"):
        subprocess.run(
            ["conda", "env", "remove", "-n", env_name, "-y"], capture_output=True
        )
        console.print(f"[green]✓ Removed local environment '{env_name}'[/green]")


@cli.command()
@click.argument(
    "action",
    type=click.Choice(
        ["install", "uninstall", "start", "stop", "status", "restart", "clean"]
    ),
)
@click.option(
    "--port",
    default=9999,
    help="Port for squid proxy (default: 9999)",
)
@click.option(
    "--remove",
    is_flag=True,
    help="Remove old squid.conf before starting (use with 'start' action)",
)
@click.option(
    "--local",
    is_flag=True,
    help="Install Squid in user mode (no sudo required)",
)
@click.option(
    "--install-dir",
    default=None,
    help="Custom installation directory for local install (default: ~/.local/squid)",
)
@click.option(
    "--compile",
    is_flag=True,
    help="Compile squid from source (no sudo required, takes ~10 min)",
)
def squid(
    action: str, port: int, remove: bool, local: bool, install_dir: str, compile: bool
):
    """Manage Squid HTTP proxy for air-gapped clusters.

    Squid provides an HTTP/HTTPS proxy that conda and pip can use, allowing
    package installation on clusters that block direct internet access (like MN5).

    \b
    Actions:
      install    - Auto-install squid (system-wide or local user mode)
      uninstall  - Remove locally compiled squid installation
      start      - Start squid proxy on specified port
      stop       - Stop squid proxy
      status     - Show squid status
      restart    - Restart squid proxy
      clean      - Remove squid configuration file

    \b
    Common Usage:
      $ atomate2siesta-cluster squid install --local --compile   # Compile from source (no sudo!)
      $ atomate2siesta-cluster squid install                     # Install system-wide (requires sudo)
      $ atomate2siesta-cluster squid install --local             # User mode with existing binary
      $ atomate2siesta-cluster squid start                       # Start on default port (9999)
      $ atomate2siesta-cluster squid start --port 8080           # Custom port
      $ atomate2siesta-cluster squid status                      # Check if running
      $ atomate2siesta-cluster squid stop                        # Stop squid
      $ atomate2siesta-cluster squid restart                     # Restart on same port
      $ atomate2siesta-cluster squid clean                       # Remove config file
      $ atomate2siesta-cluster squid uninstall                   # Remove local installation

    \b
    Air-gapped Cluster Workflow (default port 9999):
      # On LOCAL machine:
      $ atomate2siesta-cluster squid start
      $ ssh -R 9999:localhost:9999 cluster-host

      # On CLUSTER (in another terminal):
      $ export http_proxy=http://127.0.0.1:9999
      $ export https_proxy=http://127.0.0.1:9999
      $ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    """
    console.print(
        Panel.fit(
            f"[bold]Squid HTTP Proxy Manager[/bold]\nAction: {action}",
            style="bold blue",
        )
    )
    console.print()

    if action == "install":
        console.print("[bold]Installing Squid HTTP Proxy[/bold]\n")

        # Local user-mode installation
        if local:
            import os
            import shutil
            import tarfile
            import urllib.request
            from pathlib import Path

            console.print(
                "[bold cyan]User Mode Installation (No Sudo Required)[/bold cyan]\n"
            )

            # If --compile flag, compile from source
            if compile:
                console.print(
                    "[bold yellow]Compiling Squid from source (takes ~10 minutes)[/bold yellow]\n"
                )

                # Set installation directory
                if install_dir:
                    squid_dir = Path(install_dir).expanduser()
                else:
                    squid_dir = Path.home() / ".local" / "squid"

                console.print(f"[cyan]Installation directory:[/cyan] {squid_dir}\n")

                # Squid version
                squid_version = "6.10"
                squid_url = f"http://www.squid-cache.org/Versions/v6/squid-{squid_version}.tar.gz"
                squid_file = f"squid-{squid_version}.tar.gz"

                # Create build directory
                build_dir = squid_dir / "build"
                build_dir.mkdir(parents=True, exist_ok=True)

                download_path = build_dir / squid_file

                # Download squid source
                console.print(
                    f"[yellow]Downloading squid {squid_version} source...[/yellow]"
                )
                console.print(f"[dim]URL: {squid_url}[/dim]\n")

                try:

                    def reporthook(blocknum, blocksize, totalsize):
                        readsofar = blocknum * blocksize
                        if totalsize > 0:
                            percent = readsofar * 100 / totalsize
                            s = f"\r[cyan]Progress:[/cyan] {percent:.1f}% ({readsofar // (1024 * 1024)} MB / {totalsize // (1024 * 1024)} MB)"
                            console.print(s, end="")
                            if readsofar >= totalsize:
                                console.print()

                    urllib.request.urlretrieve(squid_url, download_path, reporthook)
                    console.print("[green]✓ Download complete[/green]")

                except Exception as e:
                    console.print(f"\n[red]✗ Download failed: {e}[/red]")
                    console.print(
                        "\n[yellow]Alternative: Download manually from:[/yellow]"
                    )
                    console.print(f"  {squid_url}")
                    sys.exit(1)

                # Extract source
                console.print("\n[yellow]Extracting source code...[/yellow]")
                try:
                    with tarfile.open(download_path, "r:gz") as tar:
                        tar.extractall(build_dir)
                    console.print("[green]✓ Extraction complete[/green]")
                except Exception as e:
                    console.print(f"[red]✗ Extraction failed: {e}[/red]")
                    sys.exit(1)

                # Find extracted directory
                src_dir = build_dir / f"squid-{squid_version}"
                if not src_dir.exists():
                    console.print(f"[red]✗ Source directory not found: {src_dir}[/red]")
                    sys.exit(1)

                # Check for required build tools
                console.print("\n[yellow]Checking build dependencies...[/yellow]")
                required_tools = ["gcc", "g++", "make"]
                missing_tools = []

                for tool in required_tools:
                    try:
                        result = subprocess.run(
                            ["which", tool], capture_output=True, text=True, timeout=5
                        )
                        if result.returncode == 0:
                            console.print(f"  [green]✓ {tool}[/green]")
                        else:
                            missing_tools.append(tool)
                            console.print(f"  [red]✗ {tool} not found[/red]")
                    except Exception:
                        missing_tools.append(tool)
                        console.print(f"  [red]✗ {tool} not found[/red]")

                if missing_tools:
                    console.print(
                        Panel(
                            f"[bold red]Missing build tools: {', '.join(missing_tools)}[/bold red]\n\n"
                            "[bold]You need these tools to compile squid:[/bold]\n"
                            "  Ubuntu/Debian: [white]sudo apt-get install build-essential[/white]\n"
                            "  RHEL/CentOS:   [white]sudo yum groupinstall 'Development Tools'[/white]\n"
                            "  On HPC: [white]module load gcc[/white]\n\n"
                            "[bold yellow]Alternative: Use SSH tunneling instead (no compilation needed)[/bold yellow]\n"
                            "  [white]ssh -R 9999:localhost:9999 cluster-host[/white]",
                            style="red",
                            title="Build Dependencies Required",
                        )
                    )
                    sys.exit(1)

                # Configure
                console.print("\n[yellow]Configuring squid...[/yellow]")
                console.print("[dim]This may take 2-3 minutes...[/dim]\n")

                configure_cmd = [
                    "./configure",
                    f"--prefix={squid_dir}",
                    "--disable-dependency-tracking",
                    "--enable-inline",
                    "--disable-arch-native",
                    "--without-openssl",  # Simplify build
                    "--disable-ssl",
                    "--disable-ssl-crtd",
                ]

                try:
                    result = subprocess.run(
                        configure_cmd,
                        cwd=src_dir,
                        capture_output=True,
                        text=True,
                        timeout=300,  # 5 minutes
                    )

                    if result.returncode != 0:
                        console.print("[red]✗ Configuration failed[/red]")
                        console.print(
                            f"\n[yellow]Error output:[/yellow]\n{result.stderr[-1000:]}"
                        )
                        sys.exit(1)

                    console.print("[green]✓ Configuration complete[/green]")

                except subprocess.TimeoutExpired:
                    console.print("[red]✗ Configuration timed out[/red]")
                    sys.exit(1)
                except Exception as e:
                    console.print(f"[red]✗ Configuration error: {e}[/red]")
                    sys.exit(1)

                # Compile
                console.print(
                    "\n[yellow]Compiling squid (this takes ~10 minutes)...[/yellow]"
                )
                console.print("[dim]Be patient, this is normal...[/dim]\n")

                # Detect number of CPU cores for parallel compilation
                import multiprocessing

                n_cores = multiprocessing.cpu_count()
                console.print(f"[dim]Using {n_cores} cores for compilation[/dim]\n")

                try:
                    result = subprocess.run(
                        ["make", f"-j{n_cores}"],
                        cwd=src_dir,
                        capture_output=True,
                        text=True,
                        timeout=1200,  # 20 minutes
                    )

                    if result.returncode != 0:
                        console.print("[red]✗ Compilation failed[/red]")
                        console.print(
                            f"\n[yellow]Error output:[/yellow]\n{result.stderr[-1000:]}"
                        )
                        sys.exit(1)

                    console.print("[green]✓ Compilation complete[/green]")

                except subprocess.TimeoutExpired:
                    console.print("[red]✗ Compilation timed out[/red]")
                    sys.exit(1)
                except Exception as e:
                    console.print(f"[red]✗ Compilation error: {e}[/red]")
                    sys.exit(1)

                # Install
                console.print(
                    "\n[yellow]Installing squid to user directory...[/yellow]"
                )

                try:
                    result = subprocess.run(
                        ["make", "install"],
                        cwd=src_dir,
                        capture_output=True,
                        text=True,
                        timeout=300,
                    )

                    if result.returncode != 0:
                        console.print("[red]✗ Installation failed[/red]")
                        console.print(
                            f"\n[yellow]Error output:[/yellow]\n{result.stderr[-1000:]}"
                        )
                        sys.exit(1)

                    console.print("[green]✓ Installation complete[/green]")

                except Exception as e:
                    console.print(f"[red]✗ Installation error: {e}[/red]")
                    sys.exit(1)

                # Clean up build directory
                console.print("\n[yellow]Cleaning up...[/yellow]")
                try:
                    shutil.rmtree(build_dir)
                    console.print("[green]✓ Build directory removed[/green]")
                except Exception:
                    console.print("[yellow]⚠ Could not remove build directory[/yellow]")

                # Now continue with config creation (squid is in squid_dir/sbin/squid)
                bin_dir = squid_dir / "sbin"
                squid_binary = bin_dir / "squid"

                if not squid_binary.exists():
                    console.print(
                        f"[red]✗ Squid binary not found: {squid_binary}[/red]"
                    )
                    sys.exit(1)

                console.print("\n[green]✓ Squid compiled and installed![/green]")
                console.print(f"[dim]Binary: {squid_binary}[/dim]\n")

                # Create directories for runtime
                squid_config_dir = squid_dir / "etc"
                squid_cache_dir = squid_dir / "var" / "cache"
                squid_log_dir = squid_dir / "var" / "logs"
                squid_run_dir = squid_dir / "var" / "run"

                squid_cache_dir.mkdir(parents=True, exist_ok=True)
                squid_log_dir.mkdir(parents=True, exist_ok=True)
                squid_run_dir.mkdir(parents=True, exist_ok=True)

                # Create user-mode configuration
                squid_config_file = squid_config_dir / "squid.conf"

                config_content = f"""# Squid configuration for user mode (compiled from source)
# Port for HTTP proxy
http_port {port}

# Cache directory (user-writable)
cache_dir ufs {squid_cache_dir} 100 16 256

# Log files (user-writable)
access_log {squid_log_dir}/access.log
cache_log {squid_log_dir}/cache.log

# PID file (user-writable)
pid_filename {squid_run_dir}/squid.pid

# ACLs and access control
acl localhost src 127.0.0.1/32 ::1
acl Safe_ports port 80 443 8080 8443
acl SSL_ports port 443
acl CONNECT method CONNECT

# Allow access from localhost
http_access allow localhost

# Deny requests to unknown ports
http_access deny !Safe_ports

# Deny CONNECT to non-SSL ports
http_access deny CONNECT !SSL_ports

# Default deny
http_access deny all

# Refresh patterns
refresh_pattern ^ftp:     1440  20%  10080
refresh_pattern ^gopher:  1440   0%   1440
refresh_pattern -i (/cgi-bin/|\\?) 0 0% 0
refresh_pattern .         0    20%  4320
"""

                with open(squid_config_file, "w") as f:
                    f.write(config_content)

                console.print(
                    f"[green]✓ Configuration file created:[/green] {squid_config_file}"
                )

                # Create management scripts
                start_script = squid_dir / "start-squid.sh"
                start_content = f"""#!/bin/bash
# Start Squid (compiled from source)
echo "Initializing squid cache..."
{squid_binary} -f {squid_config_file} -z
echo "Starting Squid proxy on port {port}..."
{squid_binary} -f {squid_config_file}
sleep 1
echo "Squid started. Check status with: ps aux | grep squid"
echo ""
echo "Set these environment variables on the cluster:"
echo "  export http_proxy=http://127.0.0.1:{port}"
echo "  export https_proxy=http://127.0.0.1:{port}"
"""

                with open(start_script, "w") as f:
                    f.write(start_content)
                start_script.chmod(0o755)

                stop_script = squid_dir / "stop-squid.sh"
                stop_content = f"""#!/bin/bash
# Stop Squid
echo "Stopping Squid..."
{squid_binary} -f {squid_config_file} -k shutdown
sleep 1
pkill -u $USER squid 2>/dev/null
echo "Squid stopped."
"""

                with open(stop_script, "w") as f:
                    f.write(stop_content)
                stop_script.chmod(0o755)

                status_script = squid_dir / "status-squid.sh"
                status_content = f"""#!/bin/bash
# Check Squid status
echo "Checking Squid process..."
ps aux | grep squid | grep -v grep

echo ""
echo "Checking port {port}..."
lsof -i :{port} 2>/dev/null || echo "Port {port} not in use"

echo ""
echo "Recent log entries:"
tail -n 10 {squid_log_dir}/cache.log 2>/dev/null || echo "No log file found"
"""

                with open(status_script, "w") as f:
                    f.write(status_content)
                status_script.chmod(0o755)

                console.print("[green]✓ Management scripts created[/green]")

                # Show summary
                console.print(
                    Panel(
                        f"[bold green]✓ Squid compiled and installed successfully![/bold green]\n\n"
                        f"[bold cyan]Installation:[/bold cyan]\n"
                        f"  Squid:  [white]{squid_dir}[/white]\n"
                        f"  Binary: [white]{squid_binary}[/white]\n"
                        f"  Config: [white]{squid_config_file}[/white]\n\n"
                        f"[bold cyan]Start Squid:[/bold cyan]\n"
                        f"  [white]{start_script}[/white]\n\n"
                        f"[bold cyan]Stop Squid:[/bold cyan]\n"
                        f"  [white]{stop_script}[/white]\n\n"
                        f"[bold cyan]Check Status:[/bold cyan]\n"
                        f"  [white]{status_script}[/white]\n\n"
                        f"[bold cyan]Add to PATH (optional):[/bold cyan]\n"
                        f"  Add to ~/.bashrc:\n"
                        f'  [white]export PATH="{bin_dir}:$PATH"[/white]\n\n'
                        f"[bold cyan]Use with SSH Tunnel:[/bold cyan]\n"
                        f"  Local:  [white]{start_script}[/white]\n"
                        f"  Then:   [white]ssh -R {port}:localhost:{port} cluster-host[/white]\n"
                        f"  Cluster:\n"
                        f"    [white]export http_proxy=http://127.0.0.1:{port}[/white]\n"
                        f"    [white]export https_proxy=http://127.0.0.1:{port}[/white]",
                        style="green",
                        title="Compilation Complete",
                    )
                )

                sys.exit(0)

            # Check if squid binary exists anywhere (non-compile mode)
            try:
                squid_check = subprocess.run(
                    ["which", "squid"], capture_output=True, text=True, timeout=5
                )
            except Exception as e:
                console.print(f"[red]Error checking for squid: {e}[/red]")
                squid_check = subprocess.CompletedProcess(args=[], returncode=1)

            if squid_check.returncode != 0:
                console.print(
                    Panel(
                        "[bold yellow]Squid binary not found in PATH[/bold yellow]\n\n"
                        "[bold green]NEW! You can now compile squid from source (no sudo):[/bold green]\n"
                        "  [white]atomate2siesta-cluster squid install --local --compile[/white]\n"
                        "  Takes ~10 minutes, requires gcc/g++/make\n"
                        "  Perfect for HPC where you can't install packages!\n\n"
                        "[bold cyan]Alternative Options:[/bold cyan]\n\n"
                        "[cyan]1. Check for squid module (HPC)[/cyan]\n"
                        "  [white]module avail squid[/white]\n"
                        "  [white]module load squid[/white]\n"
                        "  Then retry: [white]atomate2siesta-cluster squid install --local[/white]\n\n"
                        "[cyan]2. SSH tunneling (easiest, no proxy needed!)[/cyan]\n"
                        "  [white]ssh -R 9999:localhost:9999 cluster-host[/white]\n"
                        "  See: [white]atomate2siesta-cluster setup --help[/white]\n\n"
                        "[cyan]3. Ask administrator for system install[/cyan]\n"
                        "  Ubuntu/Debian: [white]sudo apt-get install squid[/white]\n"
                        "  RHEL/CentOS:   [white]sudo yum install squid[/white]",
                        style="yellow",
                        title="Squid Binary Required",
                    )
                )
                sys.exit(1)

            # Squid binary exists, create user-mode config
            console.print("[green]✓ Found squid binary in PATH[/green]")
            try:
                result = subprocess.run(
                    ["squid", "-v"], capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    version = result.stdout.split("\n")[0]
                    console.print(f"[dim]{version}[/dim]\n")
            except Exception:
                console.print("[yellow]Could not get squid version[/yellow]\n")

            # Set installation directory
            if install_dir:
                squid_dir = Path(install_dir).expanduser()
            else:
                squid_dir = Path.home() / ".local" / "squid"

            console.print(f"[cyan]Configuration directory:[/cyan] {squid_dir}\n")

            # Create directories
            squid_dir.mkdir(parents=True, exist_ok=True)
            squid_config_dir = squid_dir / "etc"
            squid_cache_dir = squid_dir / "cache"
            squid_log_dir = squid_dir / "log"
            squid_run_dir = squid_dir / "run"

            squid_config_dir.mkdir(exist_ok=True)
            squid_cache_dir.mkdir(exist_ok=True)
            squid_log_dir.mkdir(exist_ok=True)
            squid_run_dir.mkdir(exist_ok=True)

            # Create user-mode squid configuration
            squid_config_file = squid_config_dir / "squid.conf"
            config_content = f"""# Squid configuration for user mode (no root required)
# Port for HTTP proxy
http_port {port}

# Cache directory (user-writable)
cache_dir ufs {squid_cache_dir} 100 16 256

# Log files (user-writable)
access_log {squid_log_dir}/access.log
cache_log {squid_log_dir}/cache.log
cache_store_log {squid_log_dir}/store.log

# PID file (user-writable)
pid_filename {squid_run_dir}/squid.pid

# User running squid (current user, not 'squid')
# cache_effective_user is not set (runs as current user)

# ACLs and access control
acl localhost src 127.0.0.1/32 ::1
acl to_localhost dst 127.0.0.0/8 0.0.0.0/32 ::1
acl Safe_ports port 80 443 8080 8443
acl SSL_ports port 443
acl CONNECT method CONNECT

# Allow access from localhost
http_access allow localhost

# Deny requests to unknown ports
http_access deny !Safe_ports

# Deny CONNECT to non-SSL ports
http_access deny CONNECT !SSL_ports

# Default deny
http_access deny all

# Coredump directory
coredump_dir {squid_dir}/core

# Leave coredumps in the first cache dir
refresh_pattern ^ftp:     1440  20%  10080
refresh_pattern ^gopher:  1440   0%   1440
refresh_pattern -i (/cgi-bin/|\\?) 0 0% 0
refresh_pattern .         0    20%  4320
"""

            with open(squid_config_file, "w") as f:
                f.write(config_content)

            console.print(
                f"[green]✓ Configuration file created:[/green] {squid_config_file}"
            )

            # Create start script
            start_script = squid_dir / "start-squid.sh"
            start_content = f"""#!/bin/bash
# Start Squid in user mode
echo "Starting Squid proxy on port {port}..."
squid -f {squid_config_file} -N -z  # Initialize cache
squid -f {squid_config_file}
sleep 1
echo "Squid started. Check status with: ps aux | grep squid"
echo ""
echo "Set these environment variables on the cluster:"
echo "  export http_proxy=http://127.0.0.1:{port}"
echo "  export https_proxy=http://127.0.0.1:{port}"
"""

            with open(start_script, "w") as f:
                f.write(start_content)
            start_script.chmod(0o755)

            # Create stop script
            stop_script = squid_dir / "stop-squid.sh"
            stop_content = f"""#!/bin/bash
# Stop Squid
echo "Stopping Squid..."
squid -f {squid_config_file} -k shutdown
sleep 1
# Force kill if still running
pkill -u $USER squid 2>/dev/null
echo "Squid stopped."
"""

            with open(stop_script, "w") as f:
                f.write(stop_content)
            stop_script.chmod(0o755)

            # Create status script
            status_script = squid_dir / "status-squid.sh"
            status_content = f"""#!/bin/bash
# Check Squid status
echo "Checking Squid process..."
ps aux | grep squid | grep -v grep

echo ""
echo "Checking port {port}..."
lsof -i :{port} 2>/dev/null || echo "Port {port} not in use"

echo ""
echo "Recent log entries:"
tail -n 10 {squid_log_dir}/cache.log 2>/dev/null || echo "No log file found"
"""

            with open(status_script, "w") as f:
                f.write(status_content)
            status_script.chmod(0o755)

            console.print("[green]✓ Start/stop/status scripts created[/green]")

            # Show installation summary
            console.print(
                Panel(
                    f"[bold green]✓ Squid configured for user mode![/bold green]\n\n"
                    f"[bold cyan]Configuration Directory:[/bold cyan]\n"
                    f"  Squid: [white]{squid_dir}[/white]\n"
                    f"  Config: [white]{squid_config_file}[/white]\n"
                    f"  Cache: [white]{squid_cache_dir}[/white]\n"
                    f"  Logs: [white]{squid_log_dir}[/white]\n\n"
                    f"[bold cyan]Start Squid:[/bold cyan]\n"
                    f"  [white]{start_script}[/white]\n"
                    f"  OR\n"
                    f"  [white]squid -f {squid_config_file}[/white]\n\n"
                    f"[bold cyan]Stop Squid:[/bold cyan]\n"
                    f"  [white]{stop_script}[/white]\n"
                    f"  OR\n"
                    f"  [white]squid -f {squid_config_file} -k shutdown[/white]\n\n"
                    f"[bold cyan]Check Status:[/bold cyan]\n"
                    f"  [white]{status_script}[/white]\n"
                    f"  OR\n"
                    f"  [white]ps aux | grep squid[/white]\n"
                    f"  [white]lsof -i :{port}[/white]\n\n"
                    f"[bold cyan]Use with SSH Tunnel:[/bold cyan]\n"
                    f"  On LOCAL: [white]{start_script}[/white]\n"
                    f"  Then: [white]ssh -R {port}:localhost:{port} cluster-host[/white]\n"
                    f"  On CLUSTER:\n"
                    f"    [white]export http_proxy=http://127.0.0.1:{port}[/white]\n"
                    f"    [white]export https_proxy=http://127.0.0.1:{port}[/white]\n\n"
                    f"[bold cyan]Next Steps:[/bold cyan]\n"
                    f"1. Start Squid: [white]{start_script}[/white]\n"
                    f"2. Verify: [white]ps aux | grep squid[/white]\n"
                    f"3. Test: [white]curl -x http://127.0.0.1:{port} https://google.com[/white]",
                    style="green",
                    title="User Mode Installation Complete",
                )
            )

            sys.exit(0)

        # Check if already installed (system-wide)
        if is_squid_installed():
            console.print("[green]✓ Squid is already installed[/green]")
            result = subprocess.run(["squid", "-v"], capture_output=True, text=True)
            if result.returncode == 0:
                version = result.stdout.split("\n")[0]
                console.print(f"[dim]{version}[/dim]")
            sys.exit(0)

        # Detect OS and install (system-wide with sudo)
        import platform

        os_type = platform.system()

        if os_type == "Darwin":  # macOS
            console.print("[cyan]Detected macOS - installing with Homebrew...[/cyan]")
            console.print("\nRunning: [cyan]brew install squid[/cyan]\n")

            result = subprocess.run(
                ["brew", "install", "squid"],
                capture_output=False,  # Show output to user
                text=True,
            )

            if result.returncode == 0:
                console.print("\n[green]✓ Squid installed successfully![/green]")
                console.print("\n[bold]Next steps:[/bold]")
                console.print(
                    "  1. Start squid: [cyan]atomate2siesta-cluster squid start[/cyan]"
                )
                console.print(
                    "  2. Run cluster setup: [cyan]atomate2siesta-cluster setup --host <host> --use-squid[/cyan]"
                )
            else:
                console.print("\n[red]✗ Failed to install squid[/red]")
                console.print("\nTry manually: [cyan]brew install squid[/cyan]")
                sys.exit(1)

        elif os_type == "Linux":
            console.print("[cyan]Detected Linux - checking package manager...[/cyan]")

            # Check if apt-get is available (Ubuntu/Debian)
            try:
                apt_check = subprocess.run(
                    ["which", "apt-get"], capture_output=True, text=True, timeout=5
                )
            except Exception:
                apt_check = subprocess.CompletedProcess(args=[], returncode=1)

            # Check if yum is available (CentOS/RHEL)
            try:
                yum_check = subprocess.run(
                    ["which", "yum"], capture_output=True, text=True, timeout=5
                )
            except Exception:
                yum_check = subprocess.CompletedProcess(args=[], returncode=1)

            # Check if dnf is available (Fedora/RHEL 8+)
            try:
                dnf_check = subprocess.run(
                    ["which", "dnf"], capture_output=True, text=True, timeout=5
                )
            except Exception:
                dnf_check = subprocess.CompletedProcess(args=[], returncode=1)

            if apt_check.returncode == 0:
                # Ubuntu/Debian - use apt-get
                console.print(
                    "[cyan]Detected apt package manager (Ubuntu/Debian)[/cyan]"
                )
                console.print(
                    "\nRunning: [cyan]sudo apt-get update && sudo apt-get install -y squid[/cyan]\n"
                )

                # Update package list first
                update_result = subprocess.run(
                    ["sudo", "apt-get", "update"],
                    capture_output=False,  # Show output to user
                    text=True,
                )

                if update_result.returncode != 0:
                    console.print(
                        "\n[yellow]⚠ Warning: apt-get update failed, continuing anyway...[/yellow]\n"
                    )

                # Install squid
                result = subprocess.run(
                    ["sudo", "apt-get", "install", "-y", "squid"],
                    capture_output=False,  # Show output to user
                    text=True,
                )

                if result.returncode == 0:
                    console.print("\n[green]✓ Squid installed successfully![/green]")
                    console.print("\n[bold]Next steps:[/bold]")
                    console.print(
                        "  1. Start squid: [cyan]atomate2siesta-cluster squid start[/cyan]"
                    )
                    console.print(
                        "  2. Run cluster setup: [cyan]atomate2siesta-cluster setup --host <host> --use-squid[/cyan]"
                    )
                else:
                    console.print("\n[red]✗ Failed to install squid[/red]")
                    console.print(
                        "\nTry manually: [cyan]sudo apt-get install squid[/cyan]"
                    )
                    sys.exit(1)

            elif yum_check.returncode == 0:
                # CentOS/RHEL - use yum
                console.print("[cyan]Detected yum package manager (CentOS/RHEL)[/cyan]")
                console.print("\nRunning: [cyan]sudo yum install -y squid[/cyan]\n")

                result = subprocess.run(
                    ["sudo", "yum", "install", "-y", "squid"],
                    capture_output=False,  # Show output to user
                    text=True,
                )

                if result.returncode == 0:
                    console.print("\n[green]✓ Squid installed successfully![/green]")
                    console.print("\n[bold]Next steps:[/bold]")
                    console.print(
                        "  1. Start squid: [cyan]atomate2siesta-cluster squid start[/cyan]"
                    )
                    console.print(
                        "  2. Run cluster setup: [cyan]atomate2siesta-cluster setup --host <host> --use-squid[/cyan]"
                    )
                else:
                    console.print("\n[red]✗ Failed to install squid[/red]")
                    console.print("\nTry manually: [cyan]sudo yum install squid[/cyan]")
                    sys.exit(1)

            elif dnf_check.returncode == 0:
                # Fedora/RHEL 8+ - use dnf
                console.print(
                    "[cyan]Detected dnf package manager (Fedora/RHEL 8+)[/cyan]"
                )
                console.print("\nRunning: [cyan]sudo dnf install -y squid[/cyan]\n")

                result = subprocess.run(
                    ["sudo", "dnf", "install", "-y", "squid"],
                    capture_output=False,  # Show output to user
                    text=True,
                )

                if result.returncode == 0:
                    console.print("\n[green]✓ Squid installed successfully![/green]")
                    console.print("\n[bold]Next steps:[/bold]")
                    console.print(
                        "  1. Start squid: [cyan]atomate2siesta-cluster squid start[/cyan]"
                    )
                    console.print(
                        "  2. Run cluster setup: [cyan]atomate2siesta-cluster setup --host <host> --use-squid[/cyan]"
                    )
                else:
                    console.print("\n[red]✗ Failed to install squid[/red]")
                    console.print("\nTry manually: [cyan]sudo dnf install squid[/cyan]")
                    sys.exit(1)
            else:
                # Unknown Linux distribution
                console.print(
                    "\n[yellow]Could not detect package manager automatically[/yellow]"
                )
                console.print("\n[yellow]Please install squid manually:[/yellow]")
                console.print(
                    "  Ubuntu/Debian: [cyan]sudo apt-get install squid[/cyan]"
                )
                console.print("  CentOS/RHEL 7: [cyan]sudo yum install squid[/cyan]")
                console.print("  RHEL 8+/Fedora: [cyan]sudo dnf install squid[/cyan]")
                console.print("  Arch Linux:    [cyan]sudo pacman -S squid[/cyan]")
                console.print("  openSUSE:      [cyan]sudo zypper install squid[/cyan]")
                sys.exit(1)

        else:
            console.print(f"[yellow]Unsupported OS: {os_type}[/yellow]")
            console.print("Please install squid manually for your system")
            sys.exit(1)

    elif action == "start":
        console.print(f"[bold]Starting Squid on port {port}[/bold]\n")

        if start_squid(port, remove_old_config=remove):
            console.print("\n[bold green]✓ Squid is running![/bold green]\n")
            console.print("[bold]Proxy URL:[/bold]")
            console.print(f"  [cyan]http://127.0.0.1:{port}[/cyan]\n")
            console.print("[bold]Use with cluster setup:[/bold]")
            console.print(
                "  [cyan]atomate2siesta-cluster setup --host <host> --use-squid[/cyan]\n"
            )
            console.print("[bold]Or set environment variables:[/bold]")
            console.print(f"  [cyan]export http_proxy=http://127.0.0.1:{port}[/cyan]")
            console.print(f"  [cyan]export https_proxy=http://127.0.0.1:{port}[/cyan]")
        else:
            console.print("\n[red]✗ Failed to start squid[/red]")

            # Check if squid is already running on any port
            try:
                result = subprocess.run(
                    ["pgrep", "-l", "squid"], capture_output=True, text=True
                )
                if result.returncode == 0 and result.stdout.strip():
                    console.print("\n[yellow]⚠ Squid is already running![/yellow]")
                    console.print(f"[dim]Process: {result.stdout.strip()}[/dim]\n")

                    # Try to detect which port it's on
                    lsof_result = subprocess.run(
                        ["lsof", "-i", "-P", "-n", "-sTCP:LISTEN"],
                        capture_output=True,
                        text=True,
                    )
                    if "squid" in lsof_result.stdout:
                        for line in lsof_result.stdout.split("\n"):
                            if "squid" in line and "LISTEN" in line:
                                console.print("[dim]Currently listening:[/dim]")
                                console.print(f"  [dim]{line}[/dim]\n")

                    console.print("[bold]To fix this:[/bold]")
                    console.print("  1. Stop squid properly:")
                    console.print(
                        "     [cyan]atomate2siesta-cluster squid stop[/cyan]\n"
                    )
                    console.print("  2. Or kill the process manually:")
                    console.print("     [cyan]pkill squid[/cyan]")
                    console.print("     Or by PID:")
                    console.print(
                        f"     [cyan]kill {result.stdout.split()[0]}[/cyan]\n"
                    )
                    console.print("  3. Then start on your desired port:")
                    console.print(
                        f"     [cyan]atomate2siesta-cluster squid start --port {port}[/cyan]"
                    )
                else:
                    # Squid not running, different error
                    console.print("\n[yellow]Troubleshooting:[/yellow]")
                    console.print("  • Check if port is already in use:")
                    console.print(f"    [cyan]lsof -i :{port}[/cyan]")
                    console.print("  • Try a different port:")
                    console.print(
                        "    [cyan]atomate2siesta-cluster squid start --port <number>[/cyan]"
                    )
                    console.print("  • Check squid logs for errors")
            except Exception:
                # Fallback if pgrep/lsof not available
                console.print("\n[yellow]Troubleshooting:[/yellow]")
                console.print("  • Stop any running squid:")
                console.print("    [cyan]atomate2siesta-cluster squid stop[/cyan]")
                console.print("  • Check if port is already in use:")
                console.print(f"    [cyan]lsof -i :{port}[/cyan]")
                console.print("  • Try a different port:")
                console.print(
                    "    [cyan]atomate2siesta-cluster squid start --port <number>[/cyan]"
                )
            sys.exit(1)

    elif action == "stop":
        console.print("[bold]Stopping Squid[/bold]\n")

        if stop_squid():
            console.print("\n[green]✓ Squid stopped successfully[/green]")
        else:
            console.print("\n[yellow]Squid may not be running[/yellow]")

    elif action == "status":
        from pathlib import Path

        console.print("[bold]Squid Status[/bold]\n")

        status_info = get_squid_status(port)

        # Check for locally compiled squid
        local_squid_dir = Path.home() / ".local" / "squid"
        local_squid_binary = local_squid_dir / "sbin" / "squid"
        local_squid_installed = local_squid_binary.exists()

        # Check for squid in PATH
        system_squid_installed = status_info["installed"]

        # Determine installation type
        if local_squid_installed:
            installation_type = "Local (compiled from source)"
            squid_binary_path = str(local_squid_binary)
        elif system_squid_installed:
            # Get system squid path
            try:
                result = subprocess.run(
                    ["which", "squid"], capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    installation_type = "System"
                    squid_binary_path = result.stdout.strip()
                else:
                    installation_type = "System"
                    squid_binary_path = "Unknown"
            except Exception:
                installation_type = "System"
                squid_binary_path = "Unknown"
        else:
            installation_type = "Not installed"
            squid_binary_path = None

        status_table = Table(show_header=False, box=None)
        status_table.add_column("Field", style="cyan")
        status_table.add_column("Value")

        # Show installation status
        if local_squid_installed or system_squid_installed:
            status_table.add_row("Installed", "[green]✓ Yes[/green]")
            status_table.add_row("Type", f"[yellow]{installation_type}[/yellow]")
            if squid_binary_path:
                status_table.add_row("Binary", f"[dim]{squid_binary_path}[/dim]")
        else:
            status_table.add_row("Installed", "[red]✗ No[/red]")

        # Show running status - always show actual running port
        actual_running_port = (
            status_info["actual_port"]
            if status_info["actual_port"]
            else (status_info["port"] if status_info["running"] else None)
        )

        if actual_running_port:
            status_table.add_row("Running", "[green]✓ Yes[/green]")
            status_table.add_row("Port", f"[green]{actual_running_port}[/green]")
            status_table.add_row(
                "Proxy URL", f"[green]http://127.0.0.1:{actual_running_port}[/green]"
            )
        else:
            status_table.add_row("Running", "[red]✗ No[/red]")
            status_table.add_row("Port", "[dim]Not running[/dim]")
            status_table.add_row("Proxy URL", "[dim]Not available[/dim]")

        console.print(status_table)
        console.print()

        # Show management scripts for locally compiled squid
        if local_squid_installed:
            start_script = local_squid_dir / "start-squid.sh"
            stop_script = local_squid_dir / "stop-squid.sh"
            status_script = local_squid_dir / "status-squid.sh"

            if start_script.exists():
                console.print("[bold cyan]Management Scripts:[/bold cyan]")
                console.print(f"  Start:  [white]{start_script}[/white]")
                if stop_script.exists():
                    console.print(f"  Stop:   [white]{stop_script}[/white]")
                if status_script.exists():
                    console.print(f"  Status: [white]{status_script}[/white]")
                console.print()

        # Check if something else is using the port
        port_in_use_by_other = False
        if not actual_running_port:
            try:
                lsof_result = subprocess.run(
                    ["lsof", "-i", f":{port}"], capture_output=True, text=True
                )
                if lsof_result.returncode == 0 and lsof_result.stdout.strip():
                    port_in_use_by_other = True
                    console.print(
                        f"[yellow]⚠ Port {port} is in use by another process:[/yellow]\n"
                    )
                    # Show the processes using the port (deduplicate PIDs since lsof shows IPv4 + IPv6)
                    lines = lsof_result.stdout.strip().split("\n")
                    seen_pids = set()
                    pids = []  # Store all PIDs for later use
                    if len(lines) > 1:  # Skip header
                        for line in lines[1:]:
                            parts = line.split()
                            if len(parts) >= 2:
                                cmd = parts[0]
                                pid = parts[1]

                                # Skip if we've already shown this PID
                                if pid in seen_pids:
                                    continue
                                seen_pids.add(pid)
                                pids.append(pid)

                                console.print(
                                    f"  [yellow]Process: {cmd} (PID: {pid})[/yellow]"
                                )

                                # Show full command for this PID
                                try:
                                    ps_result = subprocess.run(
                                        ["ps", "-p", pid, "-o", "command="],
                                        capture_output=True,
                                        text=True,
                                    )
                                    if ps_result.returncode == 0:
                                        full_cmd = ps_result.stdout.strip()
                                        console.print(
                                            f"  [dim]Command: {full_cmd}[/dim]\n"
                                        )
                                except Exception:
                                    pass

                    console.print("[bold]What is this?[/bold]")
                    if "ssh" in lsof_result.stdout.lower():
                        console.print(
                            "  • [yellow]SSH SOCKS proxy tunnel[/yellow] (ssh -D)"
                        )
                        console.print(
                            "  • Created by: [cyan]atomate2siesta-cluster setup --ssh-tunnel[/cyan]"
                        )
                        console.print("\n[bold]To use squid instead:[/bold]")
                        # Show all PIDs if multiple SSH tunnels
                        if len(pids) > 1:
                            console.print(
                                f"  1. Kill the SSH tunnels: [cyan]kill {' '.join(pids)}[/cyan]"
                            )
                        else:
                            console.print(
                                f"  1. Kill the SSH tunnel: [cyan]kill {pids[0]}[/cyan]"
                            )
                        console.print(
                            "  2. Start squid: [cyan]atomate2siesta-cluster squid start[/cyan]"
                        )
                    else:
                        console.print(
                            "  • [yellow]Another application[/yellow] using this port"
                        )
                        console.print("\n[bold]Options:[/bold]")
                        # Show all PIDs if multiple processes
                        if len(pids) > 1:
                            console.print(
                                f"  1. Kill them: [cyan]kill {' '.join(pids)}[/cyan]"
                            )
                        else:
                            console.print(f"  1. Kill it: [cyan]kill {pids[0]}[/cyan]")
                        console.print(
                            "  2. Use different port: [cyan]atomate2siesta-cluster squid start --port <other-port>[/cyan]"
                        )
            except Exception:
                pass

        if not status_info["installed"]:
            console.print("[yellow]Squid is not installed.[/yellow]")
            console.print(
                "\nInstall with: [cyan]atomate2siesta-cluster squid install[/cyan]"
            )
        elif actual_running_port:
            console.print(
                f"[green]✓ Squid is running on port {actual_running_port}![/green]"
            )
            console.print(
                f"\nProxy URL: [cyan]http://127.0.0.1:{actual_running_port}[/cyan]"
            )
            console.print("\n[dim]Use with SSH reverse tunnel:[/dim]")
            console.print(
                f"  [cyan]ssh -R {actual_running_port}:localhost:{actual_running_port} cluster-host[/cyan]"
            )
        elif not actual_running_port and not port_in_use_by_other:
            console.print("[yellow]Squid is not running and port is free.[/yellow]")
            console.print(
                "\nStart with: [cyan]atomate2siesta-cluster squid start[/cyan]"
            )
            console.print(
                "Or start on specific port: [cyan]atomate2siesta-cluster squid start --port 9999[/cyan]"
            )

    elif action == "restart":
        console.print("[bold]Restarting Squid[/bold]\n")

        stop_squid()
        import time

        time.sleep(1)

        if start_squid(port, remove_old_config=remove):
            console.print("\n[green]✓ Squid restarted successfully[/green]")
        else:
            console.print("\n[red]✗ Failed to restart squid[/red]")
            sys.exit(1)

    elif action == "clean":
        console.print("[bold]Cleaning Squid Configuration[/bold]\n")

        import os
        import shutil

        config_dir = os.path.expanduser("~/.atomate2siesta-cluster")
        config_file = os.path.join(config_dir, "squid.conf")

        if not os.path.exists(config_dir):
            console.print("[yellow]No configuration directory found[/yellow]")
            console.print(f"[dim]Expected location: {config_dir}[/dim]")
            sys.exit(0)

        # Show what will be removed
        console.print(f"[cyan]Configuration directory: {config_dir}[/cyan]")

        if os.path.exists(config_file):
            # Read and show current port
            try:
                with open(config_file) as f:
                    for line in f:
                        if line.strip().startswith("http_port"):
                            parts = line.strip().split(":")
                            if len(parts) >= 2:
                                current_port = parts[-1]
                                console.print(
                                    f"[dim]Current port: {current_port}[/dim]"
                                )
                            break
            except Exception:
                pass

        # Remove the entire directory
        try:
            shutil.rmtree(config_dir)
            console.print("\n[green]✓ Configuration directory removed[/green]")
            console.print(
                "\n[dim]Next time you run 'squid start', fresh config will be created[/dim]"
            )
        except Exception as e:
            console.print(f"\n[red]✗ Failed to remove directory: {e}[/red]")
            sys.exit(1)

    elif action == "uninstall":
        console.print("[bold]Uninstalling Locally Compiled Squid[/bold]\n")

        import os
        import shutil
        from pathlib import Path

        # Check for locally compiled squid
        local_squid_dir = Path.home() / ".local" / "squid"
        local_squid_binary = local_squid_dir / "sbin" / "squid"

        if not local_squid_dir.exists():
            console.print("[yellow]No local squid installation found[/yellow]")
            console.print(f"[dim]Expected location: {local_squid_dir}[/dim]")
            console.print(
                "\n[cyan]Note:[/cyan] This only removes locally compiled squid."
            )
            console.print(
                "[dim]System squid (installed via package manager) is not affected.[/dim]"
            )
            sys.exit(0)

        # Check if squid is running
        if is_squid_running(port):
            console.print(
                f"[yellow]⚠️  Squid is currently running on port {port}[/yellow]"
            )
            console.print(
                "[yellow]   Stopping squid before uninstallation...[/yellow]\n"
            )
            stop_squid()
            console.print()

        # Show what will be removed
        console.print(f"[cyan]Installation directory:[/cyan] {local_squid_dir}")

        # Calculate directory size
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(local_squid_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
            size_mb = total_size / (1024 * 1024)
            console.print(f"[dim]Size: {size_mb:.1f} MB[/dim]")
        except Exception:
            pass

        # List contents
        console.print("\n[cyan]Contents:[/cyan]")
        try:
            subdirs = [d.name for d in local_squid_dir.iterdir() if d.is_dir()]
            for subdir in sorted(subdirs):
                console.print(f"  • {subdir}/")
        except Exception:
            pass

        # Confirmation
        console.print("\n[bold yellow]⚠️  This will permanently delete:[/bold yellow]")
        console.print(f"  {local_squid_dir}")
        console.print("\n[dim]This action cannot be undone.[/dim]")

        confirm = input("\nProceed with uninstallation? [y/N]: ").strip().lower()

        if confirm not in ["y", "yes"]:
            console.print("\n[yellow]Uninstallation cancelled[/yellow]")
            sys.exit(0)

        # Remove the squid directory
        console.print("\n[yellow]Removing squid installation...[/yellow]")
        try:
            shutil.rmtree(local_squid_dir)
            console.print("[green]✓ Local squid installation removed[/green]")
        except Exception as e:
            console.print(f"[red]✗ Failed to remove directory: {e}[/red]")
            sys.exit(1)

        # Ask about config directory
        squid_config_dir = Path.home() / ".atomate2siesta-cluster"
        if squid_config_dir.exists():
            console.print("\n[cyan]Configuration directory found:[/cyan]")
            console.print(f"  {squid_config_dir}")
            console.print("[dim]Contains squid.conf used for proxy configuration[/dim]")

            remove_config = (
                input("\nRemove configuration directory too? [y/N]: ").strip().lower()
            )

            if remove_config in ["y", "yes"]:
                try:
                    shutil.rmtree(squid_config_dir)
                    console.print("[green]✓ Configuration directory removed[/green]")
                except Exception as e:
                    console.print(f"[yellow]⚠️  Could not remove config: {e}[/yellow]")
            else:
                console.print("[dim]Configuration directory kept[/dim]")

        console.print("\n[green]✓ Uninstallation complete[/green]")
        console.print(
            "\n[cyan]To reinstall:[/cyan] [dim]atomate2siesta-cluster squid install --local --compile[/dim]"
        )


@cli.group()
def ssh_setup():
    """Manage SSH keys and config for cluster access.

    \b
    Subcommands:
      add     - Add new SSH config entry
      status  - Show SSH keys and config entries
      test    - Test SSH connections
    """


@ssh_setup.command()
@click.option(
    "--alias",
    required=True,
    help="SSH config alias name (e.g., 'mycluster')",
)
@click.option(
    "--hostname",
    required=True,
    help="Remote cluster hostname or IP address",
)
@click.option(
    "--user",
    help="Username for SSH connection (defaults to current user)",
)
@click.option(
    "--port",
    default=22,
    help="SSH port (default: 22)",
)
@click.option(
    "--key-file",
    default="~/.ssh/id_rsa",
    help="Path to SSH private key (default: ~/.ssh/id_rsa)",
)
@click.option(
    "--generate-key",
    is_flag=True,
    help="Generate new SSH key pair if it doesn't exist",
)
@click.option(
    "--copy-id",
    is_flag=True,
    help="Copy public key to remote server (enables passwordless login)",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing SSH config entry",
)
def add(
    alias: str,
    hostname: str,
    user: str | None,
    port: int,
    key_file: str,
    generate_key: bool,
    copy_id: bool,
    overwrite: bool,
):
    """Add new SSH config entry for cluster access.

    This command helps you set up passwordless SSH access to remote clusters by:
      - Generating SSH key pairs (if needed)
      - Creating/updating ~/.ssh/config entries
      - Copying public keys to remote servers

    \b
    Common Usage:
      $ atomate2siesta-cluster ssh-setup add --alias mycluster --hostname cluster.edu --user myuser
      $ atomate2siesta-cluster ssh-setup add --alias mn5 --hostname mn5.bsc.es --user myuser --generate-key
      $ atomate2siesta-cluster ssh-setup add --alias hpc --hostname hpc.uni.edu --copy-id

    \b
    After setup, you can connect with:
      $ ssh mycluster
      $ atomate2siesta-cluster setup --host mycluster --ssh-config
    """
    from pathlib import Path

    console.print("\n[bold cyan]SSH Configuration Setup[/bold cyan]\n")

    # Expand ~ in key_file path
    key_file_path = Path(key_file).expanduser()
    pub_key_path = key_file_path.with_suffix(".pub")
    ssh_dir = Path.home() / ".ssh"
    config_path = ssh_dir / "config"

    # Get username
    if not user:
        user = getpass.getuser()
        console.print(f"[dim]Using current user: {user}[/dim]\n")

    # Show configuration
    config_table = Table(show_header=False, box=None)
    config_table.add_column("Field", style="cyan")
    config_table.add_column("Value")
    config_table.add_row("Alias", alias)
    config_table.add_row("Hostname", hostname)
    config_table.add_row("User", user)
    config_table.add_row("Port", str(port))
    config_table.add_row("Key File", str(key_file_path))
    console.print(config_table)
    console.print()

    # Step 1: Check/Create .ssh directory
    if not ssh_dir.exists():
        console.print("[cyan]Creating ~/.ssh directory...[/cyan]")
        ssh_dir.mkdir(mode=0o700)
        console.print("[green]✓ Created ~/.ssh directory[/green]\n")
    else:
        console.print("[green]✓ ~/.ssh directory exists[/green]\n")

    # Step 2: Check/Generate SSH keys
    if not key_file_path.exists():
        console.print(f"[yellow]SSH key not found: {key_file_path}[/yellow]\n")

        if generate_key or Confirm.ask(
            f"Generate new SSH key at {key_file_path}?", default=True
        ):
            console.print("[cyan]Generating SSH key pair...[/cyan]")

            # Generate SSH key
            result = subprocess.run(
                [
                    "ssh-keygen",
                    "-t",
                    "rsa",
                    "-b",
                    "4096",
                    "-f",
                    str(key_file_path),
                    "-N",
                    "",  # No passphrase
                    "-C",
                    f"{user}@{hostname}",
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                console.print(f"[green]✓ Generated SSH key: {key_file_path}[/green]")
                console.print(f"[green]✓ Public key: {pub_key_path}[/green]\n")

                # Set correct permissions
                key_file_path.chmod(0o600)
                pub_key_path.chmod(0o644)
            else:
                console.print("[red]✗ Failed to generate SSH key[/red]")
                console.print(f"[red]{result.stderr}[/red]")
                sys.exit(1)
        else:
            console.print(
                "[yellow]Cancelled. Please provide an existing SSH key.[/yellow]"
            )
            sys.exit(0)
    else:
        console.print(f"[green]✓ SSH key exists: {key_file_path}[/green]\n")

    # Step 3: Check/Create SSH config file
    if not config_path.exists():
        console.print("[cyan]Creating ~/.ssh/config file...[/cyan]")
        config_path.touch(mode=0o600)
        console.print("[green]✓ Created ~/.ssh/config[/green]\n")
    else:
        console.print("[green]✓ ~/.ssh/config exists[/green]\n")

    # Step 4: Check if alias already exists in config
    alias_exists = False
    if config_path.exists():
        with open(config_path) as f:
            config_content = f.read()
            if (
                f"Host {alias}\n" in config_content
                or f"Host {alias} " in config_content
            ):
                alias_exists = True

    if alias_exists and not overwrite:
        console.print(
            f"[yellow]⚠ SSH config entry for '{alias}' already exists[/yellow]\n"
        )
        if not Confirm.ask("Overwrite existing entry?", default=False):
            console.print("[yellow]Keeping existing configuration.[/yellow]")
            console.print("\n[dim]To overwrite, use: --overwrite flag[/dim]")
        else:
            overwrite = True

    # Step 5: Add/Update SSH config entry
    if not alias_exists or overwrite:
        console.print(f"[cyan]Adding SSH config entry for '{alias}'...[/cyan]")

        # Prepare the config entry
        config_entry = f"""
# Added by atomate2siesta-cluster ssh-setup
Host {alias}
    HostName {hostname}
    User {user}
    Port {port}
    IdentityFile {key_file_path}
    ForwardAgent yes
    ServerAliveInterval 60
    ServerAliveCountMax 3

"""

        if alias_exists and overwrite:
            # Remove old entry and add new one
            with open(config_path) as f:
                lines = f.readlines()

            # Find and remove old entry
            new_lines = []
            skip = False
            for line in lines:
                if line.strip().startswith(f"Host {alias}"):
                    skip = True
                    continue
                if skip and line.strip().startswith("Host "):
                    skip = False

                if not skip:
                    new_lines.append(line)

            # Write back without old entry
            with open(config_path, "w") as f:
                f.writelines(new_lines)

        # Append new entry
        with open(config_path, "a") as f:
            f.write(config_entry)

        console.print(f"[green]✓ Added SSH config entry for '{alias}'[/green]\n")

    # Step 6: Show the config entry
    console.print(
        Panel(
            f"[bold]SSH Config Entry:[/bold]\n\n"
            f"Host {alias}\n"
            f"    HostName {hostname}\n"
            f"    User {user}\n"
            f"    Port {port}\n"
            f"    IdentityFile {key_file_path}\n"
            f"    ForwardAgent yes\n"
            f"    ServerAliveInterval 60",
            style="green",
            title="Configuration",
        )
    )
    console.print()

    # Step 7: Copy public key to remote server (optional)
    if copy_id:
        console.print(f"[cyan]Copying public key to {hostname}...[/cyan]")
        console.print("[dim]You may be prompted for your password[/dim]\n")

        result = subprocess.run(
            ["ssh-copy-id", "-i", str(pub_key_path), f"{user}@{hostname}"],
            capture_output=False,  # Show output to user
            text=True,
        )

        if result.returncode == 0:
            console.print("\n[green]✓ Public key copied successfully![/green]")
            console.print("[green]✓ Passwordless login is now enabled[/green]\n")
        else:
            console.print("\n[yellow]⚠ Failed to copy public key[/yellow]")
            console.print("\n[yellow]You can try manually:[/yellow]")
            console.print(
                f"  [cyan]ssh-copy-id -i {pub_key_path} {user}@{hostname}[/cyan]\n"
            )
    elif not alias_exists or overwrite:
        console.print(
            "[yellow]Tip: Use --copy-id to enable passwordless login[/yellow]\n"
        )

    # Step 8: Test connection
    if Confirm.ask("Test SSH connection now?", default=True):
        console.print(f"\n[cyan]Testing connection to {alias}...[/cyan]")

        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", alias, "echo", "Connection successful!"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            console.print("[green]✓ Connection successful![/green]\n")
        else:
            console.print("[yellow]⚠ Connection test failed[/yellow]")
            console.print(f"[yellow]{result.stderr}[/yellow]")
            console.print("\n[yellow]Troubleshooting:[/yellow]")
            console.print("  • Check if the hostname is correct")
            console.print("  • Verify network connectivity")
            if not copy_id:
                console.print("  • Try copying your public key: --copy-id")
            console.print(f"  • Test manually: [cyan]ssh {alias}[/cyan]\n")

    # Success message
    console.print(
        Panel(
            f"[bold]SSH Setup Complete![/bold]\n\n"
            f"You can now connect using:\n"
            f"  [cyan]ssh {alias}[/cyan]\n\n"
            f"Use with atomate2siesta-cluster:\n"
            f"  [cyan]atomate2siesta-cluster setup --host {alias} --ssh-config[/cyan]\n"
            f"  [cyan]atomate2siesta-cluster status --host {alias} --ssh-config[/cyan]",
            title="Success",
            style="green",
        )
    )
    console.print()


@ssh_setup.command(name="status")
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed information",
)
def ssh_status(verbose: bool):
    """Show SSH keys, config entries, and connection status.

    Displays information about your SSH setup including:
      - SSH keys in ~/.ssh/
      - Configured hosts in ~/.ssh/config
      - SSH agent status

    \b
    Common Usage:
      $ atomate2siesta-cluster ssh-setup status
      $ atomate2siesta-cluster ssh-setup status -v    # Verbose mode
    """
    import os
    from pathlib import Path

    console.print("\n[bold cyan]SSH Configuration Status[/bold cyan]\n")

    ssh_dir = Path.home() / ".ssh"
    config_path = ssh_dir / "config"

    # Step 1: Check SSH directory
    if not ssh_dir.exists():
        console.print("[yellow]✗ ~/.ssh directory does not exist[/yellow]")
        console.print(
            "\n[dim]Run: [cyan]mkdir -p ~/.ssh && chmod 700 ~/.ssh[/cyan][/dim]\n"
        )
        sys.exit(1)
    else:
        console.print("[green]✓ ~/.ssh directory exists[/green]\n")

    # Step 2: List SSH keys
    console.print("[bold]SSH Keys:[/bold]\n")

    key_types = ["id_rsa", "id_ed25519", "id_ecdsa", "id_dsa"]
    found_keys = []

    keys_table = Table(show_header=True, box=None)
    keys_table.add_column("Key File", style="cyan")
    keys_table.add_column("Type", style="yellow")
    keys_table.add_column("Size", style="green")
    keys_table.add_column("Public Key", style="dim")

    for key_type in key_types:
        private_key = ssh_dir / key_type
        public_key = ssh_dir / f"{key_type}.pub"

        if private_key.exists():
            found_keys.append(key_type)

            # Get file size
            size = private_key.stat().st_size
            size_str = f"{size} bytes"

            # Check if public key exists
            pub_status = "✓" if public_key.exists() else "✗ Missing"

            # Get key type from file
            try:
                result = subprocess.run(
                    ["ssh-keygen", "-l", "-f", str(private_key)],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    # Parse output: "2048 SHA256:... user@host (RSA)"
                    parts = result.stdout.split()
                    if len(parts) >= 4:
                        key_bits = parts[0]
                        key_algo = parts[-1].strip("()")
                        type_str = f"{key_algo} {key_bits}-bit"
                    else:
                        type_str = "Unknown"
                else:
                    type_str = "Unknown"
            except Exception:
                type_str = "Unknown"

            keys_table.add_row(str(private_key), type_str, size_str, pub_status)

    if found_keys:
        console.print(keys_table)
        console.print()
    else:
        console.print("[yellow]No SSH keys found in ~/.ssh/[/yellow]")
        console.print(
            "\n[dim]Generate a key: [cyan]atomate2siesta-cluster ssh-setup add --alias myhost --hostname host.edu --generate-key[/cyan][/dim]\n"
        )

    # Step 3: Check SSH agent
    console.print("[bold]SSH Agent:[/bold]\n")

    ssh_auth_sock = os.environ.get("SSH_AUTH_SOCK")
    if ssh_auth_sock:
        console.print("[green]✓ SSH agent is running[/green]")
        if verbose:
            console.print(f"[dim]  Socket: {ssh_auth_sock}[/dim]")

        # List loaded keys
        result = subprocess.run(["ssh-add", "-l"], capture_output=True, text=True)

        if result.returncode == 0:
            loaded_keys = result.stdout.strip().split("\n")
            console.print(f"[green]✓ {len(loaded_keys)} key(s) loaded[/green]")
            if verbose:
                for key in loaded_keys:
                    console.print(f"[dim]  {key}[/dim]")
        else:
            console.print("[yellow]  No keys loaded in agent[/yellow]")
            if verbose:
                console.print("[dim]  Add keys: ssh-add ~/.ssh/id_rsa[/dim]")
    else:
        console.print("[yellow]✗ SSH agent is not running[/yellow]")
        if verbose:
            console.print("[dim]  Start agent: eval $(ssh-agent)[/dim]")

    console.print()

    # Step 4: Show SSH config entries
    console.print("[bold]SSH Config Entries (~/.ssh/config):[/bold]\n")

    if not config_path.exists():
        console.print("[yellow]✗ ~/.ssh/config does not exist[/yellow]")
        console.print(
            "\n[dim]Create one: [cyan]atomate2siesta-cluster ssh-setup add --alias myhost --hostname host.edu[/cyan][/dim]\n"
        )
    else:
        # Parse SSH config file
        config_entries: list[dict[str, str | dict[str, str]]] = []
        current_entry: dict[str, str | dict[str, str]] | None = None

        with open(config_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("Host ") and not line.startswith("Host *"):
                    if current_entry:
                        config_entries.append(current_entry)
                    # Extract host alias(es)
                    aliases = line[5:].strip().split()
                    current_entry = {"alias": aliases[0], "details": {}}
                elif current_entry and line:
                    # Parse config option
                    if " " in line or "\t" in line:
                        parts = line.split(None, 1)
                        if len(parts) == 2:
                            key, value = parts
                            details_dict = current_entry["details"]
                            assert isinstance(details_dict, dict)
                            details_dict[key.lower()] = value

            # Add last entry
            if current_entry:
                config_entries.append(current_entry)

        if config_entries:
            config_table = Table(show_header=True, box=None)
            config_table.add_column("Alias", style="cyan", no_wrap=True)
            config_table.add_column("HostName", style="green")
            config_table.add_column("User", style="yellow")
            config_table.add_column("Port", style="dim")
            if verbose:
                config_table.add_column("IdentityFile", style="dim")

            for entry in config_entries:
                alias = entry["alias"]
                details = entry["details"]
                # Type cast for mypy
                assert isinstance(details, dict)
                hostname = details.get("hostname", "-")
                user = details.get("user", "-")
                port = details.get("port", "22")
                identity_file = details.get("identityfile", "-")

                if verbose:
                    config_table.add_row(alias, hostname, user, port, identity_file)
                else:
                    config_table.add_row(alias, hostname, user, port)

            console.print(config_table)
            console.print(
                f"\n[dim]Found {len(config_entries)} host(s) configured[/dim]\n"
            )

            # Show usage hint
            console.print("[bold]Usage:[/bold]")
            console.print(f"  Connect: [cyan]ssh {config_entries[0]['alias']}[/cyan]")
            console.print(
                f"  Test: [cyan]atomate2siesta-cluster ssh-setup test {config_entries[0]['alias']}[/cyan]"
            )
            console.print()
        else:
            console.print("[yellow]No host entries found in ~/.ssh/config[/yellow]")
            console.print(
                "\n[dim]Add one: [cyan]atomate2siesta-cluster ssh-setup add --alias myhost --hostname host.edu[/cyan][/dim]\n"
            )


@ssh_setup.command()
@click.argument("alias", required=False)
@click.option(
    "--all",
    "-a",
    is_flag=True,
    help="Test all configured hosts",
)
def test(alias: str | None, all: bool):
    """Test SSH connection to configured hosts.

    \b
    Common Usage:
      $ atomate2siesta-cluster ssh-setup test mycluster
      $ atomate2siesta-cluster ssh-setup test --all    # Test all hosts
    """
    from pathlib import Path

    console.print("\n[bold cyan]SSH Connection Test[/bold cyan]\n")

    ssh_dir = Path.home() / ".ssh"
    config_path = ssh_dir / "config"

    if not config_path.exists():
        console.print("[red]✗ ~/.ssh/config does not exist[/red]")
        sys.exit(1)

    # Parse SSH config to get hosts
    config_entries = []
    current_entry = None

    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("Host ") and not line.startswith("Host *"):
                if current_entry:
                    config_entries.append(current_entry)
                aliases = line[5:].strip().split()
                current_entry = {"alias": aliases[0], "hostname": None}
            elif current_entry and line.startswith("HostName"):
                parts = line.split(None, 1)
                if len(parts) == 2:
                    current_entry["hostname"] = parts[1]

        if current_entry:
            config_entries.append(current_entry)

    if not config_entries:
        console.print("[yellow]No hosts configured in ~/.ssh/config[/yellow]")
        sys.exit(1)

    # Determine which hosts to test
    hosts_to_test = []
    if all:
        hosts_to_test = config_entries
    elif alias:
        # Find specific host
        found = False
        for entry in config_entries:
            if entry["alias"] == alias:
                hosts_to_test = [entry]
                found = True
                break
        if not found:
            console.print(f"[red]✗ Host '{alias}' not found in ~/.ssh/config[/red]")
            console.print("\n[yellow]Available hosts:[/yellow]")
            for entry in config_entries:
                console.print(f"  - {entry['alias']}")
            sys.exit(1)
    else:
        # No alias specified and not --all
        console.print("[yellow]Please specify a host alias or use --all[/yellow]")
        console.print("\n[yellow]Available hosts:[/yellow]")
        for entry in config_entries:
            console.print(f"  - {entry['alias']}")
        console.print(
            "\n[dim]Example: [cyan]atomate2siesta-cluster ssh-setup test mycluster[/cyan][/dim]"
        )
        sys.exit(1)

    # Test connections
    results_table = Table(show_header=True, box=None)
    results_table.add_column("Alias", style="cyan")
    results_table.add_column("HostName", style="dim")
    results_table.add_column("Status", style="bold")
    results_table.add_column("Response Time", style="green")

    for entry in hosts_to_test:
        alias_name = entry["alias"]
        hostname = entry.get("hostname", "Unknown")

        console.print(f"[cyan]Testing {alias_name}...[/cyan]", end=" ")

        import time

        start_time = time.time()
        result = subprocess.run(
            [
                "ssh",
                "-o",
                "ConnectTimeout=10",
                "-o",
                "BatchMode=yes",
                alias_name,
                "echo",
                "OK",
            ],
            capture_output=True,
            text=True,
        )
        elapsed = time.time() - start_time

        if result.returncode == 0:
            status = "[green]✓ Connected[/green]"
            response_time = f"{elapsed:.2f}s"
            console.print("[green]✓[/green]")
        else:
            status = "[red]✗ Failed[/red]"
            response_time = "-"
            console.print("[red]✗[/red]")

        results_table.add_row(alias_name, hostname, status, response_time)

    console.print()
    console.print(results_table)
    console.print()


# ------------------------------------------------------------------ #
# profile subgroup — cluster hardware profiles
# ------------------------------------------------------------------ #


@cli.group()
def profile():
    """Manage cluster hardware profiles for auto_allocate_resources().

    \b
    Profiles describe cluster hardware (cores, memory, walltime, partitions)
    so that auto_allocate_resources() can intelligently cap resources.

    \b
    Commands:
      list     List all predefined profiles
      show     Show detailed profile information
      create   Interactive profile builder
    """


@profile.command("list")
def profile_list():
    """List all predefined cluster profiles."""
    from atomate2.siesta.cluster_profiles import ClusterProfile

    profiles = ClusterProfile.list_predefined()

    console.print(
        Panel.fit(
            "[bold]Cluster Hardware Profiles[/bold]",
            border_style="cyan",
        )
    )
    console.print()

    from rich import box

    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("Name", style="cyan", width=12)
    table.add_column("Cores/Node", style="yellow", justify="right", width=12)
    table.add_column("Memory (GB)", style="white", justify="right", width=12)
    table.add_column("Max Nodes", style="white", justify="right", width=10)
    table.add_column("Max Walltime", style="white", width=14)
    table.add_column("Partition", style="green", width=12)
    table.add_column("Account", style="green", width=12)
    table.add_column("QOS", style="green", width=12)
    table.add_column("GPUs", style="yellow", justify="right", width=6)

    for name, p in profiles.items():
        table.add_row(
            name,
            str(p.cores_per_node),
            str(p.memory_per_node_gb),
            str(p.max_nodes),
            p.max_walltime,
            p.partition or "-",
            p.account or "-",
            p.qos or "-",
            str(p.gpu_per_node),
        )

    console.print(table)
    console.print()
    console.print(
        "[dim]Usage: ClusterProfile.mn5()  or  "
        'ClusterProfile.from_dict({"cores_per_node": 48, ...})[/dim]'
    )
    console.print()


@profile.command("show")
@click.argument("name")
def profile_show(name: str):
    """Show detailed information for a specific profile.

    NAME is the profile name (e.g. mn5, agustina, generic).
    """
    from atomate2.siesta.cluster_profiles import ClusterProfile

    profiles = ClusterProfile.list_predefined()

    if name.lower() not in profiles:
        console.print(f"[red]Unknown profile: '{name}'[/red]")
        console.print(f"[dim]Available profiles: {', '.join(profiles.keys())}[/dim]")
        return

    p = profiles[name.lower()]

    console.print(
        Panel.fit(
            f"[bold]Cluster Profile: {p.name}[/bold]",
            border_style="cyan",
        )
    )
    console.print()

    from rich import box

    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    table.add_column("Field", style="cyan", width=20)
    table.add_column("Value", style="white")

    table.add_row("Name", p.name)
    table.add_row("Cores per node", str(p.cores_per_node))
    table.add_row("Memory per node", f"{p.memory_per_node_gb} GB")
    table.add_row("Max nodes", str(p.max_nodes))
    table.add_row("Max walltime", p.max_walltime)
    table.add_row("Partition", p.partition or "(not set)")
    table.add_row("Account", p.account or "(not set)")
    table.add_row("QOS", p.qos or "(not set)")
    table.add_row("GPUs per node", str(p.gpu_per_node))
    table.add_row("Modules", ", ".join(p.modules) if p.modules else "(none)")

    console.print(table)
    console.print()

    # Python usage example
    factory = name.lower()
    console.print("[bold]Python usage:[/bold]")
    console.print(
        Panel(
            f"from atomate2.siesta.cluster_profiles import ClusterProfile\n"
            f"from atomate2.siesta.powerups import auto_allocate_resources\n\n"
            f"flow = auto_allocate_resources(\n"
            f"    flow,\n"
            f"    cluster_profile=ClusterProfile.{factory}(),\n"
            f")",
            title="Example",
            border_style="green",
        )
    )
    console.print()


@profile.command("create")
def profile_create():
    """Interactively create a custom cluster profile and print the Python code."""
    try:
        import questionary
    except ImportError:
        console.print(
            "[red]The 'questionary' package is required for interactive mode.[/red]"
        )
        console.print("[dim]Install with: pip install questionary[/dim]")
        return

    console.print(
        Panel.fit(
            "[bold]Custom Cluster Profile Builder[/bold]",
            border_style="cyan",
        )
    )
    console.print()

    name = questionary.text("Profile name:", default="my_cluster").ask()
    if name is None:
        return

    cores = questionary.text("Cores per node:", default="48").ask()
    if cores is None:
        return
    cores = int(cores)

    memory = questionary.text("Memory per node (GB):", default="192").ask()
    if memory is None:
        return
    memory = float(memory)

    max_nodes = questionary.text("Max nodes per job:", default="1").ask()
    if max_nodes is None:
        return
    max_nodes = int(max_nodes)

    max_walltime = questionary.text(
        "Max walltime (HH:MM:SS):", default="72:00:00"
    ).ask()
    if max_walltime is None:
        return

    partition = questionary.text(
        "SLURM partition (leave empty for none):", default=""
    ).ask()
    if partition is None:
        return
    partition = partition or None

    account = questionary.text(
        "SLURM account (leave empty for none):", default=""
    ).ask()
    if account is None:
        return
    account = account or None

    qos = questionary.text("SLURM QOS (leave empty for none):", default="").ask()
    if qos is None:
        return
    qos = qos or None

    gpus = questionary.text("GPUs per node:", default="0").ask()
    if gpus is None:
        return
    gpus = int(gpus)

    # Build the code snippet
    console.print()
    console.print("[bold]Generated Python code:[/bold]")

    args = [f'    name="{name}"']
    args.append(f"    cores_per_node={cores}")
    args.append(f"    memory_per_node_gb={memory}")
    if max_nodes > 1:
        args.append(f"    max_nodes={max_nodes}")
    if max_walltime != "72:00:00":
        args.append(f'    max_walltime="{max_walltime}"')
    if partition:
        args.append(f'    partition="{partition}"')
    if account:
        args.append(f'    account="{account}"')
    if qos:
        args.append(f'    qos="{qos}"')
    if gpus > 0:
        args.append(f"    gpu_per_node={gpus}")

    args_str = ",\n".join(args)

    code = (
        "from atomate2.siesta.cluster_profiles import ClusterProfile\n"
        "from atomate2.siesta.powerups import auto_allocate_resources\n\n"
        f"profile = ClusterProfile(\n{args_str},\n)\n\n"
        "flow = auto_allocate_resources(flow, cluster_profile=profile)"
    )

    console.print(Panel(code, title="Copy & paste", border_style="green"))
    console.print()


if __name__ == "__main__":
    cli()
