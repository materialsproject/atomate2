=======================================
Remote Cluster Setup CLI
=======================================

The ``atomate2siesta-cluster`` command-line tool helps you set up remote HPC clusters for atomate2siesta calculations by automating the process of creating conda environments and installing necessary packages via SSH.

Overview
========

This tool provides a streamlined way to:

* SSH to remote HPC clusters
* Create conda environments in ``$HOME``
* Install jobflow-remote and atomate2siesta from GitHub
* Optionally install SIESTA from conda-forge
* Generate ``.atomate2siesta.yaml`` configuration file
* Verify installations and check environment status

Commands
========

The ``atomate2siesta-cluster`` tool provides the following commands:

* **setup** - Set up conda environment on remote cluster
* **status** - Check status of remote environment
* **ssh-setup** - Manage SSH keys and config (add, status, test)
* **squid** - Manage Squid HTTP proxy (install, start, stop, status, restart)
* **build-offline** - Build offline environment for air-gapped clusters
* **info** - Show comprehensive usage information

----

setup
-----

Set up a conda environment on a remote cluster with atomate2siesta installed.

**Usage:**

.. code-block:: bash

   atomate2siesta-cluster setup --host <hostname> [OPTIONS]

**Options:**

* ``--host TEXT`` (required): Remote cluster hostname, IP address, or SSH config alias
* ``--user TEXT``: Username for SSH connection (not needed if using SSH config)
* ``-i, --identity-file TEXT``: Path to SSH private key file
* ``--password``: Prompt for password authentication
* ``--ssh-config``: Use SSH config alias (no user@ prefix needed)
* ``--env-name TEXT``: Name for conda environment (default: ``jobflow-remote``)
* ``--python-version TEXT``: Python version (default: ``3.11``)
* ``--git-ssh``: Use SSH URL for Git clone (requires SSH key on cluster)
* ``--git-token TEXT``: GitHub personal access token for private repo (HTTPS only)
* ``--install-siesta``: Install SIESTA from conda-forge (``siesta=*=*mpich*``)
* ``--proxy TEXT``: HTTP/HTTPS proxy URL (e.g., ``http://proxy.cluster.edu:8080``)
* ``--auto-proxy``: Try to auto-detect proxy settings from remote cluster
* ``--ssh-tunnel``: Create SSH tunnel for internet access (uses local machine as proxy)
* ``--tunnel-port INT``: Local port for SSH tunnel (default: 3129)
* ``-v, --verbose``: Show detailed command output (stdout/stderr)

**Examples:**

.. code-block:: bash

   # Using SSH config alias with Git SSH (RECOMMENDED for private repos)
   atomate2siesta-cluster setup --host mycluster --ssh-config --git-ssh

   # Using SSH config with GitHub personal access token
   atomate2siesta-cluster setup --host mycluster --ssh-config --git-token ghp_xxxxx

   # Install with SIESTA included
   atomate2siesta-cluster setup --host mycluster --ssh-config --git-ssh --install-siesta

   # Using SSH key authentication with Git SSH
   atomate2siesta-cluster setup --host cluster.university.edu --user myuser -i ~/.ssh/id_rsa --git-ssh

   # Setup with password authentication
   atomate2siesta-cluster setup --host cluster.university.edu --user myuser --password --git-ssh

   # Custom environment name and Python version
   atomate2siesta-cluster setup --host mycluster --ssh-config --env-name myenv --python-version 3.11 --git-ssh

   # Verbose mode to see detailed output
   atomate2siesta-cluster setup --host mycluster --ssh-config --git-ssh --verbose

   # Setup with proxy configuration (for clusters that block internet access)
   atomate2siesta-cluster setup --host mn5 --ssh-config --git-ssh --proxy http://proxy.bsc.es:8080

   # Auto-detect proxy from remote cluster environment
   atomate2siesta-cluster setup --host mn5 --ssh-config --git-ssh --auto-proxy

   # Complete setup with proxy and SIESTA installation
   atomate2siesta-cluster setup --host mn5 --ssh-config --git-ssh --proxy http://proxy.bsc.es:8080 --install-siesta

   # SSH tunnel for air-gapped clusters (MN5 recommended approach)
   atomate2siesta-cluster setup --host mn5 --ssh-config --git-ssh --ssh-tunnel --install-siesta

   # SSH tunnel with custom port
   atomate2siesta-cluster setup --host mn5 --ssh-config --git-ssh --ssh-tunnel --tunnel-port 3130

**What it does:**

1. Tests SSH connection to the remote host
2. Checks for conda/miniconda installation
3. Creates a new conda environment (or removes existing one if requested)
4. Installs jobflow-remote from PyPI
5. Installs atomate2siesta from GitHub (main branch)
6. Optionally installs SIESTA from conda-forge
7. Verifies installations (checks package versions)
8. Creates ``~/.atomate2siesta.yaml`` configuration file with default SIESTA settings

**Configuration File:**

The setup command automatically creates ``~/.atomate2siesta.yaml`` on the remote cluster with these default settings:

.. code-block:: yaml

   SIESTA_CMD: siesta < siesta.fdf > siesta.out
   SIESTA_PP_PATH: '$HOME/.siesta/pseudos/ONCVPSP-PBEsol-FR-PDv0.4-Standard/'
   FLOS_PATH: "$HOME/apps/flos"
   OPTICAL_INPUT_CMD: optical_input < siesta.EPSIMG
   OPTICAL_CMD: optical < siesta.EPSIMG

All paths use ``$HOME`` for portability across different user accounts. You can edit this file later to customize paths for your cluster setup.

----

status
------

Check the status of a remote cluster environment.

**Usage:**

.. code-block:: bash

   atomate2siesta-cluster status --host <hostname> [OPTIONS]

**Options:**

* ``--host TEXT`` (required): Remote cluster hostname or IP address
* ``--user TEXT``: Username for SSH connection (defaults to current user)
* ``-i, --identity-file TEXT``: Path to SSH private key file
* ``--password``: Prompt for password authentication
* ``--env-name TEXT``: Name of environment to check (default: ``atomate2siesta``)

**Examples:**

.. code-block:: bash

   # Check default environment
   atomate2siesta-cluster status --host cluster.university.edu --user myuser

   # Check custom environment
   atomate2siesta-cluster status --host cluster.university.edu --env-name myenv

**What it shows:**

* SSH connection status
* Conda version
* Environment existence
* Installed packages (atomate2siesta, jobflow-remote, pymatgen, sisl)
* **Internet connectivity** (NEW!):

  * Direct internet access status
  * Proxy configuration (environment variables)
  * Proxy configuration (.condarc file)
  * Recommendations for air-gapped clusters

**Example Output:**

.. code-block:: text

   Internet Connectivity:

   Direct Access         ✗ Blocked
   Proxy (Environment)   Not configured
   Proxy (.condarc)      Not configured

   ⚠ No internet access detected!

   This cluster appears to be air-gapped. Use one of these solutions:
     1. SSH Tunnel (recommended):
        atomate2siesta-cluster setup --host mn5 --ssh-config --ssh-tunnel
     2. Squid Proxy:
        atomate2siesta-cluster squid install && squid start
        atomate2siesta-cluster setup --host mn5 --ssh-config --use-squid
     3. Offline Environment:
        atomate2siesta-cluster build-offline --install-siesta

----

info
----

Show comprehensive information about the cluster setup tool, including usage examples, authentication methods, and workflow guidance.

**Usage:**

.. code-block:: bash

   atomate2siesta-cluster info

----

ssh-setup
---------

Manage SSH keys and configuration for cluster access. This command group provides three subcommands to help you set up and verify SSH connections.

**Subcommands:**

* ``add`` - Add new SSH config entry with optional key generation
* ``status`` - Show SSH keys, config entries, and connection status
* ``test`` - Test SSH connections to configured hosts

ssh-setup add
^^^^^^^^^^^^^

Add a new SSH config entry for cluster access, with optional key generation and passwordless login setup.

**Usage:**

.. code-block:: bash

   atomate2siesta-cluster ssh-setup add --alias <name> --hostname <host> [OPTIONS]

**Options:**

* ``--alias TEXT`` (required): SSH config alias name (e.g., 'mycluster')
* ``--hostname TEXT`` (required): Remote cluster hostname or IP address
* ``--user TEXT``: Username for SSH connection (defaults to current user)
* ``--port INTEGER``: SSH port (default: 22)
* ``--key-file TEXT``: Path to SSH private key (default: ~/.ssh/id_rsa)
* ``--generate-key``: Generate new SSH key pair if it doesn't exist
* ``--copy-id``: Copy public key to remote server (enables passwordless login)
* ``--overwrite``: Overwrite existing SSH config entry

**Examples:**

.. code-block:: bash

   # Basic setup (uses existing SSH key)
   atomate2siesta-cluster ssh-setup add --alias mycluster --hostname cluster.edu --user myuser

   # Generate SSH key automatically
   atomate2siesta-cluster ssh-setup add --alias mn5 --hostname mn5.bsc.es --user myuser --generate-key

   # Set up passwordless login
   atomate2siesta-cluster ssh-setup add --alias hpc --hostname hpc.edu --user myuser --copy-id

   # Complete setup with key generation and passwordless login
   atomate2siesta-cluster ssh-setup add \
       --alias newcluster \
       --hostname cluster.university.edu \
       --user myuser \
       --generate-key \
       --copy-id

   # Use custom SSH key file
   atomate2siesta-cluster ssh-setup add \
       --alias special \
       --hostname special.edu \
       --key-file ~/.ssh/id_rsa_special

**What it does:**

1. Creates ``~/.ssh`` directory if it doesn't exist (with proper permissions: 700)
2. Generates SSH key pair if requested or if key doesn't exist (4096-bit RSA, no passphrase)
3. Creates or updates ``~/.ssh/config`` entry with:

   * HostName, User, Port, IdentityFile
   * ForwardAgent yes (for SSH agent forwarding)
   * ServerAliveInterval 60 (keep connection alive)
   * ServerAliveCountMax 3

4. Optionally copies public key to remote server using ``ssh-copy-id``
5. Tests the connection to verify setup

**Safety Features:**

* Won't overwrite existing SSH keys without explicit confirmation
* Prompts before generating new keys
* Won't overwrite existing config entries unless ``--overwrite`` is used
* Sets correct file permissions automatically (600 for private keys, 644 for public keys)

**After setup:**

.. code-block:: bash

   # Connect using the alias
   ssh mycluster

   # Use with cluster setup
   atomate2siesta-cluster setup --host mycluster --ssh-config

ssh-setup status
^^^^^^^^^^^^^^^^

Show comprehensive information about your SSH configuration, including keys, SSH agent status, and configured hosts.

**Usage:**

.. code-block:: bash

   atomate2siesta-cluster ssh-setup status [OPTIONS]

**Options:**

* ``-v, --verbose``: Show detailed information (SSH agent socket, identity files)

**Examples:**

.. code-block:: bash

   # Basic status check
   atomate2siesta-cluster ssh-setup status

   # Detailed view with all information
   atomate2siesta-cluster ssh-setup status -v

**What it shows:**

1. **SSH Keys**: Lists all SSH keys in ``~/.ssh/`` with:

   * Key type (RSA, ED25519, ECDSA, DSA)
   * Key size (bits)
   * Whether public key exists

2. **SSH Agent**: Shows if SSH agent is running and which keys are loaded

3. **SSH Config Entries**: Displays all configured hosts from ``~/.ssh/config`` including:

   * Alias name
   * Hostname
   * Username
   * Port
   * IdentityFile (in verbose mode)

**Example Output:**

.. code-block:: text

   SSH Configuration Status

   ✓ ~/.ssh directory exists

   SSH Keys:

    Key File                        Type             Size       Public Key
    /Users/user/.ssh/id_rsa         RSA 4096-bit     3247 bytes ✓
    /Users/user/.ssh/id_ed25519     ED25519 256-bit  432 bytes  ✓

   SSH Agent:

   ✓ SSH agent is running
   ✓ 2 key(s) loaded

   SSH Config Entries (~/.ssh/config):

    Alias         HostName              User      Port
    mn5-glogin1   glogin1.bsc.es        icn2001   22
    mycluster     cluster.university.…  myuser    22
    hpc           hpc.edu               myuser    22

   Found 3 host(s) configured

   Usage:
     Connect: ssh mn5-glogin1
     Test: atomate2siesta-cluster ssh-setup test mn5-glogin1

ssh-setup test
^^^^^^^^^^^^^^

Test SSH connections to configured hosts to verify connectivity and measure response times.

**Usage:**

.. code-block:: bash

   atomate2siesta-cluster ssh-setup test [ALIAS] [OPTIONS]

**Options:**

* ``ALIAS``: SSH config alias to test (optional if using ``--all``)
* ``-a, --all``: Test all configured hosts

**Examples:**

.. code-block:: bash

   # Test specific host
   atomate2siesta-cluster ssh-setup test mycluster

   # Test all configured hosts
   atomate2siesta-cluster ssh-setup test --all

**What it does:**

1. Reads ``~/.ssh/config`` to find configured hosts
2. Attempts SSH connection with 10-second timeout
3. Measures connection response time
4. Reports success/failure status for each host

**Example Output:**

.. code-block:: text

   SSH Connection Test

   Testing mycluster... ✓
   Testing mn5-glogin1... ✓
   Testing oldcluster... ✗

    Alias        HostName              Status          Response Time
    mycluster    cluster.edu           ✓ Connected     1.23s
    mn5-glogin1  glogin1.bsc.es        ✓ Connected     2.45s
    oldcluster   old.cluster.edu       ✗ Failed        -

**Troubleshooting:**

If a connection test fails, check:

* Hostname is correct and reachable
* Network connectivity (firewall, VPN)
* SSH key permissions (should be 600)
* Public key is installed on remote server (use ``--copy-id`` to fix)

----

squid
-----

Manage Squid HTTP proxy for air-gapped clusters. This command group provides installation and management of Squid proxy server, which enables package installation on clusters that block direct internet access.

**Subcommands:**

* ``install`` - Auto-install squid (supports macOS, Ubuntu/Debian, CentOS/RHEL, Fedora)
* ``start`` - Start squid proxy on specified port
* ``stop`` - Stop squid proxy
* ``status`` - Show squid status
* ``restart`` - Restart squid proxy

**Usage:**

.. code-block:: bash

   atomate2siesta-cluster squid <action> [OPTIONS]

**Options:**

* ``--port INTEGER``: Port for squid proxy (default: 3129)
* ``--remove``: Remove old squid.conf before starting (use with 'start' action)

**Examples:**

.. code-block:: bash

   # Install squid (auto-detects OS)
   atomate2siesta-cluster squid install

   # Start squid on default port (3129)
   atomate2siesta-cluster squid start

   # Start with custom port
   atomate2siesta-cluster squid start --port 8080

   # Remove old config before starting
   atomate2siesta-cluster squid start --remove

   # Check status
   atomate2siesta-cluster squid status

   # Stop squid
   atomate2siesta-cluster squid stop

   # Restart squid
   atomate2siesta-cluster squid restart

**Automatic Installation:**

The ``install`` command automatically detects your operating system and uses the appropriate package manager:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Operating System
     - Package Manager
     - Command Used
   * - **macOS**
     - Homebrew
     - ``brew install squid``
   * - **Ubuntu/Debian**
     - apt-get
     - ``apt-get update && apt-get install -y squid``
   * - **CentOS/RHEL 7**
     - yum
     - ``yum install -y squid``
   * - **RHEL 8+/Fedora**
     - dnf
     - ``dnf install -y squid``
   * - **Other Linux**
     - Manual
     - Shows instructions

**How Squid Works:**

Squid provides an HTTP/HTTPS proxy that conda and pip can use:

.. code-block:: text

   Cluster → Squid (localhost:3129) → Internet
            ↑
            Your local machine

**Use with Cluster Setup:**

Once squid is installed and running:

.. code-block:: bash

   # Option 1: Use with --use-squid flag
   atomate2siesta-cluster setup --host mn5 --ssh-config --git-ssh --use-squid

   # Option 2: Set environment variables manually
   export http_proxy=http://127.0.0.1:3129
   export https_proxy=http://127.0.0.1:3129

**When to Use Squid:**

* Air-gapped clusters that block outgoing connections
* Alternative to SSH tunnel (``-D`` flag)
* When you need persistent local proxy
* For multiple simultaneous cluster setups

**Advantages:**

* ✅ Works on completely air-gapped clusters
* ✅ Persistent (survives terminal closures)
* ✅ Can serve multiple connections simultaneously
* ✅ Professional caching capabilities

----

build-offline
-------------

Build an offline conda environment package for air-gapped clusters (like MareNostrum 5).

**IMPORTANT**: Use this command when the target cluster blocks ALL outgoing internet connections, including proxy access.

**Usage:**

.. code-block:: bash

   atomate2siesta-cluster build-offline [OPTIONS]

**Options:**

* ``--output, -o TEXT``: Output filename for packed environment (default: ``jobflow-remote.tar.gz``)
* ``--env-name TEXT``: Name for conda environment (default: ``jobflow-remote``)
* ``--python-version TEXT``: Python version (default: ``3.11``)
* ``--git-ssh``: Use SSH URL for Git clone (requires GitHub SSH key configured locally)
* ``--git-token TEXT``: GitHub personal access token for private repo
* ``--install-siesta``: Include SIESTA in the packed environment

**Requirements:**

* Linux x86_64 architecture (must match target cluster)
* conda or miniconda installed locally
* conda-pack package (will be installed automatically if missing)
* GitHub access for atomate2siesta

**Examples:**

.. code-block:: bash

   # Basic offline environment
   atomate2siesta-cluster build-offline --git-ssh

   # Include SIESTA and use custom output name
   atomate2siesta-cluster build-offline --output mn5-env.tar.gz --install-siesta --git-ssh

   # With GitHub token authentication
   atomate2siesta-cluster build-offline --git-token ghp_xxxxx --install-siesta

**What it does:**

1. Checks system compatibility (Linux x86_64 recommended)
2. Creates local conda environment with specified Python version
3. Installs jobflow-remote from PyPI
4. Installs atomate2siesta from GitHub
5. Optionally installs SIESTA from conda-forge
6. Packs entire environment using conda-pack (~500 MB - 1.5 GB)
7. Displays transfer and installation instructions

**Transfer to Cluster:**

After building, transfer the packed environment to the cluster:

.. code-block:: bash

   # Transfer using scp
   scp jobflow-remote.tar.gz user@mn5-glogin1:~/

   # Or using rsync for resume capability
   rsync -avz --progress jobflow-remote.tar.gz user@mn5-glogin1:~/

**Install on Cluster:**

SSH to the cluster and unpack:

.. code-block:: bash

   ssh user@mn5-glogin1
   mkdir -p ~/miniconda3/envs/jobflow-remote
   tar -xzf jobflow-remote.tar.gz -C ~/miniconda3/envs/jobflow-remote
   source ~/miniconda3/envs/jobflow-remote/bin/activate
   conda-unpack

   # Verify installation
   python -c "import jobflow_remote; import atomate2.siesta; print('Success!')"

**When to Use:**

* ✅ **MareNostrum 5** (blocks all outgoing connections)
* ✅ Air-gapped clusters (no internet access)
* ✅ When proxy configuration is unavailable or doesn't work
* ✅ For reproducible environments across multiple clusters

Authentication Methods
======================

SSH Configuration (Easiest - Recommended)
------------------------------------------

The **easiest way** to set up SSH access is using the ``ssh-setup`` command group:

.. code-block:: bash

   # Set up SSH config entry with key generation and passwordless login
   atomate2siesta-cluster ssh-setup add \
       --alias mycluster \
       --hostname cluster.edu \
       --user myuser \
       --generate-key \
       --copy-id

   # Then use it everywhere
   ssh mycluster
   atomate2siesta-cluster setup --host mycluster --ssh-config

**Benefits:**

* ✅ Automatically generates SSH keys if needed
* ✅ Creates ``~/.ssh/config`` entries
* ✅ Sets up passwordless login
* ✅ Tests connection
* ✅ No need to remember hostnames, usernames, or key files

See the ``ssh-setup`` command section above for full documentation.

SSH Key Authentication (Manual)
---------------------------------

Alternatively, use your SSH private key directly:

.. code-block:: bash

   atomate2siesta-cluster setup --host cluster.edu -i ~/.ssh/id_rsa

Password Authentication
-----------------------

Prompt for password (requires ``sshpass`` to be installed locally):

.. code-block:: bash

   atomate2siesta-cluster setup --host cluster.edu --password

SSH Agent
---------

If you have SSH keys configured and loaded in ssh-agent, you can omit authentication flags:

.. code-block:: bash

   atomate2siesta-cluster setup --host cluster.edu

Prerequisites
=============

Local Machine
-------------

* SSH access to the remote cluster
* ``sshpass`` (if using password authentication): ``brew install sshpass`` (macOS) or ``apt-get install sshpass`` (Linux)

Remote Cluster
--------------

* Conda or Miniconda installed and in PATH
* Internet access for package installation (may require proxy configuration on secure clusters)
* Sufficient disk space in ``$HOME`` for conda environment

Installing Conda on Remote Cluster
-----------------------------------

If conda is not installed on the remote cluster, SSH to it and run:

.. code-block:: bash

   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
   source $HOME/miniconda3/bin/activate
   conda init bash

Log out and log back in for the changes to take effect.

Proxy Configuration
-------------------

Many HPC clusters block direct internet access from login nodes for security reasons and require HTTP/HTTPS proxy servers for internet connectivity. This is especially common on supercomputing centers like BSC's MN5.

**When to Use Proxy Options**

You need proxy configuration if:

* Your cluster blocks direct internet access from login nodes
* You see errors like ``CondaHTTPError: HTTP 000 CONNECTION FAILED``
* Package installation fails with ``Connection refused`` or ``Connection reset by peer``
* Your cluster documentation mentions proxy servers for internet access

**Using Proxy Options**

Option 1: Specify proxy URL directly (recommended if you know the proxy):

.. code-block:: bash

   atomate2siesta-cluster setup --host mn5 --ssh-config --git-ssh --proxy http://proxy.bsc.es:8080

Option 2: Auto-detect proxy from remote environment:

.. code-block:: bash

   atomate2siesta-cluster setup --host mn5 --ssh-config --git-ssh --auto-proxy

**How Proxy Configuration Works**

The setup command configures proxy for conda and pip by:

1. Creating ``~/.condarc`` with proxy settings:

   .. code-block:: yaml

      proxy_servers:
        http: http://proxy.cluster.edu:8080
        https: http://proxy.cluster.edu:8080

2. Creating ``~/.config/pip/pip.conf`` with proxy settings:

   .. code-block:: ini

      [global]
      proxy = http://proxy.cluster.edu:8080

3. Setting environment variables for all conda/pip commands

**Common Proxy URLs**

* BSC MN5: ``http://proxy.bsc.es:8080``
* Ask your cluster administrator for the correct proxy URL

**Note:** If proxy auto-detection finds existing proxy settings in your remote environment, it will use those automatically.

SSH Tunnel for Air-Gapped Clusters
-----------------------------------

**RECOMMENDED for MareNostrum 5** and other clusters that block ALL outgoing connections.

**Problem:** Traditional proxy configuration requires outgoing connections to the proxy server. MN5 blocks ALL outgoing connections, so proxy won't work.

**Solution:** SSH tunnel with SOCKS proxy - route internet through your local machine!

**How it works:**

.. code-block:: text

   MN5 (localhost:3129) → SSH Tunnel → Your Local Machine → Internet

**Automated Approach** (recommended):

.. code-block:: bash

   # One command handles everything
   atomate2siesta-cluster setup --host mn5 --ssh-config --git-ssh --ssh-tunnel

This automatically:

1. Creates SSH tunnel with dynamic port forwarding (port 3129)
2. Configures MN5 to use ``http://127.0.0.1:3129`` as proxy
3. Installs all packages through the tunnel
4. Cleans up tunnel when done

**Manual Approach** (for more control):

Step 1 - Create tunnel on your local machine:

.. code-block:: bash

   # Create SOCKS proxy on port 3129
   ssh -D 3129 -N -f mn5-glogin1

   # Explanation:
   #   -D 3129  : Create SOCKS proxy on local port 3129
   #   -N       : Don't execute commands (just tunnel)
   #   -f       : Run in background

Step 2 - Configure proxy on MN5:

.. code-block:: bash

   ssh mn5-glogin1

   # Set proxy environment variables
   export http_proxy=http://127.0.0.1:3129
   export https_proxy=http://127.0.0.1:3129

   # Now internet access works!
   conda create -n jobflow-remote python=3.11 -y

Step 3 - Make it permanent (optional):

.. code-block:: bash

   # On MN5, add to ~/.bashrc
   echo 'export http_proxy=http://127.0.0.1:3129' >> ~/.bashrc
   echo 'export https_proxy=http://127.0.0.1:3129' >> ~/.bashrc

**Managing the tunnel:**

.. code-block:: bash

   # Check if tunnel is running
   pgrep -f "ssh.*-D.*3129"

   # Kill tunnel
   kill $(pgrep -f "ssh.*-D.*3129")

   # Restart tunnel
   ssh -D 3129 -N -f mn5-glogin1

**Advantages:**

* ✅ Works on completely air-gapped clusters
* ✅ No additional software needed (uses built-in SSH)
* ✅ Secure (all traffic encrypted through SSH)
* ✅ Flexible (can install/update packages anytime)

**Compared to SquidMan:**

SquidMan is a macOS GUI application that creates a local proxy. However, SSH's built-in dynamic port forwarding (``-D`` flag) provides the same functionality without additional software:

* **SSH Tunnel**: ``ssh -D 3129 -N -f mn5`` (built-in, cross-platform)
* **SquidMan**: GUI app + SSH (macOS only, extra software)

Both work identically - SSH tunnel is simpler!

Complete Workflow
=================

Here's a complete workflow for setting up a remote cluster for atomate2siesta calculations:

0. Set Up SSH Access (New - Recommended!)
------------------------------------------

**First time?** Set up SSH configuration for easy access:

.. code-block:: bash

   # One-time SSH setup (generates keys, creates config, enables passwordless login)
   atomate2siesta-cluster ssh-setup add \
       --alias mycluster \
       --hostname cluster.university.edu \
       --user myuser \
       --generate-key \
       --copy-id

   # Check your SSH setup anytime
   atomate2siesta-cluster ssh-setup status

   # Test the connection
   atomate2siesta-cluster ssh-setup test mycluster

This creates a convenient alias so you can use ``--host mycluster --ssh-config`` instead of typing the full hostname, username, and key file path every time.

1. Set Up Cluster Environment
------------------------------

.. code-block:: bash

   # Using SSH config alias (much simpler!)
   atomate2siesta-cluster setup \
       --host mycluster \
       --ssh-config \
       --git-ssh \
       --install-siesta

   # Or the traditional way with full details
   atomate2siesta-cluster setup \
       --host cluster.university.edu \
       --user myuser \
       -i ~/.ssh/id_rsa \
       --git-ssh \
       --install-siesta

This automatically:

* Creates conda environment with jobflow-remote and atomate2siesta
* Installs SIESTA from conda-forge (if ``--install-siesta`` is used)
* Generates ``~/.atomate2siesta.yaml`` configuration file

2. SSH to Cluster and Activate Environment
-------------------------------------------

.. code-block:: bash

   ssh myuser@cluster.university.edu
   conda activate jobflow-remote

3. Configure Jobflow-Remote on Cluster
---------------------------------------

.. code-block:: bash

   # Generate project configuration
   jf project generate myproject

   # Edit configuration file
   nano ~/.jfremote/myproject.yaml

   # Initialize database
   jf admin reset

4. Verify and Customize Configuration (Optional)
-------------------------------------------------

The ``~/.atomate2siesta.yaml`` file has been automatically created. You may need to edit it to match your cluster setup:

.. code-block:: bash

   nano ~/.atomate2siesta.yaml

Edit paths if needed:

* ``SIESTA_CMD``: Path to SIESTA executable (default assumes it's in PATH)
* ``SIESTA_PP_PATH``: Path to pseudopotential directory
* ``FLOS_PATH``: Path to FLOS library for advanced MD/optimization
* ``OPTICAL_INPUT_CMD`` and ``OPTICAL_CMD``: Commands for optical calculations

If you used ``--install-siesta``, SIESTA is already installed from conda-forge and should be available in your PATH.

5. Start Runner on Cluster
---------------------------

.. code-block:: bash

   # Start runner in background
   jf runner start -d

   # Check status
   jf runner status

6. Submit Jobs from Local Machine
----------------------------------

On your local machine, create a Python script to submit workflows:

.. code-block:: python

   from atomate2.siesta.jobs.core import RelaxMaker
   from jobflow_remote import submit_flow
   from pymatgen.core import Structure

   # Create structure
   structure = Structure.from_file("structure.cif")

   # Create job
   maker = RelaxMaker.fixed_cell_relaxation()
   job = maker.make(structure)

   # Submit to remote cluster
   job_id = submit_flow(job, worker="my_worker")
   print(f"Submitted job: {job_id}")

7. Monitor Jobs
---------------

.. code-block:: bash

   # List jobs
   jf job list

   # Check specific job
   jf job info <job_id>

   # Get job output
   jf job output <job_id>

Troubleshooting
===============

SSH Connection Fails
--------------------

**Error:** ``SSH connection failed!``

**Quick Diagnosis:**

.. code-block:: bash

   # Check your SSH configuration
   atomate2siesta-cluster ssh-setup status

   # Test the connection
   atomate2siesta-cluster ssh-setup test mycluster

**Solutions:**

* **Easy setup:** Use ``atomate2siesta-cluster ssh-setup add`` to configure SSH properly
* Verify hostname and username are correct
* Check that SSH key has correct permissions: ``chmod 600 ~/.ssh/id_rsa``
* Test manual SSH connection: ``ssh user@host``
* Ensure firewall allows SSH connections
* If using SSH config, verify entry exists: ``cat ~/.ssh/config``

Conda Not Found
---------------

**Error:** ``conda not found on remote host!``

**Solution:** Install Miniconda/Anaconda on the remote cluster (see Prerequisites section above)

sshpass Not Found
-----------------

**Error:** ``sshpass not found``

**Solutions:**

* Install sshpass: ``brew install sshpass`` (macOS) or ``apt-get install sshpass`` (Linux)
* Use SSH key authentication instead: ``--identity-file ~/.ssh/id_rsa``

Environment Already Exists
---------------------------

The tool will detect existing environments and prompt you to remove and recreate them. Answer "yes" to proceed or "no" to cancel.

Package Installation Fails
---------------------------

**Common causes:**

* Network connectivity issues on cluster (may need proxy configuration)
* Insufficient disk space
* Package conflicts

**Solutions:**

* Check internet connectivity on cluster
* If cluster blocks internet access, use ``--proxy`` or ``--auto-proxy`` flag
* Clear conda cache: ``conda clean --all``
* Try development installation with ``--dev`` flag

Internet Connection Errors (Proxy Required)
--------------------------------------------

**Error:** ``CondaHTTPError: HTTP 000 CONNECTION FAILED`` or ``Connection reset by peer``

**Cause:** Many HPC clusters block direct internet access and require proxy configuration.

**Solutions:**

1. **Ask your cluster administrator** for the proxy URL

2. **Use proxy flag** with the setup command:

   .. code-block:: bash

      atomate2siesta-cluster setup --host cluster --ssh-config --git-ssh --proxy http://proxy.cluster.edu:8080

3. **Try auto-detection** to use existing proxy settings:

   .. code-block:: bash

      atomate2siesta-cluster setup --host cluster --ssh-config --git-ssh --auto-proxy

4. **Check cluster documentation** for proxy configuration

**Common Proxy URLs:**

* BSC MN5: ``http://proxy.bsc.es:8080``
* Your cluster: Ask administrator

MareNostrum 5 Setup (Air-Gapped Cluster)
-----------------------------------------

**Issue:** MareNostrum 5 (MN5) blocks **ALL outgoing internet connections**, including proxy access.

**Official Documentation:** According to BSC:

   *"Once you are logged into MareNostrum you cannot make outgoing connections for security reasons. Only incoming connections are allowed in the whole cluster."*

**Why Standard Setup Fails:**

* Regular ``setup`` command requires internet access to install packages
* Proxy configuration (``--proxy`` or ``--auto-proxy``) does NOT work on MN5
* All conda/pip commands will fail with connection errors

**Solution 1: SSH Tunnel (RECOMMENDED)** ✅

Use your local machine as a proxy via SSH tunneling:

.. code-block:: bash

   # Automated (one command)
   atomate2siesta-cluster setup --host mn5 --ssh-config --git-ssh --ssh-tunnel --install-siesta

   # Manual (more control)
   # Terminal 1: Create tunnel
   ssh -D 3129 -N -f mn5-glogin1

   # Terminal 2: SSH and configure
   ssh mn5-glogin1
   export http_proxy=http://127.0.0.1:3129
   export https_proxy=http://127.0.0.1:3129
   conda create -n jobflow-remote python=3.11 -y
   # ... continue with installation

See "SSH Tunnel for Air-Gapped Clusters" section above for full details.

**Solution 2: Offline Environment Transfer** (if SSH tunnel doesn't work)

1. **Build on Linux Machine** (your local machine or another cluster):

   .. code-block:: bash

      atomate2siesta-cluster build-offline --output mn5-env.tar.gz --install-siesta --git-ssh

2. **Transfer to MN5:**

   .. code-block:: bash

      scp mn5-env.tar.gz user@mn5-glogin1:~/

3. **Install on MN5:**

   .. code-block:: bash

      ssh user@mn5-glogin1
      mkdir -p ~/miniconda3/envs/jobflow-remote
      tar -xzf mn5-env.tar.gz -C ~/miniconda3/envs/jobflow-remote
      source ~/miniconda3/envs/jobflow-remote/bin/activate
      conda-unpack

4. **Verify:**

   .. code-block:: bash

      python -c "import jobflow_remote; import atomate2.siesta; print('✓ Success!')"

**Alternative: Contact BSC Support**

Ask about:

* Internal package mirrors or conda channels
* Pre-installed Python/conda modules
* BSC-recommended package installation procedures

Email: support@bsc.es

GitHub Clone Fails (Private Repository)
----------------------------------------

**Error:** Failed to install atomate2siesta - authentication failed

**Solutions:**

**Option 1: Use SSH (Recommended)**

1. Ensure SSH key is set up on cluster for GitHub:

   .. code-block:: bash

      ssh user@cluster
      ssh-keygen -t ed25519 -C "your_email@example.com"
      cat ~/.ssh/id_ed25519.pub  # Add this to GitHub
      ssh -T git@github.com  # Test connection

2. Use ``--git-ssh`` flag: ``atomate2siesta-cluster setup --host cluster --git-ssh``

**Option 2: Use Personal Access Token**

1. Generate token at https://github.com/settings/tokens (requires ``repo`` scope)
2. Use ``--git-token`` flag: ``atomate2siesta-cluster setup --host cluster --git-token ghp_xxxxx``

SIESTA Installation Fails
--------------------------

**Error:** Failed to install SIESTA from conda-forge

**Solutions:**

* Check internet connectivity on cluster
* Verify conda-forge channel is accessible
* Try manual installation: ``conda install -c conda-forge "siesta=*=*mpich*"``
* If unsuccessful, install SIESTA manually and update ``SIESTA_CMD`` in ``~/.atomate2siesta.yaml``

Configuration File Creation Fails
----------------------------------

**Error:** Failed to create ``.atomate2siesta.yaml``

**Solution:** Create manually on the cluster:

.. code-block:: bash

   cat > ~/.atomate2siesta.yaml << 'EOF'
   SIESTA_CMD: siesta < siesta.fdf > siesta.out
   SIESTA_PP_PATH: '$HOME/.siesta/pseudos/ONCVPSP-PBEsol-FR-PDv0.4-Standard/'
   FLOS_PATH: "$HOME/apps/flos"
   OPTICAL_INPUT_CMD: optical_input < siesta.EPSIMG
   OPTICAL_CMD: optical < siesta.EPSIMG
   EOF

Advanced Usage
==============

Custom Worker Configuration
----------------------------

After setting up the cluster, you may want to configure custom workers in jobflow-remote:

1. Edit ``~/.jfremote/<project>.yaml``
2. Add worker configurations for different partitions/queues
3. Specify resource requirements (nodes, cores, memory, time)
4. Configure scheduler settings (SLURM, PBS, etc.)

Example worker configuration for SLURM:

.. code-block:: yaml

   workers:
     - name: compute_worker
       type: remote
       scheduler_type: slurm
       work_dir: $HOME/jobflow_remote_work
       resources:
         partition: compute
         nodes: 1
         ntasks: 32
         time: "24:00:00"
         memory: "128GB"
       pre_run: |
         conda activate atomate2siesta
         export SIESTA_CMD="/apps/siesta/5.0/bin/siesta"

Multiple Environments
---------------------

You can set up multiple environments on the same cluster for different purposes:

.. code-block:: bash

   # Production environment
   atomate2siesta-cluster setup --host cluster.edu --env-name production --python-version 3.10

   # Development environment
   atomate2siesta-cluster setup --host cluster.edu --env-name development --python-version 3.11 --dev

   # Testing environment with older Python
   atomate2siesta-cluster setup --host cluster.edu --env-name testing --python-version 3.9

Batch Setup for Multiple Clusters
----------------------------------

Create a shell script to set up multiple clusters:

.. code-block:: bash

   #!/bin/bash

   clusters=("cluster1.edu" "cluster2.edu" "cluster3.edu")

   for cluster in "${clusters[@]}"; do
       echo "Setting up $cluster..."
       atomate2siesta-cluster setup \
           --host "$cluster" \
           --user myuser \
           -i ~/.ssh/id_rsa \
           --install-jobflow-remote
   done

Integration with Existing Workflows
====================================

This tool integrates seamlessly with:

* **Database CLI** (``atomate2siesta-database``): Set up MongoDB for storing calculation results
* **Jobflow-Remote CLI** (``atomate2siesta-jobflow-remote``): Configure job submission from local machine
* **Input Generator** (``atomate2siesta-inputs``): Generate SIESTA input files
* **Pseudopotential Manager** (``atomate2siesta-pseudos``): Manage pseudopotential libraries

See Also
========

* :doc:`cli-database` - Database CLI Documentation
* :doc:`cli-jobflow-remote` - Jobflow-Remote Setup
* Tutorial 14: Job Submission
* `Jobflow-Remote Documentation <https://matgenix.github.io/jobflow-remote/>`_
