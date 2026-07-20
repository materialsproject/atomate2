"""Cluster hardware profiles for resource allocation.

Provides the :class:`ClusterProfile` dataclass that describes cluster hardware
so that :func:`~atomate2.siesta.powerups.auto_allocate_resources` can cap cores,
spread across nodes, and inject SLURM metadata automatically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ClusterProfile:
    """Description of cluster hardware for automatic resource allocation.

    Parameters
    ----------
    name : str
        Human-readable profile name (e.g. ``"mn5"``, ``"agustina"``).
    cores_per_node : int
        Number of CPU cores available per compute node.
    memory_per_node_gb : float
        Total RAM per node in gigabytes.
    max_nodes : int
        Maximum number of nodes to request for a single job.
    partition : str or None
        SLURM partition name (e.g. ``"RES"``, ``"gp_bsccase"``).
    account : str or None
        SLURM account string for billing.
    qos : str or None
        SLURM quality-of-service level.
    max_walltime : str
        Maximum walltime in ``"HH:MM:SS"`` format.
    gpu_per_node : int
        Number of GPUs per node (0 for CPU-only clusters).
    modules : list[str] or None
        Environment modules to load before running (informational).

    Examples
    --------
    Use a predefined profile:

    >>> from atomate2.siesta.cluster_profiles import ClusterProfile
    >>> profile = ClusterProfile.mn5()
    >>> print(profile.summary())
    mn5: 112 cores/node, 256.0 GB, max 1 node, 72:00:00 walltime

    Create a custom profile:

    >>> profile = ClusterProfile(
    ...     name="my_cluster",
    ...     cores_per_node=64,
    ...     memory_per_node_gb=256.0,
    ...     max_nodes=4,
    ...     partition="compute",
    ...     max_walltime="48:00:00",
    ... )

    Use a plain dict (no import needed):

    >>> profile = ClusterProfile.from_dict({
    ...     "cores_per_node": 48,
    ...     "memory_per_node_gb": 192,
    ...     "partition": "RES",
    ... })
    """

    name: str = "generic"
    cores_per_node: int = 48
    memory_per_node_gb: float = 192.0
    max_nodes: int = 1
    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    max_walltime: str = "72:00:00"
    gpu_per_node: int = 0
    modules: list[str] | None = None

    # ------------------------------------------------------------------ #
    # Factory class methods (predefined profiles)
    # ------------------------------------------------------------------ #

    @classmethod
    def generic(cls, **overrides: Any) -> ClusterProfile:
        """Create a customizable generic profile.

        Parameters
        ----------
        **overrides
            Any :class:`ClusterProfile` field to override.

        Returns
        -------
        ClusterProfile
        """
        defaults: dict[str, Any] = dict(
            name="generic",
            cores_per_node=48,
            memory_per_node_gb=192.0,
            max_nodes=1,
            max_walltime="72:00:00",
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def mn5(cls, **overrides: Any) -> ClusterProfile:
        """MareNostrum 5 (BSC) profile.

        112 cores/node (2x Intel Sapphire Rapids), 256 GB RAM,
        ``gp_bsccase`` QOS, 72 h max walltime.

        Returns
        -------
        ClusterProfile
        """
        defaults: dict[str, Any] = dict(
            name="mn5",
            cores_per_node=112,
            memory_per_node_gb=256.0,
            max_nodes=1,
            partition=None,
            account=None,
            qos="gp_bsccase",
            max_walltime="72:00:00",
            gpu_per_node=0,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def agustina(cls, **overrides: Any) -> ClusterProfile:
        """Agustina (ICN2) profile.

        48 cores/node (2x Intel Xeon), 192 GB RAM,
        ``RES`` partition, ``icn2100`` account, 72 h max walltime.

        Returns
        -------
        ClusterProfile
        """
        defaults: dict[str, Any] = dict(
            name="agustina",
            cores_per_node=48,
            memory_per_node_gb=192.0,
            max_nodes=1,
            partition="RES",
            account="icn2100",
            qos=None,
            max_walltime="72:00:00",
            gpu_per_node=0,
        )
        defaults.update(overrides)
        return cls(**defaults)

    # ------------------------------------------------------------------ #
    # Utility methods
    # ------------------------------------------------------------------ #

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ClusterProfile:
        """Construct a :class:`ClusterProfile` from a plain dict.

        Unknown keys are silently ignored so users can pass dicts that
        contain extra metadata without causing errors.

        Parameters
        ----------
        d : dict
            Dictionary with profile fields.

        Returns
        -------
        ClusterProfile
        """
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict.

        Returns
        -------
        dict
        """
        return asdict(self)

    @classmethod
    def list_predefined(cls) -> dict[str, ClusterProfile]:
        """Return all predefined cluster profiles.

        Returns
        -------
        dict[str, ClusterProfile]
            Mapping of profile name to :class:`ClusterProfile` instance.
        """
        return {
            "generic": cls.generic(),
            "mn5": cls.mn5(),
            "agustina": cls.agustina(),
        }

    def summary(self) -> str:
        """One-line human-readable summary.

        Returns
        -------
        str
        """
        parts = [
            f"{self.name}: {self.cores_per_node} cores/node",
            f"{self.memory_per_node_gb} GB",
            f"max {self.max_nodes} node{'s' if self.max_nodes > 1 else ''}",
            f"{self.max_walltime} walltime",
        ]
        if self.partition:
            parts.append(f"partition={self.partition}")
        if self.account:
            parts.append(f"account={self.account}")
        if self.qos:
            parts.append(f"qos={self.qos}")
        if self.gpu_per_node > 0:
            parts.append(f"{self.gpu_per_node} GPU/node")
        return ", ".join(parts)
