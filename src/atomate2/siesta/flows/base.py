"""Base flow maker for SIESTA workflows with automatic dry-run propagation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from jobflow import Maker

logger = logging.getLogger(__name__)


@dataclass
class BaseSiestaFlowMaker(Maker):
    """
    Base flow maker with automatic dry-run, custodian, and tier propagation.

    This base class automatically propagates dry-run, custodian, and tier settings to all
    child makers that support these features. When a flow has `dry_run=True`,
    `use_custodian=True`, or `tier="basic"`, all child job makers automatically inherit
    these settings without manual configuration.

    Parameters
    ----------
    dry_run : bool
        If True, skip all SIESTA calculations and only save structures to files.
        This setting is automatically propagated to all child makers.
    dry_run_output_dir : str
        Directory to save dry-run structure files. Propagated to all child makers.
    dry_run_format : str
        Output format for dry-run structures (e.g., "cif", "xsf", "json").
        Propagated to all child makers.
    use_custodian : bool
        If True, enable custodian error handling for all child makers.
        This setting is automatically propagated to all child makers.
    custodian_handlers : list, optional
        List of custom error handlers. Propagated to all child makers.
    custodian_max_errors : int
        Maximum number of errors before giving up. Propagated to all child makers.
    tier : str, optional
        Calculation tier: "basic", "intermediate", "advanced", or "expert".
        This setting is automatically propagated to all child makers' input_set_generators.
        If None, child makers use their default tier settings.
    manager_config : dict[str, Any], optional
        Configuration for jobflow-remote resource management. When set, this dict is
        automatically propagated to all child makers. Format:
        ``{"resources": {"ntasks_per_node": 24, "time": "24:00:00", "partition": "RES"}}``.
        Useful for setting base HPC resources across all jobs in a workflow.
        If None, child makers keep their own manager_config settings.

    Examples
    --------
    >>> from atomate2.siesta.flows.surface import AdsorptionScanMaker
    >>> from atomate2.siesta.jobs.core import StaticMaker
    >>>
    >>> # Create flow with dry-run enabled
    >>> flow_maker = AdsorptionScanMaker(
    ...     slab_static_maker=StaticMaker(),
    ...     adsorbate_static_maker=StaticMaker(),
    ...     dry_run=True,  # ← Automatically propagates to both StaticMakers
    ... )
    >>>
    >>> # Both child makers now have dry_run=True automatically!
    >>> assert flow_maker.slab_static_maker.dry_run is True
    >>> assert flow_maker.adsorbate_static_maker.dry_run is True
    >>>
    >>> # Create flow with custodian enabled
    >>> flow_maker = AdsorptionScanMaker(
    ...     slab_static_maker=StaticMaker(),
    ...     adsorbate_static_maker=StaticMaker(),
    ...     use_custodian=True,  # ← Automatically propagates to both StaticMakers
    ...     custodian_max_errors=10,
    ... )
    >>>
    >>> # Both child makers now have use_custodian=True automatically!
    >>> assert flow_maker.slab_static_maker.use_custodian is True
    >>> assert flow_maker.adsorbate_static_maker.use_custodian is True
    >>>
    >>> # Create flow with tier set
    >>> flow_maker = AdsorptionScanMaker(
    ...     slab_static_maker=StaticMaker(),
    ...     adsorbate_static_maker=StaticMaker(),
    ...     tier="basic",  # ← Automatically propagates to both StaticMakers
    ... )
    >>>
    >>> # Both child makers now have tier="basic" automatically!
    >>> assert flow_maker.slab_static_maker.input_set_generator.tier == "basic"
    >>> assert flow_maker.adsorbate_static_maker.input_set_generator.tier == "basic"

    Notes
    -----
    - Propagation happens in __post_init__() after dataclass initialization
    - dry_run/use_custodian propagate to makers with those attributes
    - tier propagates to makers with input_set_generator.tier attribute
    - Safe to use with makers that don't support these features (they're ignored)
    - Propagation is recursive: works with nested flows
    """

    dry_run: bool = False
    dry_run_output_dir: str = "dry_run_output"
    dry_run_format: str = "cif"
    use_custodian: bool = False
    custodian_handlers: list | None = None
    custodian_max_errors: int = 5
    tier: str | None = None
    manager_config: dict[str, Any] | None = None

    def __post_init__(self):
        """Propagate dry-run, custodian, and tier settings to all child makers after initialization."""
        if self.dry_run:
            logger.info(
                f"{self.__class__.__name__}: dry_run=True, propagating to child makers"
            )
            self._propagate_dry_run()

        if self.use_custodian:
            logger.info(
                f"{self.__class__.__name__}: use_custodian=True, propagating to child makers"
            )
            self._propagate_custodian()

        if self.tier is not None:
            logger.info(
                f"{self.__class__.__name__}: tier='{self.tier}', propagating to child makers"
            )
            self._propagate_tier()

        if self.manager_config is not None:
            logger.info(
                f"{self.__class__.__name__}: manager_config set, propagating to child makers"
            )
            self._propagate_manager_config()

    def _propagate_dry_run(self) -> None:
        """
        Enable dry-run for all child makers that support it.

        This method iterates through all attributes of the flow maker and:
        1. Identifies attributes that are Maker instances
        2. Checks if they have dry_run support (have 'dry_run' attribute)
        3. Enables dry_run and sets output directory/format

        Handles both single makers and lists of makers (for multi-maker flows).
        """
        # Get all field names from the dataclass
        if hasattr(self, "__dataclass_fields__"):
            field_names = self.__dataclass_fields__.keys()
        else:
            # Fallback for non-dataclass makers
            field_names = [
                name
                for name in dir(self)
                if not name.startswith("_") and not callable(getattr(self, name))
            ]

        for field_name in field_names:
            try:
                field_value = getattr(self, field_name)
            except AttributeError:
                continue

            # Handle single maker
            if hasattr(field_value, "dry_run"):
                self._enable_dry_run_for_maker(field_value, field_name)

            # Handle list of makers
            elif isinstance(field_value, list):
                for i, item in enumerate(field_value):
                    if hasattr(item, "dry_run"):
                        self._enable_dry_run_for_maker(item, f"{field_name}[{i}]")

    def _enable_dry_run_for_maker(self, maker: Maker, maker_name: str) -> None:
        """
        Enable dry-run for a specific maker.

        Parameters
        ----------
        maker : Maker
            The maker to enable dry-run for.
        maker_name : str
            Name of the maker (for logging).
        """
        maker.dry_run = True
        maker.dry_run_output_dir = self.dry_run_output_dir
        maker.dry_run_format = self.dry_run_format

        logger.info(
            f"  → Enabled dry_run for {maker_name} (type: {maker.__class__.__name__})"
        )

        # Recursively propagate if this maker is also a flow
        if isinstance(maker, BaseSiestaFlowMaker):
            maker._propagate_dry_run()

    def _propagate_custodian(self) -> None:
        """
        Enable custodian for all child makers that support it.

        This method iterates through all attributes of the flow maker and:
        1. Identifies attributes that are Maker instances
        2. Checks if they have custodian support (have 'use_custodian' attribute)
        3. Enables custodian and sets handlers/max_errors

        Handles both single makers and lists of makers (for multi-maker flows).
        """
        # Get all field names from the dataclass
        if hasattr(self, "__dataclass_fields__"):
            field_names = self.__dataclass_fields__.keys()
        else:
            # Fallback for non-dataclass makers
            field_names = [
                name
                for name in dir(self)
                if not name.startswith("_") and not callable(getattr(self, name))
            ]

        for field_name in field_names:
            try:
                field_value = getattr(self, field_name)
            except AttributeError:
                continue

            # Handle single maker
            if hasattr(field_value, "use_custodian"):
                self._enable_custodian_for_maker(field_value, field_name)

            # Handle list of makers
            elif isinstance(field_value, list):
                for i, item in enumerate(field_value):
                    if hasattr(item, "use_custodian"):
                        self._enable_custodian_for_maker(item, f"{field_name}[{i}]")

    def _enable_custodian_for_maker(self, maker: Maker, maker_name: str) -> None:
        """
        Enable custodian for a specific maker (internal use during __post_init__).

        Parameters
        ----------
        maker : Maker
            The maker to enable custodian for.
        maker_name : str
            Name of the maker (for logging).
        """
        self.propagate_custodian_to_maker(maker)

        logger.info(
            f"  → Enabled custodian for {maker_name} "
            f"(type: {maker.__class__.__name__}, max_errors={self.custodian_max_errors})"
        )

        # Recursively propagate if this maker is also a flow
        if isinstance(maker, BaseSiestaFlowMaker):
            maker._propagate_custodian()

    def propagate_custodian_to_maker(self, maker: Maker):
        """
        Propagate custodian settings from this flow maker to a child maker.

        Use this method when dynamically creating makers in make() that need
        to inherit custodian settings from the flow.

        Parameters
        ----------
        maker : Maker
            The maker to propagate custodian settings to.

        Example
        -------
        >>> # In a flow maker's make() method:
        >>> maker = self.static_maker.scf()  # Creates new maker
        >>> self.propagate_custodian_to_maker(maker)  # Propagate settings
        """
        if self.use_custodian:
            maker.use_custodian = True
            maker.custodian_max_errors = self.custodian_max_errors
            if self.custodian_handlers is not None:
                maker.custodian_handlers = self.custodian_handlers

    def _propagate_tier(self) -> None:
        """
        Set tier for all child makers that support it.

        This method iterates through all attributes of the flow maker and:
        1. Identifies attributes that are Maker instances
        2. Checks if they have an input_set_generator with tier attribute
        3. Sets the tier parameter

        Handles both single makers and lists of makers (for multi-maker flows).
        """
        # Get all field names from the dataclass
        if hasattr(self, "__dataclass_fields__"):
            field_names = self.__dataclass_fields__.keys()
        else:
            # Fallback for non-dataclass makers
            field_names = [
                name
                for name in dir(self)
                if not name.startswith("_") and not callable(getattr(self, name))
            ]

        for field_name in field_names:
            try:
                field_value = getattr(self, field_name)
            except AttributeError:
                continue

            # Handle single maker
            if hasattr(field_value, "input_set_generator"):
                self._set_tier_for_maker(field_value, field_name)

            # Handle list of makers
            elif isinstance(field_value, list):
                for i, item in enumerate(field_value):
                    if hasattr(item, "input_set_generator"):
                        self._set_tier_for_maker(item, f"{field_name}[{i}]")

    def _set_tier_for_maker(self, maker: Maker, maker_name: str) -> None:
        """
        Set tier for a specific maker's input set generator.

        IMPORTANT: This method recreates the input_set_generator to ensure
        tier defaults are properly applied, while preserving ONLY the user's
        explicit parameters (not tier defaults from the maker's default tier).

        We use _explicit_user_params to distinguish user-provided params from
        tier defaults, ensuring tier propagation works correctly.

        Parameters
        ----------
        maker : Maker
            The maker to set tier for.
        maker_name : str
            Name of the maker (for logging).
        """
        if hasattr(maker.input_set_generator, "tier"):
            # Get the current generator and its EXPLICIT user params only
            current_gen = maker.input_set_generator
            generator_class = current_gen.__class__

            # Use _explicit_user_params to preserve ONLY user-set params,
            # not tier defaults from the maker's original tier
            explicit_params = getattr(current_gen, "_explicit_user_params", {})

            # Recreate the generator with new tier and only explicit params
            # The new tier defaults will be applied, and explicit params will override
            maker.input_set_generator = generator_class(
                user_params=explicit_params,  # Only explicit user params
                tier=self.tier,  # New tier
            )

            if explicit_params:
                logger.info(
                    f"  → Set tier='{self.tier}' for {maker_name} "
                    f"(preserving {len(explicit_params)} explicit user params, "
                    f"type: {maker.__class__.__name__})"
                )
            else:
                logger.info(
                    f"  → Set tier='{self.tier}' for {maker_name} "
                    f"(type: {maker.__class__.__name__})"
                )

        # Recursively propagate if this maker is also a flow
        if isinstance(maker, BaseSiestaFlowMaker):
            maker._propagate_tier()

    def _propagate_manager_config(self) -> None:
        """
        Set manager_config for all child makers that support it.

        This method iterates through all attributes of the flow maker and:
        1. Identifies attributes that are Maker instances
        2. Checks if they have manager_config support (have 'manager_config' attribute)
        3. Sets manager_config to the flow's value

        Handles both single makers and lists of makers (for multi-maker flows).
        """
        if hasattr(self, "__dataclass_fields__"):
            field_names = self.__dataclass_fields__.keys()
        else:
            field_names = [
                name
                for name in dir(self)
                if not name.startswith("_") and not callable(getattr(self, name))
            ]

        for field_name in field_names:
            try:
                field_value = getattr(self, field_name)
            except AttributeError:
                continue

            # Handle single maker
            if hasattr(field_value, "manager_config"):
                self._set_manager_config_for_maker(field_value, field_name)

            # Handle list of makers
            elif isinstance(field_value, list):
                for i, item in enumerate(field_value):
                    if hasattr(item, "manager_config"):
                        self._set_manager_config_for_maker(item, f"{field_name}[{i}]")

    def _set_manager_config_for_maker(self, maker: Maker, maker_name: str) -> None:
        """
        Set manager_config for a specific maker.

        Parameters
        ----------
        maker : Maker
            The maker to set manager_config for.
        maker_name : str
            Name of the maker (for logging).
        """
        self.propagate_manager_config_to_maker(maker)

        logger.info(
            f"  → Set manager_config for {maker_name} "
            f"(type: {maker.__class__.__name__})"
        )

        # Recursively propagate if this maker is also a flow
        if isinstance(maker, BaseSiestaFlowMaker):
            maker._propagate_manager_config()

    def propagate_manager_config_to_maker(self, maker: Maker):
        """
        Propagate manager_config from this flow maker to a child maker.

        Use this method when dynamically creating makers in make() that need
        to inherit manager_config settings from the flow.

        Parameters
        ----------
        maker : Maker
            The maker to propagate manager_config to.

        Example
        -------
        >>> # In a flow maker's make() method:
        >>> maker = self.static_maker.scf()  # Creates new maker
        >>> self.propagate_manager_config_to_maker(maker)  # Propagate settings
        """
        if self.manager_config is not None:
            maker.manager_config = self.manager_config
