"""Utilities to help with testing.

Some functionalities for testing in atomate2 and useful for
other projects, either downstream or parallel.

However, if these functionalities are places in the test directory,
they will not be available to other projects via direct imports.

This module will hold the core logic for those tests.
"""

from atomate2.utils.testing.common import get_job_uuid_name_map
