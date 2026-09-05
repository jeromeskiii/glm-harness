"""Conftest for the integration suite.

This file is loaded **only** for tests inside ``tests/integration/`` —
the ``testpaths`` in the project's pyproject.toml does not include
``integration`` by default, so these tests are not discovered unless the
operator runs ``pytest tests/integration`` explicitly with
``GLMH_RUN_INTEGRATION=1``.
"""

from __future__ import annotations
