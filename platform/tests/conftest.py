"""Shared platform test setup."""

import os

# Ray reads this at import time. Under `uv run`, Ray would otherwise start every worker through
# `uv run` again and re-resolve the whole project; local tests want the driver's own interpreter.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
