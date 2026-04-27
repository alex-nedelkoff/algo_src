"""Pytest runner that pre-loads torch under the KMP workaround.

Workaround for the monorace env where torch's fbgemm.dll fails to load
when initialised through pytest's normal collection path. This script is
the practical workaround until the env is rebuilt cleanly.
"""
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch  # noqa: F401, E402  -- preload before pytest collects anything

import pytest  # noqa: E402

if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv[1:] or ["tests/test_dynamics/", "-v"]))
