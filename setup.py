# Minimal shim so `pip install -e .` works on older pip versions that
# don't yet support PEP 660 editable installs from pyproject.toml alone.
# All real metadata lives in pyproject.toml.
from setuptools import setup

setup()
