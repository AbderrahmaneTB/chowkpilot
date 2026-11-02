"""Locate the SUMO installation shipped by the `eclipse-sumo` wheel.

Importing libsumo can point SUMO_HOME at the separate `sumo_data` package, so the
binaries and tools are resolved from the `sumo` package directory instead.
"""

import os
import shutil
import sys


def sumo_home() -> str:
    try:
        import sumo  # provided by the eclipse-sumo wheel

        home = os.path.dirname(os.path.abspath(sumo.__file__))
    except ImportError:
        home = os.environ.get("SUMO_HOME", "")
    if not os.environ.get("SUMO_HOME"):
        os.environ["SUMO_HOME"] = home
    return home


def binary(name: str) -> str:
    """Absolute path of a SUMO executable such as 'netconvert' or 'sumo'."""
    exe = name + (".exe" if sys.platform == "win32" else "")
    path = os.path.join(sumo_home(), "bin", exe)
    if os.path.isfile(path):
        return path
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(f"SUMO binary '{name}' not found under {sumo_home()}")


def tool(relpath: str) -> str:
    """Absolute path of a script under $SUMO_HOME/tools."""
    path = os.path.join(sumo_home(), "tools", relpath)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def data(relpath: str) -> str:
    return os.path.join(sumo_home(), "data", relpath)
