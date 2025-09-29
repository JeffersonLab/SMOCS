"""
Database utilities for SMOCS.
"""

__all__ = ["DBManager"]

def __getattr__(name):
    if name == "DBManager":
        from .mysql_api_v0 import DBManager
        return DBManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")