"""Paths to small data files distributed with Adora."""

from pathlib import Path


FE_I_6301_6302_LINE_LIST = Path(__file__).with_name(
    "kurucz_6301_6302.linelist"
)

__all__ = ["FE_I_6301_6302_LINE_LIST"]
