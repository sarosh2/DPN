"""Canonical original DPN and configurable regularized DPN implementations."""
from src.models.dpn import DPN, DPNAutoGrad
from src.models.dpn_flexible import FlexibleDPN, FlexibleDPNConfig
__all__ = ["DPN", "DPNAutoGrad", "FlexibleDPN", "FlexibleDPNConfig"]
