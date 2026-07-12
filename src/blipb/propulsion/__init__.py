"""Propulsion chain: 1-D compressible fan + turboelectric transmission."""

from blipb.propulsion.fan import FanResult, fan_jet
from blipb.propulsion.turboelectric import TurboelectricChain

__all__ = ["FanResult", "fan_jet", "TurboelectricChain"]
