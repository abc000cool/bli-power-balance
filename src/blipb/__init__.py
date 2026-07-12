"""blipb — power-balance boundary-layer-ingestion benefit model with UQ.

Modules
-------
atmosphere    ISA atmosphere and flight state
geometry      parametric axisymmetric fuselage + slender-body edge velocity
ibl           integral boundary-layer chain (Thwaites -> Head -> wake)
powerbalance  Drela 2009 ledger, Smith 1993 closed form, Hall 2017 decomposition,
              two-case (podded vs BLI) comparator
propulsion    1-D compressible fan and turboelectric transmission chain
mission       Breguet range / block-fuel accounting
uq            end-to-end model wrapper, Sobol (SALib) and PCE (chaospy) drivers
"""

__version__ = "0.1.0"

from blipb.atmosphere import FlightState, isa
from blipb.geometry import Fuselage
from blipb.powerbalance.comparator import BLIComparator, ComparatorResult

__all__ = [
    "FlightState",
    "isa",
    "Fuselage",
    "BLIComparator",
    "ComparatorResult",
    "__version__",
]
