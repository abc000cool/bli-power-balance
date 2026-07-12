"""Small numpy 1.x / 2.x compatibility shims.

numpy renamed trapz -> trapezoid in 2.0; the chaospy dependency chain
currently pins numpy < 2 on platforms without a C++ toolchain, so blipb
supports both.
"""

import numpy as np

trapezoid = getattr(np, "trapezoid", None) or np.trapz  # noqa: NPY201

__all__ = ["trapezoid"]
