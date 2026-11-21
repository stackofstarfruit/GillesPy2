# GillesPy2 is a modeling toolkit for biochemical simulation.
# Copyright (C) 2019-2024 GillesPy2 developers.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


"""
Exact observed-species solver for a two-population reservoir model.

Between changes of the observed species Y, the hidden reservoir I is a finite
birth-death chain killed at the rate of the Y-changing reactions. The solver
samples the next observed event (time, hidden endpoint, channel) from that
killed chain directly, so hidden reservoir jumps are never simulated one by
one. Only the ReservoirModel template is supported.

Requires numba, which is not a GillesPy2 dependency (``pip install numba``).
"""
try:
    import numba  # noqa: F401
except ImportError as err:
    raise ImportError(
        "ConditionalReservoirSolver requires numba; install it with `pip install numba`."
    ) from err

from .solver import ReservoirModel, ConditionalReservoirSolver

__all__ = ["ReservoirModel", "ConditionalReservoirSolver"]
