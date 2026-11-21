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


"""Single random stream: every draw in the generalized package uses Numba's
generator, which ``seed`` (identical to old_clock.seed) controls."""
import math
import numpy as np
from numba import njit


@njit(cache=True)
def seed(value): np.random.seed(value)


@njit(cache=True)
def uniform(): return np.random.random()


@njit(cache=True)
def positive_uniform():
    u = np.random.random()
    while u == 0: u = np.random.random()
    return u


@njit(cache=True)
def exponential(rate):
    u = np.random.random()
    while u == 0: u = np.random.random()
    return -math.log(u) / rate


@njit(cache=True)
def categorical(weights):
    """Index drawn with probability proportional to nonnegative weights."""
    total = weights.sum()
    u = np.random.random() * total
    acc = 0.
    last = -1
    for k in range(len(weights)):
        if weights[k] > 0:
            last = k
            acc += weights[k]
            if u < acc: return k
    return last


@njit(cache=True)
def gated_time(eigen, complement):
    """Same law and stream use as old_clock.time_from_factors (copied for reuse
    with arbitrary sorted spectra)."""
    t = 0.; phases = 0
    for k in range(len(eigen)):
        if k < len(complement) and np.random.random() < eigen[k] / complement[k]: continue
        u = np.random.random()
        while u == 0: u = np.random.random()
        t -= math.log(u) / eigen[k]; phases += 1
    return t, phases
