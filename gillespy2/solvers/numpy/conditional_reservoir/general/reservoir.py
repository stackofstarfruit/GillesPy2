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


"""Reservoir dispatcher without the reducibility rejection.

Identical to _kernels.dispatch.spectral_paths except for block construction: an
irreducible block is still the audited repair.Block (so irreducible requests are
bit-for-bit unchanged), while a block that repair.Block rejects as reducible
(b = 0 or e = 0 make some hidden births vanish) becomes a TreeBlock on the same path,
which splits it into strongly connected components instead of falling back to native
simulation.
"""
import math
import numpy as np
from .._kernels import native_only, repair
from .._kernels.dispatch import arguments
from .treeblock import TreeBlock


class ReducibleReservoirBlock:
    """Reservoir hidden block with one-way edges, via TreeBlock, in reservoir indexing."""

    def __init__(self, H, K, a, b, xi, g, e, y):
        self.lo = lo = 0 if y else 1
        ids = np.arange(K + 1)
        birth = (b * (y / H) + e * (ids / K)) * (K - ids)
        death = g * ids.astype(float)
        q = np.zeros((K + 1, 3)); q[:, 0] = a * ((H - y) / H) * ids; q[:, 1] = xi * y
        if y == 0: q[1, 2] = g
        n = K + 1 - lo
        edges = [(k - lo, k + 1 - lo, birth[k], death[k + 1]) for k in range(lo, K)]
        self.tree = TreeBlock(n, edges, q[lo:])
        self.conditional_fallbacks = 0; self.conditional_steps = 0; self.spectral_phases = 0; self.spectral_seconds = self.tree.spectral_seconds

    def draw(self, i):
        t, j, c, w = self.tree.draw(i - self.lo)
        self.spectral_phases += w
        self.conditional_fallbacks = getattr(self.tree, 'conditional_fallbacks', 0)
        return t, j + self.lo, c, w


def make_block(H, K, a, b, xi, g, e, y):
    try:
        return repair.Block(H, K, a, b, xi, g, e, y)
    except ValueError as ex:
        if 'irreducible' not in str(ex): raise
        return ReducibleReservoirBlock(H, K, a, b, xi, g, e, y)


def spectral_paths(c, N):
    H, K, a, b, xi, g, e, y0, i0, threshold, T, _, cap = arguments(c, N)
    rows = np.full((N, cap + 1, 4), np.nan); counts = np.zeros(N, np.int64); status = np.full(N, -2, np.int8)
    rows[:, 0] = [0., y0, i0, -1.]; cache = {}; failed = {}; queries = 0; phases = 0; fallbacks = 0; native_work = 0; abort = None
    for rep in range(N):
        y = y0; i = i0; t = 0.; events = 0; status[rep] = -1
        while y < threshold and y + i > 0 and events < cap:
            if y not in cache and y not in failed:
                try: cache[y] = make_block(H, K, a, b, xi, g, e, y)
                except (ArithmeticError, ValueError, np.linalg.LinAlgError) as ex: failed[y] = repr(ex)
            if y in failed:
                rest = dict(c, y0=y, i0=i, T=T - t, cap=cap - events, N=1)
                rr, nn, ss, ww = native_only.native_paths(*arguments(rest)); n = int(nn[0])
                if n: rows[rep, events + 1:events + n + 1] = rr[0, 1:n + 1]; rows[rep, events + 1:events + n + 1, 0] += t
                events += n; status[rep] = ss[0]; native_work += ww; fallbacks += 1
                break
            try:
                dt, j, channel, num = cache[y].draw(i)
                if not math.isfinite(dt) or dt < 0: raise ArithmeticError('nonfinite/negative conditional time')
            except (ArithmeticError, ValueError, np.linalg.LinAlgError) as ex:
                status[rep] = -2; abort = dict(path=rep, Y=y, I=i, time=t, error=repr(ex), stage='after endpoint; no retry')
                break
            queries += 1; phases += num
            if t + dt > T: status[rep] = 2; break
            t += dt; i = j
            if channel == 0: y += 1
            elif channel == 1: y -= 1
            else: i = 0; break
            events += 1; rows[rep, events] = [t, y, i, channel]
        if y >= threshold: status[rep] = 1
        elif y + i == 0: status[rep] = 0
        counts[rep] = events
        if abort is not None: break
    meta = dict(queries=queries, phases=phases, blocks=len(cache), failed_setup=failed, fallback_paths=fallbacks,
                native_fallback_work=int(native_work),
                reducible_blocks=sum(isinstance(z, ReducibleReservoirBlock) for z in cache.values()),
                conditional_fallbacks=sum(z.conditional_fallbacks for z in cache.values()), abort=abort)
    return rows, counts, status, meta
