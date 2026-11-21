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


"""Exact moving windows for unbounded (or very large) hidden birth-death species.

Take a window [lo, hi] containing the current hidden state and treat the two jumps
out of it (birth at hi, death at lo) as extra killing channels.  Leaving the window
is a stopping time, so by the strong Markov property the clock for the window plus a
restart from the post-exit state reproduces the full chain's joint law of the next
visible time, hidden endpoint and channel.  No truncation is involved: the chain can
wander arbitrarily far, it just pays one window draw per crossing.

Windows come from a fixed overlapping tiling (tile m covers [(m-1)W, (m+2)W-1], used
for states in [mW, (m+1)W-1]), so at most one block per (observed state, tile) is ever
built and cached.
"""
import numpy as np
from .pathblock import PathBlock


class WindowedChain:
    """birth(i), death(i): internal rates; kills(i): vector of visible-channel rates.

    K: optional capacity (inclusive); None means unbounded.  W: tile stride.
    """

    def __init__(self, birth, death, kills, W=32, K=None):
        if W < 1: raise ValueError('W must be >= 1')
        self.birth, self.death, self.kills, self.W, self.K = birth, death, kills, int(W), K
        self.C = len(np.atleast_1d(kills(0)))
        self.tiles = {}
        self.windows_built = 0; self.window_draws = 0

    def _tile(self, m):
        if m not in self.tiles:
            W = self.W
            lo = max(0, (m - 1) * W); hi = (m + 2) * W - 1
            if self.K is not None: hi = min(hi, self.K)
            states = np.arange(lo, hi + 1)
            n = len(states)
            b = np.array([self.birth(int(s)) for s in states], float)
            d = np.array([self.death(int(s)) for s in states], float)
            q = np.zeros((n, self.C + 2))
            for k, s in enumerate(states): q[k, :self.C] = self.kills(int(s))
            top_open = self.K is None or hi < self.K
            if top_open: q[-1, self.C] = b[-1]          # exit upward: pseudo channel C
            if lo > 0: q[0, self.C + 1] = d[0]           # exit downward: pseudo channel C+1
            b[-1] = 0.; d[0] = 0.
            self.tiles[m] = (lo, hi, PathBlock(b, d, q))
            self.windows_built += 1
        return self.tiles[m]

    def draw(self, i):
        """Return (time, J, channel, work, windows) for a start at hidden state i."""
        t = 0.; work = 0; windows = 0
        while True:
            lo, hi, blk = self._tile(i // self.W)
            dt, jj, c, w = blk.draw(i - lo)
            t += dt; work += w; windows += 1; self.window_draws += 1
            if c < self.C:
                return t, lo + jj, c, work, windows
            i = hi + 1 if c == self.C else lo - 1
