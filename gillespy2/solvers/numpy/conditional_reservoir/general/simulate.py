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


"""Observed-event history driver for frozen-observed hidden blocks.

A model supplies
  next_event(y, h) -> (dt, h_pre, event, work)   exact joint law of the next visible event
  apply(y, h_pre, event) -> (y, h)                visible reaction's post-event update
Rows are [time, y, event]; hidden snapshots are not produced (same contract as the
audited adapter).  Status: 2 horizon reached, -1 event cap.
"""
import numpy as np
from . import rng


def run_paths(model, y0, h0, T, N, cap, seed):
    rng.seed(seed)
    rows = np.full((N, cap + 1, 3), np.nan); counts = np.zeros(N, np.int64); status = np.full(N, -1, np.int8)
    work = 0
    for rep in range(N):
        y, h, t, events = y0, h0, 0., 0
        rows[rep, 0] = (0., y, -1)
        while events < cap:
            dt, hp, ev, w = model.next_event(y, h); work += w
            if not (t + dt <= T): status[rep] = 2; break   # also catches an absorbing state (dt = inf)
            t += dt
            y, h = model.apply(y, hp, ev)
            events += 1; rows[rep, events] = (t, y, ev)
        counts[rep] = events
    return rows, counts, status, work


def grid_values(rows, counts, grid):
    """Right-continuous observed value on a grid for every path."""
    out = np.empty((len(counts), len(grid)))
    for k in range(len(counts)):
        hist = rows[k, :counts[k] + 1]
        out[k] = hist[np.searchsorted(hist[:, 0], grid, side='right') - 1, 1]
    return out
