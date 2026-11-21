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


"""Independent hidden blocks that interact only through observed species.

If the hidden coordinates X_1..X_m have no internal reaction in common and every
visible propensity depends on at most one of them, the frozen-observed generator is
a Kronecker sum and killing is additive.  Then the next visible time is
tau = min_a tau_a over independent block clocks.  The winning block reports its
killing endpoint and channel from its own exact clock.  Every other block b only
needs its state at tau conditional on not having fired by tau,

    P(X_b(tau) = x | tau_b > tau) ∝ [exp(T_b tau)]_{x_b, x},

which is sampled from a fresh row of the matrix exponential (the losing block's own
endpoint draw is discarded, never used).  Cost per visible event: one clock draw per
block plus one O(n_b^2) survivor row per losing block, instead of a dense block of
size prod n_a.
"""
import numpy as np
from scipy.linalg import expm, eigh
from . import rng


class Survivor:
    """Rows of exp(T t), normalized, for one block (dense; n up to a few hundred)."""

    def __init__(self, block):
        if getattr(block, 'lo', 0) != 0: raise ValueError('product sub-blocks must index states from 0')
        T = block.generator()
        self.T = T
        self.n = T.shape[0]
        pi = None
        if hasattr(block, 'detailed_balance') and getattr(block, 'lo', 0) == 0:
            pi = block.detailed_balance()
        elif getattr(block, 'delegate', None) is not None and block.delegate.lo == 0:
            pi_path = block.delegate.detailed_balance(); pi = np.empty(self.n); pi[block.order] = pi_path
        elif hasattr(block, 'members') and len(block.members) == 1:
            pi = block.pi
        self.symmetric = pi is not None
        if self.symmetric:
            s = np.sqrt(pi)
            S = (s[:, None] * (-T)) / s[None, :]
            lam, U = eigh((S + S.T) / 2)
            if not np.all(lam > 0): raise ArithmeticError('nonpositive survivor eigenvalue')
            self.lam, self.U, self.s = lam, U, s

    def row(self, i, t):
        if self.symmetric:
            # shift by the slowest rate so long survivals do not underflow
            w = (self.U[i] * np.exp(-(self.lam - self.lam[0]) * t)) @ self.U.T
            w = w * self.s / self.s[i]
            ok = self._clean(w)
            if ok is not None: return ok
        w = expm(self.T * t)[i]          # non-reversible block, or eigen expansion cancelled
        ok = self._clean(w)
        if ok is None: raise ArithmeticError('survivor row has a materially negative entry; no clipping')
        return ok

    @staticmethod
    def _clean(w):
        scale = w.sum()
        if not np.isfinite(scale) or scale <= 0: return None
        neg = w < 0
        if np.any(w[neg] < -1e-12 * scale): return None
        return np.where(neg, 0., w)   # |entry| <= 1e-12 * row mass: floating zero, documented

    def sample(self, i, t):
        return int(rng.categorical(self.row(i, t)))


class ProductBlock:
    """blocks: list of TreeBlock/PathBlock objects (each with draw(i) and generator()).

    draw(states) -> (time, winner, channel, new_states, work).  new_states holds the
    winner's killing endpoint J and the survivors' states at the event time; the
    caller applies the visible reaction's post-kill update.
    """

    def __init__(self, blocks):
        self.blocks = list(blocks)
        self.survivors = [None] * len(self.blocks)

    def _survivor(self, b):
        if self.survivors[b] is None: self.survivors[b] = Survivor(self.blocks[b])
        return self.survivors[b]

    def draw(self, states):
        draws = [blk.draw(int(x)) for blk, x in zip(self.blocks, states)]
        winner = int(np.argmin([d[0] for d in draws]))
        t, j, c, work = draws[winner]
        new = list(states)
        new[winner] = j
        for b in range(len(self.blocks)):
            if b != winner:
                new[b] = self._survivor(b).sample(int(states[b]), t)
        return t, winner, c, tuple(new), sum(d[3] for d in draws)
