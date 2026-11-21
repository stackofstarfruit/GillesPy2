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


"""Generic killed birth-death (path) block.

This is the audited reservoir clock with the model formulas removed: rates are
supplied as arrays.  The ladder builder mirrors ``_kernels.green.build`` line by
line, endpoint sampling reuses ``green.endpoint`` unchanged, the time reuses
``old_clock.time_from_factors`` unchanged, and the Doob repair reuses
``conditional.sample`` unchanged.  Fed the reservoir arrays, every table and every
draw is bitwise identical to the audited kernels (tests/test_path_identity.py).

Exactness refers to real arithmetic.  Floating interlacing violations are never
clipped: before any time randomness is consumed, the endpoint-preserving Doob
clock is used instead (same policy as ``_kernels.repair``).
"""
import math, time
import numpy as np
from numba import njit
from scipy.linalg import eigh_tridiagonal
from .._kernels import green, old_clock, conditional


@njit(cache=True)
def build_arrays(birth, death, q, lo):
    """Positive Green-ladder tables for states lo..size-1 (green.build, generic).

    birth/death/q are full-length arrays; entries below lo are ignored, death[lo]
    is treated as zero (lo is a barrier), birth[size-1] is never used.
    """
    size = len(birth)
    K = size - 1
    ell = np.zeros(size); r = np.zeros(size); left = np.zeros(size); right = np.zeros(size)
    ell[lo] = q[lo]
    for i in range(lo + 1, size):
        den = birth[i - 1] + ell[i - 1]
        if den <= 0: raise ArithmeticError('nonabsorbing left component')
        left[i] = death[i] * ell[i - 1] / den; ell[i] = q[i] + left[i]
    r[K] = q[K]
    for i in range(K - 1, lo - 1, -1):
        den = death[i + 1] + r[i + 1]
        if den <= 0: raise ArithmeticError('nonabsorbing right component')
        right[i] = birth[i] * r[i + 1] / den; r[i] = q[i] + right[i]
    hp_left = np.zeros(size + 1); hp_right = np.zeros(size + 1)
    end_left = np.zeros(size, np.int64); end_right = np.zeros(size, np.int64)
    last = lo
    for i in range(lo, size):
        if left[i] == 0: last = i
        end_left[i] = last
        hp_left[i + 1] = hp_left[i] + (math.log1p(q[i] / left[i]) if left[i] > 0 else 0.)
    for i in range(lo, size): hp_right[i + 1] = hp_right[i] + (math.log1p(q[i] / right[i]) if right[i] > 0 else 0.)
    last = K
    for i in range(K, lo - 1, -1):
        if right[i] == 0: last = i
        end_right[i] = last
    data = np.zeros((5, size)); data[0] = left; data[1] = q; data[2] = right; data[3] = hp_left[:size]; data[4] = hp_right[:size]
    ends = np.zeros((2, size), np.int64); ends[0] = end_left; ends[1] = end_right
    if np.min(data) < 0 or not np.all(np.isfinite(data)): raise ArithmeticError('invalid block table')
    return data, ends


@njit(cache=True)
def choose_channel(qrow):
    """Categorical channel draw; for three channels identical to old_clock.choose_channel."""
    u = np.random.random() * qrow.sum()
    acc = 0.
    for c in range(len(qrow) - 1):
        acc += qrow[c]
        if u < acc: return c
    return len(qrow) - 1


class PathBlock:
    """Killed birth-death block on local states lo..size-1.

    birth[i], death[i]: internal adjacent rates (death[lo] and birth[size-1] unused).
    qmat[i, c]: killing (visible-event) rate of channel c in state i.
    ladder_birth/ladder_death/ladder_q optionally give the same rates evaluated in
    another floating order for the Green ladder only; the reservoir identity test
    uses this to reproduce green.build and old_clock.Block bit for bit.
    """
    def __init__(self, birth, death, qmat, lo=0, ladder_birth=None, ladder_death=None, ladder_q=None):
        started = time.perf_counter()
        birth = np.asarray(birth, float); death = np.asarray(death, float); qmat = np.asarray(qmat, float)
        if qmat.ndim == 1: qmat = qmat[:, None]
        size = len(birth)
        if len(death) != size or qmat.shape[0] != size: raise ValueError('rate arrays disagree in length')
        if not (0 <= lo < size): raise ValueError('lo outside block')
        if np.any(birth < 0) or np.any(death < 0) or np.any(qmat < 0) or not (np.all(np.isfinite(birth)) and np.all(np.isfinite(death)) and np.all(np.isfinite(qmat))):
            raise ValueError('rates must be finite and nonnegative')
        self.size, self.lo = size, lo
        self.q = qmat
        b = birth[lo:].copy(); d = death[lo:].copy(); b[-1] = 0.; d[0] = 0.
        self.birth, self.death = b, d
        self.kill = qmat[lo:].sum(axis=1)
        self.diag = self.birth + self.death + self.kill
        if np.any(self.birth[:-1] <= 0) or np.any(self.death[1:] <= 0):
            raise ValueError('path block requires positive internal adjacent rates (use TreeBlock for one-way edges)')
        self.off = -np.sqrt(self.birth[:-1] * self.death[1:])
        # Doob repair arrays (repair.Block convention: spectral rates, zero below lo).
        self.full_birth = np.zeros(size); self.full_death = np.zeros(size)
        self.full_birth[lo:] = self.birth; self.full_death[lo:] = self.death
        n = size - lo
        self.eigen = self._eigs(0, n)
        if not np.all(self.eigen > 0): raise ArithmeticError('nonpositive killed-block eigenvalue')
        lb = self.full_birth if ladder_birth is None else np.asarray(ladder_birth, float)
        ld = self.full_death if ladder_death is None else np.asarray(ladder_death, float)
        if ladder_q is None:
            lq = np.zeros(size); lq[lo:] = self.kill
        else:
            lq = np.asarray(ladder_q, float)
        self.data, self.ends = build_arrays(lb, ld, lq, lo)
        self.prefix = {0: np.empty(0)}; self.suffix = {n: np.empty(0)}
        self.spectral_seconds = time.perf_counter() - started
        self.factor_calls = 0; self.conditional_fallbacks = 0; self.conditional_steps = 0
        self.spectral_draws = 0; self.spectral_phases = 0; self.factor_errors = []

    def _eigs(self, left, right):
        if right - left == 0: return np.empty(0)
        if right - left == 1: return self.diag[left:right].copy()
        return eigh_tridiagonal(self.diag[left:right], self.off[left:right - 1], eigvals_only=True, lapack_driver='sterf', check_finite=True)

    def factors(self, i, j):
        started = time.perf_counter(); l = min(i, j) - self.lo; r = max(i, j) - self.lo + 1
        if not 0 <= l < r <= self.size - self.lo: raise ValueError('start/endpoint outside transient block')
        if l not in self.prefix: self.prefix[l] = self._eigs(0, l)
        if r not in self.suffix: self.suffix[r] = self._eigs(r, self.size - self.lo)
        nu = np.sort(np.concatenate((self.prefix[l], self.suffix[r])))
        if np.any(nu <= 0) or np.any(nu < self.eigen[:len(nu)]): raise ArithmeticError('floating eigenvalue interlacing violation; no clipping')
        self.spectral_seconds += time.perf_counter() - started; self.factor_calls += 1
        return nu

    def endpoint(self, i):
        j, _ = green.endpoint(self.data, self.ends, i)
        return j

    def draw(self, i):
        """Return (time, J, channel, work); endpoint drawn once, then time, then channel."""
        j, _ = green.endpoint(self.data, self.ends, i)
        try:
            nu = self.factors(i, j)
        except (ArithmeticError, np.linalg.LinAlgError) as ex:
            self.factor_errors.append(dict(start=int(i), endpoint=int(j), error=repr(ex)))
            t, steps = conditional.sample(self.full_birth, self.full_death, self.data, i, j, self.lo)
            self.conditional_fallbacks += 1; self.conditional_steps += steps
            return t, j, choose_channel(self.q[j]), steps
        t, phases = old_clock.time_from_factors(self.eigen, nu)
        self.spectral_draws += 1; self.spectral_phases += phases
        return t, j, choose_channel(self.q[j]), phases

    # --- dense views for audits and survivor sampling --------------------------------
    def generator(self):
        """Dense subgenerator T on lo..size-1 (audit/survivor use only)."""
        return np.diag(-self.diag) + np.diag(self.birth[:-1], 1) + np.diag(self.death[1:], -1)

    def detailed_balance(self):
        n = self.size - self.lo; pi = np.ones(n)
        for k in range(1, n): pi[k] = pi[k - 1] * self.birth[k - 1] / self.death[k]
        return pi
