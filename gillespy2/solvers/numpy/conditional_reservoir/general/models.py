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


"""Benchmark/test models for the generalized clock, each with a compiled native SSA.

All three have an observed count y and a hidden part frozen between visible events.

* BranchedMachine: one hidden molecule moving on a star-shaped tree of conformations
  (3 arms of length L).  Inward rates are modulated by y.  Reaching an arm tip can
  emit an observed product (y+1, molecule resets to the centre); y decays at xi*y.
  Optional one-way (irreversible) step on arm 2 exercises the SCC split.
* ReceptorPools: two independent bounded receptor pools X in [0,K1], Z in [0,K2];
  binding is inhibited by y; each bound receptor signals (y+1); y decays.
* Transcript: unbounded hidden mRNA m (birth a/(1+y/H), death g m); translation
  kappa*m makes y+1; y decays.  Uses exact moving windows.
"""
import math
import numpy as np
from numba import njit
from .treeblock import TreeBlock
from .pathblock import PathBlock
from .product import ProductBlock
from .window import WindowedChain


# ----------------------------------------------------------------------------- tree
class BranchedMachine:
    def __init__(self, L=40, f=20., rin=20., H=50., kappa=1., xi=0.01, one_way=False, arms=3):
        self.L, self.f, self.rin, self.H, self.kappa, self.xi, self.one_way, self.arms = L, f, rin, H, kappa, xi, one_way, arms
        self.n = 1 + arms * L
        self.cache = {}
        nbr = -np.ones((self.n, 3 if arms <= 2 else arms), np.int64); kind = np.zeros_like(nbr); tip = np.zeros(self.n, np.bool_)
        self.edge_list = []
        for a in range(arms):
            prev = 0
            for k in range(1, L + 1):
                node = 1 + a * L + (k - 1)
                blocked = one_way and a == 2 % arms and k == L // 2
                self.edge_list.append((prev, node, blocked))
                prev = node
            tip[1 + a * L + L - 1] = True
        deg = np.zeros(self.n, np.int64)
        for u, v, blocked in self.edge_list:
            nbr[u, deg[u]] = v; kind[u, deg[u]] = 1; deg[u] += 1                 # outward u->v
            nbr[v, deg[v]] = u; kind[v, deg[v]] = 3 if blocked else 2; deg[v] += 1  # inward v->u
        self.nbr, self.kind, self.tip = nbr, kind, tip

    def block(self, y):
        if y not in self.cache:
            inward = self.rin * (1 + y / self.H)
            edges = [(u, v, self.f, 0. if blocked else inward) for u, v, blocked in self.edge_list]
            q = np.zeros((self.n, 2)); q[self.tip, 0] = self.kappa; q[:, 1] = self.xi * y
            self.cache[y] = TreeBlock(self.n, edges, q)
        return self.cache[y]

    def next_event(self, y, h):
        t, j, c, w = self.block(y).draw(h)
        return t, j, c, w

    def apply(self, y, h, ev):
        return (y + 1, 0) if ev == 0 else (y - 1, h)

    def native(self, y0, h0, T, N, cap, seed):
        return _native_tree(self.nbr, self.kind, self.tip, self.f, self.rin, self.H, self.kappa, self.xi, y0, h0, T, N, cap, seed)


@njit(cache=True)
def _native_tree(nbr, kind, tip, f, rin, H, kappa, xi, y0, h0, T, N, cap, seed):
    np.random.seed(seed)
    rows = np.full((N, cap + 1, 3), np.nan); counts = np.zeros(N, np.int64); status = np.full(N, -1, np.int8); work = 0
    D = nbr.shape[1]; rates = np.zeros(D + 2)
    for rep in range(N):
        y = y0; h = h0; t = 0.; events = 0; rows[rep, 0, 0] = 0.; rows[rep, 0, 1] = y; rows[rep, 0, 2] = -1
        while events < cap:
            inward = rin * (1 + y / H); total = 0.
            for k in range(D):
                kd = kind[h, k]
                r = f if kd == 1 else (inward if kd == 2 else 0.)
                rates[k] = r; total += r
            rates[D] = kappa if tip[h] else 0.; rates[D + 1] = xi * y; total += rates[D] + rates[D + 1]
            u = np.random.random()
            while u == 0: u = np.random.random()
            t -= math.log(u) / total; work += 1
            if t > T: status[rep] = 2; break
            v = np.random.random() * total; acc = 0.; e = D + 1
            for k in range(D + 2):
                acc += rates[k]
                if v < acc: e = k; break
            if e < D: h = nbr[h, e]; continue
            if e == D: y += 1; h = 0; ev = 0
            else: y -= 1; ev = 1
            events += 1; rows[rep, events, 0] = t; rows[rep, events, 1] = y; rows[rep, events, 2] = ev
        counts[rep] = events
    return rows, counts, status, work


# ----------------------------------------------------------------------------- product
class ReceptorPools:
    def __init__(self, K1=64, K2=64, kon1=5., koff1=5., kon2=3., koff2=8., k1=0.002, k2=0.004, xi=0.05, H=20.):
        self.K1, self.K2, self.kon1, self.koff1, self.kon2, self.koff2 = K1, K2, kon1, koff1, kon2, koff2
        self.k1, self.k2, self.xi, self.H = k1, k2, xi, H
        self.cache = {}

    def block(self, y):
        if y not in self.cache:
            L = 1. / (1. + y / self.H)
            x = np.arange(self.K1 + 1.); z = np.arange(self.K2 + 1.)
            q1 = np.stack([self.k1 * x, np.full_like(x, self.xi * y)], axis=1)
            b1 = PathBlock(self.kon1 * L * (self.K1 - x), self.koff1 * x, q1)
            b2 = PathBlock(self.kon2 * L * (self.K2 - z), self.koff2 * z, (self.k2 * z)[:, None])
            self.cache[y] = ProductBlock([b1, b2])
        return self.cache[y]

    def next_event(self, y, h):
        t, winner, c, new, w = self.block(y).draw(h)
        ev = (0 if c == 0 else 1) if winner == 0 else 2
        return t, new, ev, w

    def apply(self, y, h, ev):
        return (y - 1, h) if ev == 1 else (y + 1, h)

    def native(self, y0, h0, T, N, cap, seed):
        return _native_pools(self.K1, self.K2, self.kon1, self.koff1, self.kon2, self.koff2, self.k1, self.k2, self.xi, self.H,
                             y0, h0[0], h0[1], T, N, cap, seed)


@njit(cache=True)
def _native_pools(K1, K2, kon1, koff1, kon2, koff2, k1, k2, xi, H, y0, x0, z0, T, N, cap, seed):
    np.random.seed(seed)
    rows = np.full((N, cap + 1, 3), np.nan); counts = np.zeros(N, np.int64); status = np.full(N, -1, np.int8); work = 0
    for rep in range(N):
        y = y0; x = x0; z = z0; t = 0.; events = 0; rows[rep, 0, 0] = 0.; rows[rep, 0, 1] = y; rows[rep, 0, 2] = -1
        while events < cap:
            L = 1. / (1. + y / H)
            r0 = kon1 * L * (K1 - x); r1 = koff1 * x; r2 = kon2 * L * (K2 - z); r3 = koff2 * z
            r4 = k1 * x; r5 = xi * y; r6 = k2 * z
            total = r0 + r1 + r2 + r3 + r4 + r5 + r6
            u = np.random.random()
            while u == 0: u = np.random.random()
            t -= math.log(u) / total; work += 1
            if t > T: status[rep] = 2; break
            v = np.random.random() * total
            if v < r0: x += 1; continue
            v -= r0
            if v < r1: x -= 1; continue
            v -= r1
            if v < r2: z += 1; continue
            v -= r2
            if v < r3: z -= 1; continue
            v -= r3
            if v < r4: y += 1; ev = 0
            elif v < r4 + r5: y -= 1; ev = 1
            else: y += 1; ev = 2
            events += 1; rows[rep, events, 0] = t; rows[rep, events, 1] = y; rows[rep, events, 2] = ev
        counts[rep] = events
    return rows, counts, status, work


# ----------------------------------------------------------------------------- window
class Transcript:
    def __init__(self, a=400., g=2., kappa=0.005, xi=0.02, H=50., W=32):
        self.a, self.g, self.kappa, self.xi, self.H, self.W = a, g, kappa, xi, H, W
        self.cache = {}

    def chain(self, y):
        if y not in self.cache:
            birth = self.a / (1. + y / self.H); g, kappa, decay = self.g, self.kappa, self.xi * y
            self.cache[y] = WindowedChain(lambda i: birth, lambda i: g * i, lambda i: np.array([kappa * i, decay]), W=self.W)
        return self.cache[y]

    def next_event(self, y, h):
        t, j, c, w, windows = self.chain(y).draw(h)
        return t, j, c, w

    def apply(self, y, h, ev):
        return (y + 1, h) if ev == 0 else (y - 1, h)

    def native(self, y0, h0, T, N, cap, seed):
        return _native_transcript(self.a, self.g, self.kappa, self.xi, self.H, y0, h0, T, N, cap, seed)


@njit(cache=True)
def _native_transcript(a, g, kappa, xi, H, y0, m0, T, N, cap, seed):
    np.random.seed(seed)
    rows = np.full((N, cap + 1, 3), np.nan); counts = np.zeros(N, np.int64); status = np.full(N, -1, np.int8); work = 0
    for rep in range(N):
        y = y0; m = m0; t = 0.; events = 0; rows[rep, 0, 0] = 0.; rows[rep, 0, 1] = y; rows[rep, 0, 2] = -1
        while events < cap:
            r0 = a / (1. + y / H); r1 = g * m; r2 = kappa * m; r3 = xi * y
            total = r0 + r1 + r2 + r3
            u = np.random.random()
            while u == 0: u = np.random.random()
            t -= math.log(u) / total; work += 1
            if t > T: status[rep] = 2; break
            v = np.random.random() * total
            if v < r0: m += 1; continue
            if v < r0 + r1: m -= 1; continue
            if v < r0 + r1 + r2: y += 1; ev = 0
            else: y -= 1; ev = 1
            events += 1; rows[rep, events, 0] = t; rows[rep, events, 1] = y; rows[rep, events, 2] = ev
        counts[rep] = events
    return rows, counts, status, work


# ----------------------------------------------------------------------------- published models
class EnzymeCGP:
    """Cao, Gillespie & Petzold, J. Chem. Phys. 123, 144917 (2005), stiff enzyme-substrate:
    E + S <-> ES (c1, c2), ES -> E + P (c3).  Observed: P.  Given P, conservation gives
    E = E0 - C and S = S0 - P - C, so the hidden complex count C is a killed birth-death
    chain on 0..min(E0, S0-P); catalysis (rate c3 C) is the only visible event (C -> C-1, P+1).
    Eq. 33: c=(1e-4, 1, 1e-4), E0=220, S0=3000.  Eq. 34: c=(1e-4, 6, 1e-3), E0=10, S0=3000."""

    def __init__(self, c1=1e-4, c2=1., c3=1e-4, E0=220, S0=3000):
        self.c1, self.c2, self.c3, self.E0, self.S0 = c1, c2, c3, E0, S0
        self.cache = {}

    def block(self, P):
        if P not in self.cache:
            n = min(self.E0, self.S0 - P); C = np.arange(n + 1.)
            self.cache[P] = PathBlock(self.c1 * (self.E0 - C) * (self.S0 - P - C), self.c2 * C, (self.c3 * C)[:, None])
        return self.cache[P]

    def next_event(self, P, C):
        if P >= self.S0: return math.inf, C, 0, 0
        t, j, c, w = self.block(P).draw(C)
        return t, j, c, w

    def apply(self, P, C, ev):
        return P + 1, C - 1

    def native(self, y0, h0, T, N, cap, seed):
        return _native_enzyme(self.c1, self.c2, self.c3, self.E0, self.S0, y0, h0, T, N, cap, seed)


@njit(cache=True)
def _native_enzyme(c1, c2, c3, E0, S0, P0, C0, T, N, cap, seed):
    np.random.seed(seed)
    rows = np.full((N, cap + 1, 3), np.nan); counts = np.zeros(N, np.int64); status = np.full(N, -1, np.int8); work = 0
    for rep in range(N):
        P = P0; C = C0; t = 0.; events = 0; rows[rep, 0, 0] = 0.; rows[rep, 0, 1] = P; rows[rep, 0, 2] = -1
        while events < cap:
            r0 = c1 * (E0 - C) * (S0 - P - C); r1 = c2 * C; r2 = c3 * C
            total = r0 + r1 + r2
            if total <= 0: status[rep] = 2; break
            u = np.random.random()
            while u == 0: u = np.random.random()
            t -= math.log(u) / total; work += 1
            if t > T: status[rep] = 2; break
            v = np.random.random() * total
            if v < r0: C += 1; continue
            if v < r0 + r1: C -= 1; continue
            C -= 1; P += 1
            events += 1; rows[rep, events, 0] = t; rows[rep, events, 1] = P; rows[rep, events, 2] = 0
        counts[rep] = events
    return rows, counts, status, work


class TelegraphRaj:
    """Raj, Peskin, Tranchina, Vargas & Tyagi, PLoS Biol 4:e309 (2006), construct E-YFP-M1-7x:
    lambda/delta = 2.44 (on), gamma/delta = 2.49 (off), mu/delta = 910 (transcription while on);
    delta = ln 2 / 4 h (reported mRNA half-life about 4 h).  Time unit: hours.  Observed: mRNA m.
    Hidden: promoter (0 off, 1 on), a two-state block; transcription (m+1) and decay (m-1) are visible."""

    def __init__(self, half_life_h=4.0, lam=2.44, gam=2.49, mu=910.):
        self.delta = math.log(2.) / half_life_h
        self.kon, self.koff, self.mu = lam * self.delta, gam * self.delta, mu * self.delta
        self.cache = {}

    def block(self, m):
        if m not in self.cache:
            q = np.array([[0., self.delta * m], [self.mu, self.delta * m]])
            self.cache[m] = PathBlock(np.array([self.kon, 0.]), np.array([0., self.koff]), q)
        return self.cache[m]

    def next_event(self, m, g):
        t, j, c, w = self.block(m).draw(g)
        return t, j, c, w

    def apply(self, m, g, ev):
        return (m + 1, g) if ev == 0 else (m - 1, g)

    def native(self, y0, h0, T, N, cap, seed):
        return _native_telegraph(self.kon, self.koff, self.mu, self.delta, y0, h0, T, N, cap, seed)


@njit(cache=True)
def _native_telegraph(kon, koff, mu, delta, m0, g0, T, N, cap, seed):
    np.random.seed(seed)
    rows = np.full((N, cap + 1, 3), np.nan); counts = np.zeros(N, np.int64); status = np.full(N, -1, np.int8); work = 0
    for rep in range(N):
        m = m0; g = g0; t = 0.; events = 0; rows[rep, 0, 0] = 0.; rows[rep, 0, 1] = m; rows[rep, 0, 2] = -1
        while events < cap:
            r0 = kon if g == 0 else 0.; r1 = koff if g == 1 else 0.; r2 = mu if g == 1 else 0.; r3 = delta * m
            total = r0 + r1 + r2 + r3
            u = np.random.random()
            while u == 0: u = np.random.random()
            t -= math.log(u) / total; work += 1
            if t > T: status[rep] = 2; break
            v = np.random.random() * total
            if v < r0: g = 1; continue
            if v < r0 + r1: g = 0; continue
            if v < r0 + r1 + r2: m += 1; ev = 0
            else: m -= 1; ev = 1
            events += 1; rows[rep, events, 0] = t; rows[rep, events, 1] = m; rows[rep, events, 2] = ev
        counts[rep] = events
    return rows, counts, status, work
