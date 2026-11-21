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


"""Killed CTMC block on a tree: the path-graph clock with the path assumption removed.

Mathematics (see ../../../GENERALIZATION.md):

* Endpoint.  P(J=j | start i) = G_ij q_j with G = A^{-1}, A = -T.
* Time.  For a tree, the (i,j) cofactor of sI+A is c_ij det(sI + A[V\\P]) where P is
  the unique tree path i->j and c_ij > 0 is the product of rates along it.  With
  one-way edges A is block triangular over the strongly connected components (each a
  subtree, hence reversible), so

      E[exp(-s tau) | J=j] = prod over components C met by P of
          det(sI + A_C[C\\P]) det(A_C) / (det(sI + A_C) det(A_C[C\\P])).

  Each factor is the audited Bernoulli-gated exponential sum with lambda=spec(A_C) and
  nu=spec(A_C[C\\P]), by generalized Cauchy interlacing (removing |C cap P| vertices of
  a symmetrizable matrix).  C\\P is a union of "hanging subtrees", one per directed
  edge (p,w) with p on P and w off P, so only 2(n-1) spectra can ever be needed.
* Repair.  Any failure while preparing those factors happens before time randomness;
  the endpoint is kept and the Doob h-transformed chain (h = G[:,j] q_j) supplies
  the conditional time, exactly as in _kernels.repair.

Path graphs whose edges are all two-way are delegated to PathBlock, so the audited
O(log K) ladder and its draws are reproduced bit for bit.
"""
import time
from collections import deque
import numpy as np
from . import rng
from .pathblock import PathBlock, choose_channel


def _path_order(n, adj):
    if n == 1: return [0]
    if any(len(v) > 2 for v in adj): return None
    ends = [v for v in range(n) if len(adj[v]) == 1]
    if len(ends) != 2: return None
    order = [min(ends)]; prev = -1
    while len(order) < n:
        nxt = [w for w in adj[order[-1]] if w != prev]
        prev = order[-1]; order.append(nxt[0])
    return order


class TreeBlock:
    """edges: iterable of (u, v, rate_u_to_v, rate_v_to_u); qmat[node, channel]."""

    def __init__(self, n, edges, qmat, use_path_fastpath=True, interlace_tol=1e-12, pair_cache_bytes=32 * 2**20):
        """interlace_tol: relative tolerance for ties nu_k = lambda_k.  Symmetric trees
        (identical arms) have exact eigenvalue ties that rounding can push to
        nu_k < lambda_k.  Deficits up to interlace_tol * lambda_k are treated as ties
        (that phase is never sampled) and their sum is recorded in tolerance_tv, an
        upper bound on the resulting per-draw total-variation perturbation if the
        floating eigenvalue error is itself below the tolerance.  A deficit is also
        treated as a tie when it is below 8 n eps lambda_max(A_C), the symmetric
        eigensolver's backward-error scale, where no floating computation can tell a
        tie from a violation.  0 restores the strict audited policy (violation ->
        endpoint-preserving Doob clock).
        """
        self.interlace_tol = float(interlace_tol); self.pair_cache_bytes = int(pair_cache_bytes)
        self.weyl_factor = 8. if interlace_tol > 0 else 0.
        started = time.perf_counter()
        qmat = np.asarray(qmat, float)
        if qmat.ndim == 1: qmat = qmat[:, None]
        if qmat.shape[0] != n: raise ValueError('qmat rows must equal n')
        edges = [(int(u), int(v), float(a), float(b)) for u, v, a, b in edges]
        if len(edges) != n - 1: raise ValueError('a tree on n nodes has n-1 edges')
        rate = {}
        adj = [[] for _ in range(n)]
        for u, v, a, b in edges:
            if u == v or not (0 <= u < n and 0 <= v < n): raise ValueError('bad edge')
            if (u, v) in rate: raise ValueError('duplicate edge')
            if not (np.isfinite(a) and np.isfinite(b)) or a < 0 or b < 0: raise ValueError('rates must be finite and nonnegative')
            if a == 0 and b == 0: raise ValueError('edge with no rate in either direction disconnects the block')
            rate[(u, v)] = a; rate[(v, u)] = b
            adj[u].append(v); adj[v].append(u)
        if np.any(qmat < 0) or not np.all(np.isfinite(qmat)): raise ValueError('kill rates must be finite and nonnegative')
        # connectivity (undirected) => tree, since |E| = n-1
        seen = {0}; stack = [0]
        while stack:
            x = stack.pop()
            for w in adj[x]:
                if w not in seen: seen.add(w); stack.append(w)
        if len(seen) != n: raise ValueError('edges do not form a tree')
        self.n, self.q, self.rate, self.adj = n, qmat, rate, adj
        self.qtot = qmat.sum(axis=1)
        self.delegate = None
        order = _path_order(n, adj) if use_path_fastpath else None
        if order is not None and all(rate[(a, b)] > 0 and rate[(b, a)] > 0 for a, b in zip(order, order[1:])):
            birth = np.array([rate[(order[k], order[k + 1])] if k + 1 < n else 0. for k in range(n)])
            death = np.array([rate[(order[k], order[k - 1])] if k > 0 else 0. for k in range(n)])
            self.order = np.array(order); self.position = np.empty(n, np.int64); self.position[self.order] = np.arange(n)
            self.delegate = PathBlock(birth, death, qmat[self.order])
            self.spectral_seconds = self.delegate.spectral_seconds
            return
        self._build_general()
        self.spectral_seconds = time.perf_counter() - started

    # ------------------------------------------------------------------ setup
    def _build_general(self):
        n, rate, adj = self.n, self.rate, self.adj
        A = np.zeros((n, n))
        for (u, v), r in rate.items():
            if r > 0: A[u, v] -= r; A[u, u] += r
        A[np.diag_indices(n)] += self.qtot
        self.A = A
        # strongly connected components = components of the two-way subgraph
        comp = -np.ones(n, np.int64); members = []
        for s in range(n):
            if comp[s] >= 0: continue
            cid = len(members); comp[s] = cid; group = [s]; stack = [s]
            while stack:
                x = stack.pop()
                for w in adj[x]:
                    if comp[w] < 0 and rate[(x, w)] > 0 and rate[(w, x)] > 0:
                        comp[w] = cid; group.append(w); stack.append(w)
            members.append(sorted(group))
        self.comp, self.members = comp, members
        # detailed-balance weights inside each component (trees always satisfy it)
        pi = np.zeros(n)
        for group in members:
            root = group[0]; pi[root] = 1.; stack = [root]
            while stack:
                x = stack.pop()
                for w in adj[x]:
                    if comp[w] == comp[x] and pi[w] == 0:
                        pi[w] = pi[x] * rate[(x, w)] / rate[(w, x)]; stack.append(w)
        self.pi = pi
        self.comp_eigen = [self._spectrum(group) for group in members]
        for lam in self.comp_eigen:
            if len(lam) and not np.all(lam > 0): raise ArithmeticError('nonpositive killed-component eigenvalue')
        self.hanging = {}
        # undirected rooted structure for tree paths
        parent = -np.ones(n, np.int64); depth = np.zeros(n, np.int64); dq = deque([0]); parent[0] = 0
        while dq:
            x = dq.popleft()
            for w in adj[x]:
                if parent[w] < 0: parent[w] = x; depth[w] = depth[x] + 1; dq.append(w)
        parent[0] = -1
        self.parent, self.depth = parent, depth
        self.rows = {}; self.columns = {}; self.pairs = {}; self.pair_bytes = 0
        self.tolerance_tv = 0.; self.tolerance_ties = 0
        self.factor_calls = 0; self.conditional_fallbacks = 0; self.conditional_steps = 0
        self.spectral_draws = 0; self.spectral_phases = 0; self.factor_errors = []

    def _spectrum(self, nodes):
        nodes = list(nodes)
        if not nodes: return np.empty(0)
        if len(nodes) == 1: return np.array([self.A[nodes[0], nodes[0]]])
        M = self.A[np.ix_(nodes, nodes)]
        s = np.sqrt(self.pi[nodes])
        S = (s[:, None] * M) / s[None, :]
        return np.linalg.eigvalsh((S + S.T) / 2)

    def _hanging_spectrum(self, p, w):
        key = (p, w)
        if key not in self.hanging:
            c = self.comp[w]; nodes = [w]; stack = [w]; seen = {p, w}
            while stack:
                x = stack.pop()
                for z in self.adj[x]:
                    if z not in seen and self.comp[z] == c:
                        seen.add(z); nodes.append(z); stack.append(z)
            self.hanging[key] = self._spectrum(nodes)
        return self.hanging[key]

    def _rooted(self, root):
        """Subtraction-free Green recursion rooted at `root` (the tree analogue of the
        audited positive ladder).  For each non-root v with parent p:
            h_v   = P(start at v, killed inside v's subtree before stepping to p)
            stay_v = q_v + sum_children r_vc h_c          (net leakage below v)
            den_v  = stay_v + r_vp                        (effective exit rate of v)
        and h_v = stay_v / den_v.  Only sums, products and quotients of positives."""
        n, adj, rate, q = self.n, self.adj, self.rate, self.qtot
        parent = -np.ones(n, np.int64); order = [root]; parent[root] = root
        for x in order:
            for w in adj[x]:
                if parent[w] < 0: parent[w] = x; order.append(w)
        stay = q.copy(); den = np.zeros(n); h = np.zeros(n)
        for v in reversed(order):
            if v == root: continue
            p = parent[v]
            den[v] = stay[v] + rate[(v, p)]
            if den[v] <= 0: raise ArithmeticError('non-transient subtree (no killing and no return)')
            h[v] = stay[v] / den[v]
            stay[p] += rate[(p, v)] * h[v]
        if stay[root] <= 0: raise ArithmeticError('non-transient block at root')
        return order, parent, stay, den

    def green_row(self, i):
        """Row G[i, :] by the rooted recursion: G_ii = 1/stay_i, G_iv = G_ip r_pv / den_v."""
        order, parent, stay, den = self._rooted(i)
        g = np.zeros(self.n); g[i] = 1. / stay[i]
        for v in order[1:]:
            p = parent[v]; g[v] = g[p] * self.rate[(p, v)] / den[v]
        return g

    def green_column(self, j):
        """Column G[:, j]: G_jj = 1/stay_j, G_xj = G_pj r_xp / den_x (p = parent toward j)."""
        order, parent, stay, den = self._rooted(j)
        g = np.zeros(self.n); g[j] = 1. / stay[j]
        for x in order[1:]:
            p = parent[x]; g[x] = g[p] * self.rate[(x, p)] / den[x]
        return g

    def tree_path(self, i, j):
        a, b = i, j; left = []; right = []
        while self.depth[a] > self.depth[b]: left.append(a); a = self.parent[a]
        while self.depth[b] > self.depth[a]: right.append(b); b = self.parent[b]
        while a != b:
            left.append(a); right.append(b); a = self.parent[a]; b = self.parent[b]
        return left + [a] + right[::-1]

    # ------------------------------------------------------------------ endpoint
    def _row(self, i):
        if i not in self.rows:
            w = self.green_row(i) * self.qtot
            if not np.all(np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
                raise ArithmeticError('invalid Green row; refusing to sample endpoint')
            self.rows[i] = np.cumsum(w)
        return self.rows[i]

    def endpoint(self, i):
        if self.delegate is not None:
            return int(self.order[self.delegate.endpoint(int(self.position[i]))])
        c = self._row(i)
        u = rng.uniform() * c[-1]
        return int(np.searchsorted(c, u, side='right'))

    # ------------------------------------------------------------------ time
    def factors(self, i, j):
        """([(lambda_C, nu_C) per component met by the tree path, in order], tv, ties)."""
        P = self.tree_path(i, j); onpath = set(P); out = []; tv = 0.; ties = 0
        k = 0
        while k < len(P):
            c = self.comp[P[k]]; group = []
            while k < len(P) and self.comp[P[k]] == c: group.append(P[k]); k += 1
            parts = [self._hanging_spectrum(p, w) for p in group for w in self.adj[p]
                     if w not in onpath and self.comp[w] == c]
            nu = np.sort(np.concatenate(parts)) if parts else np.empty(0)
            lam = self.comp_eigen[c]
            if len(nu) != len(lam) - len(group): raise ArithmeticError('hanging-subtree spectra have the wrong size')
            if np.any(nu <= 0): raise ArithmeticError('nonpositive complementary eigenvalue')
            gap = lam[:len(nu)] - nu
            deficit = gap / lam[:len(nu)]
            bad = gap > 0
            if np.any(bad):
                # A tie is indistinguishable from a violation below the symmetric
                # eigensolver's backward-error scale (Weyl: |error| <~ n eps ||A_C||).
                floor = self.weyl_factor * len(lam) * np.finfo(float).eps * lam[-1]
                if np.any((deficit > self.interlace_tol) & (gap > floor)):
                    raise ArithmeticError('floating eigenvalue interlacing violation beyond tolerance; no clipping')
                nu = nu.copy(); nu[bad] = lam[:len(nu)][bad]
                tv += float(deficit[bad].sum()); ties += int(bad.sum())
            out.append((lam, nu))
        self.factor_calls += 1
        return out, tv, ties

    def _doob(self, i, j):
        if j not in self.columns:
            self.columns[j] = self.green_column(j) * self.qtot[j]
        h = self.columns[j]
        x = i; t = 0.
        for step in range(10_000_000):
            if not (h[x] > 0) or not np.isfinite(h[x]): raise ArithmeticError('conditional clock left the support of h; no retry')
            nbr = self.adj[x]
            moves = np.array([self.rate[(x, w)] * h[w] / h[x] for w in nbr] + [self.qtot[x] / h[x] if x == j else 0.])
            total = self.A[x, x]
            if abs(moves.sum() - total) > 1e-10 * total: raise ArithmeticError('conditional holding-rate identity failed; no retry')
            t += rng.exponential(total)
            k = rng.categorical(moves)
            if k == len(nbr): return t, step + 1
            x = nbr[k]
        raise RuntimeError('conditional clock work cap; partial time cannot be discarded/retried')

    def draw(self, i):
        """Return (time, J, channel, work)."""
        if self.delegate is not None:
            t, jj, c, w = self.delegate.draw(int(self.position[i]))
            return t, int(self.order[jj]), c, w
        j = self.endpoint(i)
        try:
            key = (i, j)
            if key in self.pairs:
                fac, tv, ties = self.pairs[key]
            else:
                fac, tv, ties = self.factors(i, j)
                size = sum(l.nbytes + v.nbytes for l, v in fac)
                if self.pair_bytes + size <= self.pair_cache_bytes:
                    self.pairs[key] = (fac, tv, ties); self.pair_bytes += size
            self.tolerance_tv += tv; self.tolerance_ties += ties
        except (ArithmeticError, np.linalg.LinAlgError) as ex:
            self.factor_errors.append(dict(start=int(i), endpoint=int(j), error=repr(ex)))
            t, steps = self._doob(i, j)
            self.conditional_fallbacks += 1; self.conditional_steps += steps
            return t, j, choose_channel(self.q[j]), steps
        t = 0.; phases = 0
        for lam, nu in fac:
            dt, ph = rng.gated_time(lam, nu); t += dt; phases += ph
        self.spectral_draws += 1; self.spectral_phases += phases
        return t, j, choose_channel(self.q[j]), phases

    def generator(self):
        """Dense subgenerator (audit / survivor use)."""
        if self.delegate is not None:
            T = np.zeros((self.n, self.n)); Tp = self.delegate.generator(); o = self.order
            T[np.ix_(o, o)] = Tp
            return T
        return -self.A
