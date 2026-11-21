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


"""Tests for gillespy2.solvers.numpy.conditional_reservoir.general.

Exactness is checked against references that import nothing from the samplers:
direct SSA on the dense killed generator, exact moments from its inverse, and the
full finite-generator law of the reservoir model.  Statistical thresholds are
Bonferroni-style floors (p > 1e-4, |z| < 5 or 6).
"""
import math
import unittest

import numpy as np

try:
    from scipy import stats
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import expm_multiply
    from gillespy2.solvers.numpy.conditional_reservoir import ReservoirModel, ConditionalReservoirSolver
    from gillespy2.solvers.numpy.conditional_reservoir._kernels import repair
    from gillespy2.solvers.numpy.conditional_reservoir.general import rng, models, simulate
    from gillespy2.solvers.numpy.conditional_reservoir.general.pathblock import PathBlock
    from gillespy2.solvers.numpy.conditional_reservoir.general.treeblock import TreeBlock
    from gillespy2.solvers.numpy.conditional_reservoir.general.product import ProductBlock
    from gillespy2.solvers.numpy.conditional_reservoir.general.window import WindowedChain
except ImportError:
    ReservoirModel = None

N = 30000


# ---------------------------------------------------------------- references
def direct_kill(T, qmat, i, n, r):
    """Direct SSA on subgenerator T (off-diagonals = internal rates) with kill matrix qmat."""
    m = T.shape[0]; R = np.where(np.eye(m, dtype=bool), 0., T); out = np.empty((n, 3))
    hold = R.sum(1) + qmat.sum(1); cumR = np.cumsum(R, axis=1); qt = qmat.sum(1)
    for k in range(n):
        x = i; t = 0.
        while True:
            t += r.exponential(1 / hold[x]); u = r.random() * hold[x]
            if u < qt[x]:
                c = int(np.searchsorted(np.cumsum(qmat[x]), u, side='right')); break
            x = int(np.searchsorted(cumR[x], u - qt[x], side='right'))
        out[k] = (t, x, c)
    return out


def exact_moments(T, qmat, i):
    G = np.linalg.inv(-T); qt = qmat.sum(1); pJ = G[i] * qt
    G2 = G @ G; G3 = G2 @ G
    with np.errstate(divide='ignore', invalid='ignore'):
        mean = G2[i] / G[i]; second = 2 * G3[i] / G[i]
    return pJ, mean, second - mean ** 2


def chi2_two_sample(a, b, min_count=10):
    cats = np.union1d(a, b)
    ca = np.array([(a == c).sum() for c in cats]); cb = np.array([(b == c).sum() for c in cats])
    keep = (ca + cb) >= min_count
    if keep.sum() < 2:
        return 1.0
    return float(stats.chi2_contingency(np.vstack([ca[keep], cb[keep]]))[1])


def endpoint_z(J, pJ):
    n = len(J); z = 0.
    for j, p in enumerate(pJ):
        if 1e-12 < p < 1 - 1e-12:
            z = max(z, abs(((J == j).mean() - p) / math.sqrt(p * (1 - p) / n)))
    return z


def mean_z(S, mean, var, min_count=200):
    z = 0.
    for j in np.unique(S[:, 1]).astype(int):
        sel = S[S[:, 1] == j, 0]
        if len(sel) >= min_count:
            z = max(z, abs((sel.mean() - mean[j]) / math.sqrt(var[j] / len(sel))))
    return z


def reservoir_marginals(c, times):
    """Exact P(Y(t)=y) for the reservoir model from its full finite generator."""
    H, K = c['H'], c['K']; n = (H + 1) * (K + 1); q = lil_matrix((n, n))
    for y in range(H + 1):
        for i in range(K + 1):
            for ny, ni, rate in [(y + 1, i, c['a'] * ((H - y) / H) * i), (y - 1, i, c['xi'] * y),
                                 (y, i + 1, (c['b'] * y / H + c['e'] * i / K) * (K - i)), (y, i - 1, c['g'] * i)]:
                if rate > 0:
                    q[y * (K + 1) + i, ny * (K + 1) + ni] += rate; q[y * (K + 1) + i, y * (K + 1) + i] -= rate
    q = q.tocsr(); p0 = np.zeros(n); p0[c['y0'] * (K + 1) + c['i0']] = 1
    return np.array([expm_multiply(q.T * t, p0).reshape(H + 1, K + 1).sum(1) for t in times])


def reservoir_path_block(H, K, a, b, xi, g, e, y):
    """Reservoir block rates in the exact floating orders used by the audited kernels."""
    lo = 0 if y else 1; ids = np.arange(K + 1)
    birth = np.zeros(K + 1); death = np.zeros(K + 1)
    birth[lo:] = (b * (y / H) + e * (ids[lo:] / K)) * (K - ids[lo:]); death[lo:] = g * ids[lo:].astype(float)
    q = np.zeros((K + 1, 3)); q[:, 0] = a * ((H - y) / H) * np.arange(K + 1); q[:, 1] = xi * y
    if y == 0:
        q[1, 2] = g
    lb = np.zeros(K + 1); ld = np.zeros(K + 1); lq = np.zeros(K + 1)
    for i in range(lo, K + 1):
        lb[i] = (b * y / H + e * i / K) * (K - i)
        ld[i] = g * i if i > lo else 0.
        lq[i] = a * (H - y) * i / H + xi * y + (g if y == 0 and i == 1 else 0.)
    return PathBlock(birth, death, q, lo=lo, ladder_birth=lb, ladder_death=ld, ladder_q=lq)


def random_tree_block(n, seed, one_way=0, channels=2):
    r = np.random.default_rng(seed); labels = r.permutation(n)
    pairs = [(labels[int(r.integers(k))], labels[k]) for k in range(1, n)]   # random recursive tree
    oneway = set(r.choice(n - 1, size=one_way, replace=False)) if one_way else set()
    edges = []
    for k, (u, v) in enumerate(pairs):
        a, b = r.uniform(.5, 3), r.uniform(.5, 3)
        edges.append((int(u), int(v), a, 0. if k in oneway else b))
    q = np.zeros((n, channels))
    for c in range(channels):
        q[r.choice(n, size=max(1, n // 4), replace=False), c] = r.uniform(.05, .6, max(1, n // 4))
    q[:, -1] += 0.01   # every state can be killed: transient even with one-way edges
    return TreeBlock(n, edges, q, use_path_fastpath=False), q


def star_block(arms=3, L=6, f=2., g=1.5, kappa=.4, tol=1e-12):
    n = 1 + arms * L; edges = []; q = np.zeros((n, 1))
    for a in range(arms):
        prev = 0
        for k in range(1, L + 1):
            node = 1 + a * L + k - 1; edges.append((prev, node, f, g)); prev = node
        q[prev, 0] = kappa
    return TreeBlock(n, edges, q, interlace_tol=tol), q


PARAMS = [dict(H=20, K=12, a=.05, b=.15, xi=.1, g=.9, e=1.6),
          dict(H=100, K=96, a=.00006, b=.15, xi=.0011, g=.9, e=1.8),
          dict(H=3, K=4, a=.35, b=.25, xi=.4, g=.6, e=.8)]


@unittest.skipIf(ReservoirModel is None, "numba or scipy is not installed")
class TestPathIdentity(unittest.TestCase):
    """The generic path block must reproduce the audited reservoir kernels exactly."""

    def test_tables_and_draws_bitwise(self):
        for P in PARAMS:
            for y in range(0, P['H'] + 1, max(1, P['H'] // 6)):
                ref = repair.Block(P['H'], P['K'], P['a'], P['b'], P['xi'], P['g'], P['e'], y)
                new = reservoir_path_block(**P, y=y)
                for k in ['data', 'ends', 'eigen', 'diag', 'off', 'q', 'full_birth', 'full_death', 'kill']:
                    self.assertTrue(np.array_equal(getattr(ref, k), getattr(new, k)), (P, y, k))
                for i in range(ref.lo, P['K'] + 1, max(1, P['K'] // 5)):
                    rng.seed(1000 + y + i); a = [ref.draw(i) for _ in range(200)]
                    rng.seed(1000 + y + i); b = [new.draw(i) for _ in range(200)]
                    self.assertEqual(a, b)

    def test_tree_on_path_delegates(self):
        P = PARAMS[0]
        for y in [0, 7, 20]:
            ref = repair.Block(P['H'], P['K'], P['a'], P['b'], P['xi'], P['g'], P['e'], y)
            n = len(ref.ids); lo = ref.lo
            edges = [(k, k + 1, ref.birth[k], ref.death[k + 1]) for k in range(n - 1)]
            tree = TreeBlock(n, edges, ref.q[lo:])
            self.assertIsNotNone(tree.delegate)
            self.assertTrue(np.array_equal(tree.delegate.eigen, ref.eigen))
            direct = PathBlock(np.r_[ref.birth[:-1], 0.], np.r_[0., ref.death[1:]], ref.q[lo:])
            for i in range(n):
                rng.seed(77 + i); a = [tree.draw(i) for _ in range(100)]
                rng.seed(77 + i); b = [direct.draw(i) for _ in range(100)]
                self.assertEqual(a, b)

    def test_generic_tree_engine_matches_path_spectra(self):
        P = PARAMS[0]; worst = 0.
        for y in [0, 3, 11, 20]:
            ref = repair.Block(P['H'], P['K'], P['a'], P['b'], P['xi'], P['g'], P['e'], y)
            n = len(ref.ids); lo = ref.lo
            edges = [(k, k + 1, ref.birth[k], ref.death[k + 1]) for k in range(n - 1)]
            gen = TreeBlock(n, edges, ref.q[lo:], use_path_fastpath=False)
            self.assertIsNone(gen.delegate)
            worst = max(worst, np.max(np.abs(gen.comp_eigen[0] - ref.eigen) / ref.eigen))
            for i in range(n):
                for j in range(n):
                    try:
                        nu_ref = ref.factors(i + lo, j + lo)
                    except ArithmeticError:
                        continue
                    (lam, nu), = gen.factors(i, j)[0]
                    if len(nu):
                        worst = max(worst, np.max(np.abs(nu - nu_ref) / nu_ref))
        self.assertLess(worst, 1e-9)

    def test_general_method_irreducible_is_bitwise_conditional(self):
        model = ReservoirModel(H=20, K=12, a=.05, b=.15, xi=.1, g=.9, e=1.6, y0=2, i0=6, times=np.linspace(0, 10, 21))
        s1 = ConditionalReservoirSolver(model, method='conditional')
        model.run(solver=s1, number_of_trajectories=64, seed=42, max_events=256, retain_events=True)
        s2 = ConditionalReservoirSolver(model, method='general')
        model.run(solver=s2, number_of_trajectories=64, seed=42, max_events=256, retain_events=True)
        for a, b in zip(s1.event_histories, s2.event_histories):
            self.assertTrue(np.array_equal(a, b))


@unittest.skipIf(ReservoirModel is None, "numba or scipy is not installed")
class TestReducibleReservoir(unittest.TestCase):
    """method='general' on reservoir models whose blocks are reducible (b = 0 or e = 0)."""
    CASES = [dict(H=3, K=4, a=.35, b=0., xi=.4, g=.6, e=.8, y0=1, i0=2),
             dict(H=3, K=4, a=.35, b=.25, xi=.4, g=.6, e=0., y0=0, i0=3),
             dict(H=3, K=4, a=.35, b=0., xi=.4, g=.6, e=0., y0=2, i0=4),
             dict(H=8, K=24, a=.002, b=0., xi=.01, g=.9, e=1.8, y0=2, i0=12)]

    def test_matches_full_generator(self):
        for k, c in enumerate(self.CASES):
            times = np.linspace(0, 2 if k < 3 else 200, 5)
            model = ReservoirModel(**c, times=times)
            solver = ConditionalReservoirSolver(model=model, method='general')
            res = model.run(solver=solver, number_of_trajectories=6000, seed=4100 + k, max_events=128)
            Y = np.array([r['Y'] for r in res]); exact = reservoir_marginals(c, times); z = 0.
            for t in range(len(times)):
                for y in range(c['H'] + 1):
                    p = exact[t, y]
                    if 1e-14 < p < 1 - 1e-14:
                        z = max(z, abs(np.mean(Y[:, t] == y) - p) / math.sqrt(p * (1 - p) / len(Y)))
            self.assertLess(z, 6, c)
            self.assertEqual(solver.diagnostics['spectral']['fallback_paths'], 0)
            self.assertGreater(solver.diagnostics['spectral']['reducible_blocks'], 0)


@unittest.skipIf(ReservoirModel is None, "numba or scipy is not installed")
class TestTreeLaw(unittest.TestCase):

    def check(self, block, q, start, seed):
        T = block.generator()
        rng.seed(seed)
        S = np.array([block.draw(start)[:3] for _ in range(N)])
        D = direct_kill(T, q, start, N, np.random.default_rng(seed + 1))
        pJ, mean, var = exact_moments(T, q, start)
        self.assertLess(endpoint_z(S[:, 1], pJ), 5)
        self.assertLess(mean_z(S, mean, var), 5)
        self.assertGreater(chi2_two_sample(S[:, 1] * 10 + S[:, 2], D[:, 1] * 10 + D[:, 2]), 1e-4)
        self.assertGreater(stats.ks_2samp(S[:, 0], D[:, 0]).pvalue, 1e-4)

    def test_random_reversible_tree(self):
        block, q = random_tree_block(14, 1)
        self.assertEqual(len(block.members), 1)
        self.check(block, q, 1, 101)

    def test_symmetric_star_with_eigenvalue_ties(self):
        block, q = star_block()
        self.check(block, q, 5, 7)
        self.assertEqual(block.conditional_fallbacks, 0)
        self.assertGreater(block.tolerance_ties, 0)

    def test_symmetric_star_strict_policy_uses_doob(self):
        block, q = star_block(tol=0.)
        self.check(block, q, 5, 8)
        self.assertGreater(block.conditional_fallbacks, 0)

    def test_one_way_edges_scc_split(self):
        block, q = random_tree_block(14, 3, one_way=4)
        self.assertGreater(len(block.members), 1)
        self.check(block, q, 3, 203)

    def test_pure_one_way_chain(self):
        n = 8; edges = [(k + 1, k, 1.3 + .1 * k, 0.) for k in range(n - 1)]
        q = np.zeros((n, 1)); q[0, 0] = .7; q[:, 0] += .05
        block = TreeBlock(n, edges, q)
        self.assertEqual(len(block.members), n)
        self.check(block, q, n - 1, 11)

    def test_forced_doob_repair_preserves_law(self):
        block, q = random_tree_block(12, 5, one_way=2)
        def fail(i, j):
            raise ArithmeticError('forced for test')
        block.factors = fail
        self.check(block, q, 0, 300)
        self.assertEqual(block.conditional_fallbacks, N)

    def test_rejects_cycles(self):
        with self.assertRaises(ValueError):
            TreeBlock(3, [(0, 1, 1., 1.), (1, 2, 1., 1.), (2, 0, 1., 1.)], np.ones((3, 1)))


@unittest.skipIf(ReservoirModel is None, "numba or scipy is not installed")
class TestProductLaw(unittest.TestCase):

    def test_path_tree_path_matches_direct(self):
        x = np.arange(6.)
        b1 = PathBlock(1.5 * (5 - x), .8 * x, (.2 * x)[:, None])
        edges = [(0, 1, 2., 1.), (0, 2, 1., 1.5), (2, 3, 1.2, 0.), (2, 4, .7, .9)]
        q = np.zeros((5, 1)); q[[1, 3, 4], 0] = [.3, .5, .2]; q[:, 0] += .02
        b2 = TreeBlock(5, edges, q)
        z = np.arange(4.)
        b3 = PathBlock(1. * (3 - z), 2. * z, (.25 + .1 * z)[:, None])
        blocks, start = [b1, b2, b3], (2, 0, 1)
        P = ProductBlock(blocks); rng.seed(91); S = []
        for _ in range(N):
            t, w, c, new, _ = P.draw(start); S.append((t, w, c) + tuple(new))
        S = np.array(S)
        # direct SSA on the Kronecker-sum chain
        r = np.random.default_rng(92); Rs = [np.where(np.eye(b.generator().shape[0], dtype=bool), 0., b.generator()) for b in blocks]
        D = []
        for _ in range(N):
            xs = list(start); t = 0.
            while True:
                rates = []; moves = []
                for k, b in enumerate(blocks):
                    for y, rr in enumerate(Rs[k][xs[k]]):
                        if rr > 0: rates.append(rr); moves.append(('move', k, y))
                    for c, qq in enumerate(b.q[xs[k]]):
                        if qq > 0: rates.append(qq); moves.append(('kill', k, c))
                rates = np.array(rates); tot = rates.sum(); t += r.exponential(1 / tot)
                kind, k, v = moves[int(np.searchsorted(np.cumsum(rates), r.random() * tot, side='right'))]
                if kind == 'move': xs[k] = v
                else: D.append((t, k, v) + tuple(xs)); break
        D = np.array(D)
        key = lambda A: sum(A[:, k] * 10 ** (len(start) + 2 - k) for k in range(1, A.shape[1]))
        self.assertGreater(chi2_two_sample(key(S), key(D)), 1e-4)
        self.assertGreater(stats.ks_2samp(S[:, 0], D[:, 0]).pvalue, 1e-4)


@unittest.skipIf(ReservoirModel is None, "numba or scipy is not installed")
class TestWindowLaw(unittest.TestCase):

    def test_unbounded_chain_matches_direct(self):
        a, g, kap, xi = 30., 1., .02, .1
        ch = WindowedChain(lambda i: a, lambda i: g * i, lambda i: np.array([kap * i, xi]), W=8)
        rng.seed(5); start = 5
        S = np.array([ch.draw(start)[:3] for _ in range(N)])
        r = np.random.default_rng(6); D = []
        for _ in range(N):
            i = start; t = 0.
            while True:
                rates = np.array([a, g * i, kap * i, xi]); tot = rates.sum(); t += r.exponential(1 / tot)
                e = int(np.searchsorted(np.cumsum(rates), r.random() * tot, side='right'))
                if e == 0: i += 1
                elif e == 1: i -= 1
                else: D.append((t, i, e - 2)); break
        D = np.array(D)
        self.assertGreater(stats.ks_2samp(S[:, 0], D[:, 0]).pvalue, 1e-4)
        self.assertGreater(stats.ks_2samp(S[:, 1], D[:, 1]).pvalue, 1e-4)
        self.assertGreater(S[:, 1].max(), 3 * 8)

    def test_bounded_chain_exact(self):
        K = 40
        b = lambda i: 3. * (K - i) / K + .5; d = lambda i: 1.2 * i
        kills = lambda i: np.array([.03 * i + (0.2 if i == K else 0.)])
        ch = WindowedChain(lambda i: b(i) if i < K else 0., d, kills, W=5, K=K)
        T = np.zeros((K + 1, K + 1)); q = np.array([kills(i) for i in range(K + 1)])
        for i in range(K + 1):
            if i < K: T[i, i + 1] = b(i)
            if i > 0: T[i, i - 1] = d(i)
        T -= np.diag(T.sum(1) + q.sum(1))
        pJ, mean, var = exact_moments(T, q, 12)
        rng.seed(9); S = np.array([ch.draw(12)[:3] for _ in range(N)])
        self.assertLess(endpoint_z(S[:, 1], pJ), 5)
        self.assertLess(mean_z(S, mean, var), 5)


@unittest.skipIf(ReservoirModel is None, "numba or scipy is not installed")
class TestModelsAgainstNativeSSA(unittest.TestCase):
    """Observed histories of the example models against each model's compiled native SSA."""
    CASES = {
        'tree_one_way': (lambda: models.BranchedMachine(L=12, f=6., rin=6., kappa=1., xi=.05, one_way=True), 0, 0, 200.),
        'pools': (lambda: models.ReceptorPools(K1=16, K2=12, k1=.01, k2=.02), 0, (8, 6), 100.),
        'transcript': (lambda: models.Transcript(a=60., g=1., kappa=.01, xi=.05, W=12), 0, 60, 100.),
        'enzyme_cgp_eq34': (lambda: models.EnzymeCGP(c1=1e-4, c2=6., c3=1e-3, E0=10, S0=3000), 0, 0, 2.e5),
    }

    def test_observed_history_agrees(self):
        for name, (make, y0, h0, T) in self.CASES.items():
            n = 400 if name.startswith('enzyme') else 3000; grid = np.linspace(0, T, 6)[1:]
            m = make()
            r1, c1, s1, _ = simulate.run_paths(m, y0, h0, T, n, 5000, 31)
            r2, c2, s2, _ = m.native(y0, h0, T, n, 5000, 32)
            self.assertTrue(np.all(s1 == 2) and np.all(s2 == 2), name)
            v1 = simulate.grid_values(r1, c1, grid); v2 = simulate.grid_values(r2, c2, grid)
            self.assertGreater(min(stats.ks_2samp(v1[:, k], v2[:, k]).pvalue for k in range(len(grid))), 1e-4, name)


if __name__ == '__main__':
    unittest.main()
