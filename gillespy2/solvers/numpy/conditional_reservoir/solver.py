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


"""Explicitly restricted model adapter, not an automatic arbitrary-network reducer."""
import copy
import math
import secrets
import time
import numpy as np
from gillespy2.core import GillesPySolver, Model, Parameter, Reaction, Results, Species, Trajectory
from ._kernels import dispatch


def _integer(name, value, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


class ReservoirModel(Model):
    """Two-population jump process with an observed host count Y and hidden I.

    All four rates are custom GillesPy propensities, avoiding mass-action
    combinatorial conventions. Time is in arbitrary consistent model units.
    """
    def __init__(self, H, K, a, b, xi, g, e, y0, i0, times):
        super().__init__(name="ConditionalReservoir")
        H = _integer("H", H, 1); K = _integer("K", K, 1)
        y0 = _integer("y0", y0); i0 = _integer("i0", i0)
        if y0 > H or i0 > K:
            raise ValueError("initial populations exceed capacity")
        rates = dict(a=float(a), b=float(b), xi=float(xi), g=float(g), e=float(e))
        if any(not math.isfinite(v) or v < 0 for v in rates.values()):
            raise ValueError("rates must be finite and nonnegative")
        # Positive losses/contact are the current portable sampler's domain.
        if min(rates['a'], rates['xi'], rates['g']) <= 0:
            raise ValueError("current adapter requires a, xi, g > 0")
        grid = np.asarray(times, dtype=float)
        if grid.ndim != 1 or len(grid) < 2 or grid[0] != 0 or not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0):
            raise ValueError("times must be a finite increasing grid starting at zero")
        self._conditional_spec = dict(H=H, K=K, **rates, y0=y0, i0=i0)
        # GillesPy's expression parser treats bare e as Euler's constant.
        parameters = dict(H=H, K=K, a=rates['a'], b=rates['b'], xi=rates['xi'],
                          g=rates['g'], epsilon=rates['e'])
        self.add_parameter([Parameter(name=k, expression=v) for k, v in parameters.items()])
        Y = Species(name='Y', initial_value=y0, mode='discrete')
        I = Species(name='I', initial_value=i0, mode='discrete')
        self.add_species([Y, I])
        self.add_reaction([
            Reaction(name='infection', reactants={}, products={Y: 1}, propensity_function='a*((H-Y)/H)*I'),
            Reaction(name='recovery', reactants={Y: 1}, products={}, propensity_function='xi*Y'),
            Reaction(name='reservoir_birth', reactants={}, products={I: 1}, propensity_function='(b*(Y/H)+epsilon*(I/K))*(K-I)'),
            Reaction(name='reservoir_death', reactants={I: 1}, products={}, propensity_function='g*I'),
        ])
        self.timespan(grid)


def _signature(model):
    """Compare the actual model, not a trusted tag or stale cached model hash."""
    params = tuple(sorted((k, str(v.expression)) for k, v in model.listOfParameters.items()))
    species = tuple(sorted((k, v.initial_value, v.mode, v.constant, v.boundary_condition)
                           for k, v in model.listOfSpecies.items()))
    reactions = tuple(sorted((k, r.propensity_function, r.ode_propensity_function,
                               tuple(sorted((s.name, n) for s, n in r.reactants.items())),
                               tuple(sorted((s.name, n) for s, n in r.products.items())))
                              for k, r in model.listOfReactions.items()))
    return params, species, reactions


def _validate(model):
    if not isinstance(model, ReservoirModel):
        raise ValueError("ConditionalReservoirSolver currently accepts only ReservoirModel")
    if model.get_model_features() or model.listOfFunctionDefinitions:
        raise ValueError("events, rules and function definitions are unsupported")
    if model.volume != 1 or model.units != 'population':
        raise ValueError("current adapter requires unit volume and population units")
    grid = np.asarray(model.tspan, dtype=float)
    expected = ReservoirModel(**model._conditional_spec, times=grid)
    if _signature(model) != _signature(expected):
        raise ValueError("model was modified outside the supported reaction template; use an ordinary GillesPy solver")
    return dict(model._conditional_spec), grid.copy()


def _run_general(case, n, s):
    from .general import reservoir
    if case['N'] * (case['cap'] + 1) * 4 * 8 > 128 * 1024**2:
        raise ValueError('registered output memory limit')
    started = time.perf_counter()
    dispatch.native_only.seed(s)
    rows, counts, status, work = reservoir.spectral_paths(case, n)
    meta = dict(requested='general', pilot_paths=0, selected='general', spectral=work,
                elapsed_s=time.perf_counter() - started)
    return rows, counts, status, meta


class ConditionalReservoirSolver(GillesPySolver):
    """Returns GillesPy Results with keys time,Y; I is deliberately absent.

    method: auto (retained-pilot selector), conditional (force block backend),
    native (compiled reference), general (block backend that also handles
    reducible hidden blocks, b = 0 or e = 0, by strongly connected components
    instead of native fallback; irreducible blocks use the same kernels as
    conditional). Resource/numerical failures raise for the whole
    request, so successful prefixes cannot silently become selected samples.
    """
    name = "ConditionalReservoirSolver"

    def __init__(self, model=None, method='auto'):
        if method not in ('auto', 'conditional', 'native', 'general'):
            raise ValueError("method must be auto, conditional, native or general")
        _validate(model)
        self.model = copy.deepcopy(model)
        self.method = method
        self.is_instantiated = True
        self.diagnostics = None
        self.event_histories = None

    @classmethod
    def get_solver_settings(cls):
        return ('number_of_trajectories', 'seed', 'max_events', 'retain_events')

    def run(self, t=None, increment=None, timeout=0, number_of_trajectories=1,
            seed=None, max_events=1024, retain_events=False, **kwargs):
        if kwargs:
            raise ValueError(f"unsupported solver arguments: {sorted(kwargs)}")
        if t is not None or increment is not None or timeout != 0:
            raise ValueError("set times on ReservoirModel; t, increment and in-process timeout are unsupported")
        n = _integer('number_of_trajectories', number_of_trajectories, 1)
        cap = _integer('max_events', max_events, 1)
        s = secrets.randbelow(2**32) if seed is None else _integer('seed', seed)
        if s >= 2**32:
            raise ValueError("seed must fit uint32")
        spec, grid = _validate(self.model)
        case = dict(spec, threshold=spec['H']+1, T=float(grid[-1]), N=n, cap=cap)
        self.event_histories = None
        self.diagnostics = None
        if self.method == 'general':
            rows, counts, status, meta = _run_general(case, n, s)
        else:
            rows, counts, status, meta = dispatch.run(case, 'spectral' if self.method == 'conditional' else self.method, s)
        meta.update(statuses={str(int(v)): int(np.sum(status == v)) for v in np.unique(status)}, seed=s,
                    requested_output=['time', 'Y'], hidden_snapshots=False)
        self.diagnostics = meta
        if np.any(status < 0):
            raise RuntimeError(f"request incomplete; no Results returned: {meta}")
        data = []
        histories = [] if retain_events else None
        for k in range(n):
            history = rows[k, :int(counts[k])+1]
            indices = np.searchsorted(history[:, 0], grid, side='right') - 1
            y = history[indices, 1].copy()
            data.append(Trajectory({'time': grid.copy(), 'Y': y}, model=self.model, solver_name=self.name))
            if retain_events:
                histories.append(history.copy())
        self.event_histories = histories
        return Results(data)
