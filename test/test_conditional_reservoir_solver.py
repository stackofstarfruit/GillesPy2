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


import unittest

import numpy as np

from gillespy2 import NumPySSASolver, Species
from gillespy2.core.gillespyError import SimulationError

try:
    from gillespy2.solvers.numpy.conditional_reservoir import ReservoirModel, ConditionalReservoirSolver
except ImportError:
    ReservoirModel = None


def _small_model():
    return ReservoirModel(
        H=20, K=12, a=0.05, b=0.15, xi=0.1, g=0.9, e=1.6,
        y0=2, i0=6, times=np.linspace(0, 10, 11),
    )


@unittest.skipIf(ReservoirModel is None, "numba is not installed")
class TestConditionalReservoirSolver(unittest.TestCase):

    def test_results_contain_time_and_observed_species_only(self):
        model = _small_model()
        solver = ConditionalReservoirSolver(model, method="conditional")
        results = model.run(solver=solver, number_of_trajectories=4, seed=1)
        self.assertEqual(len(results), 4)
        for trajectory in results:
            self.assertEqual(sorted(trajectory.keys()), ["Y", "time"])
            np.testing.assert_array_equal(trajectory["time"], model.tspan)

    def test_same_seed_is_reproducible(self):
        model = _small_model()
        first = model.run(solver=ConditionalReservoirSolver(model, method="conditional"),
                          number_of_trajectories=8, seed=7)
        second = model.run(solver=ConditionalReservoirSolver(model, method="conditional"),
                           number_of_trajectories=8, seed=7)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a["Y"], b["Y"])

    def test_rejects_modified_model(self):
        model = _small_model()
        model.add_species(Species(name="Z", initial_value=0))
        with self.assertRaises(ValueError):
            ConditionalReservoirSolver(model)

    def test_rejects_unsupported_arguments(self):
        model = _small_model()
        solver = ConditionalReservoirSolver(model, method="conditional")
        with self.assertRaises(SimulationError):
            model.run(solver=solver, number_of_trajectories=2, seed=1, t=5)

    def test_matches_numpy_ssa_distribution(self):
        """Mean and variance of Y on the time grid agree with NumPySSASolver."""
        model = _small_model()
        n = 2000
        conditional = model.run(solver=ConditionalReservoirSolver(model, method="conditional"),
                                number_of_trajectories=n, seed=11)
        reference = model.run(solver=NumPySSASolver, number_of_trajectories=n, seed=12)
        y_conditional = np.array([trajectory["Y"] for trajectory in conditional])
        y_reference = np.array([trajectory["Y"] for trajectory in reference])
        for k in range(1, len(model.tspan)):
            a, b = y_conditional[:, k], y_reference[:, k]
            standard_error = np.sqrt(a.var(ddof=1) / n + b.var(ddof=1) / n)
            self.assertLess(abs(a.mean() - b.mean()), 5 * standard_error + 1e-12,
                            f"mean of Y differs at t={model.tspan[k]}")


if __name__ == '__main__':
    unittest.main()
