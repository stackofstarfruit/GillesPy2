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


"""Generalized frozen-observed hidden blocks for the conditional clock.

Between observed events the hidden species form a killed continuous-time Markov
chain. The audited reservoir solver handles one irreducible birth-death chain;
this subpackage samples the same exact next-event law for wider hidden blocks:

* ``PathBlock``: any birth-death chain with arbitrary rates (bitwise identical
  to the reservoir kernels on reservoir rates).
* ``TreeBlock``: tree-shaped hidden state graphs, including one-way edges,
  which are split into strongly connected components.
* ``ProductBlock``: several independent hidden blocks at once.
* ``WindowedChain``: unbounded birth-death chains through exact moving windows.

``ConditionalReservoirSolver(method='general')`` uses ``reservoir`` to run
reservoir models whose blocks are reducible (b = 0 or e = 0) without native
fallback. Other models are driven through ``simulate.run_paths`` with a model
object (see ``models`` for examples). Cyclic hidden graphs are not supported.
"""
