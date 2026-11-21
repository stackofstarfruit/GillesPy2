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


"""Copied old spectral clock repaired ONLY before its conditional-time RNG."""
import numpy as np
from . import old_clock
from . import conditional


class Block(old_clock.Block):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  self.conditional_fallbacks=0;self.conditional_steps=0;self.spectral_draws=0;self.spectral_phases=0;self.factor_errors=[]
  self.full_birth=np.zeros(self.K+1);self.full_death=np.zeros(self.K+1)
  self.full_birth[self.lo:]=self.birth;self.full_death[self.lo:]=self.death

 def draw(self,i):
  """Return(time,J,mark,work); work has backend-specific units.

  For diagnostics use separate spectral_phases and conditional_steps counters.
  """
  # The endpoint is drawn exactly once, before either clock algorithm.
  j,_=old_clock.green.endpoint(self.data,self.ends,i)
  try:nu=self.factors(i,j)
  except (ArithmeticError,np.linalg.LinAlgError) as ex:
   # factors() is deterministic: no conditional time has been sampled.
   # Keep j; replacing it with unconditional SSA would be wrong.
   self.factor_errors.append(dict(start=int(i),endpoint=int(j),error=repr(ex)))
   t,steps=conditional.sample(self.full_birth,self.full_death,self.data,i,j,self.lo)
   self.conditional_fallbacks+=1;self.conditional_steps+=steps
   channel=old_clock.choose_channel(self.q,j)
   return t,j,channel,steps
  # Errors after any time RNG are deliberately not caught or retried.
  t,phases=old_clock.time_from_factors(self.eigen,nu)
  self.spectral_draws+=1;self.spectral_phases+=phases
  channel=old_clock.choose_channel(self.q,j)
  return t,j,channel,phases
