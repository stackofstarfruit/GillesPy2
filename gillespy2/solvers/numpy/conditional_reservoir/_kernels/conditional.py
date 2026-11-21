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


"""Known Doob-conditioned clock; preserves an already selected kill endpoint."""
import math
import numpy as np
from numba import njit
from . import green


@njit(cache=True)
def build(H,K,alpha,beta,xi,gamma,epsilon,y):
 data,ends=green.build(H,K,alpha,beta,xi,gamma,epsilon,y)
 lo=0 if y>0 else 1;birth=np.zeros(K+1);death=np.zeros(K+1)
 for i in range(lo,K+1):
  birth[i]=(beta*(y/H)+epsilon*(i/K))*(K-i)
  if i>lo:death[i]=gamma*i
 return birth,death,data,lo


@njit(cache=True)
def rates(birth,death,data,i,j,lo):
 b=birth[i];d=death[i];q=data[1,i];left=data[0,i];right=data[2,i]
 away_death=0.;away_birth=0.
 if i>lo:
  den=birth[i-1]+data[1,i-1]+data[0,i-1]
  if den<=0:raise ArithmeticError('unreachable conditional left component')
  away_death=d*(birth[i-1]/den)
 if i+1<len(birth):
  den=death[i+1]+data[1,i+1]+data[2,i+1]
  if den<=0:raise ArithmeticError('unreachable conditional right component')
  away_birth=b*(death[i+1]/den)
 if i<j:return b+q+left,away_death,0.
 if i>j:return away_birth,d+q+right,0.
 return away_birth,away_death,q+left+right


@njit(cache=True)
def sample(birth,death,data,start,target,lo,max_steps=10000000):
 if start<lo or target<lo or start>=len(birth) or target>=len(birth) or data[1,target]<=0:
  raise ValueError('invalid/unmarked conditional endpoint')
 i=start;t=0.
 for step in range(max_steps):
  toward_b,toward_d,kill=rates(birth,death,data,i,target,lo)
  total=birth[i]+death[i]+data[1,i];weights=toward_b+toward_d+kill
  if not math.isfinite(total) or total<=0 or not math.isfinite(weights) or min(toward_b,toward_d,kill)<0:
   raise ArithmeticError('invalid conditional rates; no retry')
  if abs(weights-total)>1e-10*total:raise ArithmeticError('conditional holding-rate identity failed; no retry')
  u=np.random.random()
  while u==0:u=np.random.random()
  t-=math.log(u)/total
  z=np.random.random()*weights
  if z<toward_b:i+=1
  elif z<toward_b+toward_d:i-=1
  else:return t,step+1
 raise RuntimeError('conditional clock work cap; partial time cannot be discarded/retried')


@njit(cache=True)
def many(birth,death,data,start,target,lo,N,seed):
 np.random.seed(seed);times=np.empty(N);work=0
 for rep in range(N):times[rep],w=sample(birth,death,data,start,target,lo);work+=w
 return times,work
