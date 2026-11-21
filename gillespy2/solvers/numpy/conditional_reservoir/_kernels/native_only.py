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


"""Identical aggregate SSA kernel, without unused spectral/SciPy imports."""
import numpy as np
from numba import njit
@njit(cache=True)
def native_paths(H,K,a,b,xi,g,e,y0,i0,threshold,T,N,cap):
 rows=np.full((N,cap+1,4),np.nan);counts=np.zeros(N,np.int64);status=np.full(N,-1,np.int8);work=0
 for rep in range(N):
  y=y0;i=i0;t=0.;events=0;rows[rep,0]=np.array([t,y,i,-1.])
  while y<threshold and y+i>0 and events<cap:
   infect=a*((H-y)/H)*i;recover=xi*y;birth=(b*(y/H)+e*(i/K))*(K-i);death=g*i;total=infect+recover+birth+death
   u=np.random.random()
   while u==0:u=np.random.random()
   tnext=t-np.log(u)/total;work+=1
   if tnext>T:status[rep]=2;break
   t=tnext;u=np.random.random()*total
   if u<infect:y+=1;c=0
   elif u<infect+recover:y-=1;c=1
   elif u<infect+recover+birth:i+=1;continue
   else:i-=1;continue
   events+=1;rows[rep,events]=np.array([t,y,i,float(c)])
  if y>=threshold:status[rep]=1
  elif y+i==0:status[rep]=0
  counts[rep]=events
 return rows,counts,status,work

@njit(cache=True)
def seed(value): np.random.seed(value)
