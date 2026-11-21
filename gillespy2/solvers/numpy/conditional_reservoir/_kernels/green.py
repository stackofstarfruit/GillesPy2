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


"""Positive tridiagonal absorption kernels; no holding-time output."""
import math
import numpy as np
from numba import njit

@njit(cache=True)
def build(H,K,alpha,beta,xi,gamma,epsilon,y):
 lo=0 if y>0 else 1;size=K+1
 birth=np.zeros(size);death=np.zeros(size);q=np.zeros(size)
 for i in range(lo,size):
  birth[i]=(beta*y/H+epsilon*i/K)*(K-i)
  death[i]=gamma*i if i>lo else 0.
  q[i]=alpha*(H-y)*i/H+xi*y+(gamma if y==0 and i==1 else 0.)
 ell=np.zeros(size);r=np.zeros(size);left=np.zeros(size);right=np.zeros(size)
 ell[lo]=q[lo]
 for i in range(lo+1,size):
  den=birth[i-1]+ell[i-1]
  if den<=0:raise ArithmeticError('nonabsorbing left component')
  left[i]=death[i]*ell[i-1]/den;ell[i]=q[i]+left[i]
 r[K]=q[K]
 for i in range(K-1,lo-1,-1):
  den=death[i+1]+r[i+1]
  if den<=0:raise ArithmeticError('nonabsorbing right component')
  right[i]=birth[i]*r[i+1]/den;r[i]=q[i]+right[i]
 hp_left=np.zeros(size+1);hp_right=np.zeros(size+1)
 end_left=np.zeros(size,np.int64);end_right=np.zeros(size,np.int64)
 last=lo
 for i in range(lo,size):
  if left[i]==0:last=i
  end_left[i]=last
  hp_left[i+1]=hp_left[i]+(math.log1p(q[i]/left[i]) if left[i]>0 else 0.)
 for i in range(lo,size):hp_right[i+1]=hp_right[i]+(math.log1p(q[i]/right[i]) if right[i]>0 else 0.)
 last=K
 for i in range(K,lo-1,-1):
  if right[i]==0:last=i
  end_right[i]=last
 data=np.zeros((5,size));data[0]=left;data[1]=q;data[2]=right;data[3]=hp_left[:size];data[4]=hp_right[:size]
 # Prefix at K+1 is not needed: the last active index is a stopping barrier.
 ends=np.zeros((2,size),np.int64);ends[0]=end_left;ends[1]=end_right
 if np.min(data)<0 or not np.all(np.isfinite(data)):raise ArithmeticError('invalid block table')
 return data,ends

@njit(cache=True)
def positive_uniform():
 u=np.random.random()
 while u==0:u=np.random.random()
 return u

@njit(cache=True)
def endpoint(data,ends,i):
 left=data[0,i];q=data[1,i];right=data[2,i];u=np.random.random()*(left+q+right);work=1
 if u<left:
  start=i-1;low=ends[0,start];high=start;target=-math.log(positive_uniform())
  # prefix(start+1) available even when start=K-1; data[3] stores prefix at index.
  origin=data[3,start+1]
  while low<high:
   mid=(low+high+1)//2;mass=origin-data[3,mid];work+=1
   if mass>=target:low=mid
   else:high=mid-1
  return low,work
 if u<left+q:return i,work
 start=i+1;low=start;high=ends[1,start];target=-math.log(positive_uniform());origin=data[4,start]
 while low<high:
  mid=(low+high)//2;mass=data[4,mid+1]-origin;work+=1
  if mass>=target:high=mid
  else:low=mid+1
 return low,work

@njit(cache=True)
def endpoint_walk(data,ends,i):
 left=data[0,i];q=data[1,i];right=data[2,i];u=np.random.random()*(left+q+right);work=1
 if u<left:
  j=i-1
  while True:
   work+=1
   if np.random.random()*(data[0,j]+data[1,j])<data[1,j]:return j,work
   j-=1
 if u<left+q:return i,work
 j=i+1
 while True:
  work+=1
  if np.random.random()*(data[2,j]+data[1,j])<data[1,j]:return j,work
  j+=1
