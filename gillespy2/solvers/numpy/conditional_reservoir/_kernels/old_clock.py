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


"""Joint nonlinear birth-death killed-block clock.

Positive spectral-factor specialization of established phase-type methods.
green.py is copied unchanged from the previously audited event-order package.
Exactness refers to real arithmetic; floating interlacing violations raise.
"""
import math,time
from pathlib import Path
import numpy as np
from scipy.linalg import eigh_tridiagonal
from numba import njit
from . import green

@njit(cache=True)
def time_from_factors(eigen,complement):
 t=0.;phases=0
 for k in range(len(eigen)):
  if k<len(complement) and np.random.random()<eigen[k]/complement[k]:continue
  u=np.random.random()
  while u==0:u=np.random.random()
  t-=math.log(u)/eigen[k];phases+=1
 return t,phases

@njit(cache=True)
def choose_channel(q,j):
 u=np.random.random()*q[j].sum()
 if u<q[j,0]:return 0
 if u<q[j,0]+q[j,1]:return 1
 return 2

@njit(cache=True)
def seed(value):np.random.seed(value)

class Block:
 def __init__(self,H,K,alpha,beta,xi,gamma,epsilon,y):
  started=time.perf_counter()
  self.H,self.K,self.y=H,K,y;self.lo=0 if y else 1
  self.ids=np.arange(self.lo,K+1);n=len(self.ids)
  self.birth=(beta*(y/H)+epsilon*(self.ids/K))*(K-self.ids)
  self.death=gamma*self.ids.astype(float);self.death[0]=0.
  self.q=np.zeros((K+1,3));self.q[:,0]=alpha*((H-y)/H)*np.arange(K+1);self.q[:,1]=xi*y
  if y==0:self.q[1,2]=gamma
  self.kill=self.q[self.lo:].sum(axis=1)
  self.diag=self.birth+self.death+self.kill
  self.off=-np.sqrt(self.birth[:-1]*self.death[1:])
  if np.any(self.birth[:-1]<=0) or np.any(self.death[1:]<=0):raise ValueError('prototype requires irreducible transient birth-death block')
  self.eigen=self._eigs(0,n)
  if not np.all(self.eigen>0):raise ArithmeticError('nonpositive killed-block eigenvalue')
  self.data,self.ends=green.build(H,K,alpha,beta,xi,gamma,epsilon,y)
  self.prefix={0:np.empty(0)};self.suffix={n:np.empty(0)}
  self.spectral_seconds=time.perf_counter()-started;self.factor_calls=0
 @property
 def T(self):
  """Dense generator only on explicit request by the small independent audit."""
  return np.diag(-self.diag)+np.diag(self.birth[:-1],1)+np.diag(self.death[1:],-1)
 def _eigs(self,left,right):
  if right-left==0:return np.empty(0)
  if right-left==1:return self.diag[left:right].copy()
  return eigh_tridiagonal(self.diag[left:right],self.off[left:right-1],eigvals_only=True,lapack_driver='sterf',check_finite=True)
 def factors(self,i,j):
  started=time.perf_counter();l=min(i,j)-self.lo;r=max(i,j)-self.lo+1
  if not 0<=l<r<=len(self.ids):raise ValueError('start/endpoint outside transient block')
  if l not in self.prefix:self.prefix[l]=self._eigs(0,l)
  if r not in self.suffix:self.suffix[r]=self._eigs(r,len(self.ids))
  nu=np.sort(np.concatenate((self.prefix[l],self.suffix[r])))
  if np.any(nu<=0) or np.any(nu<self.eigen[:len(nu)]):raise ArithmeticError('floating eigenvalue interlacing violation; no clipping')
  self.spectral_seconds+=time.perf_counter()-started;self.factor_calls+=1
  return nu
 def draw(self,i):
  j,_=green.endpoint(self.data,self.ends,i)
  nu=self.factors(i,j)
  t,phases=time_from_factors(self.eigen,nu)
  c=choose_channel(self.q,j)
  return t,j,c,phases

@njit(cache=True)
def native_batch(H,K,a,b,xi,g,e,y,i0,N):
 out=np.empty((N,4));work=0
 for rep in range(N):
  i=i0;t=0.;steps=0
  while True:
   infect=a*((H-y)/H)*i;recover=xi*y;birth=(b*(y/H)+e*(i/K))*(K-i);death=g*i
   total=infect+recover+birth+death
   u=np.random.random()
   while u==0:u=np.random.random()
   t-=math.log(u)/total;v=np.random.random()*total;steps+=1
   if v<infect:c=0;break
   elif v<infect+recover:c=1;break
   elif v<infect+recover+birth:i+=1
   elif y==0 and i==1:c=2;break
   else:i-=1
  out[rep,0]=t;out[rep,1]=i;out[rep,2]=c;out[rep,3]=steps;work+=steps
 return out,work

@njit(cache=True)
def spectral_batch(data,ends,eigen,q,nu_table,nu_sizes,i0,N):
 out=np.empty((N,4));phases=0
 for rep in range(N):
  j,_=green.endpoint(data,ends,i0)
  t,num=time_from_factors(eigen,nu_table[j,:nu_sizes[j]])
  c=choose_channel(q,j)
  out[rep,0]=t;out[rep,1]=j;out[rep,2]=c;out[rep,3]=num;phases+=num
 return out,phases

def fixed_start_tables(block,i):
 table=np.empty((block.K+1,len(block.ids)));sizes=np.zeros(block.K+1,np.int64)
 for j in block.ids:
  nu=block.factors(i,int(j));table[j,:len(nu)]=nu;sizes[j]=len(nu)
 return table,sizes
