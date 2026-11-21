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


"""Frozen-source physical-history dispatcher; no hidden finite-T endpoint."""
import time,math
import numpy as np
from . import native_only

KEYS=['H','K','a','b','xi','g','e','y0','i0','threshold','T','N','cap']


def arguments(c,N=None):
 x=[c[k] for k in KEYS]
 if N is not None:x[-2]=N
 return x


def deterministic_gate(c):
 cache=c['threshold']*16*(c['K']+1)**2
 if c['N']<=32:return 'small request'
 if c['K']>192:return 'large K'
 if c['e']<=0 or c['b']<=0:return 'unsupported reducible blocks'
 if min(c['a'],c['xi'])/max(c['b'],c['g'],c['e'])<1e-10:return 'extreme kill scale'
 if cache>32*1024**2:return 'spectral cache estimate'
 return None


def spectral_paths(c,N):
 # Import only after selection, so native costs no unused SciPy/spectral JIT.
 from . import repair as clock
 H,K,a,b,xi,g,e,y0,i0,threshold,T,_,cap=arguments(c,N)
 rows=np.full((N,cap+1,4),np.nan);counts=np.zeros(N,np.int64);status=np.full(N,-2,np.int8)
 rows[:,0]=[0.,y0,i0,-1.];cache={};failed={};queries=0;phases=0;fallbacks=0;native_work=0;abort=None
 for rep in range(N):
  y=y0;i=i0;t=0.;events=0;status[rep]=-1
  while y<threshold and y+i>0 and events<cap:
   if y not in cache and y not in failed:
    try:cache[y]=clock.Block(H,K,a,b,xi,g,e,y)
    except (ArithmeticError,ValueError,np.linalg.LinAlgError) as ex:failed[y]=repr(ex)
   if y in failed:
    # Safe fallback: no endpoint RNG has occurred at this current state.
    rest=dict(c,y0=y,i0=i,T=T-t,cap=cap-events,N=1)
    rr,nn,ss,ww=native_only.native_paths(*arguments(rest));n=int(nn[0])
    if n:rows[rep,events+1:events+n+1]=rr[0,1:n+1];rows[rep,events+1:events+n+1,0]+=t
    events+=n;status[rep]=ss[0];native_work+=ww;fallbacks+=1
    break
   try:
    dt,j,channel,num=cache[y].draw(i)
    if not math.isfinite(dt) or dt<0:raise ArithmeticError('nonfinite/negative conditional time')
   except (ArithmeticError,ValueError,np.linalg.LinAlgError) as ex:
    # draw() selected an endpoint before factors(); discarding it is invalid.
    status[rep]=-2;abort=dict(path=rep,Y=y,I=i,time=t,error=repr(ex),stage='after endpoint; no retry')
    break
   queries+=1;phases+=num
   if t+dt>T:status[rep]=2;break
   t+=dt;i=j
   if channel==0:y+=1
   elif channel==1:y-=1
   else:i=0;break
   events+=1;rows[rep,events]=[t,y,i,channel]
  if y>=threshold:status[rep]=1
  elif y+i==0:status[rep]=0
  counts[rep]=events
  if abort is not None:break
 spectra=sum(z.eigen.nbytes+sum(v.nbytes for v in z.prefix.values())+sum(v.nbytes for v in z.suffix.values()) for z in cache.values())
 arrays=sum(sum(getattr(z,k).nbytes for k in ['ids','birth','death','q','kill','diag','off','data','ends']) for z in cache.values())
 meta=dict(queries=queries,phases=phases,blocks=len(cache),failed_setup=failed,fallback_paths=fallbacks,native_fallback_work=int(native_work),
           conditional_fallbacks=sum(z.conditional_fallbacks for z in cache.values()),conditional_steps=sum(z.conditional_steps for z in cache.values()),spectral_phases=sum(z.spectral_phases for z in cache.values()),spectral_seconds=sum(z.spectral_seconds for z in cache.values()),spectral_cache_bytes=spectra,block_array_bytes=arrays,abort=abort)
 return rows,counts,status,meta


def run(c,method,seed):
 if method not in ['native','spectral','auto']:raise ValueError('method')
 if c['N']*(c['cap']+1)*4*8>128*1024**2:raise ValueError('registered output memory limit')
 started=time.perf_counter();native_only.seed(seed);meta=dict(requested=method,pilot_paths=0,selected=method)
 if method=='native':
  rows,counts,status,work=native_only.native_paths(*arguments(c));meta['native_work']=int(work)
 elif method=='spectral':rows,counts,status,work=spectral_paths(c,c['N']);meta['spectral']=work
 else:
  gate=deterministic_gate(c);meta['deterministic_gate']=gate
  if gate:
   rows,counts,status,work=native_only.native_paths(*arguments(c));meta.update(selected='native',native_work=int(work))
  else:
   st=time.perf_counter();native_only.native_paths(*arguments(c,0));meta['native_compile_s']=time.perf_counter()-st
   pilot=8;st=time.perf_counter();pr,pc,ps,pw=native_only.native_paths(*arguments(c,pilot));elapsed=time.perf_counter()-st
   remaining=c['N']-pilot;h=float(pc.mean());native_est=.5*elapsed*remaining/pilot
   spectral_est=2e-7*c['K']*(h+1)*remaining+4e-9*c['K']**3*c['threshold']
   chosen='spectral' if np.all(ps>=0) and native_est>spectral_est+1.5 else 'native'
   meta.update(pilot_paths=pilot,pilot_seconds=elapsed,pilot_native_work=int(pw),pilot_visible_mean=h,
               estimated_native_remaining_s=native_est,estimated_spectral_remaining_s=spectral_est,extra_compile_allowance_s=1.5,selected=chosen)
   if chosen=='native':rr,rc,rs,rw=native_only.native_paths(*arguments(c,remaining));meta['native_work']=int(pw+rw)
   else:rr,rc,rs,rw=spectral_paths(c,remaining);meta['spectral']=rw
   rows=np.concatenate((pr,rr));counts=np.r_[pc,rc];status=np.r_[ps,rs]
 meta['elapsed_s']=time.perf_counter()-started;meta['output_array_bytes']=rows.nbytes+counts.nbytes+status.nbytes
 return rows,counts,status,meta


def summarize(rows,counts,status):
 return dict(paths=len(status),statuses={str(x):int(np.sum(status==x)) for x in [-2,-1,0,1,2]},
             mean_events=float(counts.mean()),mean_last_observed_time=float(np.mean(rows[np.arange(len(counts)),counts,0])))
