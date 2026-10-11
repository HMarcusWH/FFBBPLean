"""B16 second-order conditional Gaussian-cell feasibility evaluator.

Mathematical idea:
For Gaussian F restricted to independent innovation-coordinate box C,
E[f(F)|C]=f(E[F|C])+R, |R|<=.5 sum_ij H_ij sqrt(var_i var_j),
provided H_ij uniformly bounds |partial_ij f| over convex hull of C.
First moment uses the exact conditional covariance and central first derivative,
with remainder bounded by maximum centered F distance times the same R bound.

This implementation uses standard double arithmetic and scipy normal CDF;
returned intervals are **NOT RIGOROUSLY CERTIFIED**. In particular the
transcendental/CDF rounding and finite-difference derivative estimates are
unproved and separate from the frozen B15 proof obligations.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
import numpy as np
from scipy.special import ndtr
from .gaussian_cells import (ScopeError,normal_pdf,positive_model_params,
    gaussian_box_statistics,prior_affine,z_grid,
    conditional_likelihood_intervals,outer_global_sup)

@dataclass
class Bound:
    lo:np.ndarray
    hi:np.ndarray
    d:np.ndarray
    h:np.ndarray
    @classmethod
    def const(cls,val,rows,T):
        v=np.broadcast_to(np.asarray(val,dtype=float),(rows,))
        return cls(v,v,np.zeros((rows,T)),np.zeros((rows,T,T)))
    def __add__(self,other):
        return Bound(self.lo+other.lo,self.hi+other.hi,self.d+other.d,self.h+other.h)
    def __mul__(self,other):
        # All factors are nonnegative; d/h bound absolute derivatives.
        h=self.h*other.hi[:,None,None]+other.h*self.hi[:,None,None]
        h=h+np.einsum('ni,nj->nij',self.d,other.d)+np.einsum('ni,nj->nij',other.d,self.d)
        d=self.d*other.hi[:,None]+other.d*self.hi[:,None]
        return Bound(self.lo*other.lo,self.hi*other.hi,d,h)
    def scalar(self,a):
        if a<0:raise ScopeError('NEGATIVE_BOUND_SCALE')
        return Bound(a*self.lo,a*self.hi,a*self.d,a*self.h)
    def complement(self):
        return Bound(np.maximum(0,1-self.hi),np.minimum(1,1-self.lo),self.d,self.h)
    def exp_negative(self):
        lo=np.exp(-self.hi);hi=np.exp(-self.lo)
        return Bound(lo,hi,hi[:,None]*self.d,hi[:,None,None]*(self.h+np.einsum('ni,nj->nij',self.d,self.d)))


def zip_bound(lo,hi,der1,der2,t,T):
    lo=np.asarray(lo);hi=np.asarray(hi)
    n=len(lo);d=np.zeros((n,T));h=np.zeros((n,T,T))
    d[:,t]=der1;h[:,t,t]=der2
    return Bound(lo,hi,d,h)


def tanh_range_derivs(Flo,Fhi,clip):
    t0=np.tanh(Flo/clip);t1=np.tanh(Fhi/clip)
    mint=np.where(t0*t1<=0,0,np.minimum(abs(t0),abs(t1)))
    maxt=np.maximum(abs(t0),abs(t1))
    der=(1-mint*mint)/clip
    at=np.clip(1/np.sqrt(3),mint,maxt)
    der2=2*at*(1-at*at)/(clip*clip)
    return t0,t1,der,der2


def logistic_max_derivative(a,b):
    # x(1-x) attains maximum .25 at x=.5.
    return np.where((a<=.5)&(b>=.5),.25,np.maximum(a*(1-a),b*(1-b)))


def logistic_max_second(a,b):
    # Absolute sigma''(u)=sigma(1-sigma)|1-2sigma|.
    def val(x):return x*(1-x)*np.abs(1-2*x)
    out=np.maximum(val(a),val(b))
    p=(3-np.sqrt(3))/6;q=(3+np.sqrt(3))/6
    out=np.where((a<=p)&(b>=p),np.maximum(out,val(p)),out)
    out=np.where((a<=q)&(b>=q),np.maximum(out,val(q)),out)
    return out


def link_bound(p,g,za,zb,zprime,zsecond,t,T):
    n=len(za)
    if p==0:return Bound.const(0,n,T)
    if p==1:return Bound.const(1,n,T)
    a=np.exp(np.minimum(g*za,g*zb));b=np.exp(np.maximum(g*za,g*zb))
    slo=p*a/(1-p+p*a);shi=p*b/(1-p+p*b)
    first=logistic_max_derivative(slo,shi)
    second=logistic_max_second(slo,shi)
    d=first*abs(g)*zprime
    h=second*g*g*zprime*zprime+first*abs(g)*zsecond
    return zip_bound(slo,shi,d,h,t,T)


def birth_bound(b,g,za,zb,zprime,zsecond,t,T):
    a=np.exp(np.minimum(g*za,g*zb))*b
    c=np.exp(np.maximum(g*za,g*zb))*b
    d=c*abs(g)*zprime
    h=c*(g*g*zprime*zprime+abs(g)*zsecond)
    return zip_bound(a,c,d,h,t,T)


def motion_bound(m,t,x,y,za,zb,zprime,zsecond,T):
    w=m.transition[x][y]
    if w==0:return Bound.const(0,len(za),T)
    other=[j for j in range(len(m.survival)) if j!=y and m.transition[x][j]>0]
    if not other:return Bound.const(1,len(za),T)
    j=other[0];delta=m.gamma_motion[x][y]-m.gamma_motion[x][j]
    # motion_y(z) = [1 + (w_j/w_y)*exp(-delta*z)]^-1
    # not just gamma difference of numerator; proper odds shift.
    eta0=np.log(w/m.transition[x][j])+delta*za
    eta1=np.log(w/m.transition[x][j])+delta*zb
    slo=1/(1+np.exp(-np.minimum(eta0,eta1)))
    shi=1/(1+np.exp(-np.maximum(eta0,eta1)))
    first=logistic_max_derivative(slo,shi)
    second=logistic_max_second(slo,shi)
    return zip_bound(slo,shi,first*abs(delta)*zprime,
                     second*delta*delta*zprime*zprime+first*abs(delta)*zsecond,t,T)


def emission_bound(m,t,j,x,Flo,Fhi,T):
    y=m.observations[t][j];off=m.offset[x];r=m.response[x];v=m.observation_variance[x]
    y_minus_0=y-off-r*Flo;y_minus_1=y-off-r*Fhi
    a=np.minimum(y_minus_0,y_minus_1)/np.sqrt(v)
    b=np.maximum(y_minus_0,y_minus_1)/np.sqrt(v)
    minabs=np.where(a*b<=0,0,np.minimum(abs(a),abs(b)))
    maxabs=np.maximum(abs(a),abs(b))
    c=1/np.sqrt(2*np.pi*v)
    lo=c*np.exp(-.5*maxabs**2);hi=c*np.exp(-.5*minabs**2)
    # abs(g') max at |a|=1
    x1=np.clip(1.,minabs,maxabs)
    d=c*abs(r)/np.sqrt(v)*x1*np.exp(-.5*x1*x1)
    # abs(g'') candidates at endpoints, at a=0 and a=sqrt(3) when inside
    f=lambda x:abs(x*x-1)*np.exp(-.5*x*x)
    mx=np.maximum(f(minabs),f(maxabs))
    mx=np.where(minabs==0,np.maximum(mx,1.),mx)
    mx=np.where((minabs<=np.sqrt(3))&(maxabs>=np.sqrt(3)),
                np.maximum(mx, f(np.sqrt(3))),mx)
    h=c*r*r/v*mx
    return zip_bound(lo,hi,d,h,t,T)


def second_order_likelihood(m,Flo,Fhi):
    """Uniform mixed-Hessian bound for the exact conditional Janossy density."""
    T,D,N=positive_model_params(m);rows=len(Flo)
    if Flo.shape!=(rows,T) or Fhi.shape!=(rows,T):raise ScopeError('F_SHAPE')
    null=lambda:Bound.const(0,rows,T)
    one=lambda:Bound.const(1,rows,T)
    B=[];S=[];M=[];Dobs=[]
    for t in range(T):
        za,zb,zp,zpp=tanh_range_derivs(Flo[:,t],Fhi[:,t],m.clip_scale)
        B.append([birth_bound(m.births[t][x],m.gamma_birth[t][x],za,zb,zp,zpp,t,T) for x in range(D)])
        S.append([link_bound(m.survival[x],m.gamma_survival[x],za,zb,zp,zpp,t,T) for x in range(D)])
        Dobs.append([link_bound(m.detection[x],m.gamma_detection[x],za,zb,zp,zpp,t,T) for x in range(D)])
        M.append([[motion_bound(m,t,x,y,za,zb,zp,zpp,T) for y in range(D)] for x in range(D)])
    # Exact B15 undetected forward first-detection recursion with interval derivative bounds.
    lam=null();pending=B[0]
    for t in range(T):
        if t:
            pending=[B[t][y]+sum((pending[x]*S[t][x]*M[t][x][y] for x in range(D)),null()) for y in range(D)]
        lam=lam+sum((pending[x]*Dobs[t][x] for x in range(D)),null())
        pending=[pending[x]*Dobs[t][x].complement() for x in range(D)]
    # Complete B15 legal physical history / observed-origin partition sum.
    obs=tuple((t,j) for t,scan in enumerate(m.observations) for j in range(len(scan)))
    emis=[[emission_bound(m,t,j,x,Flo[:,t],Fhi[:,t],T) for x in range(D)] for t,j in obs]
    options={mask:null() for mask in range(1,1<<N) if len({obs[i][0] for i in range(N) if mask&(1<<i)})==mask.bit_count()}
    for b in range(T):
        for e in range(b,T):
            for path in product(range(D),repeat=e-b+1):
                weight=B[b][path[0]]
                for t in range(b+1,e+1):weight=weight*S[t][path[t-b-1]]*M[t][path[t-b-1]][path[t-b]]
                if e<T-1:weight=weight*S[e+1][path[-1]].complement()
                for mask in options:
                    chosen={obs[i][0]:i for i in range(N) if mask&(1<<i)}
                    if any(t<b or t>e for t in chosen):continue
                    w=weight
                    for t in range(b,e+1):
                        x=path[t-b]
                        if t in chosen:w=w*Dobs[t][x]*emis[chosen[t]][x]
                        else:w=w*Dobs[t][x].complement()
                    options[mask]=options[mask]+w
    clutter=[]
    for t,j in obs:
        y=m.observations[t][j]
        v=m.clutter_variance
        clutter.append(m.clutter_rate*np.exp(-.5*(y-m.clutter_mean)**2/v)/np.sqrt(2*np.pi*v))
    P={0:one()}
    for bits in range(1,1<<N):
        bit=bits&-bits;idx=bit.bit_length()-1
        vv=P[bits^bit].scalar(clutter[idx])
        for mask,w in options.items():
            if mask&bit and mask&bits==mask:vv=vv+w*P[bits^mask]
        P[bits]=vv
    return P[(1<<N)-1].scalar(np.exp(-T*m.clutter_rate))*lam.exp_negative()


def feasible_second_order(m,prior,*,interior=16,central=4.,far=8.,max_leaves=12000,h=1e-4):
    T,D,N=positive_model_params(m)
    lower,upper=z_grid(T,interior=interior,far=far,central=central)
    if len(lower)>max_leaves:raise ScopeError('LEAF_BUDGET_EXCEEDED')
    mass,muF,sdF,Flo,Fhi=gaussian_box_statistics(lower,upper,prior)
    jet=second_order_likelihood(m,Flo,Fhi)
    mid=conditional_likelihood_intervals(m,muF,muF)
    f=mid.lo
    # The uniform Hessian yields an exact mathematical Taylor expectation bound,
    # conditional on rigorous implementation of all interval primitives.
    err=.5*np.einsum('nij,ni,nj->n',jet.h,sdF,sdF)
    Zlo=float(np.sum(mass*np.maximum(0,f-err)));Zhi=float(np.sum(mass*(f+err)))
    # Cov(F_t,F_r | C) = L diag(truncated-normal Z variance) L^T.
    zlo=lower;zhi=upper
    pm=ndtr(zhi)-ndtr(zlo);phi_lo=normal_pdf(zlo);phi_hi=normal_pdf(zhi)
    u=(phi_lo-phi_hi)/pm
    zzvar=np.maximum(0,1+(zlo*phi_lo-zhi*phi_hi)/pm-u*u)
    _,L=prior_affine(prior,T)
    cov=np.einsum('ti,ni,ri->ntr',L,zzvar,L)
    # Finite difference at center is a *numeric probe*, not a proof of its own
    # derivative. When formalized, use interval dual arithmetic at the center.
    der=[]
    for r in range(T):
        plus=muF.copy();minus=muF.copy();plus[:,r]+=h;minus[:,r]-=h
        fp=conditional_likelihood_intervals(m,plus,plus).lo
        fm=conditional_likelihood_intervals(m,minus,minus).lo
        der.append((fp-fm)/(2*h))
    der=np.stack(der,axis=1)
    Mbase=muF*f[:,None]+np.einsum('nr,ntr->nt',der,cov)
    # Remainder in E[(F_t-mu_t)f] bounded by max centered deviation times
    # the Taylor error, and center gradient numerical difference bound by
    # h^2 H for true central differences (requires 3rd derivatives for O(h²)),
    # so conservatively use first-order h*H gradient error.
    mdev=np.maximum(muF-Flo,Fhi-muF)
    derivative_error=h*np.einsum('nr,ntr->nt',np.diagonal(jet.h,axis1=1,axis2=2),np.abs(cov))
    errM=(abs(muF)+mdev)*err[:,None]+derivative_error
    mom_lo=np.sum(mass[:,None]*(Mbase-errM),axis=0)
    mom_hi=np.sum(mass[:,None]*(Mbase+errM),axis=0)
    # Both central and Hessian routes are clipped only after computing bounds;
    # conservative tail based on sup f and global prior second moments.
    p_out=min(1.,2*T*float(ndtr(-far)))
    global_sup=outer_global_sup(m)
    Zhi+=global_sup*p_out
    prior_mu,L=prior_affine(prior,T)
    tail_num=global_sup*np.sqrt(p_out*(prior_mu**2+np.sum(L*L,axis=1)))
    mom_lo-=tail_num;mom_hi+=tail_num
    if Zlo<=0:raise ScopeError('LOWER_EVIDENCE_NONPOSITIVE')
    means=[]
    for a,b in zip(mom_lo,mom_hi):
        q=[a/Zlo,a/Zhi,b/Zlo,b/Zhi];means.append([min(q),max(q)])
    return {'status':'B16_SECOND_ORDER_EXPLORATORY_NOT_CERTIFIED',
        'frozen_model_digest':m.digest(),'leaves':len(lower),'interior':interior,
        'evidence_interval':[Zlo,Zhi],'evidence_width':Zhi-Zlo,
        'field_mean_intervals':means,'field_mean_widths':[v[1]-v[0] for v in means],
        'tail_mass_analytic_union_estimate':p_out,
        'hessian_error_contributions':float(np.sum(mass*err)),
        'max_hessian_bound':float(jet.h.max()),
        'numeric_gates_pass_heuristically': Zhi-Zlo<=.001 and all(v[1]-v[0]<=1. for v in means),
        'proof_status':'PENDING: rigorous elementary/CDF enclosures, validated derivative at cell mean, independent formal proof, exact model bridge'}
