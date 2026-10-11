"""B16 experimental finite-horizon positive-history Gaussian-cell bounds.

Integrates the canonical three-scan field-dependent marked-PPP conditional law,
without changing the frozen benchmark. Positive first-detection and association
recursions are the same as B15, but coefficients are bounded locally in affine
Gaussian coordinates, not globally in tanh(F/clip) in [-1,1].

Proof status: mathematical interval formula conditional on exact arithmetic;
implementation uses ordinary numpy/scipy libm/normal CDF and is NOT a
validated outward-rounded interval algorithm. No mathematically-certified
numerical results can be claimed from this file. Non-validated feasibility ONLY.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from typing import Iterable
import math
import numpy as np
from scipy.special import ndtr

class ScopeError(ValueError):pass


def normal_pdf(x):return np.exp(-0.5*np.asarray(x)**2)/np.sqrt(2*np.pi)

@dataclass(frozen=True)
class Interval:
    lo: np.ndarray
    hi: np.ndarray
    def __post_init__(self):
        if self.lo.shape!=self.hi.shape:raise ScopeError('IV_SHAPE')
        if np.any(self.lo>self.hi):raise ScopeError('IV_REVERSED')
    @classmethod
    def exact(cls,x):
        a=np.asarray(x,dtype=float)
        return cls(a,a)
    def __add__(self,other):
        if not isinstance(other,Interval):other=Interval.exact(other)
        return Interval(self.lo+other.lo,self.hi+other.hi)
    __radd__=__add__
    def __mul__(self,other):
        if not isinstance(other,Interval):other=Interval.exact(other)
        if np.any(self.lo<0) or np.any(other.lo<0):raise ScopeError('POSITIVE_MUL_REQUIRED')
        return Interval(self.lo*other.lo,self.hi*other.hi)
    __rmul__=__mul__
    def complement(self):
        if np.any(self.lo < -1e-12) or np.any(self.hi>1+1e-12):raise ScopeError('NOT_PROBABILITY')
        return Interval(np.maximum(0,1-self.hi),np.minimum(1,1-self.lo))


def sum_iv(parts,shape):
    out=Interval.exact(np.zeros(shape))
    for x in parts:out=out+x
    return out


def positive_model_params(model):
    T=len(model.observations);D=len(model.survival)
    N=sum(map(len,model.observations))
    if T>3 or D>2 or N>4 or N==0:raise ScopeError('B16_T3_D2_N4_SCOPE')
    if any(len(row)>1 for row in model.observations):raise ScopeError('B16_ONE_OBS_PER_SCAN')
    if not model.observation_measure=='CANONICAL_SORTED_SIMPLEX_LEBESGUE':raise ScopeError('UNKNOWN_MEASURE')
    return T,D,N


def prior_affine(prior,T):
    """F_t=mean_t+sum_j L_tj Z_j, Z_j independent standard normal."""
    mu=np.zeros(T);L=np.zeros((T,T))
    mu[0]=prior.mean
    L[0,0]=np.sqrt(prior.variance)
    for t in range(1,T):
        mu[t]=prior.ar*mu[t-1]+prior.drift
        L[t,:]=prior.ar*L[t-1,:]
        L[t,t]=np.sqrt(prior.process_variance)
    return mu,L


def gaussian_box_statistics(z_lo,z_hi,prior):
    """Independent standard-normal box mass, conditional F means/variances.

    Uses numerical normal CDF; **not a rigorous CDF enclosure**.
    """
    z_lo=np.asarray(z_lo,dtype=float);z_hi=np.asarray(z_hi,dtype=float)
    if z_lo.ndim!=2 or z_lo.shape!=z_hi.shape:raise ScopeError('BOX_SHAPE')
    if np.any(z_lo>=z_hi):raise ScopeError('BOX_EMPTY')
    mass1=ndtr(z_hi)-ndtr(z_lo)
    if np.any(mass1<=0):raise ScopeError('NUMERIC_ZERO_MASS')
    pdf_lo=normal_pdf(z_lo);pdf_hi=normal_pdf(z_hi)
    meanZ=(pdf_lo-pdf_hi)/mass1
    secZ=1+(z_lo*pdf_lo-z_hi*pdf_hi)/mass1
    varZ=np.maximum(0,secZ-meanZ**2)
    mass=np.prod(mass1,axis=1)
    mu,L=prior_affine(prior,z_lo.shape[1]);Fmean=meanZ@L.T+mu
    Fvar=varZ@(L*L).T
    F_lo=np.tile(mu,(len(z_lo),1));F_hi=np.tile(mu,(len(z_lo),1))
    for t in range(len(mu)):
        for j in range(len(mu)):
            c=L[t,j]
            F_lo[:,t]+=c*(z_lo[:,j] if c>=0 else z_hi[:,j])
            F_hi[:,t]+=c*(z_hi[:,j] if c>=0 else z_lo[:,j])
    return mass,Fmean,np.sqrt(Fvar),F_lo,F_hi


def rates(m,F_lo,F_hi):
    T,D,_=positive_model_params(m)
    F_lo=np.asarray(F_lo,dtype=float);F_hi=np.asarray(F_hi,dtype=float)
    if F_lo.shape!=F_hi.shape or F_lo.ndim!=2 or F_lo.shape[1]!=T:raise ScopeError('FIELD_BOUND_SHAPE')
    zlo=np.tanh(F_lo/m.clip_scale);zhi=np.tanh(F_hi/m.clip_scale)
    B=[];S=[];M=[];Dobs=[]
    for t in range(T):
        births=[];surs=[];dets=[];motion=[]
        for x in range(D):
            b=m.births[t][x];g=m.gamma_birth[t][x]
            bl,bu=sorted([g*zlo[:,t],g*zhi[:,t]],key=lambda v:float(v[0]))
            births.append(Interval(b*np.exp(bl),b*np.exp(bu)))
            def linked(p,g):
                if p==0:return Interval.exact(np.zeros(len(F_lo)))
                if p==1:return Interval.exact(np.ones(len(F_lo)))
                v0=g*zlo[:,t];v1=g*zhi[:,t]
                lo=np.minimum(v0,v1);hi=np.maximum(v0,v1)
                # Odds formula prevents clipping logit at large magnitude.
                e0=np.exp(lo);e1=np.exp(hi)
                return Interval(p*e0/(1-p+p*e0),p*e1/(1-p+p*e1))
            surs.append(linked(m.survival[x],m.gamma_survival[x]))
            dets.append(linked(m.detection[x],m.gamma_detection[x]))
            row=[]
            for y in range(D):
                w=m.transition[x][y]
                if w==0:
                    row.append(Interval.exact(np.zeros(len(F_lo))))
                    continue
                denom_lo=np.ones(len(F_lo));denom_hi=np.ones(len(F_lo))
                for j in range(D):
                    if j==y or m.transition[x][j]==0:continue
                    diff=m.gamma_motion[x][j]-m.gamma_motion[x][y]
                    v0,v1=diff*zlo[:,t],diff*zhi[:,t]
                    ratio=m.transition[x][j]/w
                    denom_lo+=ratio*np.exp(np.minimum(v0,v1))
                    denom_hi+=ratio*np.exp(np.maximum(v0,v1))
                row.append(Interval(1/denom_hi,1/denom_lo))
            motion.append(tuple(row))
        B.append(tuple(births));S.append(tuple(surs));Dobs.append(tuple(dets));M.append(tuple(motion))
    return B,S,M,Dobs


def lambda_plus(B,S,M,Dobs):
    T=len(B);D=len(B[0]);shape=B[0][0].lo.shape
    pending=list(B[0]);lam=Interval.exact(np.zeros(shape))
    for t in range(T):
        if t:
            pending=[B[t][j]+sum_iv((pending[x]*S[t][x]*M[t][x][j] for x in range(D)),shape) for j in range(D)]
        lam=lam+sum_iv((pending[x]*Dobs[t][x] for x in range(D)),shape)
        pending=[pending[x]*Dobs[t][x].complement() for x in range(D)]
    return lam


def gaussian_emissions(m,F_lo,F_hi):
    obs=tuple((t,j) for t,row in enumerate(m.observations) for j in range(len(row)))
    result=[]
    for t,j in obs:
        arr=[]
        for x in range(len(m.survival)):
            v=m.observation_variance[x];sd=np.sqrt(v)
            ml=m.offset[x]+m.response[x]*F_lo[:,t];mh=m.offset[x]+m.response[x]*F_hi[:,t]
            y=m.observations[t][j]
            lo=np.minimum(ml,mh);hi=np.maximum(ml,mh)
            dmin=np.maximum(0,np.maximum(lo-y,y-hi))
            dmax=np.maximum(np.abs(y-lo),np.abs(y-hi))
            const=1/(np.sqrt(2*np.pi)*sd)
            arr.append(Interval(const*np.exp(-0.5*(dmax/sd)**2),const*np.exp(-0.5*(dmin/sd)**2)))
        result.append(tuple(arr))
    return obs,result


def _detected_options(m,B,S,M,Dobs,emissions,obs):
    T=len(B);D=len(B[0]);N=len(obs);shape=B[0][0].lo.shape
    opts={mask:Interval.exact(np.zeros(shape)) for mask in range(1,1<<N)
          if len({obs[i][0] for i in range(N) if mask&(1<<i)})==mask.bit_count()}
    # Every actual physical state-history, every positive detection subset.
    for b in range(T):
        for e in range(b,T):
            for path in product(range(D),repeat=e-b+1):
                p=B[b][path[0]]
                for t in range(b+1,e+1):
                    p=p*S[t][path[t-b-1]]*M[t][path[t-b-1]][path[t-b]]
                if e<T-1:p=p*S[e+1][path[-1]].complement()
                for mask in opts:
                    chosen={obs[i][0]:i for i in range(N) if mask&(1<<i)}
                    if any(t<b or t>e for t in chosen):continue
                    w=p
                    for t in range(b,e+1):
                        x=path[t-b]
                        if t in chosen:
                            w=w*Dobs[t][x]*emissions[chosen[t]][x]
                        else:w=w*Dobs[t][x].complement()
                    opts[mask]=opts[mask]+w
    return opts


def conditional_likelihood_intervals(m,F_lo,F_hi):
    """Per-box likelihood enclosure BEFORE integrating Gaussian prior."""
    T,D,N=positive_model_params(m)
    if len(F_lo)!=len(F_hi):raise ScopeError('F_ROWS')
    B,S,M,Dobs=rates(m,F_lo,F_hi)
    lam=lambda_plus(B,S,M,Dobs)
    obs,em=gaussian_emissions(m,F_lo,F_hi)
    options=_detected_options(m,B,S,M,Dobs,em,obs)
    shape=F_lo[:,0].shape
    clutter=[]
    for t,j in obs:
        y=m.observations[t][j];v=m.clutter_variance
        clutter.append(m.clutter_rate*np.exp(-.5*(y-m.clutter_mean)**2/v)/np.sqrt(2*np.pi*v))
    P={0:Interval.exact(np.ones(shape))}
    for bits in range(1,1<<N):
        bit=bits&-bits;idx=bit.bit_length()-1
        v=P[bits^bit]*clutter[idx]
        for mask,w in options.items():
            if mask&bit and mask&bits==mask:v=v+w*P[bits^mask]
        P[bits]=v
    poly=P[(1<<N)-1];base=np.exp(-T*m.clutter_rate)
    return Interval(base*poly.lo*np.exp(-lam.hi),base*poly.hi*np.exp(-lam.lo))


def z_grid(dims: int,interior=16,far=8.,central=4.):
    """20 bins for interior=18: bins [-8,-4], 18 central, [4,8]."""
    if not(1<=dims<=3 and 2<=interior<=40 and far>central>0):raise ScopeError('GRID_INVALID')
    axis=np.concatenate(([-far],np.linspace(-central,central,interior+1),[far]))
    lowers=[];uppers=[]
    for i in product(range(len(axis)-1),repeat=dims):
        lowers.append([axis[j] for j in i]);uppers.append([axis[j+1] for j in i])
    return np.asarray(lowers),np.asarray(uppers)


def outer_global_sup(m):
    T=len(m.observations)
    lo=np.full((1,T),-1000.0);hi=np.full((1,T),1000.0)
    iv=conditional_likelihood_intervals(m,lo,hi)
    return float(iv.hi[0])


def study(m,prior,*,interior=16,far=8.,central=4.,max_leaves=12000):
    T,D,N=positive_model_params(m)
    lower,upper=z_grid(T,interior=interior,far=far,central=central)
    if len(lower)>max_leaves:raise ScopeError('LEAF_BUDGET_EXCEEDED')
    mass,muF,sdF,Flo,Fhi=gaussian_box_statistics(lower,upper,prior)
    iv=conditional_likelihood_intervals(m,Flo,Fhi)
    # p(F) is Gaussian prior exactly; enclose conditional likelihood and multiply.
    Zlo=float(np.sum(mass*iv.lo));Zhi=float(np.sum(mass*iv.hi))
    h=(iv.lo+iv.hi)/2;delta=(iv.hi-iv.lo)/2
    # E[F * likelihood|box] = muF * E[likelihood|box]
    #   + E[(F-muF)(likelihood-h)|box]; latter <= delta * sdF.
    mom_lo=np.sum(mass[:,None]*(muF*iv.lo[:,None]*(muF>=0)+muF*iv.hi[:,None]*(muF<0)
           -delta[:,None]*sdF),axis=0)
    mom_hi=np.sum(mass[:,None]*(muF*iv.hi[:,None]*(muF>=0)+muF*iv.lo[:,None]*(muF<0)
           +delta[:,None]*sdF),axis=0)
    p_out=max(0.,1-float(np.sum(mass)))
    # Tail: positive 0 <= likelihood <= global uniform supremum.
    global_sup=outer_global_sup(m)
    Zhi+=global_sup*p_out
    prior_mu,L=prior_affine(prior,T)
    prior_sd2=np.sum(L**2,axis=1)
    tail_num=global_sup*np.sqrt(p_out*(prior_mu**2+prior_sd2))
    mom_lo-=tail_num;mom_hi+=tail_num
    if Zlo<=0:raise ScopeError('EVIDENCE_DENOMINATOR_NOT_POSITIVE')
    means=[]
    for a,b in zip(mom_lo,mom_hi):
        means.append([min(a/Zlo,a/Zhi,b/Zlo,b/Zhi),max(a/Zlo,a/Zhi,b/Zlo,b/Zhi)])
    return {'status':'B16_EXPLORATORY_NUMERIC_ONLY',
            'frozen_model_digest':m.digest(),'leaves':int(len(lower)),
            'evidence_interval':[Zlo,Zhi], 'evidence_width':Zhi-Zlo,
            'field_mean_intervals':means,'field_mean_widths':[x[1]-x[0] for x in means],
            'tail_mass_estimate':p_out,'global_likelihood_sup_estimate':global_sup,
            'source_identity':'B15 field-dependent canonical marked-PPP law',
            'proof_status':'NOT_RIGOROUS_NUMERIC; scipy/NumPy transcendental/CDF rounding NOT ENCLOSED',
            'root_normalizer_likelihood':'first-detection recursion; all partitions; no dropped physical paths',
            'numeric_gates_pass_heuristically':Zhi-Zlo<=.001 and all(x[1]-x[0]<=1.0 for x in means)}
