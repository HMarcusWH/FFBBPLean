"""B16 audit-oriented directed-decimal enclosure for frozen FFBBP T3/D2/N3.

The program computes exact-input, directed-rounding interval boxes with:
* independently enumerated complete positive history / clutter partitions,
* undetected-PPP first-detection normalization,
* Gaussian-innovation cell integrals via Fraction/Taylor series,
* uniform Jacobian and Hessian absolute bounds for the conditional law,
* centered Taylor evidence and first-moment inequalities,
* explicit full-domain tail controls and strict-positive evidence division.

ASSURANCE LEVEL: B16 experimental *conditional mathematical enclosure*.
This program is not a Lean proof. Its mathematical validity requires review of
these formulae and of the model-to-Janossy correspondence. Elementary exp and
sqrt enclosures use documented Python decimal correctly rounded operations
with one representable-unit outward padding; normal CDF uses exact Fraction
alternating-series bounds. Any result must retain these stated assumptions.
It does not validate field existence, identification or operational use.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from fractions import Fraction as Q
from decimal import Decimal as D, Context, ROUND_FLOOR, ROUND_CEILING, ROUND_HALF_EVEN, localcontext
from itertools import product
from multiprocessing import get_context
from time import monotonic
import math
import numpy as np
from scipy.special import ndtr,ndtri
from .rational_normal import normal_constant_bounds, normal_cdf_bounds,truncated_conditional_bounds

PREC=48
LO=Context(prec=PREC,rounding=ROUND_FLOOR,Emin=-999999,Emax=999999)
HI=Context(prec=PREC,rounding=ROUND_CEILING,Emin=-999999,Emax=999999)
RN=Context(prec=PREC,rounding=ROUND_HALF_EVEN,Emin=-999999,Emax=999999)
ZERO=D(0);ONE=D(1);TWO=D(2);QUARTER=D('0.25');TENTH=D('0.1');TANH_SECOND_CAP=D('0.77')

def d(x):
    if isinstance(x,D):return x
    if isinstance(x,float):return D.from_float(x)
    return D(str(x))

def qd(x:Q,ceil=False):
    with localcontext(HI if ceil else LO):return D(x.numerator)/D(x.denominator)

def mul_hi(*args):
    with localcontext(HI):
        a=ONE
        for x in args:a=a*x
        return a

def sum_hi(*args):
    with localcontext(HI):return sum(args,ZERO)

def absup(x):
    return max(abs(x.lo),abs(x.hi))

@dataclass(frozen=True)
class IV:
    lo:D
    hi:D
    def __post_init__(self):
        if self.lo>self.hi:raise ValueError('REVERSED_INTERVAL')
    @classmethod
    def exact(cls,x):
        x=d(x);return cls(x,x)
    @classmethod
    def rational(cls,a:Q,b:Q):return cls(qd(a),qd(b,True))
    def __add__(self,other):
        other=as_iv(other)
        with localcontext(LO): l=self.lo+other.lo
        with localcontext(HI): u=self.hi+other.hi
        return IV(l,u)
    __radd__=__add__
    def __neg__(self):return IV(-self.hi,-self.lo)
    def __sub__(self,other):return self+(-as_iv(other))
    def __rsub__(self,other):return as_iv(other)+(-self)
    def __mul__(self,other):
        other=as_iv(other)
        with localcontext(LO):l=min(a*b for a in (self.lo,self.hi) for b in (other.lo,other.hi))
        with localcontext(HI):u=max(a*b for a in (self.lo,self.hi) for b in (other.lo,other.hi))
        return IV(l,u)
    __rmul__=__mul__
    def __truediv__(self,other):
        other=as_iv(other)
        if other.lo<=ZERO<=other.hi:raise ValueError('DENOMINATOR_NOT_EXCLUDED_ZERO')
        with localcontext(LO):l=min(ONE/x for x in (other.lo,other.hi))
        with localcontext(HI):u=max(ONE/x for x in (other.lo,other.hi))
        return self*IV(l,u)
    def exp(self):
        with localcontext(RN):
            l=self.lo.exp();u=self.hi.exp()
        with localcontext(LO):l=l.next_minus()
        with localcontext(HI):u=u.next_plus()
        return IV(l,u)
    def sqrt(self):
        if self.lo<0:raise ValueError('SQRT_NEGATIVE')
        with localcontext(RN):
            l=self.lo.sqrt();u=self.hi.sqrt()
        with localcontext(LO):l=max(ZERO,l.next_minus())
        with localcontext(HI):u=u.next_plus()
        return IV(l,u)
    def square(self):
        if self.lo<=0<=self.hi:
            with localcontext(HI):u=max(self.lo*self.lo,self.hi*self.hi)
            return IV(ZERO,u)
        with localcontext(LO):l=min(self.lo*self.lo,self.hi*self.hi)
        with localcontext(HI):u=max(self.lo*self.lo,self.hi*self.hi)
        return IV(l,u)
    def intersect(self,lo=None,hi=None):
        a=self.lo if lo is None else max(self.lo,d(lo))
        b=self.hi if hi is None else min(self.hi,d(hi))
        return IV(a,b)
    def midpoint(self):
        with localcontext(RN):return (self.lo+self.hi)/TWO
    def width(self):
        with localcontext(HI):return self.hi-self.lo

def as_iv(x):return x if isinstance(x,IV) else IV.exact(x)
ONEI=IV.exact(ONE);ZEROI=IV.exact(ZERO)

@lru_cache(maxsize=1)
def normal_const():
    x,y=normal_constant_bounds();return IV.rational(x,y)

@dataclass
class Jet:
    v:IV
    d:list[D]
    h:list[list[D]]
    @classmethod
    def const(cls,x,T):return cls(as_iv(x),[ZERO]*T,[[ZERO]*T for _ in range(T)])
    def __add__(self,other):
        T=len(self.d)
        return Jet(self.v+other.v,[sum_hi(self.d[t],other.d[t]) for t in range(T)],
            [[sum_hi(self.h[t][s],other.h[t][s]) for s in range(T)] for t in range(T)])
    def __mul__(self,other):
        T=len(self.d);v=self.v*other.v
        f,g=self.v.hi,other.v.hi
        if min(self.v.lo,other.v.lo)<0:raise ValueError('JET_POSITIVE_MULTIPLY_ONLY')
        grads=[sum_hi(mul_hi(self.d[t],g),mul_hi(other.d[t],f)) for t in range(T)]
        H=[[sum_hi(mul_hi(self.h[t][s],g),mul_hi(other.h[t][s],f),
            mul_hi(self.d[t],other.d[s]),mul_hi(other.d[t],self.d[s])) for s in range(T)] for t in range(T)]
        return Jet(v,grads,H)
    def complement(self):
        return Jet((ONEI-self.v).intersect(0,1),self.d,self.h)
    def scale(self,x):
        x=d(x)
        if x<0:raise ValueError('SCALE_NEGATIVE')
        with localcontext(HI):
            grads=[z*x for z in self.d];H=[[z*x for z in row] for row in self.h]
        return Jet(self.v*x,grads,H)
    def expneg(self):
        T=len(self.d);v=(-self.v).exp();u=v.hi
        grads=[mul_hi(u,x) for x in self.d]
        H=[[mul_hi(u,sum_hi(self.h[t][s],mul_hi(self.d[t],self.d[s])))
            for s in range(T)] for t in range(T)]
        return Jet(v,grads,H)

def scalar_link(p,g,z:IV):
    p=d(p);g=d(g)
    if p==ZERO:return ZEROI
    if p==ONE:return ONEI
    y=(z*g).exp()
    return ((y*p)/(ONEI-p+y*p)).intersect(0,1)

def tanh_iv(f:IV,scale:D):
    # monotone tanh on real line; evaluate each endpoint independently
    def at(v):
        u=(IV.exact(v)*(IV.exact(TWO)/IV.exact(scale))).exp()
        return (u-ONEI)/(u+ONEI)
    return IV(at(f.lo).lo,at(f.hi).hi).intersect(-1,1)

def factor_jet(v:IV,t,T,der1,der2):
    grads=[ZERO]*T;hess=[[ZERO]*T for _ in range(T)]
    grads[t]=der1;hess[t][t]=der2
    return Jet(v,grads,hess)

def rates(m,Fbox,T,global_z=False):
    nstate=len(m.survival);B=[];S=[];Dobs=[];M=[]
    clip=d(m.clip_scale)
    for t in range(T):
        z=IV(D(-1),D(1)) if global_z else tanh_iv(Fbox[t],clip)
        births=[];surs=[];dets=[];motion=[]
        for x in range(nstate):
            bg=d(m.gamma_birth[t][x]);b=d(m.births[t][x]);v=(z*bg).exp()*b
            dB=mul_hi(v.hi,abs(bg),ONE/clip)
            hB=mul_hi(v.hi,sum_hi(bg*bg,mul_hi(abs(bg),TANH_SECOND_CAP)),ONE/(clip*clip))
            births.append(factor_jet(v,t,T,dB,hB))
            for source,target in ((surs,(m.survival[x],m.gamma_survival[x])),
                                  (dets,(m.detection[x],m.gamma_detection[x]))):
                p,g=target;g=d(g)
                v=scalar_link(p,g,z)
                derivative=mul_hi(abs(g),QUARTER,ONE/clip)
                second=mul_hi(sum_hi(mul_hi(TENTH,g*g),mul_hi(QUARTER,TANH_SECOND_CAP,abs(g))),ONE/(clip*clip))
                source.append(factor_jet(v,t,T,derivative,second))
            row=[]
            for y in range(nstate):
                w=d(m.transition[x][y])
                if w==0:
                    row.append(Jet.const(ZEROI,T));continue
                denom=ONEI
                for j in range(nstate):
                    if j==y or m.transition[x][j]==0:continue
                    diff=d(m.gamma_motion[x][j])-d(m.gamma_motion[x][y])
                    denom=denom+((IV.exact(d(m.transition[x][j]))/IV.exact(w))*(z*diff).exp())
                v=ONEI/denom
                g=max(abs(d(m.gamma_motion[x][j])-d(m.gamma_motion[x][y])) for j in range(nstate) if j!=y) if nstate>1 else ZERO
                derivative=mul_hi(abs(g),QUARTER,ONE/clip)
                second=mul_hi(sum_hi(mul_hi(TENTH,g*g),mul_hi(QUARTER,TANH_SECOND_CAP,abs(g))),ONE/(clip*clip))
                row.append(factor_jet(v.intersect(0,1),t,T,derivative,second))
            motion.append(row)
        B.append(births);S.append(surs);Dobs.append(dets);M.append(motion)
    return B,S,M,Dobs

def emission(m,t,j,x,f:IV,T,global_pdf=False):
    v=d(m.observation_variance[x]);r=d(m.response[x]);offset=d(m.offset[x]);y=d(m.observations[t][j]);
    c=normal_const()/IV.exact(v).sqrt()
    if global_pdf:
        value=IV(ZERO,c.hi)
    else:
        delta=IV.exact(y-offset)-f*r
        dsq=delta.square()
        # Gaussian density decreasing in squared residual
        q=dsq/(2*v)
        value=(-q).exp()*c
    # Local elementary Gaussian derivative envelopes.  For a=|standardized
    # residual|: a*exp(-a²/2) peaks at a=1; |a²-1|exp(-a²/2)
    # peaks at a=0 or a=sqrt(3).  Evaluate endpoints and these critical
    # points by directed intervals, not unverified numerical differentiation.
    if global_pdf:
        f1=ONE;f2=ONE
    else:
        aa=delta/IV.exact(v).sqrt()
        if aa.lo<=ZERO<=aa.hi:amin=ZERO
        else:amin=min(abs(aa.lo),abs(aa.hi))
        amax=max(abs(aa.lo),abs(aa.hi))
        peak=max(amin,min(ONE,amax))
        f1=(IV.exact(peak)*(-IV.exact(peak).square()/TWO).exp()).hi
        def f2_at(point):
            x=IV.exact(point);z=x.square()-ONEI
            return mul_hi(absup(z),(-x.square()/TWO).exp().hi)
        f2=max(f2_at(amin),f2_at(amax))
        if amin==ZERO:f2=max(f2,ONE)
        with localcontext(RN):s3=D(3).sqrt()
        if amin<=s3<=amax:
            f2=max(f2,(IV.exact(TWO)*IV.exact(D('-1.5')).exp()).hi)
    with localcontext(HI):
        d1=c.hi*abs(r)/IV.exact(v).sqrt().lo*f1
        d2=c.hi*r*r/v*f2
    return factor_jet(value,t,T,d1,d2)

def conditional_jet(m,Fbox,global_z=False,global_pdf=False):
    T=len(m.observations);states=len(m.survival)
    N=sum(map(len,m.observations))
    if (T,states,N)!=(3,2,3) or any(len(row)!=1 for row in m.observations):
        raise ValueError('B16_FROZEN_T3_D2_N3_ONLY')
    B,S,M,Dobs=rates(m,Fbox,T,global_z=global_z)
    empty=lambda:Jet.const(ZEROI,T)
    one=lambda:Jet.const(ONEI,T)
    undetected=B[0];lam=empty()
    for t in range(T):
        if t:undetected=[B[t][j]+sum((undetected[x]*S[t][x]*M[t][x][j] for x in range(states)),empty()) for j in range(states)]
        lam=lam+sum((undetected[x]*Dobs[t][x] for x in range(states)),empty())
        undetected=[undetected[x]*Dobs[t][x].complement() for x in range(states)]
    obs=tuple((t,j) for t,row in enumerate(m.observations) for j in range(len(row)))
    em=[[emission(m,t,j,x,Fbox[t],T,global_pdf=global_pdf) for x in range(states)] for t,j in obs]
    options={mask:empty() for mask in range(1,1<<N) if len({obs[i][0] for i in range(N) if mask&(1<<i)})==mask.bit_count()}
    for b in range(T):
        for e in range(b,T):
            for path in product(range(states),repeat=e-b+1):
                weight=B[b][path[0]]
                for t in range(b+1,e+1):weight=weight*S[t][path[t-b-1]]*M[t][path[t-b-1]][path[t-b]]
                if e<T-1:weight=weight*S[e+1][path[-1]].complement()
                for mask in options:
                    chosen={obs[i][0]:i for i in range(N) if mask&(1<<i)}
                    if any(t<b or t>e for t in chosen):continue
                    w=weight
                    for t in range(b,e+1):
                        x=path[t-b]
                        if t in chosen:w=w*Dobs[t][x]*em[chosen[t]][x]
                        else:w=w*Dobs[t][x].complement()
                    options[mask]=options[mask]+w
    clutter=[]
    for t,j in obs:
        y=d(m.observations[t][j]);mu=d(m.clutter_mean);var=d(m.clutter_variance)
        exponent=(-(IV.exact(y-mu).square()/(2*var))).exp()
        density=(normal_const()/IV.exact(var).sqrt())*exponent*d(m.clutter_rate)
        clutter.append(density)
    P={0:one()}
    for mask in range(1,1<<N):
        bit=mask&(-mask);index=bit.bit_length()-1
        v=P[mask^bit]*Jet.const(clutter[index],T)
        for trackmask,w in options.items():
            if trackmask&bit and trackmask&mask==trackmask:
                v=v+w*P[mask^trackmask]
        P[mask]=v
    with localcontext(RN):kappa=d(m.clutter_rate)*T
    return P[(1<<N)-1].scale(ONE)*lam.expneg()*Jet.const(IV.exact(-kappa).exp(),T)

@dataclass(frozen=True)
class AxisCell:
    lo:D
    hi:D
    mass:IV
    mean:IV
    var_upper:D


def axis_cells(n=18,far=8.,sigma=3**.5):
    if n<4 or n>22:raise ValueError('AXIS_CELL_CAP')
    L=ndtr(-far/sigma);H=ndtr(far/sigma)
    edges=sigma*ndtri(np.linspace(L,H,n+1))
    edges[0]=-far;edges[-1]=far
    nodes=[d(float(v)) for v in edges]
    cells=[]
    for a,b in zip(nodes[:-1],nodes[1:]):
        mass,mean,var=truncated_conditional_bounds(Q(a),Q(b))
        cells.append(AxisCell(a,b,IV.rational(*mass),IV.rational(*mean),qd(var[1],True)))
    return cells


def prior_coeff(prior,T=3):
    a=d(prior.ar);q=IV.exact(d(prior.process_variance)).sqrt();r=IV.exact(d(prior.variance)).sqrt()
    coeff=[[ZEROI for _ in range(T)] for _ in range(T)]
    mean=[ZEROI for _ in range(T)]
    mean[0]=IV.exact(d(prior.mean));coeff[0][0]=r
    for t in range(1,T):
        mean[t]=mean[t-1]*a+d(prior.drift)
        for j in range(t):coeff[t][j]=coeff[t-1][j]*a
        coeff[t][t]=q
    return mean,coeff


def cell_gaussian_statistics(cells, prior):
    mean,coeff=prior_coeff(prior)
    mass=ONEI
    for s in cells:mass=mass*s.mass
    Fbox=[];central=[];sd=[];meanerr=[]
    for t in range(3):
        lohi=mean[t];center=mean[t];variance_upper=ZERO
        for j,cell in enumerate(cells):
            lohi=lohi+coeff[t][j]*IV(cell.lo,cell.hi)
            center=center+coeff[t][j]*cell.mean
            variance_upper=sum_hi(variance_upper,mul_hi(coeff[t][j].square().hi,cell.var_upper))
        point=center.midpoint()
        with localcontext(HI):error=max(point-center.lo,center.hi-point)
        second=sum_hi(variance_upper,mul_hi(error,error))
        Fbox.append(lohi)
        central.append(point)
        sd.append(IV.exact(second).sqrt().hi)
        meanerr.append(error)
    return mass,Fbox,central,sd,meanerr


def cell_contribution(m,prior,axes):
    mass,Fbox,c,sd,meanerr=cell_gaussian_statistics(axes,prior)
    jet=conditional_jet(m,Fbox)
    centerjet=conditional_jet(m,[IV.exact(v) for v in c])
    R=ZERO
    for r in range(3):
        R=sum_hi(R,mul_hi(jet.d[r],meanerr[r]))
        for s in range(3):R=sum_hi(R,mul_hi(D('.5'),jet.h[r][s],sd[r],sd[s]))
    ev=(centerjet.v+IV(-R,R)).intersect(0,None)
    mass_ev=mass*ev
    moments=[]
    for t in range(3):
        U=mul_hi(centerjet.v.hi,meanerr[t])
        for r in range(3):U=sum_hi(U,mul_hi(jet.d[r],sd[t],sd[r]))
        signed=IV(-U,U)
        moments.append(mass*(ev*c[t]+signed))
    return mass_ev,moments


def _compute_first_axis(job):
    a,m,prior,axes=job
    n=len(axes)
    evidence=ZEROI;moments=[ZEROI for _ in range(3)]
    for b,c in product(range(n),repeat=2):
        ev,mom=cell_contribution(m,prior,[axes[a],axes[b],axes[c]])
        evidence=evidence+ev
        moments=[x+y for x,y in zip(moments,mom)]
    return a,evidence,moments


def certificate(m,prior,n=18,max_leaves=12000,progress=None,workers=1):
    if m.digest()!='7004149342c03dd395ed30a35775576553ff9c091c1f0fedb90e610e9a9aa885':
        raise ValueError('B16_FROZEN_DIGEST_MISMATCH')
    axes=axis_cells(n)
    count=len(axes)**3
    if count>max_leaves:raise ValueError('B16_LEAF_BUDGET')
    t0=monotonic()
    evidence=ZEROI;moments=[ZEROI for _ in range(3)]
    jobs=((i,m,prior,axes) for i in range(n))
    if workers==1:
        partials=map(_compute_first_axis,jobs)
        for j,ev,mom in partials:
            evidence=evidence+ev
            moments=[a+b for a,b in zip(moments,mom)]
            if progress is not None:progress((j+1)*n*n,count,monotonic()-t0)
    else:
        if not 1<workers<=4:raise ValueError('WORKER_BUDGET_MAX_4')
        with get_context('fork').Pool(workers) as pool:
            # map (not unordered): reproducible summation order.
            for j,ev,mom in pool.imap(_compute_first_axis,jobs):
                evidence=evidence+ev
                moments=[a+b for a,b in zip(moments,mom)]
                if progress is not None:progress((j+1)*n*n,count,monotonic()-t0)
    # Gaussian innovation-coordinate tail: union bound P(any |Z_j|>8)
    tail=mul_hi(IV.rational(*normal_cdf_bounds(-8)).hi,d(2*len(m.observations)))
    globalbox=[IV.exact(0) for _ in range(3)]
    maxf=conditional_jet(m,globalbox,global_z=True,global_pdf=True).v.hi
    with localcontext(HI):tail_ev=tail*maxf
    evidence=evidence+IV(ZERO,tail_ev)
    mean,coeff=prior_coeff(prior)
    for t in range(3):
        varup=sum_hi(*[coeff[t][j].square().hi for j in range(3)])
        second=sum_hi(mean[t].square().hi,varup)
        tailM=mul_hi(IV.exact(mul_hi(tail,second)).sqrt().hi,maxf)
        moments[t]=moments[t]+IV(-tailM,tailM)
    if evidence.lo<=0:raise ValueError('CERTIFIED_POSITIVE_EVIDENCE_NOT_PROVED')
    post=[v/evidence for v in moments]
    result={
        'status':'B16_CONDITIONAL_DIRECTED_DECIMAL_ENCLOSURE',
        'model_digest':m.digest(),'leaves':count,'axis_cells':n,'runtime_seconds':monotonic()-t0,
        'evidence_interval':[str(evidence.lo),str(evidence.hi)],
        'evidence_width':str(evidence.width()),
        'field_mean_intervals':[[str(x.lo),str(x.hi)] for x in post],
        'field_mean_widths':[str(x.width()) for x in post],
        'tail_union_bound':str(tail),'global_likelihood_upper':str(maxf),
        'numeric_targets_within_enclosure':
            evidence.width()<=D('.001') and all(x.width()<=ONE for x in post),
        'proof_status':'CONDITIONAL; not Lean-mechanized; requires independent source-law, analytic-inequality and decimal-library review',
        'nonclaims':'no general PMBM/GLMB closure, no mechanized proof, no causal identification, no operational or privacy qualification',
    }
    return result
