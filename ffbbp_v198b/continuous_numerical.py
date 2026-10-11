"""Continuous Gaussian-field integration: **UNCERTIFIED QUADRATURE**.

Gaussian AR(1) innovation quadrature, whose exact conditional integrand is the
field-driven marked-PPP law. No a posteriori certified tail/remainder bound.
Zero-coupling may be independently compared to the frozen parent's analytic oracle.
"""
from dataclasses import dataclass
from collections import defaultdict
from itertools import product
import math
import numpy as np
from .model import Model, BudgetRefusal, ModelRefusal
from . import reference_a, reference_b

@dataclass(frozen=True)
class GaussianAR1:
    mean: float
    variance: float
    ar: float
    drift: float
    process_variance: float
    def __post_init__(self):
        if any(not math.isfinite(v) for v in (self.mean,self.variance,self.ar,self.drift,self.process_variance)):
            raise ModelRefusal('NONFINITE_GAUSSIAN_FIELD')
        if min(self.variance,self.process_variance)<=0:raise ModelRefusal('NONPOSITIVE_GAUSSIAN_VARIANCE')


def quadrature(m:Model,prior:GaussianAR1,n=12,reference='a',max_points=5000):
    if type(n) is not int or n<2 or n**len(m.observations)>max_points:raise BudgetRefusal('QUADRATURE_POINT_BUDGET')
    if reference not in ('a','b'):raise ModelRefusal('UNKNOWN_REFERENCE')
    zs, ws = np.polynomial.hermite.hermgauss(n)
    zs=zs*math.sqrt(2);ws=ws/math.sqrt(math.pi)
    compute=reference_a.conditional if reference=='a' else reference_b.conditional
    mass=0.;num=np.zeros(len(m.observations));num2=np.zeros((len(m.observations),len(m.observations)))
    origins=defaultdict(float);births=defaultdict(float);card=defaultdict(float);undetected=defaultdict(float)
    for idx in product(range(n),repeat=len(m.observations)):
        f=[];w=1.
        for t,k in enumerate(idx):
            val=(prior.mean+math.sqrt(prior.variance)*zs[k] if t==0 else
                 prior.ar*f[-1]+prior.drift+math.sqrt(prior.process_variance)*zs[k])
            f.append(val);w*=float(ws[k])
        if w==0:continue
        conditional=compute(m,tuple(f))
        density=conditional['full'];a=w*density;mass+=a
        weight_factor=w*math.exp(-conditional['exponent'])
        for (tracks,origin_hist),branch_weight in conditional['weights'].items():
            prob=weight_factor*branch_weight
            card[sum(tr[2][-1]>=0 for tr in tracks)]+=prob
            for track in tracks:births[(track[0],track[1])]+=prob
            for tt,scan in enumerate(origin_hist):
                for j,origin in enumerate(scan):origins[(tt,j,origin)]+=prob
        for bx,v in conditional['undetected'].items():undetected[bx]+=a*v
        arr=np.array(f);num+=a*arr;num2+=a*np.outer(arr,arr)
    if mass<=0 or not math.isfinite(mass):raise ModelRefusal('ZERO_NUMERIC_QUADRATURE_MASS')
    mean=num/mass;cov=num2/mass-np.outer(mean,mean)
    return {'full_evidence':mass,'posterior_field_mean':tuple(float(x) for x in mean),
            'posterior_field_covariance':tuple(tuple(float(x) for x in r) for r in cov),
            'status':'UNCERTIFIED_NUMERIC_CONTINUOUS_APPROX',
            'quadrature_points':n**len(m.observations),'nodes_per_dimension':n,'method':reference,
            'origin_probabilities':dict((k,v/mass) for k,v in origins.items()),
            'birth_probabilities':dict((k,v/mass) for k,v in births.items()),
            'detected_alive_cardinality':dict((k,v/mass) for k,v in card.items()),
            'undetected_final':dict((k,v/mass) for k,v in undetected.items())}
