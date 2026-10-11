"""Frozen finite-field input schema, canonical finite-set Gaussian measurement density.

This is a *discretized field* specialization of the v1.9.8-A conditional law,
not a Gaussian-field full RFS solution or a field-conditioned physical validation.
"""
from dataclasses import dataclass
import math
import hashlib
import json
from dataclasses import asdict

class ModelRefusal(ValueError): pass
class BudgetRefusal(RuntimeError): pass
class NumericalRefusal(RuntimeError): pass


def gaussian(y, mean, variance):
    return math.exp(-0.5*(y-mean)**2/variance) / math.sqrt(2*math.pi*variance)


def sigmoid(v):
    if v >= 0: return 1.0/(1.0+math.exp(-v))
    e=math.exp(v); return e/(1.0+e)


@dataclass(frozen=True)
class Model:
    observations: tuple[tuple[float,...], ...]
    births: tuple[tuple[float,...], ...]
    survival: tuple[float,...]
    detection: tuple[float,...]
    transition: tuple[tuple[float,...], ...]
    response: tuple[float,...]
    offset: tuple[float,...]
    observation_variance: tuple[float,...]
    clutter_rate: float
    clutter_mean: float
    clutter_variance: float
    field_values: tuple[float,...]
    field_initial: tuple[float,...]
    field_transition: tuple[tuple[float,...], ...]
    gamma_birth: tuple[tuple[float,...], ...]
    gamma_survival: tuple[float,...]
    gamma_motion: tuple[tuple[float,...], ...]
    gamma_detection: tuple[float,...]
    clip_scale: float = 2.0
    source_digest: str = 'e'*64
    partition: str = 'EXPERIMENT'
    observation_measure: str = 'CANONICAL_SORTED_SIMPLEX_LEBESGUE'
    version: str = 'field_driven_finite_marked_ppp_v1_9_8_b'

    def __post_init__(self):
        T=len(self.observations); D=len(self.survival); K=len(self.field_values)
        if not (1<=T<=4 and 1<=D<=3 and 1<=K<=4): raise BudgetRefusal('FIELD_HORIZON_STATE_CAP')
        if sum(map(len,self.observations))>5 or any(len(r)>2 for r in self.observations): raise BudgetRefusal('OBSERVATION_CAP')
        if not isinstance(self.source_digest,str) or len(self.source_digest)!=64 or any(c not in '0123456789abcdef' for c in self.source_digest): raise ModelRefusal('SOURCE_DIGEST')
        if self.partition not in ('TRAIN','EXPERIMENT'):raise ModelRefusal('PARTITION')
        if self.observation_measure != 'CANONICAL_SORTED_SIMPLEX_LEBESGUE':raise ModelRefusal('OBSERVATION_MEASURE')
        if self.version != 'field_driven_finite_marked_ppp_v1_9_8_b':raise ModelRefusal('VERSION')
        if not(all(type(row) is tuple for row in self.observations+self.births+self.transition+self.field_transition+self.gamma_birth+self.gamma_motion)
               and all(type(v) is tuple for v in (self.observations,self.births,self.survival,self.detection,self.transition,self.response,self.offset,self.observation_variance,self.field_values,self.field_initial,self.field_transition,self.gamma_birth,self.gamma_survival,self.gamma_motion,self.gamma_detection))):raise ModelRefusal('IMMUTABLE_SCHEMA')
        if not(len(self.births)==T and len(self.gamma_birth)==T and all(len(row)==D for row in self.births+self.gamma_birth)
                and all(len(row)==D for row in self.transition+self.gamma_motion)
                and len(self.transition)==len(self.gamma_motion)==D
                and all(len(x)==D for x in (self.detection,self.response,self.offset,self.observation_variance,self.gamma_survival,self.gamma_detection))
                and len(self.field_initial)==K and len(self.field_transition)==K
                and all(len(row)==K for row in self.field_transition)):
            raise ModelRefusal('SHAPE')
        numbers=[self.clutter_rate,self.clutter_mean,self.clutter_variance,self.clip_scale,*self.survival,*self.detection,*self.response,*self.offset,*self.observation_variance,*self.field_values,*self.field_initial,*self.gamma_survival,*self.gamma_detection]
        for matrix in (self.observations,self.births,self.transition,self.field_transition,self.gamma_birth,self.gamma_motion): numbers.extend(v for row in matrix for v in row)
        if any(type(x) not in (float,int) or not math.isfinite(x) for x in numbers):raise ModelRefusal('NONFINITE')
        if any(abs(x)>1e6 for x in numbers):raise ModelRefusal('NUMERIC_PARAMETER_MAGNITUDE_CAP')
        if any(abs(x)>40 for x in (*self.gamma_survival,*self.gamma_detection,*(v for row in self.gamma_motion for v in row))):raise ModelRefusal('NONBIRTH_LINK_MAGNITUDE_CAP')
        if self.clutter_rate<0 or self.clutter_variance<=0 or self.clip_scale<=0 or min(self.observation_variance)<=0:raise ModelRefusal('RANGE')
        if any(v<0 for row in self.births for v in row):raise ModelRefusal('NEGATIVE_BIRTH')
        if any(not 0<=v<=1 for v in self.survival+self.detection):raise ModelRefusal('BAD_PROBABILITY')
        for row in self.transition+self.field_transition+(self.field_initial,):
            if any(v<0 for v in row) or abs(math.fsum(row)-1)>1e-12:raise ModelRefusal('NONSTOCHASTIC')
        if any(tuple(sorted(row))!=row or len(set(row))!=len(row) for row in self.observations):raise ModelRefusal('NONCANONICAL_OBSERVATIONS')
        if any(abs(v)>20 for row in self.observations for v in row):raise ModelRefusal('OBSERVATION_BUDGET')
        if any(abs(v)>60 for row in self.gamma_birth for v in row):raise ModelRefusal('BIRTH_EXP_OVERFLOW_CAP')

    def digest(self):
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def example(*,T=2,observations=((-0.2,),(.4,)),gammas=True,source_digest='e'*64):
    observations=tuple(tuple(sorted(row)) for row in observations)
    assert len(observations)==T
    return Model(
        observations=observations,
        births=tuple((.48,.22) if t==0 else (.18,.12) for t in range(T)),
        survival=(.75,.62), detection=(.65,.48),transition=((.82,.18),(.32,.68)),
        response=(.8,-.5),offset=(.1,.85),observation_variance=(.36,.5),
        clutter_rate=.22,clutter_mean=.0,clutter_variance=2.5,
        field_values=(-1.,1.),field_initial=(.55,.45),field_transition=((.8,.2),(.3,.7)),
        gamma_birth=tuple((.6 if gammas else 0.,-.3 if gammas else 0.) for _ in range(T)),
        gamma_survival=(.7 if gammas else 0.,-.8 if gammas else 0.),
        gamma_motion=((.55 if gammas else 0.,-.25 if gammas else 0.),(-.35 if gammas else 0.,.2 if gammas else 0.)),
        gamma_detection=(.65 if gammas else 0.,-.5 if gammas else 0.),source_digest=source_digest)
