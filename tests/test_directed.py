import math
import mpmath as mp
import numpy as np
from ffbbp_v198b.model import example
from ffbbp_v198b.reference_a import conditional as A
from ffbbp_v198b.reference_b import conditional as B
from ffbbp_v198b.continuous_numerical import GaussianAR1
from b16.directed_certificate import d,IV,ONEI,ZEROI,tanh_iv,conditional_jet,axis_cells,cell_gaussian_statistics,cell_contribution


def get_model():
 return example(T=3,observations=((-0.2,),(.4,),(-.1,)))

def test_directed_basic_arithmetic():
 a=IV(d('0.03'),d('.07'));b=IV(d('0.15'),d('0.21'))
 c=(a+b)*b
 assert c.lo<=d('.18')*d('.15')
 assert c.hi>=d('.28')*d('.21')
 q=IV.exact(3)/IV.exact(7)
 assert mp.mpf(str(q.lo)) <=mp.mpf(3)/7 <=mp.mpf(str(q.hi))


def test_tanh_interval_vs_independent_mp():
 mp.mp.dps=100
 for lo,hi in [(-8,-4),(-1.2,0.4),(0.,.3),(2.,7.)]:
  a=tanh_iv(IV(d(float(lo)),d(float(hi))),d(2.))
  assert mp.mpf(str(a.lo)) <=mp.tanh(mp.mpf(lo)/2)
  assert mp.mpf(str(a.hi)) >=mp.tanh(mp.mpf(hi)/2)


def test_frozen_pointwise_history_likelihood_directed_encloses_independent_references():
 m=get_model();rng=np.random.default_rng(7)
 assert m.digest()=='7004149342c03dd395ed30a35775576553ff9c091c1f0fedb90e610e9a9aa885'
 for row in rng.normal(size=(12,3)):
  r=conditional_jet(m,[IV.exact(float(v)) for v in row])
  a=A(m,tuple(row))['full'];b=B(m,tuple(row))['full']
  assert math.isclose(a,b,rel_tol=2e-12)
  assert float(r.v.lo)<=a+1e-14
  assert float(r.v.hi)>=a-1e-14
  assert math.isclose(float(r.v.lo),a,rel_tol=2e-12)


def test_decimal_cell_taylor_contains_numeric_point_probe():
 m=get_model();prior=GaussianAR1(.1,.8,.35,.05,.7)
 cells=axis_cells(12)
 i=(5,6,5)
 mass,boxes,cent,sd,err=cell_gaussian_statistics([cells[j] for j in i],prior)
 ev,mom=cell_contribution(m,prior,[cells[j] for j in i])
 v=conditional_jet(m,[IV.exact(t) for t in cent]).v
 assert mass.lo>0
 assert ev.lo>=0
 assert ev.hi>0
 assert all(x.hi>x.lo for x in mom)
 assert v.lo>0


def test_global_enclosure_rejects_no_false_positives():
 m=get_model()
 r=conditional_jet(m,[IV.exact(0) for _ in range(3)],global_z=True,global_pdf=True)
 for s in [(0.,0.,0.),(-2.,1.,.4),(7.,-5.,8.)]:
  assert r.v.hi>=d(B(m,s)['full'])
