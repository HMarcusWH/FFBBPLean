import math
import numpy as np
import pytest
from ffbbp_v198b.model import example
from ffbbp_v198b.continuous_numerical import GaussianAR1
from ffbbp_v198b.reference_a import conditional as reference_a
from ffbbp_v198b.reference_b import conditional as reference_b
from b16.gaussian_cells import (ScopeError, gaussian_box_statistics,conditional_likelihood_intervals,study,prior_affine, z_grid)
from b16.hessian_cells import second_order_likelihood,feasible_second_order

@pytest.fixture(scope='module')
def inputs():
 return (example(T=3,observations=((-0.2,),(.4,),(-.1,))),GaussianAR1(.1,.8,.35,.05,.7))

def test_frozen_digest(inputs):
 m,p=inputs
 assert m.digest()=='7004149342c03dd395ed30a35775576553ff9c091c1f0fedb90e610e9a9aa885'

def test_pointwise_exact_independent_A_B(inputs):
 m,_=inputs
 rng=np.random.default_rng(42)
 F=rng.normal(0,.8,(48,3))
 ours=conditional_likelihood_intervals(m,F,F)
 for i,f in enumerate(F):
  a=reference_a(m,tuple(f))['full']
  b=reference_b(m,tuple(f))['full']
  assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-15),('A_B',i,a,b)
  assert math.isclose(a,ours.lo[i],rel_tol=1e-11,abs_tol=1e-15),('ours',i,a,ours.lo[i])
  assert ours.lo[i]<=ours.hi[i]+1e-15

def test_box_bounds_enclose_interior_samples(inputs):
 m,_=inputs
 rng=np.random.default_rng(123)
 lo=rng.uniform(-1.8,.6,(64,3));hi=lo+rng.uniform(.05,1.2,(64,3))
 iv=conditional_likelihood_intervals(m,lo,hi)
 for frac in [0.,.23,.57,1.]:
  test=lo+frac*(hi-lo)
  v=conditional_likelihood_intervals(m,test,test).lo
  assert np.all(v>=iv.lo-1e-14),np.min(v-iv.lo)
  assert np.all(v<=iv.hi+1e-14),np.max(v-iv.hi)

def test_hessian_uniform_bounded_by_point_finite_difference(inputs):
 m,_=inputs
 rng=np.random.default_rng(22)
 lo=rng.uniform(-1.2,.2,(64,3));hi=lo+rng.uniform(.15,.9,(64,3))
 bound=second_order_likelihood(m,lo,hi)
 assert np.all(bound.lo <= bound.hi)
 probe=lo+rng.uniform(.15,.85,(64,3))*(hi-lo)
 v=conditional_likelihood_intervals(m,probe,probe).lo
 e=1e-3
 for a in range(3):
  for b in range(3):
   dx=np.zeros(3);dy=np.zeros(3);dx[a]=e;dy[b]=e
   if a==b:
    vplus=conditional_likelihood_intervals(m,probe+dx,probe+dx).lo
    vminus=conditional_likelihood_intervals(m,probe-dx,probe-dx).lo
    h=(vplus-2*v+vminus)/(e*e)
   else:
    p1=conditional_likelihood_intervals(m,probe+dx+dy,probe+dx+dy).lo
    p2=conditional_likelihood_intervals(m,probe+dx-dy,probe+dx-dy).lo
    p3=conditional_likelihood_intervals(m,probe-dx+dy,probe-dx+dy).lo
    p4=conditional_likelihood_intervals(m,probe-dx-dy,probe-dx-dy).lo
    h=(p1-p2-p3+p4)/(4*e*e)
   assert np.all(np.abs(h)<=bound.h[:,a,b]*(1+2e-3)+1e-7), (a,b, np.max(np.abs(h)-bound.h[:,a,b]))

def test_gaussian_affine_box_moments(inputs):
 m,p=inputs
 lo=np.array([[-.5,-1.,-.25],[.2,.1,.4]]);hi=np.array([[.5,1.,.25],[1.2,.5,1.4]])
 mass,means,std,fl,fh=gaussian_box_statistics(lo,hi,p)
 mu,L=prior_affine(p,3)
 assert np.all(mass>0)&np.all(mass<1)
 assert np.all(means>=fl)&np.all(means<=fh)
 assert np.all(std>0)
 # Z bounded box ensures F expected inside convex transformed bounds.

@pytest.mark.parametrize('bins',[4,10,16,18,20])
def test_taylor_heuristic_vs_independent_uniform_midpoint_mc(inputs,bins):
 m,p=inputs
 r=feasible_second_order(m,p,interior=bins)
 assert r['frozen_model_digest']==m.digest()
 assert r['leaves'] == (bins+2)**3
 assert r['evidence_interval'][0] > 0
 # Independent quadrature estimates from the frozen B15 history; not certificates.
 assert r['evidence_interval'][0] <= .0020898073357226714 <= r['evidence_interval'][1]
 for iv,probe in zip(r['field_mean_intervals'],[-.019829570311999892,.2443228949305262,.02818659795446629]):
  assert iv[0]<=probe<=iv[1]

def test_frozen_b16_numerical_feasibility_is_not_certification(inputs):
 m,p=inputs
 r=feasible_second_order(m,p,interior=20)
 assert r['leaves'] <= 12000
 assert r['numeric_gates_pass_heuristically']
 assert r['evidence_width']<=.001
 assert all(x<=1. for x in r['field_mean_widths'])
 assert r['status'].endswith('NOT_CERTIFIED')
 assert 'PENDING' in r['proof_status']

def test_refuse_out_of_scope_four_scans():
 m=example(T=4,observations=((-0.2,),(.4,),(-.1,),(.3,)))
 with pytest.raises(ScopeError):
  conditional_likelihood_intervals(m,np.zeros((1,4)),np.ones((1,4)))
