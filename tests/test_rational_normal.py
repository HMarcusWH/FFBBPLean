from fractions import Fraction as Q
import math
import mpmath as mp
import pytest
from b16.rational_normal import (pi_bounds,exp_bounds,sqrt_bounds,normal_cdf_bounds,
                                 normal_pdf_bounds,truncated_standard_normal_bounds,truncated_conditional_bounds)
mp.mp.dps=100

@pytest.mark.parametrize('x',[-8.,-7.,-5.,-2.,-.4,0.,.4,1.4,3.,5.,8.])
def test_normal_cdf_rational_contains_independent_mpmath_value(x):
    lo,hi=normal_cdf_bounds(x)
    actual=mp.erfc(-mp.mpf(x)/mp.sqrt(2))/2
    assert mp.mpf(lo.numerator)/lo.denominator <= actual
    assert actual <= mp.mpf(hi.numerator)/hi.denominator
    assert mp.mpf(hi.numerator)/hi.denominator-mp.mpf(lo.numerator)/lo.denominator < mp.mpf('1e-32')

@pytest.mark.parametrize('x',[-8.,-3.,0.,2.4,8.])
def test_normal_pdf_rational_contains_independent_mpmath_value(x):
    lo,hi=normal_pdf_bounds(x)
    actual=mp.exp(-mp.mpf(x)**2/2)/mp.sqrt(2*mp.pi)
    assert mp.mpf(lo.numerator)/lo.denominator <= actual
    assert actual<=mp.mpf(hi.numerator)/hi.denominator

@pytest.mark.parametrize('a,b',[(-8.,-4.),(-2.,-.4),(-.3,.9),(2.,5.),(5.,8.)])
def test_truncated_moments_against_integrals(a,b):
    mass,m1,m2=truncated_standard_normal_bounds(a,b)
    phi=lambda x:mp.exp(-mp.mpf(x)**2/2)/mp.sqrt(2*mp.pi)
    p=mp.erfc(-mp.mpf(b)/mp.sqrt(2))/2-mp.erfc(-mp.mpf(a)/mp.sqrt(2))/2
    first=phi(a)-phi(b)
    sec=p+a*phi(a)-b*phi(b)
    for v,vref in zip((mass,m1,m2),(p,first,sec)):
        assert mp.mpf(v[0].numerator)/v[0].denominator <=vref
        assert vref <=mp.mpf(v[1].numerator)/v[1].denominator
    mass,mean,var=truncated_conditional_bounds(a,b)
    mu=first/p;variance=sec/p-mu**2
    assert mp.mpf(mean[0].numerator)/mean[0].denominator <=mu<=mp.mpf(mean[1].numerator)/mean[1].denominator
    assert mp.mpf(var[0].numerator)/var[0].denominator <=variance<=mp.mpf(var[1].numerator)/var[1].denominator
