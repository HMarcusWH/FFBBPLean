"""B16 exact-rational intervals for standard-normal CDF, pdf and cell moments.

The inequalities use (a) the alternating arctangent series for Machin's π,
(b) directed integer sqrt enclosures, (c) Taylor series with bounded geometric
exp tail, and (d) the alternating integrated normal-density series whose
remaining terms decrease once n >= x²/2.

These are exact Python Fraction computations *under those stated elementary
analytic inequalities*. They have not been independently formalized in Lean.
This file alone is not a certification of the FFBBP full solver.
"""
from __future__ import annotations
from fractions import Fraction as Q
from functools import lru_cache
from math import isqrt,ceil
from typing import Tuple

Interval=tuple[Q,Q]

def fraction(x):
    if isinstance(x,Q):return x
    if isinstance(x,float):return Q(*x.as_integer_ratio())
    return Q(x)

def plus(a:Interval,b:Interval)->Interval:return a[0]+b[0],a[1]+b[1]
def neg(a:Interval)->Interval:return -a[1],-a[0]
def times(a:Interval,b:Interval)->Interval:
    p=tuple(x*y for x in a for y in b)
    return min(p),max(p)
def div_positive(a:Interval,b:Interval)->Interval:
    if b[0]<=0:raise ValueError('POSITIVE_DIVISOR_REQUIRED')
    return times(a,(1/b[1],1/b[0]))

def arctan_unit_bounds(x:Q,n=100)->Interval:
    if x<0 or x>1:raise ValueError('ATAN_UNIT_DOMAIN')
    # S_n=sum_{j=0}^{n-1}(-1)^j x^(2j+1)/(2j+1)
    term=x;value=Q(0)
    for j in range(n):
        if j:term=term*(-x*x)*Q(2*j-1,2*j+1)
        value+=term
    tail=x**(2*n+1)/(2*n+1)
    return (value,value+tail) if n%2==0 else (value-tail,value)

@lru_cache(maxsize=5)
def pi_bounds(n=100)->Interval:
    a=arctan_unit_bounds(Q(1,5),n)
    b=arctan_unit_bounds(Q(1,239),n)
    return 16*a[0]-4*b[1],16*a[1]-4*b[0]

def sqrt_bounds(x:Q,decimal_places=75)->Interval:
    if x<0:raise ValueError('SQRT_NEGATIVE')
    power=10**decimal_places
    v=(x.numerator*power*power)//x.denominator
    floor=isqrt(v)
    lo=Q(floor,power)
    return (lo,lo) if lo*lo==x else (lo,Q(floor+1,power))

def _exp_positive(x:Q,n=50)->Interval:
    if x<0:raise ValueError('NEGATIVE_INPUT')
    k=0
    while x/Q(2**k)>Q(1,4):k+=1
    y=x/Q(2**k)
    s=Q(1);term=Q(1)
    for j in range(1,n+1):
        term*=y/j;s+=term
    first=term*y/(n+1)
    ratio=y/(n+2)
    lower=s;upper=s+first/(1-ratio)
    for _ in range(k):lower*=lower;upper*=upper
    return lower,upper

def exp_bounds(x,n=50)->Interval:
    x=fraction(x)
    if x>=0:return _exp_positive(x,n)
    l,u=_exp_positive(-x,n)
    return 1/u,1/l

@lru_cache(maxsize=1)
def normal_constant_bounds()->Interval:
    plo,phi=pi_bounds()
    return div_positive((Q(1),Q(1)),(sqrt_bounds(2*plo)[0],sqrt_bounds(2*phi)[1]))

@lru_cache(maxsize=300)
def normal_pdf_bounds(x)->Interval:
    x=fraction(x)
    return times(normal_constant_bounds(),exp_bounds(-x*x/2))

@lru_cache(maxsize=300)
def normal_cdf_bounds(x,order=180)->Interval:
    """Exact rational enclosure of Phi(x) for |x|<=8.5.

    If n+1 > x²/2, the integrated Taylor terms are monotonically
    decreasing from the first omitted term; alternating-tail enclosure.
    """
    x=fraction(x)
    if abs(x)>Q(17,2):raise ValueError('CDF_TAYLOR_DOMAIN_8_5')
    if order+1 <= x*x/2:raise ValueError('CDF_TAIL_NOT_DECREASING')
    if x<0:
        l,u=normal_cdf_bounds(-x,order)
        return 1-u,1-l
    s=Q(0);term=x
    for j in range(order+1):
        if j:term*=(-x*x)*Q(2*j-1,2*j*(2*j+1))
        s+=term
    next_term=term*(-x*x)*Q(2*order+1,2*(order+1)*(2*order+3))
    integ=(min(s,s+next_term),max(s,s+next_term))
    l,u=times(integ,normal_constant_bounds())
    return Q(1,2)+l,Q(1,2)+u

@lru_cache(maxsize=300)
def truncated_standard_normal_bounds(a,b):
    """Mass, first unnormalized moment and second unnormalized moment."""
    a=fraction(a);b=fraction(b)
    if not a<b:raise ValueError('EMPTY_NORMAL_INTERVAL')
    Ca=normal_cdf_bounds(a);Cb=normal_cdf_bounds(b)
    mass=plus(Cb,neg(Ca))
    pa=normal_pdf_bounds(a);pb=normal_pdf_bounds(b)
    first=plus(pa,neg(pb))
    second=plus(mass,plus(times((a,a),pa),neg(times((b,b),pb))))
    if mass[0]<=0:raise ValueError('CDF_MASS_INSUFFICIENT_PRECISION')
    return mass,first,second

@lru_cache(maxsize=300)
def truncated_conditional_bounds(a,b):
    mass,m1,m2=truncated_standard_normal_bounds(a,b)
    cm=div_positive(m1,mass)
    sec=div_positive(m2,mass)
    mean_sq=times(cm,cm)
    v=plus(sec,neg(mean_sq))
    return mass,cm,(max(Q(0),v[0]),max(Q(0),v[1]))
