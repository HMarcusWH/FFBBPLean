"""Reference B: complete marked-trajectory + observation partition enumeration.

No imports from Reference A. Computes the no-detection PPP normalizer by
complete physical history enumeration, not first-detection thinning recursion.
"""
from itertools import product,combinations
from collections import defaultdict
import math
from .model import Model, BudgetRefusal, NumericalRefusal


def density(y,mu,var):
    return math.exp(-((y-mu)**2)/(2*var)) / math.sqrt(2*math.pi*var)


def _rates(m,f):
    T=len(m.observations);D=len(m.survival)
    bs=[];sur=[];mot=[];det=[]
    for t in range(T):
        z=math.tanh(f[t]/m.clip_scale)
        bs.append(tuple(m.births[t][j]*math.exp(z*m.gamma_birth[t][j]) for j in range(D)))
        sur.append(tuple(p if p in (0.,1.) else (p*math.exp(z*m.gamma_survival[j]))/(1-p+p*math.exp(z*m.gamma_survival[j])) for j,p in enumerate(m.survival)))
        det.append(tuple(p if p in (0.,1.) else (p*math.exp(z*m.gamma_detection[j]))/(1-p+p*math.exp(z*m.gamma_detection[j])) for j,p in enumerate(m.detection)))
        rows=[]
        for j in range(D):
            raw=[m.transition[j][k]*math.exp(z*m.gamma_motion[j][k]) for k in range(D)]
            den=math.fsum(raw)
            rows.append(tuple(v/den for v in raw))
        mot.append(tuple(rows))
    return bs,sur,mot,det


def _physical_paths(m,f):
    """All finite life-history intensities prior to observation marks."""
    B,S,M,P=_rates(m,f);T=len(m.observations);D=len(m.survival)
    for b in range(T):
        for e in range(b,T):
            for states in product(range(D),repeat=e-b+1):
                w=B[b][states[0]]
                for t in range(b+1,e+1):
                    w*=S[t][states[t-b-1]]*M[t][states[t-b-1]][states[t-b]]
                if e<T-1:w*=1-S[e+1][states[-1]]
                if w:yield b,e,states,w,P


def conditional(m,f,*,max_histories=300000):
    if not isinstance(m,Model):raise TypeError('MODEL')
    if len(f)!=len(m.observations) or not all(math.isfinite(v) for v in f):raise ValueError('FIELD_PATH')
    T=len(m.observations)
    paths=list(_physical_paths(m,tuple(f)))
    b_tot=math.fsum(m.births[t][x]*math.exp(m.gamma_birth[t][x]*math.tanh(f[t]/m.clip_scale)) for t in range(T) for x in range(len(m.survival)))
    lambda0=0.;undetected=defaultdict(float)
    for b,e,states,physical,P in paths:
        w=physical
        for t in range(b,e+1):w*=1-P[t][states[t-b]]
        lambda0+=w
        if e==T-1:undetected[(b,states[-1])]+=w
    lp=b_tot-lambda0
    if lp < -1e-11:raise NumericalRefusal('NEGATIVE_FIRST_HIT_MASS')
    exponent=T*m.clutter_rate+max(0,lp)
    ids=tuple((t,j) for t,row in enumerate(m.observations) for j in range(len(row)))
    weights=defaultdict(float)
    options={}
    def tracks_for(group):
        group=tuple(sorted(group))
        if group in options:return options[group]
        chosen=dict(group)
        variants=[]
        for b,e,states,physical,P in paths:
            if not group or b>group[0][0] or e<group[-1][0]:continue
            w=physical
            for t in range(b,e+1):
                x=states[t-b]
                if t in chosen:
                    y=m.observations[t][chosen[t]]
                    w*=P[t][x]*density(y,m.offset[x]+m.response[x]*f[t],m.observation_variance[x])
                else:w*=1-P[t][x]
            if w:
                label=group[0]
                tr=(label,b,states+(-1,)*(T-1-e),group)
                variants.append((tr,w))
        options[group]=variants
        return variants

    leaves=0
    def walk(remaining,tracks,clutter,weight):
        nonlocal leaves
        if not remaining:
            leaves+=1
            if leaves>max_histories:raise BudgetRefusal('COMPLETE_TRAJECTORY_HISTORY_CAP')
            origins={s:('clutter',) for s in clutter}
            for label,b,path,dets in tracks:
                for k,(t,j) in enumerate(dets):
                    origins[(t,j)]=('new',b,path[t-b]) if k==0 else ('old',label,path[t-b])
            sig=(tuple(sorted(tracks)),tuple(tuple(origins[(t,j)] for j in range(len(row))) for t,row in enumerate(m.observations)))
            weights[sig]+=weight
            return
        first=remaining[0]
        y=m.observations[first[0]][first[1]]
        cl=m.clutter_rate*density(y,m.clutter_mean,m.clutter_variance)
        if cl:walk(remaining[1:],tracks,clutter+(first,),weight*cl)
        later=remaining[1:]
        for n in range(len(later)+1):
            for extras in combinations(later,n):
                group=(first,)+extras
                if len(set(t for t,j in group))!=len(group):continue
                remain=tuple(s for s in later if s not in group)
                for tr,w in tracks_for(group):walk(remain,tracks+(tr,),clutter,weight*w)
    walk(ids,(),(),1.)
    relative=math.fsum(weights.values())
    return {'weights':dict(weights),'relative':relative,'full':relative*math.exp(-exponent),
            'exponent':exponent,'birth_mass':b_tot,'never_detected_mass':lambda0,
            'undetected':dict(undetected),'histories':leaves}


def _field_law(m):
    """Independently enumerate finite Markov field laws (not imported from A)."""
    T=len(m.observations);K=len(m.field_values)
    for idx in product(range(K),repeat=T):
        w=m.field_initial[idx[0]]
        for t in range(1,T):w*=m.field_transition[idx[t-1]][idx[t]]
        if w>0:yield idx,tuple(m.field_values[j] for j in idx),w


def solve(m,*,max_histories=300000):
    mass={};pfield={};expo={};undetected={};cards=defaultdict(float)
    conditional_support=False
    for idx,f,fw in _field_law(m):
        res=conditional(m,f,max_histories=max_histories)
        conditional_support=conditional_support or (res['relative']>0)
        factor=fw*math.exp(-res['exponent'])
        pfield[idx]=factor*res['relative'];expo[idx]=res['exponent'];undetected[idx]=res['undetected']
        for sig,w in res['weights'].items():mass[(idx,sig)]=factor*w
    evidence=math.fsum(mass.values())
    if not conditional_support:raise NumericalRefusal('IMPOSSIBLE_EVIDENCE')
    if evidence<=0 or not math.isfinite(evidence):raise NumericalRefusal('NUMERICAL_EVIDENCE_UNDERFLOW')
    posterior={k:v/evidence for k,v in mass.items()}
    for (ids,(tracks,origins)),p in posterior.items():cards[sum(tr[2][-1]>=0 for tr in tracks)]+=p
    return {'full_evidence':evidence,'posterior':posterior,'field_posterior':{k:v/evidence for k,v in pfield.items()},
            'exponents':expo,'undetected_by_field':undetected,'detected_count_posterior':dict(cards),
            'model_sha256':m.digest(),'claim_cap':'FINITE_FIELD_CONDITIONAL_RFS_REFERENCE_ONLY'}
