"""Reference A: chronological RFS-first-detection recursion.

No B-oracle import. All weights are event densities under sorted-simplex measure.
The undetected process is PPP conditioned on *the field path*, not after marginalization.
"""
from itertools import product
from collections import defaultdict
import math
from .model import Model, BudgetRefusal, NumericalRefusal, gaussian, sigmoid


def rates(m, f):
    """A's field-conditioned rates: independent link-function implementation."""
    T=len(f); D=len(m.survival)
    births=[]; surv=[]; motion=[]; detect=[]
    for t in range(T):
        z=math.tanh(f[t]/m.clip_scale)
        births.append(tuple(m.births[t][x]*math.exp(m.gamma_birth[t][x]*z) for x in range(D)))
        surv.append(tuple(sigmoid(math.log(p/(1-p))+m.gamma_survival[x]*z) if 0<p<1 else p for x,p in enumerate(m.survival)))
        detect.append(tuple(sigmoid(math.log(p/(1-p))+m.gamma_detection[x]*z) if 0<p<1 else p for x,p in enumerate(m.detection)))
        rows=[]
        for x in range(D):
            scores=[math.log(p)+m.gamma_motion[x][j]*z if p>0 else -math.inf for j,p in enumerate(m.transition[x])]
            mx=max(scores);es=[math.exp(a-mx) if a!=-math.inf else 0. for a in scores];s=math.fsum(es)
            rows.append(tuple(v/s for v in es))
        motion.append(tuple(rows))
    return births,surv,motion,detect


def field_paths(m):
    K=len(m.field_values);T=len(m.observations)
    for ids in product(range(K),repeat=T):
        p=m.field_initial[ids[0]]
        for t in range(1,T):p*=m.field_transition[ids[t-1]][ids[t]]
        if p>0:yield ids,tuple(m.field_values[i] for i in ids),p


def _one_conditional(m,f,max_branches):
    T=len(m.observations); D=len(m.survival)
    B,S,M,P=rates(m,f)
    branches={((),()):1.0}  # key=(detected tracks, measurement origin history)
    post=[]; first=[];final_undetected={}
    for t,obs in enumerate(m.observations):
        pre=[]
        for birth,path,w in post:
            x=path[-1]
            for j in range(D):
                v=w*S[t][x]*M[t][x][j]
                if v:pre.append((birth,path+(j,),v))
        for j in range(D):
            if B[t][j]:pre.append((t,(j,),B[t][j]))
        first.append(math.fsum(w*P[t][path[-1]] for birth,path,w in pre))
        post=[(birth,path,w*(1-P[t][path[-1]])) for birth,path,w in pre if w*(1-P[t][path[-1]])>0]
        nxt=defaultdict(float)
        for (tracks,old_origins),branch_weight in branches.items():
            expansions=[((),branch_weight)]
            for label,birth,path,dets in tracks:
                nxt_exp=[]
                for existing,weight in expansions:
                    x=path[-1]
                    if x<0:nxt_exp.append((existing+((label,birth,path+(-1,),dets),),weight));continue
                    dead=1-S[t][x]
                    if dead:nxt_exp.append((existing+((label,birth,path+(-1,),dets),),weight*dead))
                    for j in range(D):
                        chance=S[t][x]*M[t][x][j]
                        if chance:nxt_exp.append((existing+((label,birth,path+(j,),dets),),weight*chance))
                expansions=nxt_exp
            if t==0:expansions=[((),branch_weight)]
            for existing,base in expansions:
                def assign(slot,tracks_now,used,origins,weight):
                    if slot==len(obs):
                        for label,birth,path,dets in existing:
                            if path[-1]>=0 and label not in used:weight*=1-P[t][path[-1]]
                        if weight:
                            key=(tuple(sorted(tracks_now)),old_origins+(tuple(origins),))
                            nxt[key]+=weight
                            if len(nxt)>max_branches:raise BudgetRefusal('CHRONOLOGICAL_BRANCH_CAP')
                        return
                    y=obs[slot]; slotid=(t,slot)
                    clutter=m.clutter_rate*gaussian(y,m.clutter_mean,m.clutter_variance)
                    if clutter:assign(slot+1,tracks_now,used,origins+(('clutter',),),weight*clutter)
                    for idx,tr in enumerate(tracks_now):
                        label,birth,path,dets=tr
                        if label in used or path[-1]<0:continue
                        x=path[-1]; w=P[t][x]*gaussian(y,m.offset[x]+m.response[x]*f[t],m.observation_variance[x])
                        if w:
                            changed=tracks_now[:idx]+((label,birth,path,dets+(slotid,)),)+tracks_now[idx+1:]
                            assign(slot+1,changed,used|frozenset((label,)),origins+(('old',label,x),),weight*w)
                    for birth,path,lam in pre:
                        x=path[-1]; w=lam*P[t][x]*gaussian(y,m.offset[x]+m.response[x]*f[t],m.observation_variance[x])
                        if w:
                            obj=(slotid,birth,path,(slotid,))
                            assign(slot+1,tracks_now+(obj,),used|frozenset((slotid,)),origins+(('new',birth,x),),weight*w)
                assign(0,existing,frozenset(),(),base)
        branches=dict(nxt)
    for birth,path,w in post:
        final_undetected[(birth,path[-1])]=final_undetected.get((birth,path[-1]),0.)+w
    exponent=T*m.clutter_rate+math.fsum(first)
    mass=math.fsum(branches.values());likelihood=math.exp(-exponent)*mass
    return {'weights':branches,'relative':mass,'full':likelihood,'exponent':exponent,
            'first_hit':tuple(first),'undetected':final_undetected}


def conditional(m,f,*,max_branches=300000):
    if not isinstance(m,Model):raise TypeError('MODEL')
    if len(f)!=len(m.observations) or not all(math.isfinite(v) for v in f):raise ValueError('FIELD_PATH')
    return _one_conditional(m,tuple(f),max_branches)


def solve(m,*,max_branches=300000):
    masses={};field_masses={};undetected={};expo={};countmass={}
    conditional_support=False
    for ids,f,p in field_paths(m):
        res=conditional(m,f,max_branches=max_branches)
        conditional_support=conditional_support or (res['relative']>0)
        fw=p*math.exp(-res['exponent']);field_masses[ids]=fw*res['relative'];expo[ids]=res['exponent'];undetected[ids]=res['undetected']
        for sig,w in res['weights'].items():
            masses[(ids,sig)]=fw*w
    evidence=math.fsum(masses.values())
    if not conditional_support:raise NumericalRefusal('IMPOSSIBLE_EVIDENCE')
    if not math.isfinite(evidence) or evidence<=0:raise NumericalRefusal('NUMERICAL_EVIDENCE_UNDERFLOW')
    posterior={key:value/evidence for key,value in masses.items()}
    field_post={key:value/evidence for key,value in field_masses.items()}
    for (ids,(tracks,origins)),q in posterior.items():
        alive=sum(tr[2][-1]>=0 for tr in tracks)
        countmass[alive]=countmass.get(alive,0.)+q
    return {'full_evidence':evidence,'posterior':posterior,'field_posterior':field_post,
            'exponents':expo,'undetected_by_field':undetected,'detected_count_posterior':countmass,
            'model_sha256':m.digest(),'claim_cap':'FINITE_FIELD_CONDITIONAL_RFS_REFERENCE_ONLY'}
