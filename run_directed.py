import json,sys,time
from pathlib import Path
from ffbbp_v198b.model import example
from ffbbp_v198b.continuous_numerical import GaussianAR1
from b16.directed_certificate import certificate
m=example(T=3,observations=((-0.2,),(.4,),(-.1,)))
p=GaussianAR1(.1,.8,.35,.05,.7)
n=int(sys.argv[1]) if len(sys.argv)>1 else 8
r=certificate(m,p,n=n,workers=4,progress=lambda i,N,t:print('PROGRESS',i,'/',N,'secs',round(t,1),flush=True))
print(json.dumps(r,indent=2))
f=Path(__file__).parent/'receipts'/f'b16_decimal_directed_n{n}.json';f.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
