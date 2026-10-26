import sys, cv2, numpy as np
sys.path.insert(0,'.'); sys.path.insert(0,'multicam360')
from calib import *
L = parse_offset_v6(offset_v6_from_metadata(read_metadata_bytes('re/tail8m.bin')))
imgs=[cv2.imread(f're/l{i}.png') for i in (0,1)]
W,H=3072,1536
lon=(np.arange(W)+.5)/W*2*np.pi-np.pi; lat=(np.arange(H)+.5)/H*np.pi-np.pi/2
LON,LAT=np.meshgrid(lon,lat)
D=np.stack([np.cos(LAT)*np.sin(LON),np.sin(LAT),np.cos(LAT)*np.cos(LON)],-1).reshape(-1,3)
order=sys.argv[1] if len(sys.argv)>1 else 'zxy'
R=[euler(*l['rot'],order).T for l in L]
M=[np.eye(3), R[1]@R[0].T@np.diag([-1.,1,-1])]
outs=[];ws=[]
for k in (0,1):
    rays=D@M[k].T
    px,valid=omni_project(rays,L[k])
    lx=((px[:,0]-k*L[k]['size'][1]-32)*0.5).reshape(H,W).astype(np.float32)
    ly=((px[:,1]-32)*0.5).reshape(H,W).astype(np.float32)
    v=cv2.remap(imgs[k],lx,ly,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    # 광축에서 멀수록 가중치 감소 → 겹침 띠에서 페더 블렌딩
    z=(rays[:,2]/np.linalg.norm(rays,axis=1)).reshape(H,W)
    w=np.clip((z-np.cos(L[k]['max_fov']))/0.12,0,1)*valid.reshape(H,W)
    outs.append(v.astype(np.float32)); ws.append(w)
    cv2.imwrite(f're/hemi{k}.jpg',v)
wsum=ws[0]+ws[1]+1e-6
pano=(outs[0]*ws[0][...,None]+outs[1]*ws[1][...,None])/wsum[...,None]
cv2.imwrite(f're/pano_calib_{order}.jpg',pano.astype(np.uint8))
