"""Write the ssd array 'celltype_pts_v8_analysis' (+ certainty) into the v8 working dir: the PTv3 prediction
'celltype_pts_v8' (INT4 = 17, glia 12-16, OPC = 18) remapped to the contiguous neuron numbering of Analysis_Params('v8')
(INT4 = 12, ASTRO..FRAG = 13-17, OPC = 18). -1 (unlabelled) stays -1.
"""
import os, sys, numpy as np
wd = os.environ['BIO_ANALYSIS_WD'] + '/ssv_0'
src = np.load(f'{wd}/celltype_pts_v8.npy')
lut = np.arange(19)
lut[17] = 12
lut[12:17] = np.arange(13, 18)
lut[18] = 18  # OPC stays 18 in both numberings
out = np.where(src >= 0, lut[np.clip(src, 0, 18)], -1).astype(src.dtype)
np.save(f'{wd}/celltype_pts_v8_analysis.npy', out)
cert = f'{wd}/celltype_pts_v8_certainty.npy'
if os.path.exists(cert) and not os.path.exists(f'{wd}/celltype_pts_v8_analysis_certainty.npy'):
    os.symlink('celltype_pts_v8_certainty.npy', f'{wd}/celltype_pts_v8_analysis_certainty.npy')
names = ['DA','LMAN','HVC','MSN','STN','TAN','GPe','GPi','LTS','INT1','INT2','INT3','INT4','ASTRO','OLIGO','MICRO','MIGR','FRAG','OPC']
for i, n in enumerate(names):
    print(f'{n:6s} {int((out == i).sum()):9d}   (classifier id {int(np.where(lut == i)[0][0])})')
print('unlabelled', int((out < 0).sum()))
