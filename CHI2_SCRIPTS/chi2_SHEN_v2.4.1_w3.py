from functions import *
import h5py
import itertools
import numpy as np
import scipy as sp
import scipy.stats
from numpy.polynomial import chebyshev as C
import timeit
import warnings

warnings.filterwarnings("ignore")

zlist = [1.0]
reso = 10
qlf_bins = 0.005
slopes = 1.
norms = 0.
Lleft = 8.95
Lright = 14.95
logMstar0 = 3.
xsigpre = 2.
xsigpost = np.linspace(1.0,8.0,reso)
local_norms = np.linspace(np.log10(0.001), np.log10(0.005), reso)
combos = np.array(list(itertools.product(xsigpost, local_norms)))



filename = "output/chi2_SHEN_r"+str(reso)+"_v2.4.1_w3_s2.h5py"



f = h5py.File(filename, "a")

f.attrs.modify('resolution', np.array(reso))
dset = f.create_dataset('logMstar0', data = np.array(logMstar0))
dset = f.create_dataset('siglnX2', data = xsigpost)
dset = f.create_dataset('siglnX1', data = np.array(xsigpre))
dset = f.create_dataset('slope_low', data = np.array(slopes))
dset = f.create_dataset('norm_from_local', data = np.array(norms))
dset = f.create_dataset('norm_of_local', data = local_norms)

f.close()

def chi2(a, z, qlf):
    qlf.get_Mbh(logMstar0, slopes, norms, approx_local=True, norm_local = 11+a[1])
    qlf.get_dNdlnL(lums, [xsigpre, a[0]])
    ym = np.log10(qlf.dNdlogL)
    presum = (ym-ya)**2
    return np.sum((ym-ya)**2)

for z in zlist:
    
    print('Begin with redshift '+str(z))
    qlf = QLF(z, qlf_bins)
    
    lums = np.zeros(150)
    
    _, _, _, _, Lbreak = Shen_fit_uncer(z, lums)
    
    lums[0:75] = np.linspace(Lleft, Lbreak, 75, endpoint = False)
    lums[75:] = np.linspace(Lbreak, Lright, 75)
    
    ya, err_ave, err_abv, err_blw, _ = Shen_fit_uncer(z, lums)
    
    print('Begin itterations...')
    start = timeit.default_timer()
    chi23d = np.apply_along_axis(chi2, 1, combos, z, qlf).reshape(reso,reso)
    stop = timeit.default_timer()

    print('Time to itterate: ', stop - start) 
    print('Writting the file...')

    f = h5py.File(filename, "a")
    
    grp = f.create_group("z="+str(z))
    dset = grp.create_dataset('chi23d_grid', data = chi23d)
    
    f.close()
    
    print('Write successful...')
    print('Done with redshift '+str(z))
    
print('Fit complete. Output written to "'+str(filename)+'".')