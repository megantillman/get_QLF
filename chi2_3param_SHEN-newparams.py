from functions_newparams import *
import h5py
import itertools
import numpy as np
import scipy as sp
import scipy.stats
from numpy.polynomial import chebyshev as C
import timeit
import warnings

warnings.filterwarnings("ignore")

zlist = [0.0, 1.0, 2.0, 3.0, 4.0]
reso = 30
qlf_bins = 0.005
sig_lnMstar = 0.7
slopes = np.linspace(0.01,1.5,reso)
norms = np.linspace(0.0,3.0,reso)
lums = np.linspace(8.95, 14.95, 150)
logMstar0 = 10.58 #10.3
xsigpre = np.linspace(1.0,10.0,reso)
xsigpost = 1.23 #2.3
combos = np.array(list(itertools.product(slopes, xsigpre, norms)))


filename = "output/chi2_3pShenfit_"+str(reso)+"_newparams.h5py"



f = h5py.File(filename, "a")

f.attrs.modify('resolution', reso)
dset = f.create_dataset('logMstar0', data = np.asarray(logMstar0))
dset = f.create_dataset('siglnX2', data = np.asarray(xsigpost))
dset = f.create_dataset('siglnX1', data = xsigpre)
dset = f.create_dataset('slope_low', data = slopes)
dset = f.create_dataset('norm_from_local', data = norms)

f.close()

def chi2(a, z, qlf):
    qlf.get_Mbh(logMstar0, a[0], a[2], approx_local=True)
    qlf.get_dNdlnL(lums, [a[1], xsigpost])

    ym = np.log10(qlf.dNdlnL * np.log(10))
    presum = (ym-ya)**2
#     print(max(presum), np.mean(presum))
    return np.sum((ym-ya)**2)

for z in zlist:
    
    print('Begin with redshift '+str(z))
    qlf = QLF(z, qlf_bins)
    qlf.get_dNdlnMstar(sig_lnMstar)
    
    ya, err_ave, err_abv, err_blw = Shen_fit_uncer(z, lums)
    
    print('Begin itterations...')
    start = timeit.default_timer()
    chi23d = np.apply_along_axis(chi2, 1, combos, z, qlf).reshape(reso, reso, reso)
    stop = timeit.default_timer()

    print('Time to itterate: ', stop - start) 
    print('Writting the file...')

    f = h5py.File(filename, "a")
    
    grp = f.create_group("z="+str(z))
    dset = grp.create_dataset('chi23d_grid', data = chi23d)
    
    f.close()
    
    print('Write successful...')
    print('Done with redshift '+str(z))