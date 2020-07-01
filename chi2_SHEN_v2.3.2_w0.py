from functions_SMFstart import *
import h5py
import itertools
import numpy as np
import scipy as sp
import scipy.stats
from numpy.polynomial import chebyshev as C
import timeit
import warnings

warnings.filterwarnings("ignore")

# zlist = [0.0, 1.0, 2.0, 3.0, 4.0]
zlist = [1.0]
reso = 100
qlf_bins = 0.005
slopes = 1.
norms = 0.
lums = np.linspace(6.5, 14.95, 150)
logMstar0 = 3.
xsigpre = 2.
xsigpost = np.linspace(1.0,10.0,reso)


filename = "output/chi2_SHEN_r"+str(reso)+"_v2.3.2_w0_s2.h5py"



f = h5py.File(filename, "a")

f.attrs.modify('resolution', np.array(reso))
dset = f.create_dataset('logMstar0', data = np.array(logMstar0))
dset = f.create_dataset('siglnX2', data = xsigpost)
dset = f.create_dataset('siglnX1', data = np.array(xsigpre))
dset = f.create_dataset('slope_low', data = np.array(slopes))
dset = f.create_dataset('norm_from_local', data = np.array(norms))

f.close()

def chi2(a, z, qlf):
    qlf.get_Mbh(logMstar0, slopes, norms, approx_local=True)
    qlf.get_dNdlnL(lums, [xsigpre, a[0]])
    ym = np.log10(qlf.dNdlogL)
    presum = (ym-ya)**2
    return np.sum((ym-ya)**2)

for z in zlist:
    
    print('Begin with redshift '+str(z))
    qlf = QLF(z, qlf_bins)
    
    ya, err_ave, err_abv, err_blw = Shen_fit_uncer(z, lums)
    
    print('Begin itterations...')
    start = timeit.default_timer()
    chi23d = np.apply_along_axis(chi2, 1, xsigpost.reshape(reso,1), z, qlf)
    stop = timeit.default_timer()

    print('Time to itterate: ', stop - start) 
    print('Writting the file...')

    f = h5py.File(filename, "a")
    
    grp = f.create_group("z="+str(z))
    dset = grp.create_dataset('chi23d_grid', data = chi23d)
    
    f.close()
    
    print('Write successful...')
    print('Done with redshift '+str(z))