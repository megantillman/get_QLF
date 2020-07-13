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

# zlist = [0.0, 1.0, 2.0, 3.0, 4.0]
zlist = [1.0]
reso = 10
qlf_bins = 0.005
Lleft = 8.95
Lright = 14.95
slopes = np.linspace(0.01,1.5,reso)
norms = np.linspace(0.0,3.0,reso)
logMstar0 = np.linspace(5,12.5,reso)
xsigpre = np.linspace(1.0,8.0,reso)
xsigpost = np.linspace(1.0,8.0,reso)
combos = np.array(list(itertools.product(logMstar0, xsigpre, xsigpost, slopes, norms)))


filename = "output/chi2_SHEN_r"+str(reso)+"_v2.2.1_w3.h5py"



f = h5py.File(filename, "a")

f.attrs.modify('resolution', np.array(reso))
dset = f.create_dataset('logMstar0', data = logMstar0)
dset = f.create_dataset('siglnX2', data = xsigpost)
dset = f.create_dataset('siglnX1', data = xsigpre)
dset = f.create_dataset('slope_low', data = slopes)
dset = f.create_dataset('norm_from_local', data = norms)

f.close()

def chi2(a, z, qlf):
    qlf.get_Mbh(a[0], a[3], a[4], approx_local=True)
    qlf.get_dNdlnL(lums, [a[1], a[2]])

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
    chi23d = np.apply_along_axis(chi2, 1, combos, z, qlf).reshape(reso, reso, reso, reso, reso)
    stop = timeit.default_timer()

    print('Time to itterate: ', stop - start) 
    print('Writting the file...')

    f = h5py.File(filename, "a")
    
    grp = f.create_group("z="+str(z))
    dset = grp.create_dataset('chi23d_grid', data = chi23d)
    
    f.close()
    
    print('Write successful...')
    print('Done with redshift '+str(z))