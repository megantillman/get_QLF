from functions import *
import h5py
import itertools
import numpy as np
import scipy as sp
import scipy.stats
from numpy.polynomial import chebyshev as C

zlist = [0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
reso = 75
qlf_bins = 0.005
sig_lnMstar = 0.7
logMstar0 = np.linspace(7,12,reso)
xsigpre = np.linspace(1.0,8.0,reso)
xsigpost = np.linspace(1.0,8.0,reso)

f = h5py.File("output/chi2_3pShenfit_nm.h5py", "w")

f.attrs.modify('resolution', reso)
dset = f.create_dataset('logMstar0', data = logMstar0)
dset = f.create_dataset('siglnX2', data = xsigpost)
dset = f.create_dataset('siglnX1', data = xsigpre)

f.close()

def chi2(a, z, qlf):
    logMstar0 = a[0]
    sig_lnX = [a[1], a[2]]
    qlf.get_SMBM(dM)
    qlf.get_dNdlnL(lums, sig_lnX)
    qlf.get_Mbh(logMstar0, approx_local=False)

    ym = np.log10(qlf.dNdlnL * np.log(10))

    return np.sum((ym-ya)**2)

for z in zlist:
    qlf = QLF(z, qlf_bins)
    qlf.get_dNdlnMstar(sig_lnMstar)
    xo, yo, yerro = grab_obs(z)
    lums = np.linspcae(min(xo), max(xo), 300)
    
    ya, err_ave, err_abv, err_blw = Shen_fit_uncer(z, lums)

    combos = np.array(list(itertools.product(logMstar0, xsigpre, xsigpost)))
    chi23d = np.apply_along_axis(chi2, 1, combos, z, qlf).reshape(reso, reso, reso)

    f = h5py.File("output/chi2_3pShenfit_nm.h5py", "a")
    
    grp = f.create_group("z="+str(z))
    dset = grp.create_dataset('chi23d_grid', data = chi23d)
    
    f.close()