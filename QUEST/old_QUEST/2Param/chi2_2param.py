from functions import *
import h5py
import itertools
from numpy.polynomial import chebyshev as C

zlist = [0.2, 0.4, 0.8, 1.2, 1.8, 2.4, 3.0, 3.6, 4.2, 4.8, 5.4, 6.0]
reso = 150
qlf_bins = 0.01
sig_lnMstar = 0.7
siglnX1 = 6.0 
siglnX2 = np.linspace(1.0,8.0, reso)
dM = np.linspace(0.01,3.3,reso)
lum = np.linspace(9,14,50)

f = h5py.File("output/chi2_2pShenfit.h5py", "w")

f.attrs.modify('resolution', reso)
dset = f.create_dataset('dM', data = dM)
dset = f.create_dataset('siglnX2', data = siglnX2)

f.close()

def shen_QLF(z, L):
    a0, a1, a2 = 0.85858, -0.26236, 0.02105
    b0, b1, b2 = 2*2.54992, -1.04735, 1.13277
    c0, c1, c2 = 2*13.01297, -0.57587, 0.45361
    d0, d1 = -3.53138, -0.39961
    zr = 2.0
    zfrac = (1 + z)/(1 + zr)
    g1 = C.chebval(1 + z, [a0, a1, a2])
    g2 = b0/(zfrac**b1 + zfrac **b2)
    logLs = c0/(zfrac**c1 + zfrac**c2)
    logPhis = C.chebval(z, [d0]) + C.chebval(1 + z, [0, d1])
    Lfrac = 10**L / 10**logLs
    Phibol = 10**logPhis/(Lfrac**g1 + Lfrac**g2)

    return np.log10(Phibol)

def chi2(a, z, qlf):
    dM = a[0]
    sig_lnX = [siglnX1, a[1]]
    qlf.get_SMBM(dM)
    qlf.get_dNdlnL(lum, sig_lnX)
    ym = np.log10(qlf.dNdlnL * np.log(10))
    ya = shen_QLF(z, lum)

    return np.sum((ym-ya)**2)

for z in zlist:
    qlf = QLF(z, qlf_bins)
    qlf.get_dNdlnMstar(sig_lnMstar)
 
    combos = np.array(list(itertools.product(dM, siglnX2)))
    chi2grid = np.apply_along_axis(chi2, 1, combos, z, qlf).reshape(reso, reso)
    
    f = h5py.File("output/chi2_2pShenfit.h5py", "a")
    
    grp = f.create_group("z="+str(z))
    dset = grp.create_dataset('chi2_grid', data = chi2grid)

    f.close()