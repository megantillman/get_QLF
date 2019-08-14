from functions import *
import h5py
import itertools
from numpy.polynomial import chebyshev as C

zlist = [0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
reso = 75
qlf_bins = 0.01
sig_lnMstar = 0.7
dM = np.linspace(0.01,3.3,reso)
xsigpre = np.linspace(1.0,8.0,reso)
xsigpost = np.linspace(1.0,8.0,reso)
lum = np.linspace(9,14,50)

f = h5py.File("output/chi2_3pShenfit.h5py", "w")

f.attrs.modify('resolution', reso)
dset = f.create_dataset('dM', data = dM)
dset = f.create_dataset('siglnX2', data = xsigpost)
dset = f.create_dataset('siglnX1', data = xsigpre)

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
    sig_lnX = [a[1], a[2]]
    qlf.get_SMBM(dM)
    qlf.get_dNdlnL(lum, sig_lnX)
    ym = np.log10(qlf.dNdlnL * np.log(10))
    ya = shen_QLF(z, lum)

    return np.sum((ym-ya)**2)


for z in zlist:
    qlf = QLF(z, qlf_bins)
    qlf.get_dNdlnMstar(sig_lnMstar)

    combos = np.array(list(itertools.product(dM, xsigpre, xsigpost)))
    chi23d = np.apply_along_axis(chi2, 1, combos, z, qlf).reshape(reso, reso, reso)

    f = h5py.File("output/chi2_3pShenfit.h5py", "a")
    
    grp = f.create_group("z="+str(z))
    dset = grp.create_dataset('chi23d_grid', data = chi23d)
    
    f.close()