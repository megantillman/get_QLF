from functions import *
import h5py
import itertools

zlist = [0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
reso = 75
qlf_bins = 0.01
sig_lnMstar = 0.7
dM = np.linspace(0.01,3.3,reso)
xsigpre = np.linspace(1.0,8.0,reso)
xsigpost = np.linspace(1.0,8.0,reso)

f = h5py.File("output/chi2_3equiweight.h5py", "w")

f.attrs.modify('resolution', reso)
dset = f.create_dataset('dM', data = dM)
dset = f.create_dataset('siglnX2', data = xsigpost)
dset = f.create_dataset('siglnX1', data = xsigpre)

f.close()

for z in zlist:
    qlf = QLF(z, qlf_bins)
    qlf.get_dNdlnMstar(sig_lnMstar)
    xa, ya, yerr = grab_obs(z)
    xa = np.array(xa)
    ya = np.array(ya)
    
    def chi2(a):
        dM = a[0]
        sig_lnX = [a[1],a[2]]
        qlf.get_SMBM(dM)
        qlf.get_dNdlnL(xa, sig_lnX)
        ym = np.log10(qlf.dNdlnL * np.log(10))

        return np.sum((ym-ya)**2)

    combos = np.array(list(itertools.product(dM, xsigpre, xsigpost)))
    chi23d = np.apply_along_axis(chi2, 1, combos).reshape(reso, reso, reso)

    f = h5py.File("output/chi2_3equiweight.h5py", "a")
    
    grp = f.create_group("z="+str(z))
    dset = grp.create_dataset('chi23d_grid', data = chi23d)
    dset = grp.create_dataset('#_observation', data = (len(xa),))
    
    f.close()