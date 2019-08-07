from functions import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py

def get_QLF(z, dM, bin_num = 300, sig_lnMstar = 0.7, sig_lnX = [3.5,2.2]):

    qlf = QLF(z, bin_num)
    qlf.LumBins = np.linspace(8.5, 16.5, bin_num)
    qlf.get_dNdlnMstar(sig_lnMstar)
    qlf.get_SMBM(dM)
    qlf.get_dNdlnL(sig_lnX)
    
    return qlf.LumBins, np.log10(qlf.dNdlnL * np.log(10))

def calc_chi2(ym, ya, yerr):
    return np.sum(((ym-ya)/yerr)**2)

def get_mdat(xsig2, dm, z, xa):
    sigs = [siglnX1,xsig2[0]]
    xm, ym = get_QLF(z, dM = dm, sig_lnX = sigs)
    ymi = np.interp(xa, xm, ym)
    return ymi

def get_chi2(dm, xsig2, z):
    ymi = get_mdat(xsig2, dm, z, xa)
    chi2 = calc_chi2(np.array(ymi), np.array(ya), np.array(yerr))
    return chi2

def get_gridline(yax, z):
    gridline = np.apply_along_axis(get_chi2, 1, xax, yax, z)
    return gridline

zlist = [0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
bins = 500
siglnX1 = 6.0 
siglnX2 = np.linspace(8.0,1.0, bins)
dM = np.linspace(0,3.0,bins)
dM[0] = 0.01

extent = [dM[0], dM[-1], siglnX2[-1], siglnX2[0]]
yax = siglnX2.reshape((bins,1))
xax = dM.reshape((bins, 1))

f = h5py.File("output/chi2_grid_data.h5py", "w")

f.attrs.modify('bins', bins)
f.attrs.modify('siglnX1', siglnX1)
f.attrs.modify('x-axis', 'dM')
f.attrs.modify('y-axis', 'siglnX2')
dset = f.create_dataset('dM', data = dM)
dset = f.create_dataset('siglnX2', data = siglnX2)
dset = f.create_dataset('extent', data = extent)

f.close()

for z in zlist:
    xa, ya, yerr = grab_obs(z)
    num_obs = len(xa)

    grid2d = np.apply_along_axis(get_gridline, 1, yax, z)

    maxl = []
    for n in grid2d:
        maxl.append(max(n))

    minl = []
    for m in grid2d:
        minl.append(min(m))

    f = h5py.File("output/chi2_grid_data.h5py", "a")
    
    grp = f.create_group("z="+str(z))
    dset = grp.create_dataset('chi2_grid', data = grid2d)
    dset = grp.create_dataset('min-max', data = [min(minl), max(maxl)])
    dset = grp.create_dataset('#_observation', data = num_obs)
    
    f.close()