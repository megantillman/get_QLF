import multiprocessing
import numpy as np
import itertools
import h5py
import timeit
import sys
from numpy.polynomial import chebyshev as C
import scipy as sp
import scipy.stats
import glob
from numba import jit

z = float(sys.argv[1])

filename = "output/chi2_fiducial-B3_2P_z"+str(z)+".h5py"

BP = [1.38000186, 11.90976007]

r1, r2, r3, r4, r5, r6 = 21, 16, 16, 29, 11, 13

logMb0 = np.linspace(9.0,11.5, r1)
xsigpre = np.linspace(1.0,4.0,r2)
xsigpost = np.linspace(1.0,4.0,r3)
slopes = np.linspace(0.1,1.5,r4)
norms = np.linspace(0.5,3.0,r5)
local_norms = np.linspace(np.log10(0.001), np.log10(0.005), r6)

qlf_bins = 0.005


##################################################################################################
files = [f.split('a')[1].split('.d')[0] for f in glob.glob('ssfrs/ssfr_a*.dat')]
ssfr_a_list = np.array([float(a) for a in files])
ssfr_mass_list = np.array([np.loadtxt("ssfrs/ssfr_a"+f+".dat")[:,0] for f in files])
ssfr_list = [np.loadtxt("ssfrs/ssfr_a"+f+".dat")[:,1] for f in files]

files = [f.split('a')[1].split('.d')[0] for f in glob.glob('smfs/smf_a*.dat')]
smf_a_list = np.array([float(a) for a in files])
smf_mass_list = np.array([np.loadtxt("smfs/smf_a"+f+".dat")[:,0] for f in files])
smf_list = [np.loadtxt("smfs/smf_a"+f+".dat")[:,1] for f in files]

param_file = np.loadtxt("smhm_params.txt")
names = "EFF_0 EFF_0_A EFF_0_A2 EFF_0_Z M_1 M_1_A M_1_A2 M_1_Z ALPHA ALPHA_A ALPHA_A2 ALPHA_Z BETA BETA_A BETA_Z DELTA GAMMA GAMMA_A GAMMA_Z CHI2".split(" ");
params = dict(zip(names, param_file[:,1]))

a = 1.0/(1.0+z)

StellBins = np.linspace(3.0, 12.5, int((12.5 - 3.0) / qlf_bins))

closest_a = np.argmin(np.abs(ssfr_a_list - a))

#### extract ssfr
ssfrs = np.array(ssfr_list[closest_a])
nonzero = (ssfrs != 0)
masses = np.array(ssfr_mass_list[closest_a])
SSFRs = 10**np.interp(StellBins, masses[nonzero], np.log10(ssfrs[nonzero]))

#### extrapolate out to high M*
slope = (np.log10(ssfrs[nonzero][-1]) - np.log10(ssfrs[nonzero][-2])) / (masses[nonzero][-1] - masses[nonzero][-2])
inter = np.log10(ssfrs[nonzero][-1]) - slope * masses[nonzero][-1]
gtzero = (StellBins >= masses[nonzero][-1])
SSFRs[gtzero] = 10**(StellBins[gtzero]*slope + inter)

#### extrapolate out to low M*
slope = (np.log10(ssfrs[1]) - np.log10(ssfrs[0])) / (masses[1] - masses[0])
inter = np.log10(ssfrs[0]) - slope * masses[0]
ltavail = (StellBins < masses[0])
SSFRs[ltavail] = 10**(StellBins[ltavail]*slope + inter)


### extract smf
closest_a = np.argmin(np.abs(smf_a_list - a))
smf = np.array(smf_list[closest_a])
nonzero = (smf != 0)
masses = np.array(smf_mass_list[closest_a])
dNdlogMstar = 10**np.interp(StellBins, masses[nonzero], np.log10(smf[nonzero]))

#### extrapolate out to high M*
slope = (np.log10(smf[nonzero][-1]) - np.log10(smf[nonzero][-2])) / (masses[nonzero][-1] - masses[nonzero][-2])
inter = np.log10(smf[nonzero][-1]) - slope * masses[nonzero][-1]
dNdlogMstar[gtzero] = 10**(StellBins[gtzero]*slope + inter)

#### extrapolate out to low M*
slope = (np.log10(smf[1]) - np.log10(smf[0])) / (masses[1] - masses[0])
inter = np.log10(smf[0]) - slope * masses[0]
dNdlogMstar[ltavail] = 10**(StellBins[ltavail]*slope + inter)

dNdlnMstar = dNdlogMstar/np.log(10)
##################################################################################################

BT = 1.0 / (1.0 + np.exp(-BP[0]*(StellBins-BP[1])))
BulgeBins = np.log10(BT * 10**StellBins) #this is bulge mass in units of solar mass
dBTdMstar = np.gradient(BT, 10**StellBins)
dMbdMstar = dBTdMstar*10**StellBins+BT
SMBARs = (SSFRs*dMbdMstar*10**StellBins) / 10**BulgeBins #this is units of per year
        
dlogMstardlogMb = np.gradient(StellBins, BulgeBins)
dNdlogMbulge = dNdlogMstar*dlogMstardlogMb
dNdlnMbulge = dNdlogMbulge/np.log(10)



@jit(nopython=True) 
def get_QLF(L, bins = 0.005, logMb0 = None, slope_low = None, norm_from_local = None, norm_local = 8.2, approx_local = True, presig = None, postsig = None):
    
    norm = [11, norm_local]

    logMb = BulgeBins
    BHBins = logMb * 0
    slopes = logMb * 0

    logMbh0 = logMb0 + norm[1] - norm[0] - norm_from_local
    my_norm = [logMb0, logMbh0]
    post_params = [1, logMbh0 - logMb0]
    pre_params = [slope_low, my_norm[1] - my_norm[0] * slope_low]

    post = (logMb > logMb0)
    pre = (logMb <= logMb0)
    Mb = 10**logMb
    Mb0 = 10**logMb0
    Mbh0 = 10**logMbh0
    beta = post_params[0]
    alpha = 10**norm[1] / 10**(norm[0]*beta)

    Mbh = Mbh0 + alpha * Mb[post]**(beta-1) * (Mb[post] - Mb0)
    BHBins[post] = np.log10(Mbh)
    slopes[post] = alpha * beta * Mb[post]**beta / Mbh

    BHBins[pre] = logMb[pre] * pre_params[0] + pre_params[1]
    slopes[pre] = pre_params[0]

        ## SSFR are in yr^-1
    SBHARs = slopes * (SMBARs / 3.154e7) ## s^-1
    MdotBH = SBHARs * (10**BHBins * 2e33) ## g/s


    lnxsig_list = StellBins * 0
    lnxsig_list[pre] = presig
    lnxsig_list[post] = postsig
    ### start transition at the M*crit value
    critpoint = np.argmin(np.abs(BulgeBins - logMb0))
    ### end transistion 0.5 dex after that value
    endtran = np.argmin(np.abs(BulgeBins - (logMb0 + 0.5)))
    lintrans = np.linspace(presig, postsig, len(lnxsig_list[critpoint-1:endtran]))
    lnxsig_list[critpoint-1:endtran] = lintrans

    
    MU_LIST = []
    SIG_LIST = []
    for lnxsig, mdotbh in zip(lnxsig_list, MdotBH):
        mu_lnx = -0.5 * lnxsig**2
        mu_lnmdotbh = mu_lnx + np.log(mdotbh) #g/s
        MU_LIST.append(mu_lnmdotbh)
        SIG_LIST.append(lnxsig)
    MU_LIST = np.array(MU_LIST)
    SIG_LIST = np.array(SIG_LIST)
        
    lnMdotbh_list = (np.asarray(L) + np.log10(3.9e33)) * np.log(10) - np.log(0.1*2.99e10**2)

    intvals = []
    for mdot in lnMdotbh_list:
        y = ( 1/np.sqrt(2.0 * np.pi * SIG_LIST*SIG_LIST) ) * np.exp( -(mdot - MU_LIST)**2.0 / (2.0 * SIG_LIST*SIG_LIST) )
        intvals.append(y * dNdlnMstar * (StellBins[1] - StellBins[0]) * np.log(10))

    dNdlnL = []
    for val in intvals:
        totval = 0.0 
        for n in val:
            totval = totval + n
        dNdlnL.append(totval)

    dNdlogL = np.array(dNdlnL) * np.log(10)
    return dNdlogL



def TimeComplete(secs):
    
    days = secs//86400
    hours = (secs - days*86400)//3600
    minutes = (secs - days*86400 - hours*3600)//60
    seconds = (secs - days*86400 - hours*3600 - minutes*60)
    result = ("{}:".format(int(days)) if days else "") + \
    ("{}:".format(int(hours)) if hours>10 else "0{}:".format(int(hours))) + \
    ("{}:".format(int(minutes)) if minutes>10 else "0{}:".format(int(minutes))) + \
    ("{}".format(int(seconds)) if seconds>10 else "0{}".format(int(seconds)))
    
    return result


def chi2(a):
    
    dNdlogL = get_QLF(lums, bins = qlf_bins, logMb0 = a[0], slope_low = a[3], norm_from_local = a[4], norm_local = 11+a[5], presig = a[1], postsig = a[2])

    ym = np.log10(dNdlogL)
    leastsq = ( (ym-ya)/yaerr )**2
    
    return np.sum(leastsq)


def process_chunk(all_args):
    
    axis, combo_slice = all_args
    
    return np.apply_along_axis(chi2, axis, combo_slice)

#### Setting up variables and luminosity bins to fit.
    
print('--------------------------------------')
print(f'Producing fit for redshift z = {z}')
print('--------------------------------------')

print('\tGrabbing observational data for fit.')
    
    
xtot = []
ya = []
yaerr = []
fobs = h5py.File("SHEN_obs_collect.h5py", "r")
for band in fobs["z="+str(z)]:
    index = "z="+str(z)+"/"+band
    xtot.extend(fobs[index]['x'][:]) ##log10 of bolometric L in erg/s
    ya.extend(fobs[index]['y'][:]) ##log10 of QLF
    yaerr.extend(fobs[index]['yerr'][:]) ##error on that
fobs.close()

ya = np.array(ya)
yaerr = np.array(yaerr)
lums = np.log10(10**np.asarray(xtot)/3.8e33)

print('\tCreating combination array.')

start = timeit.default_timer()

combos = np.array(list(itertools.product(logMb0, xsigpre, xsigpost, slopes, norms, local_norms)))

stop = timeit.default_timer()

print(f'\t\tCombination array produced after time {TimeComplete(stop-start)}.\n')

#### Write various data to the file.
f = h5py.File(filename, "w")
f.attrs.modify('resolutions', np.array([r1, r2, r3, r4, r5, r6]))
f.attrs.modify('redshift', np.array([z]))
f.attrs.modify('qlf_bin_size', np.array([qlf_bins]))
dset = f.create_dataset('luminosities', data = lums)
dset = f.create_dataset('logMb0', data = logMb0)
dset = f.create_dataset('siglnX2', data = xsigpost)
dset = f.create_dataset('siglnX1', data = xsigpre)
dset = f.create_dataset('slope_low', data = slopes)
dset = f.create_dataset('norm_from_local', data = norms)
dset = f.create_dataset('norm_of_local', data = local_norms)
f.close()

#### Slice data to use all available processors. 
print(f'\tProcessors available: {multiprocessing.cpu_count()}')
print('\tSplitting up parameter combinations for parallel calculation.')
chunks = [(1, combo_slice)
              for combo_slice in np.array_split(combos, multiprocessing.cpu_count())]

#### Begin doing calculations on different processors.
print('\t\tBeginning calculations...')
pool = multiprocessing.Pool()

start = timeit.default_timer()

chunk_results = pool.map(process_chunk, chunks)

stop = timeit.default_timer()

#### Free up the workers.
print(f'\tCalculations complete after time {TimeComplete(stop-start)}')
print('\tBegining re-combination process.')
pool.close()
pool.join()

#### Reshape the grid.
chi2_grid = np.concatenate(chunk_results).reshape((r1, r2, r3, r4, r5, r6))

#### Write final output to file.
print('\tWritting to file.')

f = h5py.File(filename, "a")
dset = f.create_dataset('chi2_grid', data = chi2_grid)
f.close()

print(f'Proccess complete. Output written to {filename}.\n')
