import multiprocessing
import numpy as np
from functions_numba import *
import itertools
import h5py
import timeit
import sys


z = float(sys.argv[1])

filename = "output/chi2_SHEN_r"+str(reso)+"_6param_eL_adjreso.h5py"

r1, r2, r3, r4, r5, r6 = 20, 15, 15, 20, 15, 15

logMstar0 = np.linspace(8.5,12.0,r1)
xsigpre = np.linspace(1.0,4.0,r2)
xsigpost = np.linspace(1.0,4.0,r3)
slopes = np.linspace(0.01,1.3,r4)
norms = np.linspace(0.5,3.0,r5)
local_norms = np.linspace(np.log10(0.001), np.log10(0.005), r6)


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
    
    dNdlogL = get_QLF(lums, bins = qlf_bins, logMstar0 = a[0], slope_low = a[3], norm_from_local = a[4], norm_local = 11+a[5], presig = a[1], postsig = a[2])

    ym = np.log10(dNdlogL)
    presum = (ym-ya)**2
    
    return np.sum((ym-ya)**2)


def process_chunk(all_args):
    
    func, axis, combo_slice, z = all_args
    
    return np.apply_along_axis(chi2, axis, combo_slice)




#### Setting up variables and luminosity bins to fit.
#### I fit an equal number of luminosities before and after the knee but all within the Shen study's range.
qlf_bins = 0.005

Lleft = 7.5 #change left bound of L
Lright = 14.95 #change right bound of L
points_to_fit = 100 #change number of L to fit
midpoint = points_to_fit//2+points_to_fit%2

lums = np.zeros(points_to_fit)
    
_, _, _, _, Lbreak = Shen_fit_uncer(z, lums)
    
lums[0:midpoint] = np.linspace(Lleft, Lbreak, midpoint, endpoint = False)
lums[midpoint:] = np.linspace(Lbreak, Lright, points_to_fit//2)



print('\nCreating combination array.')

start = timeit.default_timer()

combos = np.array(list(itertools.product(logMstar0, xsigpre, xsigpost, slopes, norms, local_norms)))

stop = timeit.default_timer()

print(f'Combination array produced after time {TimeComplete(stop-start)}.\n')

#### Write various data to the file.
f = h5py.File(filename, "w")
f.attrs.modify('resolutions', np.array([r1, r2, r3, r4, r5, r6]))
f.attrs.modify('indices', np.array(['logMstar0', 'siglnX1', 'siglnX2', 'slope_low', 'norm_from_local', 'norm_of_local']))
f.attrs.modify('redshift', np.array([z]))
f.attrs.modify('qlf_bin_size', np.array([qlf_bins]))
dset = f.create_dataset('luminosities', data = lums)
dset = f.create_dataset('logMstar0', data = logMstar0)
dset = f.create_dataset('siglnX2', data = xsigpost)
dset = f.create_dataset('siglnX1', data = xsigpre)
dset = f.create_dataset('slope_low', data = slopes)
dset = f.create_dataset('norm_from_local', data = norms)
dset = f.create_dataset('norm_of_local', data = local_norms)
f.close()

#### Produce values to fit to from Shen.
ya, err_ave, err_abv, err_blw, _ = Shen_fit_uncer(z, lums) #Shen points to fit.

#### Slice data to use all available processors. 
print(f'\nProcessors available: {multiprocessing.cpu_count()}')
print('\tSplitting up parameter combinations for parallel calculation.')
chunks = [(chi2, 1, combo_slice, z)
              for combo_slice in np.array_split(combos, multiprocessing.cpu_count())]

#### Begin doing calculations on different processors.
print('\tBeginning calculations...')
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

print(f'\nProccess complete. Output written to {filename}.\n')
