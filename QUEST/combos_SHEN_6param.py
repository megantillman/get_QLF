from functions import *
import h5py
import itertools
import numpy as np
import sys
import timeit

reso = int(sys.argv[1])
z = float(sys.argv[2])

def TimeComplete(secs):
    days = secs//86400
    hours = (secs - days*86400)//3600
    minutes = (secs - days*86400 - hours*3600)//60
    seconds = int(secs - days*86400 - hours*3600 - minutes*60)
    result = ("{}:".format(days) if days else "") + \
    ("{}:".format(hours) if hours>10 else "0{}:".format(hours)) + \
    ("{}:".format(minutes) if minutes>10 else "0{}:".format(minutes)) + \
    ("{}".format(seconds) if seconds>10 else "0{}".format(seconds))
    return result

slopes = np.linspace(0.01,1.3,reso)
norms = np.linspace(0.5,3.0,reso)
logMstar0 = np.linspace(8.0,12.0,reso)
xsigpre = np.linspace(1.0,4.0,reso)
xsigpost = np.linspace(1.0,4.0,reso)
local_norms = np.linspace(np.log10(0.001), np.log10(0.005), reso)

Lleft = 8.95
Lright = 14.95
points_to_fit = 100
midpoint = points_to_fit//2+points_to_fit%2

lums = np.zeros(points_to_fit)
    
_, _, _, _, Lbreak = Shen_fit_uncer(z, lums)
    
lums[0:midpoint] = np.linspace(Lleft, Lbreak, midpoint, endpoint = False)
lums[midpoint:] = np.linspace(Lbreak, Lright, points_to_fit//2)

print('\nCreating combination array.')

start = timeit.default_timer()

combos = np.array(list(itertools.product(logMstar0, xsigpre, xsigpost, slopes, norms, local_norms)))

stop = timeit.default_timer()

print(f'\t Combination array produced after {TimeComplete(stop-start)}.')

print('\t Writting to files.')

f = h5py.File("output/combinations_r"+str(reso)+".h5py", "w")
dset = f.create_dataset('combinations', data = combos)
dset = f.create_dataset('luminosities', data = lums)
f.close()

f = h5py.File("output/chi2_SHEN_r"+str(reso)+"_6param.h5py", "w")
f.attrs.modify('resolution', np.array([reso]))
f.attrs.modify('redshift', np.array([z]))
dset = f.create_dataset('luminosities', data = lums)
dset = f.create_dataset('logMstar0', data = logMstar0)
dset = f.create_dataset('siglnX2', data = xsigpost)
dset = f.create_dataset('siglnX1', data = xsigpre)
dset = f.create_dataset('slope_low', data = slopes)
dset = f.create_dataset('norm_from_local', data = norms)
dset = f.create_dataset('norm_of_local', data = local_norms)
f.close()

print('Complete.\n')