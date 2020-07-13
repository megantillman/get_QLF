from functions import *
import h5py
import itertools
import numpy as np
import timeit
import time
import warnings
import sys

index = int(sys.argv[1])
reso = int(sys.argv[2])
div = int(sys.argv[3])
z = float(sys.argv[4])
iters = int(sys.argv[5])
qlf_bins = 0.005

def chi2(a, qlf):
    qlf.get_Mbh(a[0], a[3], a[4], approx_local=True, norm_local = 11+a[5])
    qlf.get_dNdlnL(lums, [a[1], a[2]])

    ym = np.log10(qlf.dNdlogL)
    presum = (ym-ya)**2
    return np.sum((ym-ya)**2)

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

if reso**6%div == 0 and index == reso**6/div:
    pass

else:
    print(f'({index}) Beginning slice operations.')

    f = h5py.File("output/combinations_r"+str(reso)+".h5py", "r")
    combos = f['combinations'][:]
    lums = f['luminosities'][:]
    f.close()


    if int((index+1)*div) >= reso**6:
        combo_slice = combos[int(index*div):,:]
    else:
        combo_slice = combos[int(index*div):int((index+1)*div),:]

    print(f'({index})\t Successfully retrieved slice of shape {combo_slice.shape}.')
    qlf = QLF(z, qlf_bins)

    ya, err_ave, err_abv, err_blw, _ = Shen_fit_uncer(z, lums)


    print(f'({index})\t Begining itterations...')
    start = timeit.default_timer()
    chi2_slice = np.apply_along_axis(chi2, 1, np.array(combo_slice), qlf)
    stop = timeit.default_timer()

    print(f'({index})\t Iteration complete after {TimeComplete(stop-start)}.') 
    print(f'({index})\t Writting to temp file "output/temp_{index}.h5py"...')

    f = h5py.File("output/temp_r"+str(reso)+"_"+str(index)+".h5py", "w")
    dset = f.create_dataset('combo_slice', data = combo_slice)
    dset = f.create_dataset('chi2_slice', data = chi2_slice)
    f.close()

    print(f'({index}) Write successful.\n')