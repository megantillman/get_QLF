import h5py
import numpy as np
import sys

iters = int(sys.argv[1])
reso = int(sys.argv[2])
div = int(sys.argv[3])

print('Begin re-combination process...')

chi21d = np.zeros(reso**6)
combos = np.zeros((reso**6,6))

for i in range(iters+1):
    
    if reso**6%div == 0 and i == reso**6/div:
        pass
    
    else:
        print(f'\t Adding slice ({i}) to the grid.')
        f = h5py.File("output/temp_r"+str(reso)+"_"+str(i)+".h5py", "r")
        chi2_slice = f['chi2_slice'][:]
        combo_slice = f['combo_slice'][:]
        f.close()

        if int((i+1)*div) >= reso**6:
            chi21d[int(i*div):] = chi2_slice
            combos[int(i*div):,:] = combo_slice
        else:
            chi21d[int(i*div):int((i+1)*div)] = chi2_slice
            combos[int(i*div):int((i+1)*div),:] = combo_slice


chi2_grid = chi21d.reshape(reso, reso, reso, reso, reso, reso)
combo_grid = combos.reshape(reso, reso, reso, reso, reso, reso, 6)


print('\t Writting to file...')

f = h5py.File("output/chi2_SHEN_r"+str(reso)+"_6param.h5py", "a")
dset = f.create_dataset('chi2_grid', data = chi2_grid)
dset = f.create_dataset('combo_grid', data = combo_grid)
f.close()

print(f'Complete, output found in "output/chi2_SHEN_r{reso}"_6param.h5py".\n')