import sys
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

file = sys.argv[1]
params = int( sys.argv[2] )
redshift = str( sys.argv[3] )

f = h5py.File(file,'r') 
dM = f['dM'][:]
Xsig2 = f['siglnX2'][:]
Xsig1 = f['siglnX1'][:]
walkid = f['walker-id'][:]
f.close()

stepnum = int(len(dM)/len(set(walkid)))
steps = np.linspace(1,stepnum,stepnum)

if params == 1:
    
    fig = plt.figure(figsize=(12,2))
    plt.ylabel('dM')
    plt.xlabel('Step')
    plt.suptitle('Walker Paths')
    for i in set(walkerid):
        plt.plot(steps, dM[walkid == i], c='k', lw=0.2)
    plt.savefig('plots/1Param_walkers_z='+redshift+'.pdf')
        
if params == 2:
    
gs = gridspec.GridSpec(2, 1)
fig = plt.figure(figsize=(12,4))

    ax1 = fig.add_subplot(gs[1, 0])
    ax0 = fig.add_subplot(gs[0, 0], sharex = ax1)
    plt.setp(ax0.get_xticklabels(), visible=False)
    ax0.set_ylabel('dM')
    ax1.set_ylabel(r'$\sigma_{\ln{X}}$ post-disk')
    ax1.set_xlabel('Step')
    plt.suptitle('Walker Paths')

    for i in set(walkerid):
        ax0.plot(steps, dM[walkid == i], c='k', lw=0.2)
        ax1.plot(steps, Xsig2[walkid == i], c='k', lw=0.2)
    plt.savefig('plots/2Param_walkers_z='+redshift+'.pdf')


if params == 3:
    
gs = gridspec.GridSpec(3, 1)
fig = plt.figure(figsize=(12,6))

    ax2 = fig.add_subplot(gs[2, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex = ax2)
    ax0 = fig.add_subplot(gs[0, 0], sharex = ax2)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax0.get_xticklabels(), visible=False)
    ax0.set_ylabel('dM')
    ax1.set_ylabel(r'$\sigma_{\ln{X}}$ post-disk')
    ax2.set_ylabel(r'$\sigma_{\ln{X}}$ pre-disk')
    ax2.set_xlabel('Step')
    plt.suptitle('Walker Paths')

    for i in set(walkerid):
        ax0.plot(steps, dM[walkid == i], c='k', lw=0.2)
        ax1.plot(steps, Xsig2[walkid == i], c='k', lw=0.2)
        ax2.plot(steps, Xsig1[walkid == i], c='k', lw=0.2)
    plt.savefig('plots/3Param_walkers_z='+redshift+'.pdf')


