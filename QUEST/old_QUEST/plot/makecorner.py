import sys
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

file = sys.argv[1]
params = int( sys.argv[2] )
redshift = str( sys.argv[3] )

ls = 12
ts = 16
nbin = 40

if params == 1:

    f = h5py.File(file,'r') 
    dM = f['dM'][:]
    f.close()
    
    fig = plt.figure(figsize=(4,4))
    
    plt.hist(dM, bins = nbin, color='gray', density=True)
    plt.xlabel('dM', fontsize = ls)
    plt.ylabel('Probability of Best Fit', fontsize = ls)
    plt.title('1 Parameter MCMC', fontsize = ts)
    plt.axis([0,3,0,1])
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig('plots/1Param_Corner_z'+redshift+'.pdf')
    
if params == 2:
    
    f = h5py.File(file,'r') 
    dM = f['dM'][:]
    Xsig2 = f['siglnX2'][:]
    f.close()
    
    line = 1
    slope = (1.5-4)/3
    inter = 1.5 - 3*slope
    
    dMabv = dM[(dM > (Xsig2-inter)/slope)]
    dMblw = dM[(dM < (Xsig2-inter)/slope)]
    Xsig2abv = Xsig2[(Xsig2 > dM*slope+inter)]
    Xsig2blw = Xsig2[(Xsig2 < dM*slope+inter)]
        
    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure(figsize=(8,8))

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    ax0.hist(dM, bins = nbin, color = 'gray', density=True)
    ax1.hist2d(dM, Xsig2, bins = nbin, cmap = 'binary', density=True)
    ax2.hist(Xsig2, bins = nbin, color = 'gray', density=True)

    ax0.set_xlabel('dM', fontsize = ls)
    ax0.set_ylabel('Probability of Best Fit', fontsize = ls)
    ax0.axvline(np.median(dMabv), color='r', lw = line)
    ax0.axvline(np.median(dMblw), color='b', lw = line)
    ax0.text(6,5,'dM = '+str(np.median(dMabv)), color = 'r')
    ax0.text(6,4,'dM = '+str(np.median(dMblw)), color = 'b')
    ax0.text(6,3,'Xsig2 = '+str(np.median(Xsig2abv)), color = 'r')
    ax0.text(6,2,'Xsig2 = '+str(np.median(Xsig2blw)), color = 'b')
    
    ax0.axis([0,3,0,6])

    ax1.set_xlabel('dM', fontsize = ls)
    ax1.set_ylabel(r'$\sigma_{\ln{X}}$ post-disk', fontsize = ls)
    ax1.axvline(np.median(dMabv), color='r', lw = line)
    ax1.axvline(np.median(dMblw), color='b', lw = line)
    ax1.axhline(np.median(Xsig2abv), color='r', lw = line)
    ax1.axhline(np.median(Xsig2blw), color='b', lw = line)
    ax1.plot([0,3],[1.5,4], color='gray', linestyle='dashed', linewidth = 1.0)
    ax1.axis([0,3,4,1.5])

    ax2.set_xlabel(r'$\sigma_{\ln{X}}$ post-disk', fontsize = ls)
    ax2.set_ylabel('Probability of Best Fit', fontsize = ls)
    ax2.axvline(np.median(Xsig2abv), color='r', lw = line)
    ax2.axvline(np.median(Xsig2blw), color='b', lw = line)
    ax2.axis([1.5,4,0,2])

    gs.tight_layout(fig, rect=[0, 0, 1, 0.97])
    plt.suptitle('2-Parameter MCMC', fontsize = ts)
    
    fig.savefig('plots/2Param_Corner_z'+redshift+'.pdf')
    
if params == 3:
    
    f = h5py.File(file,'r') 
    dM = f['dM'][:]
    Xsig2 = f['siglnX2'][:]
    Xsig1 = f['siglnX1'][:]
    f.close()
    
    gs = gridspec.GridSpec(3, 3)
    fig = plt.figure(figsize=(12,12))

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[2, 1])
    ax5 = fig.add_subplot(gs[2, 2])

    ax0.hist(dM, bins = nbin, color = 'gray', density=True)
    ax1.hist2d(dM, Xsig2, bins = nbin, cmap = 'binary', density=True)
    ax2.hist(Xsig2, bins = nbin, color = 'gray', density=True)
    ax3.hist2d(dM, Xsig1, bins = nbin, cmap = 'binary', density=True)
    ax4.hist2d(Xsig2, Xsig1, bins = nbin, cmap = 'binary', density=True)
    ax5.hist(Xsig1, bins = nbin, color = 'gray', density=True)

    ax0.set_xlabel('dM', fontsize = ls)
    ax0.set_ylabel('Probability of Best Fit', fontsize = ls)
    ax0.axvline(np.median(dM), color='r')

    ax1.set_xlabel('dM', fontsize = ls)
    ax1.set_ylabel(r'$\sigma_{\ln{X}}$ post-disk', fontsize = ls)
    ax1.axvline(np.median(dM), color='r')
    ax1.axhline(np.median(Xsig2), color='r')

    ax2.set_xlabel(r'$\sigma_{\ln{X}}$ post-disk', fontsize = ls)
    ax2.set_ylabel('Probability of Best Fit', fontsize = ls)
    ax2.axvline(np.median(Xsig2), color='r')

    ax3.set_xlabel('dM', fontsize = ls)
    ax3.set_ylabel(r'$\sigma_{\ln{X}}$ pre-disk', fontsize = ls)
    ax3.axvline(np.median(dM), color='r')

    ax4.set_xlabel(r'$\sigma_{\ln{X}}$ post-disk', fontsize = ls)
    ax4.set_ylabel(r'$\sigma_{\ln{X}}$ pre-disk', fontsize = ls)
    ax4.axvline(np.median(Xsig2), color='r')

    ax5.set_xlabel(r'$\sigma_{\ln{X}}$ pre-disk', fontsize = ls)
    ax5.set_ylabel('Probability of Best Fit', fontsize = ls)

    gs.tight_layout(fig, rect=[0, 0, 1, 0.97])
    plt.suptitle('3-Parameter MCMC', fontsize = ts)
    
    fig.savefig('plots/3Param_Corner_z'+redshift+'.pdf')