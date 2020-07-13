import sys
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

params = str(sys.argv[1])
zlist = [0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
files = ['/projects/b1026/mtillman/QLF_emcee/output/chain-data_'+params+'Param_z='+str(z)+'.h5py' for z in zlist]

ls = 12
ts = 16
nbin = 40

with PdfPages('plots/'+params+'Param_MCMC_indz.pdf') as pdf:
    if params == '1':     
        for file in files:
            redshift = file.split('=')[1].split('.h')[0]
            f = h5py.File(file,'r') 
            dM = f['dM'][:]
            f.close()

            fig = plt.figure(figsize=(4,4))

            plt.hist(dM, bins = nbin, color='gray', density=True)
            plt.xlabel('dM', fontsize = ls)
            plt.ylabel('Probability of Best Fit', fontsize = ls)
            plt.title('1 Parameter MCMC z='+redshift, fontsize = ts)
            plt.axis([0,3,0,1])
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            
            pdf.savefig()
            plt.close()
    
    if params == '2':
        for file in files:
            redshift = file.split('=')[1].split('.h')[0]
            f = h5py.File(file,'r') 
            dM = f['dM'][:]
            Xsig2 = f['siglnX2'][:]
            f.close()

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
            ax0.axvline(np.median(dM), color='r')
            ax0.axis([0,3,0,1])
            ax1.set_xlabel('dM', fontsize = ls)
            ax1.set_ylabel(r'$\sigma_{\ln{X}}$ post-disk', fontsize = ls)
            ax1.axvline(np.median(dM), color='r')
            ax1.axhline(np.median(Xsig2), color='r')
            ax1.axis([0,3,0,10])
            ax2.set_xlabel(r'$\sigma_{\ln{X}}$ post-disk', fontsize = ls)
            ax2.set_ylabel('Probability of Best Fit', fontsize = ls)
            ax2.axvline(np.median(Xsig2), color='r')
            ax2.axis([0,10,0,10])
            gs.tight_layout(fig, rect=[0, 0, 1, 0.97])
            plt.suptitle('2-Parameter MCMC z='+redshift, fontsize = ts)

            pdf.savefig()
            plt.close()
    
    if params == '3':
        for file in files:
            redshift = file.split('=')[1].split('.h')[0]
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
            plt.suptitle('3-Parameter MCMC z='+redshift, fontsize = ts)
    
            pdf.savefig()
            plt.close()