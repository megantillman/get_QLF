from functions import *
import emcee
import scipy.optimize as op
import timeit
import h5py

bin_num = 100
sig_lnMstar = 0.7
ndim, nwalkers, steps, runs = 3, 300, 5000, 10
threads = 18
burn_in = 100


#### Defining functions I need to actually run emcee.
def lnprior(p):
    if 0.0 < p[0] <= 3.0 and 0.0 < p[1] <= 10.0 and 0.0 < p[2] <= 10.0:
        return 0.0
    return -np.inf

def lnprob(p, z, x, y, err):
    lp = lnprior(p)
    if np.isfinite(lp):
        return lp + lnchi2(p, z, x, y, err)
    return -np.inf

def lnchi2(p, z, x, y, err): #lnlike
    chi2 = np.sum((model_func(x, p, z) - y) ** 2 / (err ** 2)) / (len(x)-1)
    return -chi2

def model_func(x, p, z):
    y = []
    z = np.array(z)
    for zs, i in zip(zlist, qlf_list):
        i.get_SMBM(p[0])
        i.get_dNdlnL(lnxsigs = [p[2], p[1]])
        xm, ym = i.LumBins, np.log10(i.dNdlnL * np.log(10))
        where = (z == zs)  
        y.extend(np.interp(np.array(x)[where], xm, ym))
    return y



#### Creating QLF classes for each redshift I have observational data for.
qlf_00 = QLF(0.0, bin_num)
qlf_01 = QLF(0.1, bin_num)
qlf_02 = QLF(0.2, bin_num)
qlf_05 = QLF(0.5, bin_num)    
qlf_10 = QLF(1.0, bin_num)
qlf_15 = QLF(1.5, bin_num)
qlf_20 = QLF(2.0, bin_num)
qlf_25 = QLF(2.5, bin_num)
qlf_30 = QLF(3.0, bin_num)
qlf_35 = QLF(3.5, bin_num)
qlf_40 = QLF(4.0, bin_num)
qlf_45 = QLF(4.5, bin_num)
qlf_50 = QLF(5.0, bin_num)
qlf_55 = QLF(5.5, bin_num)
qlf_60 = QLF(6.0, bin_num)

qlf_list = [qlf_00, qlf_01, qlf_02, qlf_05, qlf_10, qlf_15, qlf_20, qlf_25, qlf_30, qlf_35, qlf_40, qlf_45, qlf_50, qlf_55, qlf_60]
zlist = [0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]

for i in qlf_list:
    i.LumBins = np.linspace(8.5, 16.5, bin_num)
    i.get_dNdlnMstar(sig_lnMstar)

    
#### Grabbing all the observational data I have access to.
xt_tot = []
yt_tot = []
yerr_tot = []
z_tot = []
for z in zlist:
    x, y, yerr = grab_obs(z)
    xt_tot.extend(x)
    yt_tot.extend(y)
    yerr_tot.extend(yerr)
    for j in range(len(x)):
        z_tot.append(z)

        

#### Setting up my emcee walkers and running them.
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads = threads, args=(np.array(z_tot), np.array(xt_tot), np.array(yt_tot), np.array(yerr_tot)))


#### Loops over runs and writting out to file each loop.

pos = [np.array([np.random.uniform(low=0, high=3), np.random.uniform(low=0, high=10), np.random.uniform(low=0, high=10)]) for i in range(nwalkers)]
pos = np.array(pos)
pos[(pos <= 0.0)] = 0.01

tot_time = 0
tot_steps = 0

filename = "output/chain-data2_3Param.h5py"

f = h5py.File(filename, "w")

### Attributes to describe this particular file's contents and parameters.
f.attrs.modify('num-walkers', nwalkers)
f.attrs.modify('threads', threads)
f.attrs.modify('burn-in', burn_in)
f.attrs.modify('model-info', 'QLF fit uses the universal mdot distribution model as well as the luminosity dependant obscuration factor model. sigmalnM* = '+str(sig_lnMstar)+' bin-num = '+str(bin_num)+'. We fit the parameter across all redshifts.')


### Setting up datasets to overwrite later.
dset = f.create_dataset('walker-id', data = 0)
dset = f.create_dataset('dM', data = 0)
dset = f.create_dataset('siglnX2', data = 0)
dset = f.create_dataset('siglnX1', data = 0)
dset = f.create_dataset('acceptance-fraction', data = 0)

### Setting up datasets for initial burn-in positions.
dset = f.create_dataset('pre-burnin-position:dM', data = pos[:,0])
dset = f.create_dataset('pre-burnin-position:siglnX2', data = pos[:,1])
dset = f.create_dataset('pre-burnin-position:siglnX1', data = pos[:,2])

f.close()

sampler.run_mcmc(pos, burn_in)
pos = sampler.chain[:,-1,:]
sampler.reset()

run = 1

for n in range(runs):

    ### run the emcee sampler for the number of steps defined
    ### set new pos to pos of last step
    ### add values to the file
    
    start = timeit.default_timer()
    sampler.run_mcmc(pos, steps)
    stop = timeit.default_timer()
    
    walker_id = np.zeros(nwalkers * steps * run)
    dM = np.zeros(nwalkers * steps * run)
    siglnX2 = np.zeros(nwalkers * steps * run)
    siglnX1 = np.zeros(nwalkers * steps * run)
    
    for i in range(nwalkers):
        walker_id[i * steps * run : (i+1) * steps * run] = i+1
        dM[i * steps * run : (i+1) * steps * run] = sampler.chain[i,:,0]
        siglnX2[i * steps * run : (i+1) * steps * run] = sampler.chain[i,:,1]
        siglnX1[i * steps * run : (i+1) * steps * run] = sampler.chain[i,:,2]
        
    tot_time += (stop - start)
    tot_steps += steps
    run += 1
    
    pos = sampler.chain[:,-1,:]
    
    f = h5py.File(filename, "a")
    
    f.__delitem__('walker-id')
    f.__delitem__('dM')
    f.__delitem__('siglnX2')
    f.__delitem__('siglnX1')
    f.__delitem__('acceptance-fraction')
    
    dset = f.create_dataset('walker-id', data = walker_id)
    dset = f.create_dataset('dM', data = dM)
    dset = f.create_dataset('siglnX2', data = siglnX2)
    dset = f.create_dataset('siglnX1', data = siglnX1)
    dset = f.create_dataset('acceptance-fraction', data = sampler.acceptance_fraction)
    f.attrs.modify('emcee_run_time', tot_time)
    f.attrs.modify('steps', tot_steps)
    
    f.close()
