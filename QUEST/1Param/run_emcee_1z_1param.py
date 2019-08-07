from functions import *
import emcee
import timeit
import h5py

bin_num = 100
sig_lnMstar = 0.7
lnxsig1 = 3.5
lnxsig2 = 2.7
ndim, nwalkers, steps, runs = 1, 200, 5000, 10
burn_in = 100


def run_emcee(z, filename):
    
    #### Defining functions I need to actually run emcee.
    def lnprior(p):
        if 0.0 < p <= 3.0:
            return 0.0
        return -np.inf

    def lnprob(p, x, y, err):
        lp = lnprior(p)
        if np.isfinite(lp):
            return lp + lnchi2(p, x, y, err)
        return -np.inf

    def lnchi2(p, x, y, err): #lnlike
        chi2 = np.sum((model_func(x, p) - y) ** 2 / (err ** 2)) / (len(x)-1)
        return -chi2

    def model_func(x, p):
        qlf.get_SMBM(p)
        qlf.get_dNdlnL(lnxsigs = [lnxsig1,lnxsig2])
        xm, ym = qlf.LumBins, np.log10(qlf.dNdlnL * np.log(10))  
        y = np.interp(x, xm, ym)
        return y

    qlf = QLF(z, bin_num)
    qlf.LumBins = np.linspace(8.5, 16.5, bin_num)
    qlf.get_dNdlnMstar(sig_lnMstar)
    
    x, y, yerr = grab_obs(z)
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(np.array(x), np.array(y), np.array(yerr)))
    
    pos = [np.array([np.random.uniform(low=0.0, high=3.0)]) for i in range(nwalkers)]
    pos = np.array(pos)
    pos[(pos <= 0.0)] = 0.01
    
    tot_time = 0
    tot_steps = 0
    
    f = h5py.File(filename, "w")
    
    ### Attributes to describe this particular file's contents and parameters.
    f.attrs.modify('num-walkers', nwalkers)
    f.attrs.modify('burn-in', burn_in)
    f.attrs.modify('model-info', 'QLF fit uses the universal mdot distribution model as well as the luminosity dependant obscuration factor model. sigmalnM* = '+str(sig_lnMstar)+' bin-num = '+str(bin_num)+' sigmalnX1 = '+str(3.5)+'sigmalnX1 = '+str(2.2)+'. We fit the parameter across all redshifts.')
    f.attrs.modify('redshift', z)
    
    ### Setting up datasets to overwrite later.
    dset = f.create_dataset('walker-id', data = 0)
    dset = f.create_dataset('dM', data = 0)
    dset = f.create_dataset('acceptance-fraction', data = 0)
    
    print('redshift-'+str(z)+' burn-in phase')
    sampler.run_mcmc(pos, burn_in)
    pos = sampler.chain[:,-1,:]
    
    dset = f.create_dataset('pre-burnin-position', data = sampler.chain[:,0,0])
    f.close()
    
    sampler.reset()

    run = 1
    
    for n in range(runs):
        
        print('redshift-'+str(z)+', run number: '+str(run))
        ### run the emcee sampler for the number of steps defined
        ### set new pos to pos of last step
        ### add values to the file

        start = timeit.default_timer()
        sampler.run_mcmc(pos, steps)
        stop = timeit.default_timer()

        walker_id = np.zeros(nwalkers * steps * run)
        dM = np.zeros(nwalkers * steps * run)

        for i in range(nwalkers):
            walker_id[i * steps * run : (i+1) * steps * run] = i+1
            dM[i * steps * run : (i+1) * steps * run] = sampler.chain[i,:,0]

        tot_time += (stop - start)
        tot_steps += steps
        run += 1

        pos = sampler.chain[:,-1,:]

        f = h5py.File(filename, "a")

        f.__delitem__('walker-id')
        f.__delitem__('dM')
        f.__delitem__('acceptance-fraction')

        dset = f.create_dataset('walker-id', data = walker_id)
        dset = f.create_dataset('dM', data = dM)
        dset = f.create_dataset('acceptance-fraction', data = sampler.acceptance_fraction)
        f.attrs.modify('emcee_run_time', tot_time)
        f.attrs.modify('steps', tot_steps)

        f.close()
