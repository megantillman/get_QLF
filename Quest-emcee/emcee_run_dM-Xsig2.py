from functions import *
import emcee
import scipy.optimize as op
import timeit

bin_num = 100
sig_lnMstar = 0.7
ndim, nwalkers, steps, runs = 2, 300, 10000, 10
threads = 18


#### Defining functions that I need to find my optimal initial guess for emcee.
def get_QLF_mcmc(dM, sigpost, i, sig_lnMstar = 0.7, sigpre = 3.5):
    sig_lnX = [sigpre, sigpost]
    i.get_SMBM(dM)
    i.get_dNdlnL(sig_lnX)
    return i.LumBins, np.log10(i.dNdlnL * np.log(10))

def model_func_op(x, p, i):
    xm, ym = get_QLF_mcmc(p[0], p[1], i)
    y = np.interp(x, xm, ym)
    return y

def lnchi2_op(p, i, x, y, err): #lnlike
    chi2 = np.sum((model_func_op(x, p, i) - y) ** 2 / (err**2)) / (len(x)-1)
    return -chi2


#### Defining functions I need to actually run emcee.
def lnprior(p):
    if 0.0 < p[0] <= 3.0 and p[1] > 0.0:
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
        i.get_dNdlnL(lnxsigs = [3.5, p[1]])
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


#### Using scipy to create an optimal intial guess for emcee.
p = [1.5, 2.5]
nll = lambda *args: -lnchi2_op(*args)
results = []
bond = ((0.01, 3.0), (0.0,5.0))

for i, z in zip(qlf_list, zlist):
    xt, yt, yerr = np.array(grab_obs(z))
    results.append(op.minimize(nll, p, args=(i, xt, yt, yerr), method = 'L-BFGS-B', bounds = bond)["x"])

pinit = np.sum(np.array(results), axis=0) / len(results)


#### Setting up my emcee walkers and running them.
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads = threads, args=(np.array(z_tot), np.array(xt_tot), np.array(yt_tot), np.array(yerr_tot)))


#### Loops over runs and writting out to file each loop.

pos = [pinit + .3*np.random.randn(ndim) for i in range(nwalkers)]
pos = np.array(pos)
pos[(pos <= 0.0)] = 0.01

f = open("output/dM-Xsig2-chain-v1.0_n"+str(nwalkers)+"_s"+str(steps)+"_t"+str(threads)+".dat","w")
f.write('###  dM      sigma_lnX (post-disk)\n')
f.write('###  nwalkers = '+str(nwalkers)+'   steps/run = '+str(steps)+'    threads = '+str(threads)+'\n')
f.write('###  first '+str(nwalkers)+' values per line corrspond to dM values \n')
f.write('###  second '+str(nwalkers)+' values per line corrspond to sigma_lnX (post-disk) values \n')
f.close()

index = 0
tot_time = 0
for n in range(1,runs+1):

    start = timeit.default_timer()
    sampler.run_mcmc(pos, steps)
    stop = timeit.default_timer()
    
    f = open("output/dM-Xsig2-chain-v1.0_n"+str(nwalkers)+"_s"+str(steps)+"_t"+str(threads)+".dat","a")
    for i in range(index, index+steps):
        for j in sampler.chain[:,i,0]:
            f.write(str(j)+'    ')
        for j in sampler.chain[:,i,1]:
            f.write(str(j)+'    ')
        f.write('\n')
    tot_time += float(stop - start)
    f.write('######### run: '+str(n)+'    time/run: '+str(stop - start)+'    total time: '+str(tot_time)+'\n')
    f.close()
    pos = sampler.chain[:,-1,:]
    index += steps
    
f = open("output/dM-Xsig2-chain-v1.0_n"+str(nwalkers)+"_s"+str(steps)+"_t"+str(threads)+".dat","a")
f.write('#### run attempts completed')
f.close()
