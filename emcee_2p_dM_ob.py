import emcee
import scipy.optimize as op
from functions import *


ndim, nwalkers, steps = 2, 30, 10
bin_num = 200
smhm_scat = 0.3
standev = [.85, .75, .35]
prob_zero = [0.0, 0.0, 0.0]
mmid = 10.3
bonds = ((0.01,3.0),(0.4,0.8))
p = [0.2, 0.65]
variance = 0.1
throw = 0
threads = 2


#### Defining functions that I need to find my optimal initial guess for emcee.
def get_QLF_op(p, i, mmid = mmid, prob_zero = prob_zero, standev = standev):
    ob = .5
    dm, obs = p
    i.get_SMBM(dm, mmid)
    i.get_dNdMbh()
    i.get_dNdL(prob_zero, standev, ob)
    
    return i.LumBins, i.dNdL

def model_func_op(x, p, i):
    xm, ym = get_QLF_op(p, i)
    y = np.interp(x, xm, ym)
    return y

def lnchi2_op(p, i, x, y, err): #lnlike
    
    chi2 = np.sum((y - model_func_op(x, p, i)) ** 2 / (err ** 2)) / (len(x)-1)
    return -chi2


#### Defining functions I need to actually run emcee.
def lnprior(p):
    dm, ob = p
    if 0.0 < dm <= 3.0 and 0.0 < ob < 1.0:
        return 0.0
    return -np.inf

def lnprob(p, i, x, y, err):
    lp = lnprior(p)
    if np.isfinite(lp):
        return lp + lnchi2(p, i, x, y, err)
    return -np.inf

def lnchi2(p, z, x, y, err): #lnlike
    
    chi2 = np.sum((model_func(x, p, z) - y) ** 2 / (err ** 2)) / (len(x)-1)
    return -chi2

def model_func(x, p, z, mmid = mmid, prob_zero = prob_zero, standev = standev):
    y = []
    dm, ob = p
    for zs, i in zip(zlist, qlf_list):
        i.get_SMBM(dm, mmid)
        i.get_dNdMbh()
        i.get_dNdL(prob_zero, standev, ob)
        xm, ym = i.LumBins, i.dNdL
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
    i.get_dNdMstar(smhm_scat)

    
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
nll = lambda *args: -lnchi2_op(*args)
results = []
results_unbound = []
for i, z in zip(qlf_list, zlist):
    xt, yt, yerr = np.array(grab_obs(z))
    results.append(op.minimize(nll, p, args=(i, xt, yt, yerr), method = 'L-BFGS-B', bounds = bonds)["x"])
pinit = np.sum(np.array(results),axis=0)/len(results)


#### Setting up my emcee walkers and running them.
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads = threads, args=(np.array(z_tot), np.array(xt_tot), np.array(yt_tot), np.array(yerr_tot)))

pos = [pinit + variance*np.random.randn(ndim) for i in range(nwalkers)]
for i in range(len(pos)): ## Make sure none of the walkers start at a negative position.
    if pos[i][0] <= 0.0:
        pos[i][0] = 0.05
    if pos[i][1] <= 0.4:
        pos[i][1] = 0.41
    if pos[i][0] > 3.0:
        pos[i][0] = 3.0
    if pos[i][1] > 0.8:
        pos[i][1] = 0.79

sampler.run_mcmc(pos, steps)

f = open("output/chain-v1.0_n"+str(nwalkers)+"_s"+str(steps)+".dat","w")
f.write('###  dM          obscured fraction \n')
f.write('###  nwalkers = '+str(nwalkers)+'   steps = '+str(steps)+'\n')
f.write('###  **each walker is seperated by three pound signs** \n')
for i in sampler.chain[:,throw:,:]:
    for j in i:
        f.write(str(j[0])+' '+str(j[1])+'\n')
    f.write('### \n')
f.close()