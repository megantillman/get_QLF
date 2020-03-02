from numpy.polynomial import chebyshev as C
import numpy as np
import scipy as sp
import scipy.stats
from colossus.cosmology import cosmology
cosmology.setCosmology('planck15')
from colossus.lss import mass_function as mf
import glob
import numexpr as ne


files = [f.split('a')[1].split('.d')[0] for f in glob.glob('ssfrs/ssfr_a*.dat')]
a_list = np.array([float(a) for a in files])
mass_list = np.array([np.loadtxt("ssfrs/ssfr_a"+f+".dat")[:,0] for f in files])
ssfr_list = [np.loadtxt("ssfrs/ssfr_a"+f+".dat")[:,1] for f in files]

param_file = np.loadtxt("smhm_params.txt")
names = "EFF_0 EFF_0_A EFF_0_A2 EFF_0_Z M_1 M_1_A M_1_A2 M_1_Z ALPHA ALPHA_A ALPHA_A2 ALPHA_Z BETA BETA_A BETA_Z DELTA GAMMA GAMMA_A GAMMA_Z CHI2".split(" ");
params = dict(zip(names, param_file[:,1]))

def create_ranges_numexpr(start, stop, N):

    divisor = N-1
    s0 = start[:,None]
    s1 = stop[:,None]
    r = np.arange(N)

    return ne.evaluate('((1.0/divisor) * (s1 - s0))*r + s0')

def grab_obs(redshift): ###observational data from Hopkins+2006

    obs_points = []
    with open("obs_collect.txt", 'r') as f:
        for line in f:
            if line[0:4] != ';;//':
                obs_points.append(line.split())
    x = []
    y = []
    yerr = []
    for i in obs_points:
        if float(i[0]) == redshift:
            x.append(float(i[1]))
            y.append(float(i[2]))
            yerr.append(float(i[3]))

    return(x,y,yerr)



def Shen_fit_uncer(z, lums): ###best fit data from Shen+2020

    def get_params():
        rand_params = np.zeros((NUM, len(params)))
        ind = 0
        for p in params:
            i = np.random.randint(1,3,NUM)
            rand_params[:,ind][i == 1] = param_list[ind][0] + np.abs(np.random.normal(0, param_list[ind][1], size = len(i[i==1])))
            rand_params[:,ind][i == 2] = param_list[ind][0] - np.abs(np.random.normal(0, param_list[ind][2], size = len(i[i==2])))
            ind += 1
        return rand_params

    def shen_func(p):
        L = lums
        a0, a1, a2, b0, b1, b2, c0, c1, c2, d0, d1 = p
        zr = 2.0
        zfrac = (1 + z)/(1 + zr)
        g1 = C.chebval(1 + z, [a0, a1, a2])
        g2 = 2*b0/(zfrac**b1 + zfrac **b2)
        logLs = 2*c0/(zfrac**c1 + zfrac**c2)
        logPhis = C.chebval(z, [d0]) + C.chebval(1 + z, [0, d1])
        Lfrac = 10**L / 10**logLs
        Phibol = 10**logPhis/(Lfrac**g1 + Lfrac**g2)

        return np.log10(Phibol)

    params = {'a0':[0.85858, 0.03092, 0.02876], 'a1':[-0.26236, 0.02003, 0.01753], 'a2':[0.02105, 0.00136, 0.00113],\
        'b0':[2.54992, 0.01915, 0.02949], 'b1':[-1.04735, 0.01815, 0.02999], 'b2':[1.13277, 0.01988, 0.03891],\
        'c0':[13.01297, 0.00943, 0.01354], 'c1':[-0.57587, 0.00205, 0.00261], 'c2':[0.45361, 0.00290, 0.00434],\
        'd0':[-3.53138, 0.02694, 0.02690], 'd1':[-0.39961, 0.00871, 0.00896]}
    param_list = np.array([params[i] for i in params])

    NUM = int(1e4)

    rand_params = get_params()
    ys = np.apply_along_axis(shen_func, 1, rand_params).T
    ya = shen_func(param_list[:,0])

    fracs = sp.stats.norm.cdf([-2, -1, 0, 1, 2])
    percs = np.percentile(ys, 100*fracs, axis=1)

    std_ave = np.std(ys, axis=1)
    std_blw = ya-percs[1,:]
    std_abv = percs[3,:]-ya

    return ya, std_ave, std_abv, std_blw



class QLF():
    def __init__(self, z, bins):


        self.z = float(z)
        self.a = 1.0/(1.0+self.z)
        self.bins = bins
        self.get_zparams()

        self.HaloBins = np.linspace(7.0, 15.0, int((15.0 - 7.0) / self.bins))

        self.fp = self.HaloBins
        self.xp = self.get_Mstar(self.fp)

        self.StellBins = np.linspace(7.0, 12.2, int((12.2 - 8.0) / self.bins))


        closest_a = np.argmin(np.abs(a_list - self.a))
        self.ssfrs = np.array(ssfr_list[closest_a])
        self.nonzero = (self.ssfrs != 0)
        self.masses = np.array(mass_list[closest_a])
        self.minm = np.min(self.masses[self.nonzero])
        self.maxm = np.max(self.masses[self.nonzero])
        
        self.SSFRs = np.interp(self.StellBins, self.masses[self.nonzero], self.ssfrs[self.nonzero])



    def get_zparams(self):
        a1 = self.a - 1.0
        lna = np.log(self.a)
        self.zparams = {}
        self.zparams['m_1'] = params['M_1'] + a1*params['M_1_A'] - lna*params['M_1_A2'] + self.z*params['M_1_Z']
        self.zparams['sm_0'] = self.zparams['m_1'] + params['EFF_0'] + a1*params['EFF_0_A'] - lna*params['EFF_0_A2'] + self.z*params['EFF_0_Z']
        self.zparams['alpha'] = params['ALPHA'] + a1*params['ALPHA_A'] - lna*params['ALPHA_A2'] + self.z*params['ALPHA_Z']
        self.zparams['beta'] = params['BETA'] + a1*params['BETA_A'] + self.z*params['BETA_Z']
        if self.zparams['beta'] < 0.05:
            self.zparams['beta'] = 0.05
        self.zparams['delta'] = params['DELTA']
        self.zparams['gamma'] = 10**(params['GAMMA'] + a1*params['GAMMA_A'] + self.z*params['GAMMA_Z'])




    def get_slope(self, logMhalo): #returns dlogMstar/dlogMhalo slope is same in log10 and ln space

        dm = logMhalo-self.zparams['m_1'];
        term1 = (self.zparams['alpha']*10.**(self.zparams['beta']*dm)+self.zparams['beta']*10.**(self.zparams['alpha']*dm))/(10.**(self.zparams['beta']*dm) + 10.**(self.zparams['alpha']*dm))
        term2 = -self.zparams['gamma']*dm*np.exp(-(dm/self.zparams['delta'])**2/2.)/self.zparams['delta']**2
        slope = term1 + term2

        return slope

    def get_Mstar(self,logMhalo):

        dm = logMhalo-self.zparams['m_1']
        dm2 = dm/self.zparams['delta']
        logMstar = self.zparams['sm_0'] - np.log10(10**(-self.zparams['alpha']*dm) + 10**(-self.zparams['beta']*dm)) + self.zparams['gamma']*np.exp(-0.5*(dm2*dm2))

        return logMstar


    def get_Mhalo(self, logMstar):

        logMhalo = np.interp(logMstar, self.xp, self.fp)

        return logMhalo


    def gauss_array(self, vals, std):

        y = (1/np.sqrt(2.0*np.pi*std**2.0))*np.exp((-(vals[:-1]-vals[-1])**2.0)/(2.0*std**2))

        return y


    def convolve_smhm(self, sigma):

        lnten = np.log(10)
        logMh = self.get_Mhalo(self.StellBins)
        plus_mins = (7.0 * sigma)
        mins = (logMh * lnten - plus_mins)
        maxs = (logMh * lnten + plus_mins)
        mins[mins < 3 * lnten] = 3 * lnten
        maxs[maxs > 18 * lnten] = 18 * lnten
        bin_num = int(max(maxs - mins)/self.bins)
        lnMh = create_ranges_numexpr(mins, maxs, bin_num)
        dNdlnMhalo = mf.massFunction(np.e**lnMh, self.z, q_in='M', q_out='dndlnM', mdef='vir', model='despali16')
        logMstar = np.apply_along_axis(self.get_Mstar, 1, lnMh/lnten)
        vals = np.zeros((len(self.StellBins),bin_num+1))
        vals[:,-1] = self.StellBins * lnten
        vals[:,:-1] = logMstar * lnten
        Mstar_prob = np.apply_along_axis(self.gauss_array, 1, vals, sigma)
        dNdlnMstar = np.sum(Mstar_prob * dNdlnMhalo, axis = 1) * (lnMh[:,1] - lnMh[:,0])

        return dNdlnMstar



    def get_dNdlnMstar(self, sig_lnMstar):

        if sig_lnMstar == 0.:
            self.dNdlnMstar = mf.massFunction(10.**self.get_Mhalo(self.StellBins), self.z, q_in='M', q_out='dndlnM', mdef='vir', model='despali16') / self.get_slope(self.get_Mhalo(self.StellBins))
        else:
            self.dNdlnMstar = self.convolve_smhm(sig_lnMstar)


    def get_Mbh(self, logMstar0, approx_local = False):
        norm = [11, 8.2]

        logMstar = self.StellBins
        logMbh = logMstar * 0
        slopes = logMstar * 0

        if approx_local == False:
            post_params = [1.12, norm[1] - 1.12*norm[0]]
            init = [7., 7.*post_params[0]+post_params[1]]
            pre_params = [0.2, init[1] - init[0] * 0.2]

        else:
            post_params = [1, -2.8]
            init = [7., 7.*post_params[0]+post_params[1]]
            pre_params = [0.2, init[1] - init[0] * 0.2]

        post = (logMstar > logMstar0)
        pre = (logMstar <= logMstar0)
        Ms = 10**logMstar
        Ms0 = 10**logMstar0
        Mbh0 = 10**(logMstar0 * pre_params[0] + pre_params[1])
        beta = post_params[0]
        alpha = 10**norm[1] / 10**(norm[0]*beta)

        Mbh = Mbh0 + alpha * Ms[post]**(beta-1) * (Ms[post] - Ms0)
        logMbh[post] = np.log10(Mbh)
        slopes[post] = alpha * beta * Ms[post]**beta / Mbh

        logMbh[pre] = logMstar[pre] * pre_params[0] + pre_params[1]
        slopes[pre] = pre_params[0]

        self.slopes = slopes
        self.BHBins = logMbh
        self.pre = pre
        self.post = post
        self.mmax = post_params[0]

        self.SBHARs = self.slopes * (self.SSFRs / 3.154e7)
        self.MdotBH = self.SBHARs * (10**self.BHBins * 2e33)
        self.Ledd = 1.3e38 * 10**self.BHBins
        self.Mdotedd = self.Ledd / (.1 * (2.99e10)**2)


    def get_Mdotbh(self, vals, files = files):

        lnxsig = vals[0]
        Mdotbh = vals[1]
        
        mu_lnX = -0.5 * lnxsig**2
        mu_lnMdotbh = mu_lnX + np.log(Mdotbh) #g/s
        
        return mu_lnMdotbh, lnxsig


    def gauss_Mdot(self, lnMdotbh):

        x = lnMdotbh
        mu = self.Mdot_mu_sig[:,0]
        sig = self.Mdot_mu_sig[:,1]
        y = ( 1/np.sqrt(2.0 * np.pi * sig*sig) ) * np.exp( -(x - mu)**2.0 / (2.0 * sig*sig) )

        return y


    def get_dNdlnL(self, L, lnxsigs): 

        lnxsig_list = self.StellBins * 0
        lnxsig_list[self.pre] = lnxsigs[0]
        lnxsig_list[self.post] = lnxsigs[1]
        tenper = int( 0.4 * len(self.slopes[self.post][self.slopes[self.post] >= 1.05 * self.mmax] ) )
        tranpoint = np.argmin(self.pre)

        try:
            lintrans = np.linspace(lnxsigs[0], lnxsigs[1], tenper*2, endpoint = False)
            lnxsig_list[tranpoint - tenper : tranpoint + tenper] = lintrans
        except:
            lintrans = np.linspace(lnxsigs[0], lnxsigs[1], len(lnxsig_list[0 : tranpoint + tenper]), endpoint = False)
            lnxsig_list[0 : tranpoint + tenper] = lintrans


        vals = np.zeros((len(self.StellBins), 2))
        vals[:,0] = lnxsig_list
        vals[:,1] = self.MdotBH
        self.Mdot_mu_sig = np.apply_along_axis(self.get_Mdotbh, 1, vals)

        self.lnMdotbh_list = (np.asarray(L) + np.log10(3.9e33)) * np.log(10) - np.log(0.1*2.99e10**2)

        Rl = 0.8
        Rh = 0.2
        Lc = 10**43.7
        Lx = 0.037*10**(np.asarray(L) + np.log10(3.9e33))
        self.FOb = Rl * np.e**(-Lx/Lc) + Rh * (1 - np.e**(-Lx/Lc))

        self.intvals = np.apply_along_axis(self.gauss_Mdot, 1, self.lnMdotbh_list.reshape(len(self.lnMdotbh_list),1)) * self.dNdlnMstar * (self.StellBins[1] - self.StellBins[0])

        self.dNdlnL = np.sum(self.intvals, axis = 1)
