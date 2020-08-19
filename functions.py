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
ssfr_a_list = np.array([float(a) for a in files])
ssfr_mass_list = np.array([np.loadtxt("ssfrs/ssfr_a"+f+".dat")[:,0] for f in files])
ssfr_list = [np.loadtxt("ssfrs/ssfr_a"+f+".dat")[:,1] for f in files]

files = [f.split('a')[1].split('.d')[0] for f in glob.glob('smfs/smf_a*.dat')]
smf_a_list = np.array([float(a) for a in files])
smf_mass_list = np.array([np.loadtxt("smfs/smf_a"+f+".dat")[:,0] for f in files])
smf_list = [np.loadtxt("smfs/smf_a"+f+".dat")[:,1] for f in files]

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
        logPhis = C.chebval(1 + z, [d0, d1])
        Lfrac = 10**L / 10**logLs
        Phibol = 10**logPhis/(Lfrac**g1 + Lfrac**g2)
        return np.log10(Phibol)

    params = {'a0':[0.8569, 0.0247, 0.0253], 'a1':[-0.2614, 0.0162, 0.0164], 'a2':[0.0200,0.0011,0.0011],\
        'b0':[2.5375, 0.0177, 0.0187], 'b1':[-1.0425,0.0164, 0.0182], 'b2':[1.1201, 0.0199, 0.0207],\
        'c0':[13.0088, 0.0090, 0.0091], 'c1':[-0.5759, 0.0018, 0.0020], 'c2':[0.4554, 0.0028, 0.0027],\
        'd0':[-3.5426, 0.0235, 0.0209], 'd1':[-0.3936, 0.0070, 0.0073]}
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
    
    zr = 2.0
    zfrac = (1 + z)/(1 + zr)
    logLs = 2*params['c0'][0]/(zfrac**params['c1'][0] + zfrac**params['c2'][0])

    return ya, std_ave, std_abv, std_blw, logLs



class QLF():
    def __init__(self, z, bins):


        self.z = float(z)
        self.a = 1.0/(1.0+self.z)
        self.bins = bins

        self.StellBins = np.linspace(3.0, 12.5, int((12.5 - 3.0) / self.bins))
        
        #### extract ssfr
        closest_a = np.argmin(np.abs(ssfr_a_list - self.a))
        ssfrs = np.array(ssfr_list[closest_a])
        nonzero = (ssfrs != 0)
        masses = np.array(ssfr_mass_list[closest_a])
        self.SSFRs = 10**np.interp(self.StellBins, masses[nonzero], np.log10(ssfrs[nonzero]))
        
        #### extrapolate out to high M*
        slope = (np.log10(ssfrs[nonzero][-1]) - np.log10(ssfrs[nonzero][-2])) / (masses[nonzero][-1] - masses[nonzero][-2])
        inter = np.log10(ssfrs[nonzero][-1]) - slope * masses[nonzero][-1]
        gtzero = (self.StellBins >= masses[nonzero][-1])
        self.SSFRs[gtzero] = 10**(self.StellBins[gtzero]*slope + inter)
        
        #### extrapolate out to low M*
        slope = (np.log10(ssfrs[1]) - np.log10(ssfrs[0])) / (masses[1] - masses[0])
        inter = np.log10(ssfrs[0]) - slope * masses[0]
        ltavail = (self.StellBins < masses[0])
        self.SSFRs[ltavail] = 10**(self.StellBins[ltavail]*slope + inter)
        
        
        ### extract smf
        closest_a = np.argmin(np.abs(smf_a_list - self.a))
        smf = np.array(smf_list[closest_a])
        nonzero = (smf != 0)
        masses = np.array(smf_mass_list[closest_a])
        self.dNdlogMstar = 10**np.interp(self.StellBins, masses[nonzero], np.log10(smf[nonzero]))
        
        #### extrapolate out to high M*
        slope = (np.log10(smf[nonzero][-1]) - np.log10(smf[nonzero][-2])) / (masses[nonzero][-1] - masses[nonzero][-2])
        inter = np.log10(smf[nonzero][-1]) - slope * masses[nonzero][-1]
        self.dNdlogMstar[gtzero] = 10**(self.StellBins[gtzero]*slope + inter)
        
        #### extrapolate out to low M*
        slope = (np.log10(smf[1]) - np.log10(smf[0])) / (masses[1] - masses[0])
        inter = np.log10(smf[0]) - slope * masses[0]
        self.dNdlogMstar[ltavail] = 10**(self.StellBins[ltavail]*slope + inter)
        
        self.dNdlnMstar = self.dNdlogMstar/np.log(10)
        


    def get_Mbh(self, logMstar0, slope_low = 0.2, norm_from_local =4.0, approx_local = True, norm_local = 8.2):
        norm = [11, norm_local]
        
        self.logMstar0 = logMstar0
        logMstar = self.StellBins
        logMbh = logMstar * 0
        slopes = logMstar * 0

        if approx_local == False:
            logMbh0 = logMstar0*1.12 - norm[0]*1.12 + norm[1] - norm_from_local
            my_norm = [logMstar0, logMbh0]
            post_params = [1.12, logMbh0 - 1.12*logMstar0]
            pre_params = [slope_low, my_norm[1] - my_norm[0] * slope_low]

        else:
            logMbh0 = logMstar0 + norm[1] - norm[0] - norm_from_local
            my_norm = [logMstar0, logMbh0]
            post_params = [1, logMbh0 - logMstar0]
            pre_params = [slope_low, my_norm[1] - my_norm[0] * slope_low]

        post = (logMstar > logMstar0)
        pre = (logMstar <= logMstar0)
        Ms = 10**logMstar
        Ms0 = 10**logMstar0
        Mbh0 = 10**logMbh0
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
        ## SSFR are in yr^-1
        self.SBHARs = self.slopes * (self.SSFRs / 3.154e7) ## s^-1
        self.MdotBH = self.SBHARs * (10**self.BHBins * 2e33) ## g/s
        self.Ledd = 1.3e38 * 10**self.BHBins ## erg/s
        self.Mdotedd = self.Ledd / (.1 * (2.99e10)**2) ## g/s


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


    def get_dNdlnL(self, L, lnxsigs): #input luminosity in log10 space in units of solar mass
        
        lnxsig_list = self.StellBins * 0
        lnxsig_list[self.pre] = lnxsigs[0]
        lnxsig_list[self.post] = lnxsigs[1]
        ### start transition at the M*crit value
        critpoint = np.argmin(np.abs(self.StellBins - self.logMstar0))
        ### end transistion 0.5 dex after that value
        endtran = np.argmin(np.abs(self.StellBins - (self.logMstar0 + 0.5)))
        lintrans = np.linspace(lnxsigs[0], lnxsigs[1], len(lnxsig_list[critpoint-1:endtran]))
        lnxsig_list[critpoint-1:endtran] = lintrans
        


        vals = np.zeros((len(self.StellBins), 2))
        vals[:,0] = lnxsig_list
        vals[:,1] = self.MdotBH
        self.Mdot_mu_sig = np.apply_along_axis(self.get_Mdotbh, 1, vals)

        self.lnMdotbh_list = (np.asarray(L) + np.log10(3.9e33)) * np.log(10) - np.log(0.1*2.99e10**2)

        ### I convert the stellar bins size to log base e for the calculation with the log base e stellar mass function and log base e sigmas
        self.intvals = np.apply_along_axis(self.gauss_Mdot, 1, self.lnMdotbh_list.reshape(len(self.lnMdotbh_list),1)) * self.dNdlnMstar * (self.StellBins[1] - self.StellBins[0]) * np.log(10)
     
        #### I produce two QLFs one in log base e and the other in log base 10
        self.dNdlnL = np.sum(self.intvals, axis = 1)
        self.dNdlogL = self.dNdlnL * np.log(10)

