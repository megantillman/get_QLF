import numpy as np
from colossus.cosmology import cosmology
cosmology.setCosmology('planck15') 
from colossus.lss import mass_function as mf 
import glob
import numexpr as ne
from scipy.optimize import newton


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

def grab_obs(redshift):
    
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



class QLF():
    def __init__(self, z, bins):
        
        
        self.z = float(z)
        self.a = 1.0/(1.0+self.z)
        self.bins = bins
        self.get_zparams()
        
        self.HaloBins = np.linspace(7.0, 15.0, int((15.0 - 7.0) / self.bins))

        self.fp = self.HaloBins
        self.xp = self.get_Mstar(self.fp)
        
        self.StellBins = np.linspace(8.0, 12.2, int((12.2 - 8.0) / self.bins))
        
        
    def get_zparams(self): ##converting this to ln????
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
    
    
    def get_SMBM(self, dM, Mmid = 10.3, slope1 = 0.2, slope3 = 1.):

        start = [7., np.log10(1.4*10**4.)]
        stop = [12., np.log10(1.4*10**9.)]
        mstar1 = Mmid - dM
        mstar2 = Mmid + dM
        int1 = start[1] - start[0] * slope1
        int3 = stop[1] - stop[0] * slope3
        x = (int3 - int1) / (slope1 - slope3)
        y = slope1 * x + int1
        if mstar1 < x:
            mstar1 = x
        mbh1 = slope1 * mstar1 + int1
        mbh2 = mstar2 + int3
        slope2 = (mbh2 - mbh1) / (mstar2 - mstar1)
        int2 = mbh2 - mstar2 * slope2
        
        self.dM = dM

        self.slope_list, self.int_list, self.mass_cuts = [slope1, slope2, slope3], [int1, int2, int3], [mstar1, mstar2]
        
        self.early = (self.StellBins <= self.mass_cuts[0])
        self.growth = ((self.StellBins > self.mass_cuts[0]) & (self.StellBins < self.mass_cuts[1]))
        self.late = (self.StellBins > self.mass_cuts[1])
        
        self.m = self.StellBins * 0
        self.m[self.early] = self.slope_list[0]
        self.m[self.growth] = self.slope_list[1]
        self.m[self.late] = self.slope_list[2]
        
        self.b = self.StellBins * 0
        self.b[self.early] = self.int_list[0]
        self.b[self.growth] = self.int_list[1]
        self.b[self.late] = self.int_list[2]
    
    
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


        
    
    def get_Mdotbh(self, vals, files = files):

        Mstar = vals[0]
        slope = vals[1]
        inter = vals[2]
        lnxsig = vals[3]
        a = self.a
        Mbh = 10**(Mstar*slope+inter)
        
        
        closest_a = np.argmin(np.abs(a_list - a))
        masses = np.array(mass_list[closest_a])
        ssfrs = np.array(ssfr_list[closest_a])
        closest_m = np.argmin(np.abs(masses - Mstar))
        
        nonzero = (ssfrs != 0)
        minm = np.min(masses[nonzero])
        maxm = np.max(masses[nonzero])
        if minm < Mstar < maxm:
            ssfr = np.interp(Mstar, masses[nonzero], ssfrs[nonzero])
        else:
            ssfr = ssfr_list[closest_a][closest_m]
        
        

        Ledd = 1.3e38 * Mbh #ergs/s 
        Mdotedd = Ledd / (.1 * (2.99e10)**2) #g/s
        sbhr = slope * (ssfr / 3.154e7) #1/s
        Mdotbh = sbhr * (Mbh * 2e33) #g/s
        
        
        mu_lnX = -0.5 * lnxsig**2
        mu_lnMdotbh = mu_lnX + np.log(Mdotbh) 
        
        lnMdotsig = lnxsig
        
        return mu_lnMdotbh, lnMdotsig, np.log(Mdotedd), np.log(Mdotbh), np.log(sbhr), np.log(ssfr)
    
    
    def gauss_Mdot(self, lnMdotbh):
  
        x = lnMdotbh
        mu = self.Mdot_mu_sig[:,0]
        sig = self.Mdot_mu_sig[:,1]
        y = ( 1/np.sqrt(2.0 * np.pi * sig**2.0) ) * np.exp( -(x - mu)**2.0 / (2.0 * sig**2) )

        return y
    
    
    def get_dNdlnL(self, L, lnxsigs):
        
        lnxsig_list = self.StellBins * 0
        lnxsig_list[self.early] = lnxsigs[0]
        lnxsig_list[self.growth] = lnxsigs[1]
        lnxsig_list[self.late] = lnxsigs[1]
        tenper = int(0.4 * sum(self.growth))
        tranpoint = np.argmin(self.early)

        try:
            lintrans = np.linspace(lnxsigs[0], lnxsigs[1], tenper*2, endpoint = False)
            lnxsig_list[tranpoint - tenper : tranpoint + tenper] = lintrans
        except:
            lintrans = np.linspace(lnxsigs[0], lnxsigs[1], len(lnxsig_list[0 : tranpoint + tenper]), endpoint = False)
            lnxsig_list[0 : tranpoint + tenper] = lintrans
#             try:
#                 lintrans = np.linspace(lnxsigs[0], lnxsigs[1], len(lnxsig_list[0 : tranpoint + tenper]), endpoint = False)
#                 lnxsig_list[0 : tranpoint + tenper] = lintrans
#             except:
#                 try:
#                     lintrans = np.linspace(lnxsigs[0], lnxsigs[1], len(lnxsig_list[tranpoint - tenper :]))
#                     lnxsig_list[tranpoint - tenper :] = lintrans
#                 except:
#                     lnxsig_list = np.linspace(lnxsigs[0], lnxsigs[1], len(self.StellBins))
        
        vals = np.zeros((len(self.StellBins), 4))
        vals[:,0] = self.StellBins
        vals[:,1] = self.m
        vals[:,2] = self.b
        vals[:,3] = lnxsig_list
        self.Mdot_mu_sig = np.apply_along_axis(self.get_Mdotbh, 1, vals)
        
        self.lnMdotbh_list = (np.asarray(L) + np.log10(3.9e33)) * np.log(10) - np.log(0.1*2.99e10**2)
        
        Rl = 0.8
        Rh = 0.2
        Lc = 10**43.7
        Lx = 0.037*10**(np.asarray(L) + np.log10(3.9e33))
        self.FOb = Rl * np.e**(-Lx/Lc) + Rh * (1 - np.e**(-Lx/Lc))
        
        self.intvals = np.apply_along_axis(self.gauss_Mdot, 1, self.lnMdotbh_list.reshape(len(self.lnMdotbh_list),1)) * self.dNdlnMstar * (self.StellBins[1] - self.StellBins[0])
                                 
        self.dNdlnL = (1-self.FOb) * np.sum(self.intvals, axis = 1)