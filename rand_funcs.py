from pprint import pprint
import numpy as np
import h5py
from numpy.polynomial import chebyshev as C
import scipy as sp
import scipy.stats
import glob

def get_null(FILE):
    f = h5py.File(FILE, "r") 
    siglnX2 = f['siglnX2'][:]
    siglnX1 = f['siglnX1'][:]
    chi2_grid = f['chi2_grid'][:].T
    f.close()

    for index, value in np.ndenumerate(chi2_grid):
        if siglnX2[index[3]] > siglnX1[index[4]]:
            chi2_grid[index] = 1e10

    null = np.where(chi2_grid == 1e10)
    
    return null

def best_fit_params_VARIED(redshifts, filename, null = False, bulge = False):
    
    print('Recording the best fit values for the varied fits "'+filename+'".')
    
    if not null:
        print('\tGenerating null grid.')
        null = get_null(filename+'_z'+str(redshifts[0])+'.h5py')
        
    allz = {}
#     print('\n')
    for z in redshifts:
        f = h5py.File(filename+'_z'+str(z)+'.h5py', 'r') 
        if bulge:
            transition = f['logMb0'][:]
        else:
            transition = f['logMstar0'][:]
        siglnX2 = f['siglnX2'][:]
        siglnX1 = f['siglnX1'][:]
        slope_low = f['slope_low'][:]
        norm_from_local = f['norm_from_local'][:]
        norm_of_local = f['norm_of_local'][:]
        chi2_grid = f['chi2_grid'][:].T
        f.close()

        print('\tPost-disk norm values explore:', norm_of_local)
        
        chi2_grid[null] = 1e10

        minval = np.amin(chi2_grid)
        minind = np.where(chi2_grid == minval)

        bestlocal = norm_of_local[minind[0][0]] 
        bestnorm = norm_from_local[minind[1][0]]
        bestslope = slope_low[minind[2][0]]
        bestpost = siglnX2[minind[3][0]]
        bestpre = siglnX1[minind[4][0]]
        bestcrit = transition[minind[5][0]]

        qlf_params = {'Transition point':bestcrit, 'Pre-disk normalization':bestnorm, 'Pre-disk slope':bestslope, 'Pre-disk sigma':bestpre,\
                      'Post-disk sigma':bestpost, 'Post-disk normalization':bestlocal, 'Chi2 value':minval}
        
#         print('Best fit parameter values for z ='+str(z)+':')
#         pprint(qlf_params)
#         print('\n')
        
        allz[str(z)] = qlf_params
    
    print('Return: (1) dictionary of best fit values for individual redshifts, (2) null grid.\n')
    
    return allz, null


def best_fit_params_FIXED(redshifts, filename, null = False, bulge = False):
    
    print('Reporting the best fit values for the fixed fits for files "'+filename+'".')

    if not null:
        print('\tGenerating null grid.')
        null = get_null(filename+'_z'+str(redshifts[0])+'.h5py')
       
#     print('\n')
    for z in redshifts:
        f = h5py.File(filename+'_z'+str(z)+'.h5py', "r") 
        if bulge:
            transition = f['logMb0'][:]
        else:
            transition = f['logMstar0'][:]
        siglnX2 = f['siglnX2'][:]
        siglnX1 = f['siglnX1'][:]
        slope_low = f['slope_low'][:]
        norm_from_local = f['norm_from_local'][:]
        norm_of_local = f['norm_of_local'][:]
        if z == redshifts[0]:
            chi2_grid = f['chi2_grid'][:].T
        else:
            chi2_grid += f['chi2_grid'][:].T
        f.close()
    
    chi2_grid[null] = 1e10
    
    minval_tot = np.amin(chi2_grid)
    minind = np.where(chi2_grid == minval_tot)

    bestlocal = norm_of_local[minind[0][0]] 
    bestnorm = norm_from_local[minind[1][0]]
    bestslope = slope_low[minind[2][0]]
    bestpost = siglnX2[minind[3][0]]
    bestpre = siglnX1[minind[4][0]]
    bestcrit = transition[minind[5][0]]

    qlf_params = {'Transition point':bestcrit, 'Pre-disk normalization':bestnorm, 'Pre-disk slope':bestslope, 'Pre-disk sigma':bestpre,\
                  'Post-disk sigma':bestpost, 'Post-disk normalization':bestlocal, 'Chi2 value':minval_tot}
    
#     print('Best fit parameter values for stacked z:')
#     pprint(qlf_params)
#     print('\n')
    
    print('Return: (1) dictionary of best fit values for stacked redshifts, (2) null grid.\n')
    
    return qlf_params, null


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


class QLF_wB():
    def __init__(self, z, bins, BN):

        if BN == 1:
            BP = [1.63126656, 10.08708065]
        elif BN == 2:
            BP = [11.29777953, 8.99873218]
        elif BN == 3:
            BP = [1.38000186, 11.90976007]
        else:
            print('Invalid choice for BN.')
        
        
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
 ############################       
        ##### bulge calculations

        BT = 1.0 / (1.0 + np.exp(-BP[0]*(self.StellBins-BP[1])))
        self.BulgeBins = np.log10(BT * 10**self.StellBins) #this is bulge mass in units of solar mass
        dBTdMstar = np.gradient(BT, 10**self.StellBins)
        dMbdMstar = dBTdMstar*10**self.StellBins+BT
        self.SMBARs = (self.SSFRs*dMbdMstar*10**self.StellBins) / 10**self.BulgeBins #this is units of per year
        
        dlogMstardlogMb = np.gradient(self.StellBins, self.BulgeBins)
        self.dNdlogMbulge = self.dNdlogMstar*dlogMstardlogMb
        self.dNdlnMbulge = self.dNdlogMbulge/np.log(10)


    def get_Mbh(self, logMb0, slope_low = 0.2, norm_from_local =4.0, norm_local = 8.2):
        norm = [11, norm_local]
        
        self.logMb0 = logMb0
        logMb = self.BulgeBins
        logMbh = logMb * 0
        slopes = logMb * 0

        logMbh0 = logMb0 + norm[1] - norm[0] - norm_from_local
        my_norm = [logMb0, logMbh0]
        post_params = [1, logMbh0 - logMb0]
        pre_params = [slope_low, my_norm[1] - my_norm[0] * slope_low]

        post = (logMb > logMb0)
        pre = (logMb <= logMb0)
        Mb = 10**logMb
        Mb0 = 10**logMb0
        Mbh0 = 10**logMbh0
        beta = post_params[0]
        alpha = 10**norm[1] / 10**(norm[0]*beta)

        Mbh = Mbh0 + alpha * Mb[post]**(beta-1) * (Mb[post] - Mb0)
        logMbh[post] = np.log10(Mbh)
        slopes[post] = alpha * beta * Mb[post]**beta / Mbh

        logMbh[pre] = logMb[pre] * pre_params[0] + pre_params[1]
        slopes[pre] = pre_params[0]

        
        self.slopes = slopes
        self.BHBins = logMbh
        self.pre = pre
        self.post = post
        ## SSFR are in yr^-1
        self.SBHARs = self.slopes * (self.SMBARs / 3.154e7) ## s^-1
        self.MdotBH = self.SBHARs * (10**self.BHBins * 2e33) ## g/s

#############################

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
        
        
        
        ###############################
        lnxsig_list = self.StellBins * 0
        lnxsig_list[self.pre] = lnxsigs[0]
        lnxsig_list[self.post] = lnxsigs[1]
        ### start transition at the M*crit value
        critpoint = np.argmin(np.abs(self.BulgeBins - self.logMb0))
        ### end transistion 0.5 dex after that value
        endtran = np.argmin(np.abs(self.BulgeBins - (self.logMb0 + 0.5)))
        lintrans = np.linspace(lnxsigs[0], lnxsigs[1], len(lnxsig_list[critpoint-1:endtran]))
        lnxsig_list[critpoint-1:endtran] = lintrans
        ################################
        


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
