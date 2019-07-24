# get_QLF
Northwestern CIERA QLF project

obs_collect.dat: collection of observational data points collected in Hopkins et al. 2006.

smhm_params.txt: originally smhm_true_med_params.txt holds information to be used in converting from halo mass to stellar and vice versa.

ssfrs: folder containing all the ssfr files needed for calculations in constructing your qlf.

get_QLF_notebook.ipynb: contains my messings with the qlf function

functions_pre-uni-model.py: holds functions and class needed to construct the QLF
    ---pre-universal accretion rate model QLF--- (Note: The documentation here is largely unnecissary for those wishing to just use the get_QLF function for which seperate documentation explains important technical aspects. This seperate documentation is found in the get_QLF user guide. The documentation below remains for those interested in the interworkings of get_QLF QLF class code etc.)
    ##I need to go back later and describe for each function what self.variables are defined and thus stored.
    ##I also need to be more rigorous about the object types (int, float, etc.) and make everything usable with a list when an array is required.
    ##Honestly can probably get rid of the pre-universal model later but this is for me
    
    Required packages: numpy, scipy, glob, numexpr, colossus
        -these are imported with the user imports functions_pre-uni-model.py
        
    Additional data: some other data is required for the calculations as is stored when the user imports functions_pre-uni-model.py
        -<SSFR> data is stored for later use from the folder ssfrs
        -z-parameter information regarding the stellar-halo mass relation is stored from the file smhm_params.txt
        
        functions:
            create_ranges_numexpr(start, stop, N):
                 Creates an array which consists of np.linspace like ranges constructed for the input values.
                 - start (list or array): starting points of the ranges
                 - stop (list or array): stopping points of the ranges
                 - N (int): number of bins for each array
                 returns: 2D array consisting of the ranges constructed 
                 
            grab_obs(redshift):
                Grabs observational data from Hopkins et al. 2006. Used to compare to the developed QLF.
                - redshift (int or float): redshift value for which you want to grab observational data for
                    available redshift --- 0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0
                returns: x-values, y-values, y-errors
                    x-values = log_{10}[L_bol/L_solar]  (solar L_bol==3.9x10^33 erg/s adopted here) (list)
                    y-values = log_{10}[dphi(L_bol)/dlog(L_bol)]   [Mpc^-3 log(L_bol)^{-1}] (list)
                    y-errors = +/- 1sigma uncertainty in log_{10}[dphi(L_bol)/dlog(L_bol)] above (list)
                    
        QLF class functions:
            __init__(self, z, bin_num):
                Class init function requires the initial input of a redshift and bin number. The init function sets up necissary references for stellar-halo mass conversions, and creates population ranges. 
                - z (int or float): redshift value for which you want to do calculations for.
                - bin_num (int): number of bins to integrate over in calculations (in general the more bins the better but it does begin to impact performance and no longer provides much benifit past a point)
                
            get_zparams(self):
                Calculates and stores the paramters used to convert between halo and stellar mass etc. and is dependant on the class' redshift. This function is run in the class' __init__() function.
                
            get_slope(self, Mhalo):
                Returns the slope(s) for given halo mass value(s). Model is from Universe Machine.
                - Mhalo (int, float, or array - log_{10}[M_H/M_solar]): halo mass value(s) for which you want the slope(s) for
                returns: slope(s) 
                    slope(s) = dln[M_*]/dln[M_H] (float or array depending on input)
                
                
            get_Mstar(self, Mhalo):
                Returns the stellar mass value(s) associated with given halo mass value(s). Model is from Universe Machine.
                - Mhalo (int, float, or array - log_{10}[M_H/M_solar]): halo mass(es) for which you want the stellar mass(es) for
                returns: Mstar(s)
                    Mstar(s) = log_{10}[M_*/M_solar] (float or array depending on input)
                    
            get_Mhalo(self, Mstar):
                Returns the halo mass value(s) associated with given stellar mass value(s). Model is from Universe Machine.
                - Mstar (int, float, or array - log_{10}[M_*/M_solar]): stellar mass(es) for which you want the halo mass(es) for
                returns: Mhalo(s)
                    Mhalo(s) = log_{10}[M_H/M_solar] (float or array depending on input)
                    
            get_SMBM(self, dM, Mmid = 10.3, slope1 = 0.2, slope3 = 1.):
                Calculates values associated with the stellar-black hole mass relation based on the "kink" model.
                - dM (int, float - log_{10}[M_*/M_solar]): displacement value used to determine the range of the growth regime
                - Mmid (int, float - log_{10}[M_*/M_solar]): value on which the growth regime is centered
                - slope1 (int, float - dln[M_BH]/dln[M_*]): slope of the early regime
                - slope2 (int, float - dln[M_BH]/dln[M_*]): slope of the late regime
            
            gauss_array(self, vals, std, amp):
                Returns the y-values of a normal distrbution based on the values inputted.
                - vals (array, list): vals[:-1] are mean values for the distribution, vals[-1] is the x-value for the distribution
                - std (float, int): standard deviation or sigma value of the distribution
                - amp (float, int): amplitude of the distribution
                returns: y
                    y = y-values of the normal distribtion for the given x-value for a range of means a set sigma and set amplitude (array)
                    
            convolve_smhm(self, StellBins, sigma):
                Calculates the Stellar Mass Function for a given range of stellar masses, redshift, and set scatter on the stellar-halo mass relation.
                - StellBins (array - log_{10}[M_*/M_solar]): stellar mass for which the SMF will be calculated for
                - sigma (int, float): scatter on the stellar-halo mass relation (sigma_{log_{10}M_*})
                returns: dN/dlog_10{M_*/M_solar}
            
            get_dNdMstar(self, smhm_scat):
                Retrieves the SMF. If the scatter is zero a direct conversion from dN/dlog_10{M_H/M_solar} is conducted by multiplying by the slope dln[M_*]/dln[M_H]. If the scatter is a value other than zero the SMF is convolved using convolve_smhm() in which sigma = smhm_scat.
                - smhm_scat (int, float): scatter on the stellar-halo mass relation (sigma_{log_{10}M_*})
                
            get_dNdMbh(self):
                Calculates the Black Hole Mass Function from the class' SMF.
                
            etas(self, Mbh):
                Calculates Eddington ratio (eta) values for given black hole mass value(s) based on the class' luminosity bins. Using an equation from Conroy & White 2011.
                - Mbh (int, float - log_{10}[M_BH/M_solar])
                returns: eta
                    eta = log_{10}[eta] values (array)
                    
             get_mean_etas(self, vals):
                 Gets a <eta> value using the input values, redshift, and collection of <SSFR> from Universe Machine. This function requires information from files that is stored when the user imports the functions_pre-uni-model.py code. 
                 - vals (array, list):
                     vals[0] = log_{10}[M_BH/M_solar] associated with the stellar mass
                     vals[1] = log_{10}[M_*/M_solar]
                     vals[2] = dln[M_BH]/dln[M_*] associated with the stellar mass
                     vals[3] = sigma_{log_{10}X} associated with the stellar mass X = <Mdot_BH>/Mdot_BH
                 returns: eta, sigma
                     eta = log_{10}[<eta>] (float)
                     sigma = sigma_{log_{10}[eta]} (float)
                    
              gauss(self, x, *var):
                  Returns the y-value(s) associated with the given x-value(s) for a normal distribution of the given parameters.
                  - x (int, float, array): x-value(s) for which the corresponding y-value(s) are sought
                  - *var = mean, std, amp (ints or floats): parameters for the normal distribution - mean, standard deviation, and amplitude
                  returns: y
                      y = the y-value(s) for given x of the normal distribution (float or array depending on input objects)
                      
              prob_eddratios(self, vals):
                  Returns a probability density from a normal distribution based on input values.
                  - vals (array, list):
                      vals[:-2] = x-value(s) of the normal distribution
                      vals[-2] = mean value of the normal distribution
                      vals[-1] = standard deviation of the normal distribution
                  returns: probdens
                      probdens = the y-value(s) for given x of the normal distribution (float or array depending on input objects)
              
              get_dNdL(self, xsigs, dX, obscured):
                  Calculates the QLF from the class' information.
                  - xsigs (list): list containing the two sigma_{log_{10}X} values, one for pre-disk and one for post-disk
                  - dX (int, float - log_{10}[M_*/M_solar]): width of the linear transition region between sigma_{log_{10}X} values
                  - obscured (float - ranging between 0 and 1): fraction of obscured AGN
                 