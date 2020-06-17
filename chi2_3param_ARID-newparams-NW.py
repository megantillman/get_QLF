from functions_newparams import *
import h5py
import itertools
import numpy as np
import scipy as sp
import scipy.stats
from numpy.polynomial import chebyshev as C
import timeit
import warnings

warnings.filterwarnings("ignore")

#### this collects the arid duty data
duty_arid = open("plot_data/fduty_vs_z.dat",'r')
all_gal, zl, zh, ml, mh, per, el, eh = [], [], [], [], [], [], [], []
### read through file
for i in duty_arid.readlines():
    s = i.split()
    if s[0] == 'All':
        zl.append(float(s[1])), zh.append(float(s[2])), ml.append(float(s[3])), mh.append(float(s[4]))
        per.append(float(s[5])), el.append(float(s[6])), eh.append(float(s[7].split("\n")[0]))
### collect the error values
yerr = np.zeros([2,len(el)])
yerr[0,:], yerr[1,:] = el, eh
### set up mass bins and list of z values to evaluate
mass = [8.5,9.0,9.5,10.0,10.5,11.0]
ztot = sorted(set((np.array(zl) + np.array(zh))/2))
### set up duty arid data to compare
DUTY_ARID = np.zeros((len(mass),len(ztot)))
DUTY_ARID_errup = np.zeros((len(mass),len(ztot)))
DUTY_ARID_errdown = np.zeros((len(mass),len(ztot)))
aveETA_ARID = np.zeros((len(mass)-2, len(ztot)))
aveETA_ARID_errup = np.zeros((len(mass)-2, len(ztot)))
aveETA_ARID_errdown =  np.zeros((len(mass)-2, len(ztot)))
###
### collect duty for compare
for m, i in zip(mass, range(len(mass))):
    ind = np.where(np.array(ml) == m)[0]
    DUTY_ARID[i,0:len(ind)] = np.array(per)[ind]
    DUTY_ARID_errup[i,0:len(ind)] = yerr[1,ind]
    DUTY_ARID_errdown[i,0:len(ind)] = yerr[0,ind]
###
### collect ave eta for compare
colors = ['teal', 'gold', 'brown', 'r']
for i, c in zip(range(len(mass)-2), colors):
    x = []
    y = []
    yerrup = []
    yerrdown = []
    duty_arid = open("plot_data/pledd_all_extracted.dat",'r')
    for line in duty_arid.readlines():
        s = line.split()
        if s[0][0] != '#' and s[0] == c:
            x.append(float(s[1]))
            y.append(float(s[2]))
            yerrup.append(float(s[3])-float(s[2]))
            yerrdown.append(float(s[2])-float(s[4].split("\n")[0]))
    aveETA_ARID[i,0:len(x)] = np.array(y)
    aveETA_ARID_errup[i,0:len(x)] = np.array(yerrup)
    aveETA_ARID_errdown[i,0:len(x)] = np.array(yerrdown)
    duty_arid.close()  

DUTY_ARID = np.log10(DUTY_ARID)
DUTY_ARID_errup = np.log10(DUTY_ARID_errup)
DUTY_ARID_errdown = np.log10(DUTY_ARID_errdown)

criteria = np.log(0.01)


def get_chi2(ym, ya, err_abv, err_blw):
    abv = (ym[ym > ya] - ya[ym > ya])**2# / err_abv[ym > ya]**2
    blw = (ym[ym < ya] - ya[ym < ya])**2# / err_blw[ym < ya]**2
    abv[np.where((abv>80.))] = 80.
    blw[np.where((blw>80.))] = 80.
    return np.sum(abv) + np.sum(blw)


def partial_chi2(combo):
    pre, slope, norm = combo[1], combo[2], combo[0]
    qlf.get_Mbh(logMstar0, slope, norm, approx_local=True)
    qlf.get_dNdlnL(L, [pre, xsigpost])
    
    DUTY = np.zeros(len(mass))
    aveETA = np.zeros((len(mass)-2))

    MdotBH = qlf.MdotBH
    MU_MdotBH = qlf.Mdot_mu_sig[:,0]
    SIG_MdotBH = qlf.Mdot_mu_sig[:,1]
    lnMs = qlf.StellBins
    dNdlnMstar = qlf.dNdlnMstar

    lnLbol = np.log(10**L*3.83e33) #log of erg/s
    MU_lnLbol = MU_MdotBH + np.log(0.1 * (2.99e10)**2) #log of erg/s
    
    def dut_eta(Ms, ind):
        MU_lnlambda = MU_lnLbol[ind] - np.log(1.3e38*0.002*(10**Ms)) #log dimensionless
        lnlambda = lnLbol - np.log(1.3e38*0.002*(10**Ms))

        y = ( 1/np.sqrt(2.0 * np.pi * SIG_MdotBH[ind]**2.0) ) * np.exp( -(lnlambda - MU_lnlambda)**2.0 / (2.0 * SIG_MdotBH[ind]**2) )
        duty = np.trapz(y[lnlambda>=criteria]*dNdlnMstar[ind], x=lnlambda[lnlambda>=criteria])
        aveeta = np.sum(y[lnlambda>=criteria]*np.e**lnlambda[lnlambda>=criteria]*(lnlambda[1]-lnlambda[0])*dNdlnMstar[ind])

        return duty, aveeta

    Mcount = 0
    for Ms in mass:
        inds = np.where((lnMs > Ms) & (lnMs < Ms+0.5))[0]
        lnMstar = np.trapz(dNdlnMstar[inds], x=lnMs[inds])
        h = (lnMs[inds[-1]] - lnMs[inds[0]]) / len(inds)
        for i in inds:
            dut, eta = dut_eta(lnMs[i], i)
            DUTY[Mcount] += dut*h
            if Mcount >= 2:
                aveETA[Mcount-2] += eta*h
        if Mcount >= 2:
            aveETA[Mcount-2] = aveETA[Mcount-2]/DUTY[Mcount]
        DUTY[Mcount] = DUTY[Mcount]/lnMstar
        Mcount += 1

#     print('\n \t\t\t before Duty Cycle:', DUTY,'\n')
    DUTY = np.log10(DUTY*100)
    DUTY[DUTY==-np.inf] = 0
#     print('\n \t\t\t after Duty Cycle:', DUTY,'\n')
#     print('\n \t\t\t before Eta:', aveETA,'\n')
    aveETA = np.log10(aveETA)
    aveETA[np.isnan(aveETA)] = 0
#     print('\n \t\t\t after Eta:', aveETA,'\n')
    
    chi2 = 0
    ym, ya = DUTY[dutyinds], DUTY_ARID[:,zcount][dutyinds]
    err_abv, err_blw = DUTY_ARID_errup[:,zcount][dutyinds], DUTY_ARID_errdown[:,zcount][dutyinds]
    chi2_dut = get_chi2(ym, ya, err_abv, err_blw)
    chi2 += chi2_dut
#     print('Duty model v actual: ',ym,'\n', ya)
#     print('Duty chi2: ',chi2_dut)
    
    ym, ya = aveETA[etainds], aveETA_ARID[:,zcount][etainds]
    err_abv, err_blw = aveETA_ARID_errup[:,zcount][etainds], aveETA_ARID_errdown[:,zcount][etainds]
    chi2_eta = get_chi2(ym, ya, err_abv, err_blw)
    chi2 += chi2_eta
#     print('Eta model v actual: ',ym,'\n',ya)
#     print('Eta chi2: ', chi2_eta)
    
#     for valm, vala, err_a, err_b, m in zip(ym, ya, err_abv, err_blw, np.asarray(mass)[etainds]):
#         indm_chi2_list.append(get_chi2(valm, vala, err_a, err_b))
        
        
    return chi2

#set the M*crit and post-disk sig values to be constant
reso = 30
b = 0.005
SIG_lnMs = 0.7
L = np.linspace(5,18,100)
logMstar0 = 10.3
xsigpost = 2.3
xsigpre = np.linspace(1.0,10.0,reso)
slopes = np.linspace(0.0,1.5,reso)
norms = np.linspace(0.0,3.0,reso)
combos = np.array(list(itertools.product(slopes, xsigpre, norms)))

filename = "output/chi2_3pARIDfit_"+str(reso)+"_newparams_NW.h5py"



f = h5py.File(filename, "w")

f.attrs.modify('resolution', reso)
dset = f.create_dataset('logMstar0', data = np.asarray(logMstar0))
dset = f.create_dataset('siglnX2', data = np.asarray(xsigpost))
dset = f.create_dataset('norm_from_local', data = norms)
dset = f.create_dataset('siglnX1', data = xsigpre)
dset = f.create_dataset('slope_low', data = slopes)

f.close()


#### loop through z values first because it saves time
print('Collected the ARID data, begining iterations of best fit parameters.')
print('This may take a while...')

chi2_matrix = np.zeros((reso, reso, reso))
zcount = 0
start = timeit.default_timer()
for z in ztot:
    start2 = timeit.default_timer() 
    
    print('Begin calculations for redshift = '+str(z))
    qlf = QLF(z, b)
    qlf.get_dNdlnMstar(SIG_lnMs)
    dutyinds = np.where((DUTY_ARID[:,zcount] != -np.inf))
    etainds = np.where((aveETA_ARID[:,zcount] != -np.inf))
    chi2_martix_part = np.apply_along_axis(partial_chi2, 1, combos).reshape(reso,reso,reso)
    chi2_matrix += chi2_martix_part
    
    
    print('Saving data to output file...')
    f = h5py.File(filename, "a")
    grp = f.create_group("z="+str(z))
    dset = grp.create_dataset('chi23d_grid', data = chi2_martix_part)
    f.close()
    
    stop2 = timeit.default_timer()
    print('Dont with calculation in time: ', stop2 - start2)
    
    zcount += 1


stop = timeit.default_timer()
print('Completed all calculations after time: ', stop - start )
 
print('Saving data to output file...')
f = h5py.File(filename, "a")
dset = f.create_dataset('chi23d_grid', data = chi2_matrix)
f.close()
print('Complete!')
