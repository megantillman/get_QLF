from functions import *
import matplotlib.pyplot as plt
import corner
import matplotlib.gridspec as gridspec
import h5py
import matplotlib





'''
_________________________________________________________________________________________________________________________________________________________

Generates a two by four plot of eta and mdot distributions for four different redshift values.

    Arguments:

        fit_params = A LIST (SIZE 5) containing the QLF fit parameters in the following order:
            siglnM - VALUE scatter on the SMHM relation
            bins - VALUE size of the bins for the numerical integrations 
            start - VALUE critical stellar mass value
            siglnX - LIST (SIZE: 2) scatter on the accretion relations (pre-disk then post-disk)
            lums - ARRAY (SIZE: ANY) bolometric luminosity values to develop the QLF for (must be in log base 10)
            
        zplot = A LIST (SIZE 4) z values to produce the distributions for
        
        name = FLOAT file name to save the pdf plot as

'''

def twoXfour_mdot_eta(fit_params = None, zplot = [0, 0, 0, 0], name = 'mdot-eta-dist.pdf'):
    
    ### what fit params are we using
    if not fit_params:
        siglnM = 0.7
        bins = 0.005
        start = 10.0
        siglnX = [3.0, 2.0]
        lums = np.linspace(5,18,200)
    else:
        siglnM, bins, start, siglnX, lums = fit_params
        
    ### axes
    fig = plt.figure(figsize=(22,8))
    ax1 = fig.add_axes([0.5, 0.5, 0.2, 0.4])
    ax2 = fig.add_axes([0.7, 0.5, 0.2, 0.4], sharey = ax1, sharex = ax1)
    ax3 = fig.add_axes([0.5, 0.1, 0.2, 0.4], sharey = ax1, sharex = ax1)
    ax4 = fig.add_axes([0.7, 0.1, 0.2, 0.4], sharey = ax1, sharex = ax1)
    ax5 = fig.add_axes([0.1, 0.5, 0.2, 0.4], sharey = ax1)
    ax6 = fig.add_axes([0.3, 0.5, 0.2, 0.4], sharey = ax5, sharex = ax5)
    ax7 = fig.add_axes([0.1, 0.1, 0.2, 0.4], sharey = ax5, sharex = ax5)
    ax8 = fig.add_axes([0.3, 0.1, 0.2, 0.4], sharey = ax5, sharex = ax5)
    axes = [[ax1, ax5], [ax2, ax6], [ax3, ax7], [ax4, ax8]]
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax5.get_xticklabels(), visible=False)
    plt.setp(ax6.get_xticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax4.get_yticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels(), visible=False)
    plt.setp(ax6.get_yticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)
    plt.setp(ax8.get_yticklabels(), visible=False)



    ### masses to plot for
    mass_l = np.array([8.0, 9.0, 10.0, 11.0, 12.0])
    mass_s = np.array([8.0, 9.0, 10.0, 11.0, 11.5])
    
    ### formatting
    color = ['r','orange','green','blue','violet']
    lw, fs, tw, tl, textx, texty = 1, 11, 1, 5, 10**-1.2, 0.175

    
    ### plotting eta and mdot for each z
    for z, ax in zip(zplot, axes):
        
        if z >= 1.0:
            mass = mass_s
        else:
            mass = mass_l
            
        qlf = QLF(z, bins)
        STELL = qlf.StellBins
        qlf.get_dNdlnMstar(siglnM)
        qlf.get_Mbh(start, approx_local=True)
        qlf.get_dNdlnL(lums, siglnX)
        
        lnMdot = qlf.Mdot_mu_sig[:,0]
        sigs = qlf.Mdot_mu_sig[:,1]
        
        lnMdotedd = qlf.Mdot_mu_sig[:,2]
        lneta = lnMdot - lnMdotedd
        lnMdot = np.log( np.e**lnMdot / (3.17098e-8 * 2e33) )
    
        ### plot eta
        for M, c in zip(mass, color):
            i = np.argmin(np.abs(STELL - M))
            x = np.linspace(-40, 1, 200)
            y = ( 1 / np.sqrt(2.0 * np.pi * sigs[ int(i) ]**2.0) ) * np.exp( - (x - lneta[ int(i) ])**2.0 / (2.0 * sigs[ int(i) ]**2) )

            ax[0].plot(np.e**x, y, color=c, lw = lw, label = r'M$_*/M_{\odot} = 10^{'+str(M)+'}$')
        
        ### some pretty stuff
        ax[0].text(textx,texty, 'z = '+str(z), fontsize = fs)
        ax[0].set_xscale('log')
        ax[0].axis([10**-10.5, np.e**x[-1], 0, max(y)+0.01])
        ax[0].legend(loc='upper left', fontsize = fs)
        ax[0].tick_params(direction='in', width = tw, length = tl, right = True)
    
        ### plot mdot
        for M, c in zip(mass, color):
            i = np.argmin(np.abs(STELL - M))
            x = np.linspace(-50, 3, 200)
            y = ( 1 / np.sqrt(2.0 * np.pi * sigs[ int(i) ]**2.0) ) * np.exp( - (x - lnMdot[ int(i) ])**2.0 / (2.0 * sigs[ int(i) ]**2) )

            ax[1].plot(np.e**x, y, color=c, lw = lw, label = r'M$_*/M_{\odot} = 10^{'+str(M)+'}$')
        
        ### more pretty stuff
        ax[1].text(textx, texty, 'z = '+str(z), fontsize = fs) 
        ax[1].set_xscale('log')
        ax[1].axis([10**-13.5, np.e**x[-1], 0, max(y)+0.01])
        ax[1].legend(loc='upper left', fontsize = fs)
        ax[1].tick_params(direction='in', width = tw, length = tl, right = True)
        
        ### axes labels
        if ax[1] in [ax7, ax8]:
            ax[1].set_xlabel(r'$\dot{M}_{BH}/M_{\odot} yr^{-1}$', fontsize = fs)
        if ax[0] in [ax3, ax4]:
            ax[0].set_xlabel(r'$\eta$', fontsize = fs)
        if ax[1] in [ax5, ax7]:
            ax[1].set_ylabel(r'Probability', fontsize = fs)  
    
    ### save
    plt.savefig('plots/paper-plots/'+name)
    print('\n Saved figure as "plots/paper-plots/'+name+'"\n')
       
'''
_________________________________________________________________________________________________________________________________________________________












_________________________________________________________________________________________________________________________________________________________

Equation used to plot the Shen QLF fit.

    Arguments: 

        z - VALUE redshift to produce the fit for
        L - ARRAY bolometric luminosity values to produce the fit for (must be in log base 10)

''' 
    
def shen_QLF(z, L):
    a0, a1, a2 = 0.85858, -0.26236, 0.02105
    b0, b1, b2 = 2*2.54992, -1.04735, 1.13277
    c0, c1, c2 = 2*13.01297, -0.57587, 0.45361
    d0, d1 = -3.53138, -0.39961
    zr = 2.0
    zfrac = (1 + z)/(1 + zr)
    g1 = C.chebval(1 + z, [a0, a1, a2])
    g2 = b0/(zfrac**b1 + zfrac **b2)
    logLs = c0/(zfrac**c1 + zfrac**c2)
    logPhis = C.chebval(z, [d0]) + C.chebval(1 + z, [0, d1])
    Lfrac = 10**L / 10**logLs
    Phibol = 10**logPhis/(Lfrac**g1 + Lfrac**g2)
    
    return np.log10(Phibol)   
    
'''
_________________________________________________________________________________________________________________________________________________________












_________________________________________________________________________________________________________________________________________________________

Generates a single QLF plot with the Shen fit overplotted.

    Arguments:

        fit_params = A LIST (SIZE 5) containing the QLF fit parameters in the following order:
            siglnM - VALUE scatter on the SMHM relation
            bins - VALUE size of the bins for the numerical integrations 
            start - VALUE critical stellar mass value
            siglnX - LIST (SIZE: 2) scatter on the accretion relations (pre-disk then post-disk)
            lums - ARRAY (SIZE: ANY) bolometric luminosity values to develop the QLF for (must be in log base 10)

        z = VALUE redshift value to produce the plot for

        name = STRING name of the pdf plot produced

        Hopkins = BOOLEAN whether or not to over plot the old shen scatter data

''' 
    
def QLFwShen(fit_params = None, z = 0.0, name = 'z0-QLF-v-Shen.pdf', Hopkins = False):
   

    ### what fit params are we using
    if not fit_params:
        siglnM = 0.7
        bins = 0.005
        start = 10.0
        siglnX = [3.0, 2.0]
        lums = np.linspace(5,18,200)
    else:
        siglnM, bins, start, siglnX, lums = fit_params
        
    ### figure set up
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    
    ### collecting QLF data
    qlf = QLF(z, bins)
    qlf.get_dNdlnMstar(siglnM)
    qlf.get_Mbh(start, approx_local=True)
    
    m = qlf.slopes
    
    qlf.get_dNdlnL(lums, siglnX)
    lumsp = 10**lums*3.8e33
    prea = np.zeros(len(lumsp))
    posta = np.zeros(len(lumsp))
    
    for i, pre, m in zip(np.transpose(qlf.intvals), qlf.pre, qlf.slopes):
        dens = i
        if pre == True:
            prea += dens
        else:
            posta += dens
    
    ### collecting Shen QLF data
    lumsshen = np.linspace(8.95,14.95,200) ## this is tenative and an approximate range of valid observational data
    xshen = 10**lumsshen*3.8e33
    dens, stanave, stanab, stanb = Shen_fit_uncer(z, lumsshen)
            
    ### plotting QLF data
    l1, = ax.plot(lumsp, np.log10(prea*np.log(10)), lw=1, c='r', linestyle='dashed', label='Pre-Disk')
    l2, = ax.plot(lumsp, np.log10(posta*np.log(10)), lw=1, c='b', linestyle='dashed', label='Post-Disk')
    
    ### plotting Hopkins data (if told to)
    if Hopkins == True:
        x,y,yerr = grab_obs(z)
        ax.errorbar(10**np.asarray(x)*3.8e33,y,yerr=yerr,markersize=1,fmt='o',c='gray',label='Hopkins+2006')
    
    ### plotting Shen data and our QLF
    ax.plot(lumsp, np.log10(qlf.dNdlnL * np.log(10)), c='k', label = 'Predicted QLF',linestyle='dashed')
    ax.plot(xshen, dens, label='Shen+submitted',c='k',lw=2)
    ax.fill_between(xshen, dens-stanab, dens+stanb, color='gray', alpha=.75)
    ax.axvline(xshen[0],c='k',linestyle='dotted')
    ax.axvline(xshen[-1],c='k',linestyle='dotted')
    
    ### formatting and save
    ax.axis([10**6*3.8e33,10**18*3.8e33,-10,0])
    ax.set_xlabel(r'$L_{bol} (erg\ s^{-1})$', fontsize=14)
    ax.set_ylabel(r'$\log_{10} \Phi (Mpc^{-3} \log_{10} [L_{bol}]^{-1})$', fontsize =14)
    ax.text(10**7.5*3.8e33,-1.5,'z = '+str(z),fontsize = 14)
    ax.set_xscale('log')
    ax.tick_params(axis='both', which='both', labelsize=14, direction='in')
    ax.legend()
    
    plt.savefig('plots/paper-plots/'+name)
    print('\n Saved figure as "plots/paper-plots/'+name+'"\n')
    
'''
_________________________________________________________________________________________________________________________________________________________












_________________________________________________________________________________________________________________________________________________________

Generates 9 QLF plot with the Shen fit overplotted.

    Arguments:

        fit_params = A LIST (SIZE 5) containing the QLF fit parameters in the following order:
            siglnM - VALUE scatter on the SMHM relation
            bins - VALUE size of the bins for the numerical integrations 
            start - VALUE critical stellar mass value
            siglnX - LIST (SIZE: 2) scatter on the accretion relations (pre-disk then post-disk)
            lums - ARRAY (SIZE: ANY) bolometric luminosity values to develop the QLF for (must be in log base 10)

        z = A LIST (SIZE 9) redshift values to produce the plots for... main redshift value should go first

        name = STRING name of the pdf plot produced

        Hopkins = BOOLEAN whether or not to over plot the old shen scatter data

'''     
        
def QLF9wShen(fit_params = None, z = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], name = '9-QLF-v-Shen.pdf', Hopkins = False):
    
    ### what fit params are we using
    if not fit_params:
        siglnM = 0.7
        bins = 0.005
        start = 10.0
        siglnX = [3.0, 2.0]
        lums = np.linspace(5,18,200)
    else:
        siglnM, bins, start, siglnX, lums = fit_params
    
    ### figure set-up
    fig = plt.figure(figsize=(30,8))
    gs = gridspec.GridSpec(2, 6)
    ax1 = fig.add_subplot(gs[0:, 0:2])
    
    ### begin plotting the main plot
    
    ### collect QLF data
    qlf = QLF(z[0], bins)
    qlf.get_dNdlnMstar(siglnM)
    qlf.get_Mbh(start, approx_local=True)
    qlf.get_dNdlnL(lums, siglnX)
    lumsp = 10**lums*3.8e33
    prea = np.zeros(len(lumsp))
    posta = np.zeros(len(lumsp))
    
    for dens, pre, m in zip(np.transpose(qlf.intvals), qlf.pre, qlf.slopes):
        if pre == True:
            prea += dens
        else:
            posta += dens

    ### plot main plot QLF data    
    ax1.plot(lumsp, np.log10(np.asarray(prea) * np.log(10)), c='r', label='Pre-Disk',lw=1, linestyle='dashed')
    ax1.plot(lumsp, np.log10(np.asarray(posta) * np.log(10)), c='b', label='Post-Disk',lw=1, linestyle='dashed')
    
    ax1.plot(lumsp, np.log10(qlf.dNdlnL * np.log(10)), c='k',lw=2, label='Predicted QLF', linestyle='dashed')

    ### plot the Shen QLF
    lumsshen = np.linspace(8.95,14.95,200) ## this is tenative and an approximate range of valid observational data
    xshen = 10**lumsshen*3.8e33
    dens, stanave, stanab, stanb = Shen_fit_uncer(z[0], lumsshen)
    
    ax1.plot(xshen, dens, label='Shen+submitted',c='black',linestyle='solid',lw = 2)
    ax1.fill_between(xshen, dens-stanab, dens+stanb, color='gray', alpha=.75)
    ax1.axvline(xshen[0],c='k',linestyle='dotted')
    ax1.axvline(xshen[-1],c='k',linestyle='dotted')

    ### plotting Hopkins data (if told to)
    if Hopkins == True:
        x,y,yerr = grab_obs(z[0])
        ax.errorbar(10**np.asarray(x)*3.8e33,y,yerr=yerr,markersize=1,fmt='o',c='gray',label='Hopkins+2006')

    ### formatting axes
    ax1.axis([10**6*3.8e33,10**18*3.8e33,-10,0])
    ax1.set_xlabel(r'$\log_{10} [L_{bol}] \ \ \ (erg \ s^{-1})$', fontsize=18)
    ax1.set_ylabel(r'$\log_{10} \Phi (Mpc^{-3} \log_{10} [L_{bol}]^{-1})$', fontsize =18)
    ax1.legend(fontsize = 20)
    ax1.text(10**7.5*3.8e33,-1.5,'z = '+str(z[0]),fontsize = 14)
    ax1.set_xscale('log')
    ax1.tick_params(axis='both', which='both', labelsize=14, direction='in')


    for r, i, j in zip(z[1:],[0,0,0,0,1,1,1,1],[2,3,4,5,2,3,4,5]):
        
        ### set up plot
        ax = fig.add_subplot(gs[i, j], sharex = ax1, sharey = ax1)
        
        ### get the QLF data
        qlf = QLF(r, bins)
        qlf.get_dNdlnMstar(siglnM)
        qlf.get_Mbh(start, approx_local=True)
        qlf.get_dNdlnL(lums, siglnX)
    
        prea = np.zeros(len(lumsp))
        posta = np.zeros(len(lumsp))
        latea = np.zeros(len(lumsp))
        for dens, pre, m in zip(np.transpose(qlf.intvals), qlf.pre, qlf.slopes):
            if pre == True:
                prea += dens
            else:
                posta += dens
        
        ### plot mini QLF data                    
        ax.plot(lumsp, np.log10(np.asarray(prea) * np.log(10)), c='r', label='Pre-Disk',lw=.5, linestyle='dashed')
        ax.plot(lumsp, np.log10(np.asarray(posta) * np.log(10)), c='b', label='Post-Disk',lw=.5, linestyle='dashed') 
        
        ax.plot(lumsp, np.log10(qlf.dNdlnL * np.log(10)), c='k',lw=1,linestyle='dashed')

        ### plot the Shen QLF
        dens, stanave, stanab, stanb = Shen_fit_uncer(r, lumsshen)

        ax.plot(xshen, dens, label='Shen+submitted',c='black',linestyle='solid',lw =1)
        ax.fill_between(xshen, dens-stanab, dens+stanb, color='gray', alpha=.75)
        ax.axvline(xshen[0],c='k',linestyle='dotted')
        ax.axvline(xshen[-1],c='k',linestyle='dotted')
        
                
        ### plotting Hopkins data (if told to)
        if Hopkins == True:
            x,y,yerr = grab_obs(r)
            ax.errorbar(10**np.asarray(x)*3.8e33,y,yerr=yerr,markersize=1,fmt='o',c='gray',label='Hopkins+2006')
    
        ### formatting axes
        ax.axis([10**6*3.8e33,10**18*3.8e33,-10,0])
        ax.text(10**9.5*3.8e33,-1.5,'z = '+str(r),fontsize = 12)
        ax.set_xscale('log')
        ax.tick_params(axis='both', which='both', labelsize=12, direction='in')
        
    plt.tight_layout()
    plt.savefig('plots/paper-plots/'+name)
    print('\n Saved figure as "plots/paper-plots/'+name+'"\n')
    
'''
_________________________________________________________________________________________________________________________________________________________












_________________________________________________________________________________________________________________________________________________________

Generates a plot displaying the effects of free parameters on the overall shape of the QLF.

    Arguments:

        fit_params = A LIST (SIZE 5) containing the QLF fit parameters in the following order:
            siglnM - VALUE scatter on the SMHM relation
            bins - VALUE size of the bins for the numerical integrations 
            start - VALUE critical stellar mass value
            siglnX - LIST (SIZE: 2) scatter on the accretion relations (pre-disk then post-disk)
            lums - ARRAY (SIZE: ANY) bolometric luminosity values to develop the QLF for (must be in log base 10)

        z = VALUE redshift value to produce the plot for

        name = STRING name of the pdf plot produced

        Hopkins = BOOLEAN whether or not to over plot the old shen scatter data

''' 

def free_param_effects(fit_params = None, z = 0.0, name = 'free_param_effects_v3.pdf', Hopkins = False):
    ### setting up figure
    fig = plt.figure(figsize=(22,6))
    gs = gridspec.GridSpec(1, 3)
    gs.update(wspace=0.01)
    
    ### setting up axes
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharex = ax0, sharey = ax0)
    ax2 = fig.add_subplot(gs[0, 2], sharex = ax0, sharey = ax0)
    
    ### defining some lists to loop over later
    lines = ['dashdot','dashed','dotted',(0,(5, 10))]
    colors = ['b', 'r', 'b']
    params = {'crit':1.0, 'xsig1':1.0, 'xsig2':0.5}
    labels = [[r'$M_{*crit} = 10^{', '}$ M$_{\odot}$'], [r'pre-disk $\sigma_{\ln \rmX} = $', ' dex'], \
             [r'post-disk $\sigma_{\ln \rmX} = $', ' dex']]
    axes = [ax0, ax1, ax2]
    
    ### defining function to make retreieveing different qlf data easier
    def qlf_data(crit, xsig1, xsig2):
        qlf.get_Mbh(crit, approx_local=True)
        qlf.get_dNdlnL(lums, [xsig1, xsig2])
    
    ### defining different luminosity arrays for retrieveing data and plotting
    lums = np.linspace(5,18,200)
    lumsshen = np.linspace(8.95,14.95,200)
    lumsp = 10**lums*3.8e33
    lumss = 10**lumsshen*3.8e33

    ### retrieveing the QLF fit parameters
    if not fit_params:
        siglnM = 0.7
        bins = 0.005
        start = 10.0
        siglnX = [3.0, 2.0]
        lums = np.linspace(5,18,200)
    else:
        siglnM, bins, start, siglnX, lums = fit_params
       
    ### initialize a QLF and get Hopkins obs data
    qlf = QLF(z, bins)
    qlf.get_dNdlnMstar(siglnM)
    x, y , yerr = grab_obs(z)

    ### loop over different parameters
    for param, ax, label, c in zip(params, axes, labels, colors):
        
        ### plot three different variations of the different parameters
        for i, ls in zip([-1, 0, 1], lines):
            
            ### redefine values and retreive data
            p = {'crit':start, 'xsig1':siglnX[0], 'xsig2':siglnX[1]}
            p[param] += i*params[param]
            
            qlf_data(**p)

            ### find the late and early lines
            xm, ym = lumsp, np.log10(qlf.dNdlnL * np.log(10))
            totearly = []
            totlate = []
            for i in qlf.intvals:
                totearly.append(np.sum(i[qlf.pre]))
                totlate.append(np.sum(i[qlf.post]))

            ### plot
            ax.plot(xm, ym, c='k',lw=1, label=label[0]+str(p[param])+label[1], ls=ls)
            if c == 'r':
                ax.plot(xm, np.log10(np.asarray(totearly) * np.log(10)), c=c,lw=1, ls=ls) 
            else:
                ax.plot(xm, np.log10(np.asarray(totlate) * np.log(10)), c=c,lw=1, ls=ls) 
            
    ### retrieve Shen data
    dens, stanave, stanab, stanb = Shen_fit_uncer(z, lumsshen)
    
    ### plot Shen data on all axes, and maybe the Hopkins
    for ax in [ax0, ax1, ax2]:
        ax.plot(lumss, dens, label='Shen+submitted',c='black',linestyle='solid',lw = 2)
        ax.fill_between(lumss, dens-stanab, dens+stanb, color='gray', alpha=.75)
        if Hopkins == True:
            ax.errorbar(x, y, yerr = yerr, fmt = 'o', markersize = .15, c='gray', label = 'observed')
        
        ### make plots look better
        ax.legend(fontsize=14)
        ax.set_xlabel(r'$\log_{10} [L_{bol}/L_{\odot}]$', fontsize=18)
        ax.axis([min(xm),max(xm),-11,-2])
        ax.set_xscale('log')
        ax.tick_params(axis='both', which='both', direction='in')
    
    ### one y label since they share axes
    ax0.set_ylabel(r'$\log_{10} \Phi (Mpc^{-3} \log_{10} [L_{bol}]^{-1})$', fontsize =18)

    ### make plots look nicer and save plot
    plt.setp(ax1.get_yticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.tight_layout
    plt.savefig('plots/paper-plots/'+name)
    print('\n Saved figure as "plots/paper-plots/'+name+'"\n')