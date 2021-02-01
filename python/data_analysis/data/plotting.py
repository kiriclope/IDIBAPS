from .libs import * 
from . import constants as gv 
from datetime import date

shade_alpha = 0.2
lines_alpha = 0.8

def figDir():

    gv.figdir = gv.path + '/figs' 
    gv.filedir = gv.path + '/data' 
    
    today = date.today() 
    today = today.strftime("/%y-%m-%d") 
    gv.figdir = gv.figdir + today 
        
    if gv.laser_on: 
        gv.figdir = gv.figdir + '/laser_on'
    else:
        gv.figdir = gv.figdir + '/laser_off'
        
    if gv.correct_trial :
        gv.figdir = gv.figdir + '/correct_trials'
        
    if gv.pair_trials :
        gv.figdir = gv.figdir + '/pair_trials' 
        
    if 'ND_D1' in gv.trials:
        gv.figdir = gv.figdir + '/ND_D1_ND_D2' 
    
    if gv.SYNTHETIC :
        gv.figdir = gv.figdir + '/synthetic' 
                
    if gv.trialsXepochs: 
        gv.figdir = gv.figdir + '/trialsXepochs'
        
    if gv.CONCAT_BINS: 
        gv.figdir = gv.figdir + '/concat_bins' 
        
    if gv.F0_THRESHOLD is not None: 
        gv.figdir = gv.figdir + '/F0_thresh_%.2f' % gv.F0_THRESHOLD 
        if gv.AVG_F0_TRIALS:
            gv.figdir = gv.figdir + '_avg_trials'
    elif gv.DECONVOLVE:
        gv.figdir = gv.figdir + '/deconvolve_th_%.2f' % gv.DCV_THRESHOLD
    else:
        gv.figdir = gv.figdir + '/rawF' 
        
    if gv.EDvsLD : 
        gv.figdir = gv.figdir + '/EDvsLD'
    else : 
        gv.figdir = gv.figdir + '/stimVsDelay' 
        
    if gv.T_WINDOW!=0 :
        gv.figdir = gv.figdir + '/t_window_%.1f' % gv.T_WINDOW
        
        
    elif gv.DETREND :
        gv.figdir = gv.figdir + '/detrend'  
        
    if gv.SAVGOL :
        gv.figdir = gv.figdir + '/savgol' 
        
    if gv.Z_SCORE : 
        gv.figdir = gv.figdir + '/z_score'
        
    elif gv.Z_SCORE_BL : 
        gv.figdir = gv.figdir + '/z_score_bl'
        
                
    if gv.TIBSHIRANI_TRICK:
        gv.figdir = gv.figdir + '/tibshirani_trick'
        
    if gv.FEATURE_SELECTION:
        gv.figdir = gv.figdir + '/feature_selection'

    if gv.pca_model is not None:
        
            
        gv.figdir = gv.figdir + '/dim_red/%s/%s' % (gv.pca_model, gv.pca_method)
        
        if gv.AVG_BEFORE_PCA: 
            gv.figdir = gv.figdir + '/pca_averaged_epochs' 
        else:
            gv.figdir = gv.figdir + '/pca_all_bins_epochs' 
        
        if gv.pca_model=='supervised': 
            gv.figdir = gv.figdir + '/explained_variance_%.2f/threshold_%d_Cs_%d' % (gv.explained_variance, gv.max_threshold, gv.n_thresholds ) 
        elif gv.pca_model == 'sparsePCA': 
            gv.figdir = gv.figdir + '/explained_variance_%.2f/alpha_%d_ridge_alpha_%.2f' % (gv.explained_variance, gv.sparse_alpha, gv.ridge_alpha) 
        else: 
            if gv.inflection: 
                gv.figdir = gv.figdir + '/inflection_point' 
            elif gv.minka_mle: 
                gv.figdir = gv.figdir + '/minka_mle' 
            else:            
                gv.figdir = gv.figdir + '/explained_variance_%.2f' % gv.explained_variance 
        
        if gv.ED_MD_LD :
            gv.figdir = gv.figdir + '/ED_MD_LD' 
        if gv.DELAY_ONLY:
            gv.figdir = gv.figdir + '/delay_only'
            
    if gv.pls_method is not None: 
        if isinstance(gv.pls_max_comp, str): 
            gv.figdir = gv.figdir + '/dim_red/pls/%s/max_comp_%s_cv_%.2f' % (gv.pls_method, gv.pls_max_comp, gv.pls_cv) 
        else: 
            gv.figdir = gv.figdir + '/dim_red/pls/%s/max_comp_%d_cv_%.2f' % (gv.pls_method, gv.pls_max_comp, gv.pls_cv) 
            
        if gv.ED_MD_LD : 
            gv.figdir = gv.figdir + '/ED_MD_LD' 
        if gv.DELAY_ONLY:
            gv.figdir = gv.figdir + '/delay_only'
        
    if not os.path.isdir(gv.figdir):
        os.makedirs(gv.figdir)

    print(gv.figdir)
    
def add_stim_to_plot(ax, bin_start=0):

    ax.axvspan(gv.bins_ED[0]-bin_start, gv.bins_ED[-1]-bin_start, alpha=shade_alpha, color='gray') 
    ax.axvspan(gv.bins_MD[0]-bin_start, gv.bins_MD[-1]-bin_start, alpha=shade_alpha, color='blue')    
    ax.axvspan(gv.bins_LD[0]-bin_start, gv.bins_LD[-1]-bin_start, alpha=shade_alpha, color='gray') 
    
    ax.axvline(gv.bins_DIST[0]-bin_start, alpha=lines_alpha, color='k', ls='-')
    ax.axvline(gv.bins_DIST[-1]-bin_start, alpha=lines_alpha, color='k', ls='-')

    ax.axvline(gv.bins_cue[0]-bin_start, alpha=lines_alpha, color='k', ls='--')
    ax.axvline(gv.bins_cue[-1]-bin_start, alpha=lines_alpha, color='k', ls='--')

def vlines_delay(ax):
    
    ax.axvline(gv.t_ED[0]-2, color='k', ls='--')
    ax.axvline(gv.t_ED[-1]-2, color='k', ls='--')

    ax.axvline(gv.t_MD[0]-2, color='r', ls='--')
    ax.axvline(gv.t_MD[-1]-2, color='r', ls='--')

    ax.axvline(gv.t_LD[0]-2, color='k', ls='--')
    ax.axvline(gv.t_LD[-1]-2, color='k', ls='--') 

def hlines_delay(ax):
    
    ax.axhline(gv.t_ED[0]-2, color='k', ls='--')
    ax.axhline(gv.t_ED[-1]-2, color='k', ls='--')

    ax.axhline(gv.t_MD[0]-2, color='r', ls='--')
    ax.axhline(gv.t_MD[-1]-2, color='r', ls='--')

    ax.axhline(gv.t_LD[0]-2, color='k', ls='--')
    ax.axhline(gv.t_LD[-1]-2, color='k', ls='--')

def add_orientation_legend(ax):
    custom_lines = [Line2D([0], [0], color=gv.pal[k], lw=4) for
                    k in range(len(gv.trials))]
    labels = [t for t in gv.trials]
    ax.legend(custom_lines, labels,
              frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0,0,0.9,1])    

def bar_trials_epochs(mean, lower=None, upper=None, var_name='cos_alp'):
    
    labels = np.arange(len(gv.epochs)-1) 
    width=0.25 

    figtitle = '%s_%s_bars_%s' % (gv.mouse, gv.session, var_name) 

    ax = plt.figure(figtitle).add_subplot() 
    
    for n_trial, trial in enumerate(gv.trials): 
        values = mean[n_trial][1:]
        if lower is not None: 
            error = np.absolute(np.vstack([ lower[n_trial][1:], upper[n_trial][1:] ] )) 
            ax.bar(labels + n_trial*width, values , yerr=error,  color = gv.pal[n_trial], width = width) 
        else:
            ax.bar(labels + n_trial*width, values , color = gv.pal[n_trial], width = width)  

    day = 'day %d' % (list(gv.sessions).index(gv.session) + 1 ) 
    ax.set_title(day)

    epochs = ['Early', 'Middle', 'Late'] 
    tasks = ['DPA', 'Dual Go', 'Dual NoGo'] 
    
    if gv.cos_trials:
        plt.xticks([i + width for i in range(len(gv.trials)-1)], tasks[1:]) 
        plt.xlabel('Task')
    else:
        plt.xticks([i + width for i in range(len(gv.epochs)-1)], epochs[1:]) 
        plt.xlabel('Delay')
        
    if 'cos_alp' in var_name:
        if gv.cos_trials:
            plt.ylabel('cos($\\beta_{DPA}$,$\\beta_{task}$)') 
        else: 
            plt.ylabel('cos($\\beta_{early}$,$\\beta_{delay}$)') 
    else:
        plt.ylabel('Corr')

def save_fig(figname):
    plt.figure(figname)
    if not os.path.isdir(gv.figdir):
        os.makedirs(gv.figdir)
    
    if gv.IF_SAVE:
        plt.savefig(gv.figdir + '/' + figname +'.svg',format='svg')
        print('save fig to', gv.figdir)

def save_dat(array, filename):
    if not os.path.isdir(gv.filedir):
        os.makedirs(gv.filedir)
        
    with open(gv.filedir + '/' + filename + '.pkl','wb') as f:
        pickle.dump(array, f) 
        print('saved to', gv.filedir + '/' + filename + '.pkl' )

def open_dat(filename):
    if not os.path.isdir(gv.filedir):
        os.makedirs(gv.filedir)
        
    with open(gv.filedir + '/' + filename + '.pkl','rb') as f:
        print('opening', gv.filedir + '/' + filename + '.pkl' )
        return pickle.load(f) 

def add_vlines():
    plt.axvline(gv.t_STIM[0], c='k', ls='-') # sample onset

    plt.axvline(gv.t_ED[0], c='r', ls='--') 
    plt.axvline(gv.t_ED[1], c='r', ls='--') # DPA early delay

    plt.axvline(gv.t_DIST[0], color='k', ls='-')
    plt.axvline(gv.t_DIST[1], color='k', ls='-')
    
    plt.axvline(gv.t_MD[0], c='g', ls='--') #DRT delay
    plt.axvline(gv.t_MD[1], c='g', ls='--') 
        
    plt.axvline(gv.t_LD[0], c='b', ls='--')
    plt.axvline(gv.t_LD[1], c='b', ls='--') # DPA late delay

    plt.axvline(gv.t_test[0], color='k', ls='-')
    plt.axvline(gv.t_test[1], color='k', ls='-')
    
    
def add_hlines():
    plt.axhline(gv.t_STIM[0], c='k', ls='-') # sample onset

    plt.axhline(gv.t_ED[0], c='k', ls='--') 
    plt.axhline(gv.t_ED[1], c='k', ls='--') # DPA early delay

    plt.axhline(gv.t_MD[0], c='r', ls='--') #DRT delay
    plt.axhline(gv.t_MD[1], c='r', ls='--') 
        
    plt.axhline(gv.t_LD[0], c='k', ls='--')
    plt.axhline(gv.t_LD[1], c='k', ls='--') # DPA late delay 
