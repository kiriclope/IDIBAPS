from .libs import * 
from . import constants as gv 
from datetime import date

shade_alpha = 0.2
lines_alpha = 0.8

def figDir():

    gv.figdir = gv.path + '/figs' 
    
    today = date.today() 
    today = today.strftime("/%d-%m-%y") 
    gv.figdir = gv.figdir + today 
        
    if gv.laser_on: 
        gv.figdir = gv.figdir + '/laser_on'
    else:
        gv.figdir = gv.figdir + '/laser_off'
        
    if gv.correct_trial :
        gv.figdir = gv.figdir + '/correct_trials' 
        
    if 'ND_D1' in gv.trials:
        gv.figdir = gv.figdir + '/ND_D1_ND_D2' 
    
    if gv.SYNTHETIC :
        gv.figdir = gv.figdir + '/synthetic' 
        
    if gv.pca_method is not None: 
        if gv.pca_method in 'supervised': 
            gv.figdir = gv.figdir + '/dim_red/pca/%s/explained_variance_%.2f/threshold_%d_Cs_%d' % (gv.pca_method, gv.explained_variance,
                                                                                                    gv.max_threshold, gv.n_thresholds ) 
        else:
            gv.figdir = gv.figdir + '/dim_red/pca/%s/explained_variance_%.2f' % (gv.pca_method, gv.explained_variance) 
            
        if gv.inflexion: 
            gv.figdir = gv.figdir + '/inflection_point'
            
    if gv.pls_method is not None: 
        if isinstance(gv.pls_max_comp, str): 
            gv.figdir = gv.figdir + '/dim_red/pls/%s/max_comp_%s_cv_%.2f' % (gv.pls_method, gv.pls_max_comp, gv.pls_cv) 
        else: 
            gv.figdir = gv.figdir + '/dim_red/pls/%s/max_comp_%d_cv_%.2f' % (gv.pls_method, gv.pls_max_comp, gv.pls_cv) 
            
        if gv.ED_MD_LD : 
            gv.figdir = gv.figdir + '/ED_MD_LD' 
        if gv.DELAY_ONLY:
            gv.figdir = gv.figdir + '/delay_only'

    if gv.ED_MD_LD :
        gv.figdir = gv.figdir + '/ED_MD_LD' 
    if gv.DELAY_ONLY:
        gv.figdir = gv.figdir + '/delay_only'
        
    if gv.trialsXepochs: 
        gv.figdir = gv.figdir + '/trialsXepochs' 
        
    if gv.detrend :
        gv.figdir = gv.figdir + '/detrend' 
        
    if gv.EDvsLD : 
        gv.figdir = gv.figdir + '/EDvsLD'
    else : 
        gv.figdir = gv.figdir + '/stimVsDelay' 
        
    if gv.T_WINDOW!=0 :
        gv.figdir = gv.figdir + '/t_window_%.1f' % gv.T_WINDOW 

    if gv.SAVGOL :
        gv.figdir = gv.figdir + '/savgol' 

    if gv.Z_SCORE : 
        gv.figdir = gv.figdir + '/z_score' 
        
    if gv.scaling is not None:
        gv.figdir = gv.figdir + '/%s' % gv.scaling
        
    if gv.TIBSHIRANI_TRICK:
        gv.figdir = gv.figdir + '/tibshirani_trick'

    if gv.FEATURE_SELECTION:
        gv.figdir = gv.figdir + '/feature_selection'
        
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
            error = np.array([ lower[n_trial][1:], upper[n_trial][1:] ] ) 
            ax.bar(labels + n_trial*width, values , yerr=error,  color = gv.pal[n_trial], width = width) 
        else:
            ax.bar(labels + n_trial*width, values , color = gv.pal[n_trial], width = width)  

    day = 'day %d' % (list(gv.sessions).index(gv.session) + 1 ) 
    ax.set_title(day)
    
    plt.xticks([i + width for i in range(len(gv.epochs)-1)], gv.epochs[1:]) 

    plt.xlabel('Epochs')
    if 'cos_alp' in var_name:
        plt.ylabel('Cos($\\alpha$)')
    else:
        plt.ylabel('Corr')

def save_fig(figname):
    plt.figure(figname)
    if not os.path.isdir(gv.figdir):
        os.makedirs(gv.figdir)
    
    if gv.IF_SAVE:
        plt.savefig(gv.figdir + '/' + figname +'.svg',format='svg')
        print('saved to', gv.figdir + '/' + figname + '.svg')

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

def add_vlines(figname):
    plt.figure(figname) 
    plt.ahvline(y=gv.t_STIM[0]-2, c='k', ls='-') # sample onset

    plt.ahvline(y=gv.t_ED[0]-2, c='k', ls='--') 
    plt.ahvline(y=gv.t_ED[1]-2, c='k', ls='--') # DPA early delay
    
    plt.ahvline(y=gv.t_MD[0]-2, c='r', ls='--') #DRT delay
    plt.ahvline(y=gv.t_MD[1]-2, c='r', ls='--') 
        
    plt.ahvline(y=gv.t_LD[0]-2, c='k', ls='--')
    plt.ahvline(y=gv.t_LD[1]-2, c='k', ls='--') # DPA late delay

def add_hlines(figname):
    plt.figure(figname) 
    plt.axvline(x=gv.t_STIM[0]-2, c='k', ls='-') # sample onset

    plt.axvline(x=gv.t_ED[0]-2, c='k', ls='--') 
    plt.axvline(x=gv.t_ED[1]-2, c='k', ls='--') # DPA early delay
    
    plt.axvline(x=gv.t_MD[0]-2, c='r', ls='--') #DRT delay
    plt.axvline(x=gv.t_MD[1]-2, c='r', ls='--') 
        
    plt.axvline(x=gv.t_LD[0]-2, c='k', ls='--')
    plt.axvline(x=gv.t_LD[1]-2, c='k', ls='--') # DPA late delay
