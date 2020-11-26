from .libs import * 
from . import constants as gv 

shade_alpha = 0.2
lines_alpha = 0.8
pal = ['r','b','y']

def figDir():
    gv.figdir = '/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/python/data_analysis/figs/last/'

    if gv.trialsXepochs:
        gv.figdir = gv.figdir + '/trialsXepochs' 
    
    if gv.DELAY_ONLY:
        gv.figdir = gv.figdir + '/delay_only'    
    
    if gv.laser_on: 
        gv.figdir = gv.figdir + '/laser_on'
    else:
        gv.figdir = gv.figdir + '/laser_off'

    if gv.detrend :
        gv.figdir = gv.figdir + '/detrend' 

    if gv.IF_PCA:
        if gv.pca_concat:
            gv.figdir = gv.figdir + '/dim_red/pca/concat/explained_variance%.2f' % gv.explained_variance 
        else: 
            gv.figdir = gv.figdir + '/dim_red/pca/hybrid/explained_variance_%.2f' % gv.explained_variance

    if gv.ED_MD_LD :
        gv.figdir = gv.figdir + '/ED_MD_LD' 
    elif gv.EDvsLD : 
        gv.figdir = gv.figdir + '/EDvsLD/' 
    else : 
        gv.figdir = gv.figdir + '/stimVsDelay/' 
        
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
    custom_lines = [Line2D([0], [0], color=pal[k], lw=4) for
                    k in range(len(gv.trials))]
    labels = [t for t in gv.trials]
    ax.legend(custom_lines, labels,
              frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0,0,0.9,1])    

def barCosAlp(mean, lower=None, upper=None):

        
    labels = np.arange(len(gv.epochs)-1) 
    width=0.25

    figtitle = '%s_%s_cos_alpha' % (gv.mouse, gv.session)

    ax = plt.figure(figtitle).add_subplot() 
    
    for n_trial, trial in enumerate(gv.trials): 
        values = mean[n_trial][1:]
        # if any(lower==None) :
        #     ax.bar(labels + n_trial*width, values , color = pal[n_trial], width = width) 
        # else:
        error = np.array([ lower[n_trial][1:], upper[n_trial][1:] ] ) 
        ax.bar(labels + n_trial*width, values , yerr=error,  color = pal[n_trial], width = width) 
            
    plt.xticks([i + width for i in range(len(gv.epochs)-1)], gv.epochs[1:]) 

    plt.xlabel('Epochs') 
    plt.ylabel('Cos($\\alpha$)') 

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

