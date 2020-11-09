from libs import *
sys.path.append('../../')
import data.constants as gv

shade_alpha      = 0.2
lines_alpha      = 0.8
pal = ['r','b','y']

def figDir():
    script_dir = os.path.dirname(__file__) 
    figdir = script_dir + '/figs/'
    print('saving in', figdir)

    if(clf_name == 'LogisticRegression'): 
        clf_param = '/C_%.1f_penalty_%s_solver_%s/' % (C, penalty, solver) 
        figs_dir = os.path.join(figs_dir, clf_name + clf_param) 
    else: 
        figs_dir = os.path.join(figs_dir, clf_name) 

    if gv.laser_on: 
        figs_dir = os.path.join(figs_dir, 'laser_on/') 

    if not os.path.isdir(figdir):
        os.makedirs(figdir)

    return figdir

def add_stim_to_plot(ax):
    ax.axvspan(gv.t_early_delay[0], gv.t_early_delay[1], alpha=shade_alpha,
               color='gray')

    ax.axvspan(gv.t_late_delay[0], gv.t_late_delay[1], alpha=shade_alpha,
               color='gray')
    
    ax.axvspan(gv.t_DRT_delay[0], gv.t_DRT_delay[1], alpha=shade_alpha,
               color='blue')
    
    ax.axvline(gv.t_sample[0], alpha=lines_alpha, color='gray', ls='-')
    # ax.axvline(gv.t_sample[1], alpha=lines_alpha, color='gray', ls='-')

    ax.axvline(gv.t_early_delay[0], alpha=lines_alpha, color='gray', ls='--')
    ax.axvline(gv.t_early_delay[1], alpha=lines_alpha, color='gray', ls='--')

    ax.axvline(gv.t_DRT_delay[0], alpha=lines_alpha, color='gray', ls='--')
    ax.axvline(gv.t_DRT_delay[1], alpha=lines_alpha, color='gray', ls='--')

    ax.axvline(gv.t_late_delay[0], alpha=lines_alpha, color='gray', ls='--')
    ax.axvline(gv.t_late_delay[1], alpha=lines_alpha, color='gray', ls='--')

    # ax.axvline(gv.t_distractor[0], alpha=lines_alpha, color='gray', ls='-')
    # ax.axvline(gv.t_distractor[1], alpha=lines_alpha, color='gray', ls='-')

    # ax.axvline(gv.t_cue[0], alpha=lines_alpha, color='gray', ls='--')
    # ax.axvline(gv.t_cue[1], alpha=lines_alpha, color='gray', ls='--')

def add_orientation_legend(ax):
    custom_lines = [Line2D([0], [0], color=pal[k], lw=4) for
                    k in range(len(gv.trials))]
    labels = [t for t in gv.trials]
    ax.legend(custom_lines, labels,
              frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0,0,0.9,1])    

def plot_cosine_bars(cos_alp, mean_cos=[], y_err=[], IF_SHUFFLE=0):

    if gv.laser_on:
        figtitle = '%s_%s_cos_alpha_pca_laser_on' % (gv.mouse, gv.session)
    else:
        figtitle = '%s_%s_cos_alpha_pca_laser_off' % (gv.mouse, gv.session)

    ax = plt.figure(figtitle).add_subplot()
    xticks = np.arange(0,len(gv.epochs)-1) 
    
    if IF_SHUFFLE: 
        width = 1/10. 
    else: 
        width = 2/10. 
        
    if('ND' in gv.trial): 
        ax.bar(xticks - 4/10, cos_alp[1:], width, yerr=y_err, label=gv.trial, color='r') ; 
        if IF_SHUFFLE: 
            ax.bar(xticks - 3/10, mean_cos[1:], width, yerr=y_err,color='r', alpha=0.5) ; 
                    
    if('D1' in gv.trial): 
        ax.bar(xticks - 1/10, cos_alp[1:], width, yerr=y_err, label=gv.trial, color='b') ; 
        if IF_SHUFFLE:
            ax.bar(xticks, mean_cos[1:], width, yerr=y_err, color='b', alpha=0.5) ; 
                
    if('D2' in gv.trial):
        ax.bar(xticks + 2/10 , cos_alp[1:], width, yerr=y_err, label=gv.trial, color='y') ; 
        if IF_SHUFFLE:
            ax.bar(xticks + 3/10, mean_cos[1:], width, yerr=y_err, color='y', alpha=0.5) ;
                    
    plt.ylabel('cos($\\alpha$)') 
    plt.xlabel('epochs') 
    labels = gv.epochs ; 
    ax.set_xticks(xticks) ; 
    ax.set_xticklabels(labels[1:]) ; 
                
    ax.legend()

def save_fig(figname, figdir):
    plt.figure(figname) 
    plt.savefig(figdir + figname +'.svg',format='svg') 
