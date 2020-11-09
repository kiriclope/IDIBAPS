from libs import * 

sys.path.insert(1, '/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/python/data_analysis') 

import data.constants as gv 
from plotting import *

# import the animation and the HTML module to create and render the animation
from matplotlib import animation 
from IPython.display import HTML

def animated_scatter(X_proj):

    # smooth the single projected trials 
    for i in range(X_proj.shape[0]): # trial type
        for j in range(X_proj.shape[1]): # sample 
            for k in range(X_proj.shape[2]): # trials 
                for pc in range(X_proj.shape[3]): # pcs
                    X_proj[i,j,k,pc] = gaussian_filter1d(X_proj[i,j,k,pc],sigma=3)
        
    subspace = (0,1) # pick the subspace given by the second and third components 
    
    # set up the figure
    fig, ax = plt.subplots(1, 1, figsize=[6, 6]); plt.close()
    ax.set_xlim((-50, 50))
    ax.set_ylim((-50, 50))
    ax.set_xlabel('PC 1')
    # ax.set_xticks([-20, 0, 20])
    # ax.set_yticks([-20, 0, 20])
    ax.set_ylabel('PC 2')
    sns.despine(fig=fig, top=True, right=True)

    # generate empty scatter plot to be filled by data at every time point
    scatters = []
    for i, sample in enumerate(gv.samples):    
        scatter, = ax.plot([], [], 'o', lw=2, color=pal[i]);
        scatters.append(scatter)

    # red dot to indicate when stimulus is being presented
    stimdot, = ax.plot([], [], 'o', c='r', markersize=35, alpha=0.5)

    # annotate with stimulus and time information
    text = ax.text(30, 30, 'Stimulus OFF \nt = {:.2f}'.format(gv.time[0]), fontdict={'fontsize':14}) 

    # this is the function to be called at every animation frame
    def animate(t_bin):
        for i, sample in enumerate(gv.samples): 
            # find the x and y position of all trials of a given sample
            x = X_proj[1,i,:,subspace[0], t_bin]
            y = X_proj[1,i,:,subspace[1], t_bin]
            # update the scatter
            scatters[i].set_data(x, y) 

        # update stimulus and time annotation
        if ((t_bin > gv.bins_stim[0]) and (t_bin < gv.bins_stim[-1])):
            stimdot.set_data(30, 40)
            text.set_text('Stimulus ON \nt = {:.2f}'.format(gv.time[t_bin]))
            
        elif ((t_bin > gv.bins_ED[0]) and (t_bin < gv.bins_ED[-1])):
            stimdot.set_data([], [])
            text.set_text('Early Delay \nt = {:.2f}'.format(gv.time[t_bin]))
            
        elif ((t_bin > gv.bins_dist[0]) and (t_bin < gv.bins_dist[-1])):
            stimdot.set_data(30, 40)
            text.set_text('Distractor \nt = {:.2f}'.format(gv.time[t_bin]))
            
        elif ((t_bin > gv.bins_MD[0]) and (t_bin < gv.bins_MD[-1])):
            stimdot.set_data([], [])
            text.set_text('Middle Delay \nt = {:.2f}'.format(gv.time[t_bin]))
            
        elif ((t_bin > gv.bins_cue[0]) and (t_bin < gv.bins_cue[-1])): 
            stimdot.set_data(30, 40)
            text.set_text('Cue \nt = {:.2f}'.format(gv.time[t_bin]))
            
        elif ((t_bin > gv.bins_DRT_rwd[0]) and (t_bin < gv.bins_DRT_rwd[-1])):
            stimdot.set_data([], [])
            text.set_text('DRT reward \nt = {:.2f}'.format(gv.time[t_bin]))
            
        elif ((t_bin > gv.bins_LD[0]) and (t_bin < gv.bins_LD[-1])):
            stimdot.set_data([], [])
            text.set_text('Late Delay \nt = {:.2f}'.format(gv.time[t_bin]))
            
        elif ((t_bin > gv.bins_test[0]) and (t_bin < gv.bins_test[-1])):
            stimdot.set_data(30, 40)
            text.set_text('Test \nt = {:.2f}'.format(gv.time[t_bin])) 
        else:
            stimdot.set_data([], [])
            text.set_text(' \nt = {:.2f}'.format(gv.time[t_bin]))
            
        return (scatter,)

    # generate the animation
    anim = animation.FuncAnimation(fig, animate, frames=gv.trial_size, interval=166.66, blit=False)
    HTML(anim.to_jshtml()) # render animation

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=gv.frame_rate, metadata=dict(artist='Me'), bitrate=1800)
    anim.save('animated_scatter.mp4', writer=writer) 
