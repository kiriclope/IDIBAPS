from libs import * 

sys.path.insert(1, '/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/python/data_analysis') 

import data.constants as gv 
from plotting import *

import data.utils as data 

# gv.trial_size = 3 
AVG_EPOCHS = 0

def trajectories_2D(X_avg_pc):
    
    # pick the components corresponding to the x, y, and z axes
    component_x = 0
    component_y = 2

    # create a boolean mask so we can plot activity during stimulus as 
    # solid line, and pre and post stimulus as a dashed line
    
    # utility function to clean up and label the axes
    def style_2d_ax(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')

    sigma = 1 # smoothing amount

    # set up a figure with two 3d subplots, so we can have two different views
    fig = plt.figure(figsize=[9, 4])

    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    axs = [ax1]

    for ax in axs:
        for i, trial in enumerate(gv.trials):
            for j, sample in enumerate(gv.samples):

                # for every trial type, select the part of the component 
                # which corresponds to that trial type: 
                x = X_avg_pc[component_x, (i*(j+1)) * gv.trial_size :(i*(j+1)) * gv.trial_size]
                y = X_avg_pc[component_y, (i*(j+j)) * gv.trial_size :(i*(j+1)) * gv.trial_size]

                # apply some smoothing to the trajectories
                x = gaussian_filter1d(x, sigma=sigma)
                y = gaussian_filter1d(y, sigma=sigma)

                # use the mask to plot stimulus and pre/post stimulus separately
                y_ED = y.copy()
                y_ED[np.delete(gv.bins,gv.bins_ED)] = np.nan

                y_MD = y.copy()
                y_MD[np.delete(gv.bins,gv.bins_MD)] = np.nan
                
                y_LD = y.copy()
                y_LD[np.delete(gv.bins,gv.bins_LD)] = np.nan

                y_other = y.copy()
                y_other[gv.bins_ED] = np.nan
                y_other[gv.bins_MD] = np.nan
                y_other[gv.bins_LD] = np.nan

                ax.plot(x, y_ED, c = pal[i]) 
                ax.plot(x, y_MD, c = pal[i])
                ax.plot(x, y_LD, c = pal[i])
                ax.plot(x, y_other, c=pal[i], ls=':')
        
                # plot dots at initial point
                ax.scatter(x[0], y[0], c=pal[i], s=20)
                # ax.scatter(np.mean(x[gv.bins_ED]), np.mean(y_ED[gv.bins_ED]), c=pal[t], s=14)
                # ax.scatter(np.mean(x[gv.bins_MD]), np.mean(y_MD[gv.bins_MD]), c=pal[t], s=14)
                # ax.scatter(np.mean(x[gv.bins_LD]), np.mean(y_LD[gv.bins_LD]), c=pal[t], s=14)

                style_2d_ax(ax)
    plt.tight_layout()

def trajectories_3D(X_avg_pc):
    
    # pick the components corresponding to the x, y, and z axes
    component_x = 0
    component_y = 1
    component_z = 2
    
    # utility function to clean up and label the axes
    def style_3d_ax(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')

    sigma = 3 # smoothing amount

    # set up a figure with two 3d subplots, so we can have two different views
    fig = plt.figure(figsize=[9, 4])
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    axs = [ax1, ax2]

    for ax in axs:
        for i, trial in enumerate(gv.trials):
            for j, sample in enumerate(gv.samples):

                # for every trial type, select the part of the component
                # which corresponds to that trial type:
                
                # x = X_avg_pc[component_x, (len(gv.samples)*i+j) * gv.trial_size : (len(gv.samples)*i+j+1) * gv.trial_size]
                # y = X_avg_pc[component_y, (len(gv.samples)*i+j) * gv.trial_size : (len(gv.samples)*i+j+1) * gv.trial_size]
                # z = X_avg_pc[component_z,  (len(gv.samples)*i+j) * gv.trial_size : (len(gv.samples)*i+j+1) * gv.trial_size]

                # x = X_avg_pc[i,j,component_x]
                # y = X_avg_pc[i,j,component_y]
                # z = X_avg_pc[i,j,component_z]

                x = np.mean(X_avg_pc[i,j,:,component_x,:], axis=0) 
                y = np.mean(X_avg_pc[i,j,:,component_y,:], axis=0) 
                z = np.mean(X_avg_pc[i,j,:,component_z,:], axis=0) 
                
                # apply some smoothing to the trajectories
                x = gaussian_filter1d(x, sigma=sigma) 
                y = gaussian_filter1d(y, sigma=sigma) 
                z = gaussian_filter1d(z, sigma=sigma) 

                # use the mask to plot stimulus and pre/post stimulus separately
                z_ED = z.copy() 
                z_ED[np.delete(gv.bins,gv.bins_ED)] = np.nan 

                z_MD = z.copy()
                z_MD[np.delete(gv.bins,gv.bins_MD)] = np.nan 
                
                z_LD = z.copy()
                z_LD[np.delete(gv.bins,gv.bins_LD)] = np.nan

                z_other = z.copy()
                z_other[gv.bins_ED] = np.nan
                z_other[gv.bins_MD] = np.nan
                z_other[gv.bins_LD] = np.nan

                ax.plot(x, y, z_ED, c = pal[i]) 
                ax.plot(x, y, z_MD, c = pal[i])
                ax.plot(x, y, z_LD, c = pal[i])
                ax.plot(x, y, z_other, c=pal[i], ls=':')
        
                # plot dots at initial point
                ax.scatter(x[0], y[0], z[0], c=pal[i], s=14)
        
                # make the axes a bit cleaner
                style_3d_ax(ax)
        
    # specify the orientation of the 3d plot        
    ax1.view_init(elev=22, azim=-120)
    ax2.view_init(elev=22, azim=120)
    plt.tight_layout()

def avg_trajectories_3D(X_avg_pc):
    
    # pick the components corresponding to the x, y, and z axes
    component_x = 0
    component_y = 1
    component_z = 2
    
    # utility function to clean up and label the axes
    def style_3d_ax(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')

    sigma = 3 # smoothing amount

    # set up a figure with two 3d subplots, so we can have two different views
    fig = plt.figure(figsize=[9, 4])
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    axs = [ax1, ax2]

    for ax in axs:
        for i, trial in enumerate(gv.trials):
            for j, sample in enumerate(gv.samples):

                # for every trial type, select the part of the component
                # which corresponds to that trial type:
                
                # x = X_avg_pc[component_x, (len(gv.samples)*i+j) * gv.trial_size : (len(gv.samples)*i+j+1) * gv.trial_size]
                # y = X_avg_pc[component_y, (len(gv.samples)*i+j) * gv.trial_size : (len(gv.samples)*i+j+1) * gv.trial_size]
                # z = X_avg_pc[component_z,  (len(gv.samples)*i+j) * gv.trial_size : (len(gv.samples)*i+j+1) * gv.trial_size]

                x = X_avg_pc[i,j,component_x]
                y = X_avg_pc[i,j,component_y]
                z = X_avg_pc[i,j,component_z]

                if AVG_EPOCHS :
                    x = data.avgOverEpochs(x)
                    y = data.avgOverEpochs(y)
                    z = data.avgOverEpochs(z)
                    
                # apply some smoothing to the trajectories
                x = gaussian_filter1d(x, sigma=sigma)
                y = gaussian_filter1d(y, sigma=sigma)
                z = gaussian_filter1d(z, sigma=sigma)
                        

                # print((len(gv.samples)*i+j), 'x', x.shape, 'y', y.shape, 'z', z.shape)
                
                ax.plot(x, y, z, '-o', c=pal[i], ls='-')
        
                # plot dots at initial point
                ax.scatter(x[0], y[0], z[0], c=pal[i], s=14)
        
                # make the axes a bit cleaner
                # style_3d_ax(ax)
        
    # specify the orientation of the 3d plot        
    ax1.view_init(elev=22, azim=-120)
    ax2.view_init(elev=22, azim=120)
    plt.tight_layout()
