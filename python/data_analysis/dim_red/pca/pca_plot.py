def pca(X_proj, pc_shift=0):
    
    if gv.laser_on:
        figname = '%s_%s_pca_laser_on_%d' % (gv.mouse, gv.session, pc_shift)
    else:
        figname = '%s_%s_pca_laser_off_%d' % (gv.mouse, gv.session, pc_shift)

    plt.figure(figname, figsize=[10, 2.8])    
    x = gv.time 
    for n_pc in range(np.amin([gv.n_components,3])): 
            ax = plt.figure(figname).add_subplot(1, 3, n_pc+1) 
            for i, trial in enumerate(gv.trials): 
                for j, sample in enumerate(gv.samples): 
                    dum = X_proj[i,j,:,n_pc+pc_shift,:].transpose() 
                    y = np.mean( dum, axis=1) 
                    y = gaussian_filter1d(y, sigma=1) 
                    
                    ax.plot(x, y, color=pal[i]) 
                    ci = pp.conf_inter(dum) 
                    ax.fill_between(x, ci[0], ci[1] , color=pal[i], alpha=.1) 
                    
                add_stim_to_plot(ax) 
                ax.set_xlim([0, gv.t_test[1]+1]) 
                    
            ax.set_ylabel('PC {}'.format(n_pc+pc_shift+1)) 
            ax.set_xlabel('Time (s)') 
            sns.despine(right=True, top=True)
            if n_pc == np.amin([gv.n_components,3])-1: 
                add_orientation_legend(ax) 

        figdir = pl.figDir(scriptdir)
        pl.save_fig(figname, figdir) 
