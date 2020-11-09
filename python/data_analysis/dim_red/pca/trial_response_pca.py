from libs import *

def trial_response_pca(X, n_components):
    Xr_sc = z_score(Xr)

    pca = PCA(n_components=n_components)
    Xp = pca.fit_transform(Xr_sc.T).T

    projections = [(0, 1), (1, 2), (0, 2)]
    fig, axes = plt.subplots(1, 3, figsize=[9, 3], sharey='row', sharex='row')
    for ax, proj in zip(axes, projections):
        for t, t_type in enumerate(trial_types):
            x = Xp[proj[0], t_type_ind[t]]
            y = Xp[proj[1], t_type_ind[t]]
            ax.scatter(x, y, c=pal[t], s=25, alpha=0.8)
            ax.set_xlabel('PC {}'.format(proj[0]+1))
            ax.set_ylabel('PC {}'.format(proj[1]+1))
    sns.despine(fig=fig, top=True, right=True)
    add_orientation_legend(axes[2])
