from libs import * 
sys.path.insert(1, '/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/python/data_analysis') 

import data.constants as gv 
import data.plotting as pl
import data.preprocessing as pp 

from joblib import Parallel, delayed, parallel_backend
import multiprocessing 

from sklearn.cluster import KMeans 
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.kmedians import kmedians
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer, random_center_initializer

from pyclustering.utils.metric import distance_metric, type_metric
manhattan_metric = distance_metric(type_metric.MANHATTAN)

pal = ['r','b','y'] 
    
def get_centers_kmean(X_S1, X_S2): 
    
    model1 = KMeans(n_clusters=1, init='random', n_init = 1000).fit(X_S1) 
    model2 = KMeans(n_clusters=1, init='random', n_init = 1000).fit(X_S2) 

    # model1 = KMeans(n_clusters=1, init='k-means++', n_init = 1000).fit(X_S1) 
    # model2 = KMeans(n_clusters=1, init='k-means++', n_init = 1000).fit(X_S2) 

    Dmean = model1.cluster_centers_- model2.cluster_centers_

    return Dmean

def get_centers_kmediansCpp(X_S1, X_S2): 
    init1 = kmeans_plusplus_initializer(X_S1, 1).initialize();
    init2 = kmeans_plusplus_initializer(X_S2, 1).initialize();

    # init1 = random_center_initializer(X_S1, 1).initialize();
    # init2 = random_center_initializer(X_S2, 1).initialize(); 

    model1 = kmedians(X_S1, init1)
    model2 = kmedians(X_S2, init2)  
    Dmedian = np.array(model1.get_medians())-np.array(model2.get_medians()) 

    return Dmedian

def get_centers_kmeanCpp(X_S1, X_S2):

    init1 = kmeans_plusplus_initializer(S1, 1).initialize();
    init2 = kmeans_plusplus_initializer(S2, 1).initialize();

    # init1 = random_center_initializer(X_S1, 1).initialize();
    # init2 = random_center_initializer(X_S2, 1).initialize(); 
    
    model1 = kmeans(S1, init1, metric=manhattan_metric) 
    model2 = kmeans(S2, init2, metric=manhattan_metric) 

    model1.process()
    model2.process()
    
    Dmean = np.array(model1.get_centers())-np.array(model2.get_centers())

    return Dmean

def bootstrap_distance(S1,S2,dum):
    
    if dum==1: 
        print('no boot') 
        idx_trials = np.arange(0, S1.shape[0]) 
    else:
        #standard bootstrap
        idx_trials = np.random.randint(0, S1.shape[0], S1.shape[0])

    S1_boot = S1[idx_trials]
    S2_boot = S2[idx_trials]
    
    D_center_boot = get_centers_kmean(S1_boot, S2_boot) 
    # Dcenter[n_trial, n_epochs] = get_centers_kmedian(S1, S2) 
    # Dcenter[n_trial, n_epochs] = get_centers_kmeanCpp(S1, S2)
    
    return D_center_boot

def cluster_distance(X_trials):
    gv.n_boot = int(1e3) 
    gv.num_cores = int(0.9*multiprocessing.cpu_count()) 
    
    gv.IF_PCA = 0
    if X_trials.shape[3]!=gv.n_neurons: 
        X_trials = X_trials[:,:,:,0:gv.n_components,:] 
        gv.IF_PCA = 1 
        
    if gv.EDvsLD: 
        gv.epochs = ['ED', 'MD', 'LD'] 
        print('angle btw ED and other epochs') 
    else: 
        gv.epochs = ['Stim', 'ED', 'MD', 'LD'] 
        print('angle btw STIM and other epochs') 
        
    D_center = np.empty((len(gv.trials), len(gv.epochs), gv.n_boot, X_trials.shape[3])) 
    
    for n_trial, gv.trial in enumerate(gv.trials): 
        X_S1 = X_trials[n_trial,0] 
        X_S2 = X_trials[n_trial,1] 
        
        X_S1 = pp.avg_epochs(X_S1) 
        X_S2 = pp.avg_epochs(X_S2) 
        
        for n_epochs in range(X_S1.shape[2]):
            S1 = X_S1[:,:,n_epochs] 
            S2 = X_S2[:,:,n_epochs] 
            
            D_center_boot = np.array( Parallel(n_jobs=gv.num_cores, verbose=True)(delayed(bootstrap_distance)(S1, S2, gv.n_boot) for _ in range(gv.n_boot)) ) 
            D_center[n_trial, n_epochs] = D_center_boot[:,0,:] # boots x neurons 
            
            # D_center_boot = get_centers_kmean(S1, S2)
            # D_center[n_trial, n_epochs] = D_center_boot 
            
            print(D_center_boot.shape)
            
    return D_center 

def cluster_norm(D_center):

    # mean over boots 
    D_center_mean = np.mean(D_center, axis=2) 
    lower = D_center_mean - np.percentile(D_center, 25, axis=2) 
    upper = np.percentile(D_center, 75, axis=2) - D_center_mean 

    # normalize over neurons 
    D_norm = np.linalg.norm(D_center_mean, axis=2) 
    D_norm = (D_norm.T/D_norm.T[0]).T 
    
    lower = np.linalg.norm(lower, axis=-1) 
    lower = (lower.T/lower.T[0]).T 
    
    upper = np.linalg.norm(upper, axis=-1) 
    upper = (upper.T/upper.T[0]).T 
    
    return D_norm, lower, upper 

def barNorm(D_norm, lower, upper): 
    pl.figDir() 

    labels = np.arange(len(gv.epochs)-1) 
    width=0.25
    
    figtitle = '%s_%s_cluster_distance' % (gv.mouse, gv.session)
    ax = plt.figure(figtitle).add_subplot() 

    for n_trial, trial in enumerate(gv.trials):
        values = D_norm[n_trial][1:] 
        error = np.array([ lower[n_trial][1:], upper[n_trial][1:] ] ) 
        plt.bar(labels + n_trial*width, values , yerr=error, color = pal[n_trial], width = width) 
            
    plt.xticks([i + width for i in range(len(gv.epochs)-1)], gv.epochs[1:]) 

    plt.xlabel('Epochs') 
    plt.ylabel('distance') 
    plt.ylim([0,1])

    pl.save_fig(figtitle) 
    
def clustering(X_trials):
    D_center = cluster_distance(X_trials) 
    print('center', D_center.shape) 
    D_norm, lower, upper = cluster_norm(D_center) 
    print('distance', D_norm) 
    barNorm(D_norm, lower, upper) 
    
