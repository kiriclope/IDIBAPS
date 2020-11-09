from std_lib import * 
sys.path.insert(1, '/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/python/data_analysis') 
from sklearn_lib import * 

import data.constants as gv
importlib.reload(gv) ;

import data.utils as data 
importlib.reload(data) ; 

import utils as fct 
importlib.reload(fct) ; 

import data.fct_facilities as fac
importlib.reload(fac) ;
fac.SetPlotParams()

from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.kmedians import kmedians
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer, random_center_initializer

from pyclustering.utils.metric import distance_metric, type_metric
manhattan_metric = distance_metric(type_metric.MANHATTAN)

gv.mouse = gv.mice[2] 

data.get_sessions_mouse() 
data.get_stimuli_times() 
data.get_delays_times() 

gv.session = gv.sessions[-1] 

X_data, y_labels = data.get_fluo_data() 
print('mouse', gv.mouse, 'session', gv.session, 'data X', X_data.shape,'y', y_labels.shape) 
data.get_bins(t_start=0)

from sklearn.cluster import KMeans 
from sklearn.mixture import GaussianMixture 

gv.epochs = ['ED','LD'] 

for gv.trial in gv.trials: 

    X_S1_trials, X_S2_trials = data.get_S1_S2_trials(X_data, y_labels) 
    X_trials, y_trials = data.get_X_y_epochs(X_S1_trials, X_S2_trials) 
    print('# trials, neurons, time') 
    print('X', X_trials.shape) 

    X_S1, X_S2 = data.get_dX_epochs(X_S1_trials, X_S2_trials) 
    
    dC = [] 
    labels = [] 

    # idx = np.where(np.fabs(X_S1-X_S2)<.025) 
    # X_S1 = np.delete(X_S1, idx, axis=1) 
    # X_S2 = np.delete(X_S2, idx, axis=1) 
    
    # idx = np.where(X_S2<.1) 
    # X_S1 = np.delete(X_S1, idx, axis=1) 
    # X_S2 = np.delete(X_S2, idx, axis=1) 
    
    for i in range(X_trials.shape[2]):
        
        S1 = X_S1[:,:,i] 
        S2 = X_S2[:,:,i] 

        # S1 = StandardScaler().fit_transform(S1)
        # S2 = StandardScaler().fit_transform(S2)

        # plt.figure()
        # plt.scatter(S1[:,48], S1[:,65],c='r') 
        # plt.scatter(S2[:,48], S2[:,65],c='b') 
        # plt.xlim([0,2]) 
        # plt.ylim([0,2])
        
        mean_S1 = np.asarray(np.mean(S1,axis=0))
        mean_S2 = np.asarray(np.mean(S2,axis=0))

        mean_S1 = np.reshape(mean_S1, (1, mean_S1.shape[0])) 
        mean_S2 = np.reshape(mean_S1, (1, mean_S2.shape[0])) 
        
        # dC.append(mean_S1-mean_S2) 

        # model1 = KMeans(n_clusters=1, n_init = 10).fit(S1) 
        # model2 = KMeans(n_clusters=1, n_init = 10).fit(S2) 
        # cluster_means = model1.cluster_centers_- model2.cluster_centers_
        # dC.append(cluster_means[0]) 

        init1 = kmeans_plusplus_initializer(S1, 1).initialize();
        init2 = kmeans_plusplus_initializer(S2, 1).initialize();

        # init1 = random_center_initializer(S1, 1).initialize();
        # init2 = random_center_initializer(S2, 1).initialize(); 

        # model1 = kmedians(S1, init1)
        # model2 = kmedians(S2, init2)  

        model1 = kmeans(S1, init1, metric=manhattan_metric) 
        model2 = kmeans(S2, init2, metric=manhattan_metric) 

        model1.process()
        model2.process()
        
        # cluster_means = np.array(model1.get_medians())-np.array(model2.get_medians()) 

        cluster_means = np.array(model1.get_centers())-np.array(model2.get_centers())
 
        dC.append(cluster_means[0]) 

        # model1 = GaussianMixture(n_components=1).fit(S1) 
        # model2 = GaussianMixture(n_components=1).fit(S2) 
        # cluster_means = model1.means_ - model2.means_ 
        # dC.append(cluster_means[0]) 

        # X = X_trials[:,:,i] 
        # X = StandardScaler().fit_transform(X) 
        
        # model = KMeans(n_clusters=2, n_init=10).fit(X)
        # cluster_means = model.cluster_centers_

        # # model = GaussianMixture(n_components=2).fit(X) 
        # # cluster_means = model.means_
        # # print(cluster_means.shape)
        
        # dC.append(cluster_means[0]-cluster_means[1]) 
        # labels.append(model.predict(X)) 

        # figtitle = '%s_%s_%s_%s' % (gv.mouse, gv.session, gv.trial, gv.epochs[i])
        # ax = plt.figure(figtitle).add_subplot() 
        # plt.scatter(X[:,0], X[:,1], c=model.predict(X)) 
        # plt.xlabel('<F> neuron 1') 
        # plt.ylabel('<F> neuron 2') 

    dC = np.asarray(dC) 
    labels = np.asarray(labels) 

    # print(labels.shape) 
    # idx = np.where(abs(dC)<.2) 
    # dC[idx]=0

    # print(dC.shape) 

    alpha, cos_alp = fct.get_cos(dC) 
    print('trial', gv.trial, 'cos_alp', cos_alp) 

