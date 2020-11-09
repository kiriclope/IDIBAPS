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

gv.mouse = gv.mice[2] 

data.get_sessions_mouse() 
data.get_stimuli_times() 
data.get_delays_times() 

gv.session = gv.sessions[-1] 

X_data, y_labels = data.get_fluo_data() 
print('mouse', gv.mouse, 'session', gv.session, 'data X', X_data.shape,'y', y_labels.shape) 
data.get_bins(t_start=0)


from sklearn.cluster import KMeans, SpectralClustering

for gv.trial in gv.trials: 

    X_S1_trials, X_S2_trials = data.get_S1_S2_trials(X_data, y_labels) 
    print(X_S1_trials.shape) 

    X_trials, y_trials = data.get_X_y_epochs(X_S1_trials, X_S2_trials)
    print('# trials, neurons, time')
    print('X', X_trials.shape)

    
    dC = []
    for i in range(X_trials.shape[2]):
        X = StandardScaler().fit_transform(X_trials[:,:,i])
        kmeans = KMeans(n_clusters=2, n_init=100).fit(X)
        cluster_center = kmeans.cluster_centers_
        # print(cluster_center.shape)
        
        dC.append(cluster_center[0]-cluster_center[1])

    dC = np.asarray(dC)
    # print(dC.shape)
    
    alpha, cos_alp = fct.get_cos(dC) 
    print('trial', gv.trial, 'cos_alp', cos_alp) 

# idx = np.where(X_trials<0)
# X = np.delete(X_trials, idx, axis=1)

# print('# trials, neurons, time')
# print('X', X.shape,'y', y_trials.shape)


# # for epoch in range(0, len(gv.epochs)):
# #     dum = X[:,:,epoch]
# #     X.append(StandardScaler().fit_transform(dum))

# # X = np.asarray(X)
# # X = np.rollaxis(X,2,1).transpose()

# clf = LogisticRegression(solver='liblinear', penalty='l1') 

# coefs = fct.coefs_clf(X, y_trials, clf=clf) 
# alpha, cos_alp = fct.get_cos(coefs) 
# print('trial', gv.trial, 'cos_alp', cos_alp) 

# idx = np.where(coefs[0]!=0)[0] 

# print(coefs.shape) 
# v1 = [ coefs[0,idx[0]], coefs[0,idx[1]] ]
# v2 = [ coefs[2,idx[0]], coefs[2,idx[1]] ]
# vec = np.asarray([v1, v2]) 

# x = np.mean(X[:,idx[0],0])
# y = np.mean(X[:,idx[1],0])

# figtitle = '%s_%s_%s_EDvsLD' % (gv.mouse, gv.session, gv.trial)
# ax = plt.figure(figtitle).add_subplot() 

# plt.arrow(x, y, *vec[0]*20, head_width=0.05, head_length=0.2, color='r') 
# plt.arrow(x, y, *vec[1]*20, head_width=0.05, head_length=0.2, color = 'b') 

# plt.xlim([np.amin([X[:,idx[0],0],X[:,idx[0],2]])-.1,np.amax([X[:,idx[0],0],X[:,idx[0],2]])+.1]) 
# plt.ylim([np.amin([X[:,idx[1],0],X[:,idx[1],2]])-.1,np.amax([X[:,idx[1],0],X[:,idx[1],2]])+.1]) 

# plt.plot(X_S1_trials[:,idx[0],0], X_S1_trials[:,idx[1],0],'ro',label='ED') 
# plt.plot(X_S2_trials[:,idx[0],0], X_S2_trials[:,idx[1],0],'bo',label='ED') 

# plt.plot(X_S1_trials[:,idx[0],2], X_S1_trials[:,idx[1],2],'r*',label='LD') 
# plt.plot(X_S2_trials[:,idx[0],2], X_S2_trials[:,idx[1],2],'b*',label='LD') 

# plt.xlabel('<F> neuron 1') 
# plt.ylabel('<F> neuron 2') 
# ax.legend() 
