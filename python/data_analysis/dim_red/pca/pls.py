import warnings
warnings.filterwarnings("ignore")

from libs import * 
sys.path.insert(1, '/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/python/data_analysis') 
from sys import stdout

import data.constants as gv 
import data.plotting as pl
import data.preprocessing as pp 
import data.angle as agl
import data.progressbar as pg

from scipy.signal import savgol_filter

from sklearn.cross_decomposition import PLSRegression, PLSSVD, PLSCanonical
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import StandardScaler

import data.constants as gv 
import data.preprocessing as pp 

from joblib import Parallel, delayed, parallel_backend
import multiprocessing 

def get_X_y_trials(n_trial, X_trials): 
        
    if X_trials.shape[3]!=gv.n_neurons: 
        X_trials = X_trials[:,:,:,0:gv.n_components,:] 
    
    gv.AVG_EPOCHS = 0 
    gv.trial_size = X_trials.shape[-1] 
    
    if gv.AVG_EPOCHS: 
        gv.trial_size = len(['ED','MD','LD']) 
    
    y = np.array([np.zeros(X_trials.shape[2]), np.ones(X_trials.shape[2])]).flatten() 
    
    X_S1 = X_trials[n_trial,0] 
    X_S2 = X_trials[n_trial,1] 
    X_S1_S2 = np.vstack((X_S1, X_S2)) 

    return X_S1_S2, y

# def optimise_pls_cv(X, y, n_comp):
#     # Define PLS object
#     pls = PLSRegression(n_components=n_comp)

#     # Cross-validation
#     y_cv = cross_val_predict(pls, X, y, cv=10)

#     # Calculate scores
#     r2 = r2_score(y, y_cv)
#     mse = mean_squared_error(y, y_cv)
#     rpd = y.std()/np.sqrt(mse)
    
#     return (y_cv, r2, mse, rpd)

def plot_metrics(vals, ylabel, objective):
    with plt.style.context('ggplot'):
        plt.plot(xticks, np.array(vals), '-v', color='blue', mfc='blue')
        if objective=='min':
            idx = np.argmin(vals)
        else:
            idx = np.argmax(vals)
        plt.plot(xticks[idx], np.array(vals)[idx], 'P', ms=10, mfc='red')

        plt.xlabel('Number of PLS components')
        plt.xticks = xticks
        plt.ylabel(ylabel)
        plt.title('PLS')

    plt.show()

def test_pls(X_trials, n_trial, n_comp):

    X, y = get_X_y_trials(n_trial, X_trials)
    X = pp.avg_epochs(X) 
    
    for n_epochs in range(X.shape[2]):
        r2s = []
        mses = []
        rpds = []
        
        for n_comp in np.arange(1, n_comp):
            X_epochs = X[:,:,n_epochs] 
            # X_epochs = savgol_filter(X_epochs, 17, polyorder=2, deriv=2) 
            X_epochs = StandardScaler().fit_transform(X_epochs) 
            y_cv, r2, mse, rpd = optimise_pls_cv(X_epochs, y, n_comp) 
            r2s.append(r2)
            mses.append(mse)
            rpds.append(rpd)
    
        # plot_metrics(mses, 'MSE', 'min')
        # plot_metrics(rpds, 'RPD', 'max')
        # plot_metrics(r2s, 'R2', 'max') 

        print('n_comp', np.argmax(r2s), np.argmax(mses), np.argmax(rpds))
        
# def test_pls(X_trials, n_trial, n_comp):

#     X, y = get_X_y_trials(n_trial, X_trials)
#     X = pp.avg_epochs(X) 
    
#     for n_epochs in range(X.shape[2]):
#         r2s = []
#         mses = []
#         rpds = []
        
#         for n_comp in np.arange(1, n_comp):
#             X_epochs = X[:,:,n_epochs] 
#             # X_epochs = savgol_filter(X_epochs, 17, polyorder=2, deriv=2) 
#             X_epochs = StandardScaler().fit_transform(X_epochs) 
#             y_cv, r2, mse, rpd = optimise_pls_cv(X_epochs, y, n_comp) 
#             r2s.append(r2)
#             mses.append(mse)
#             rpds.append(rpd)
    
#         # plot_metrics(mses, 'MSE', 'min')
#         # plot_metrics(rpds, 'RPD', 'max')
#         # plot_metrics(r2s, 'R2', 'max') 

#         print('n_comp', np.argmax(r2s), np.argmax(mses), np.argmax(rpds))
        
def optimise_pls_cv(X, y, n_comp_max):
    '''Run PLS including a variable number of components, up to n_comp,
       and calculate MSE '''

    mse = []
    component = np.arange(1, n_comp_max)
    for i in component:
        pls = PLSRegression(n_components=i)
        # Cross-validation
        y_cv = cross_val_predict(pls, X, y, cv=10)
        mse.append(mean_squared_error(y, y_cv))

        comp = 100*(i+1)/n_comp_max 
        # Trick to update status on the same line
        stdout.write("\r pls %d %d%% completed" % (i, comp) )
        stdout.flush() 
    stdout.write("\n") 

    # def loop_comp(X, y, n_comp) :
    #     pls = PLSRegression(n_components=n_comp)
    #     # pls = PLSCanonical(n_components=n_comp)
    #     y_cv = cross_val_predict(pls, X, y, cv=10)
    #     mse = mean_squared_error(y, y_cv)
    #     return mse
    
    # with pg.tqdm_joblib(pg.tqdm(desc= 'pls optimization', total=n_comp_max-1)) as progress_bar:             
    #     mse = Parallel(n_jobs=20)(delayed(loop_comp)(X, y, n_comp) for n_comp in range(1, n_comp_max)) 
    
    # Calculate and print the position of minimum in MSE
    msemin = np.argmin(mse)
    print("Suggested number of components: ", msemin+1)

    return msemin+1
