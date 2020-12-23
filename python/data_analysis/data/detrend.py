from libs import * 
sys.path.insert(1, '/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/python/data_analysis')

import data.constants as gv
importlib.reload(gv) ;

import data.utils as data
importlib.reload(data) ;

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def detrend_data(X_trial, poly_fit=1, degree=7): 
    """ Detrending of the data, if poly_fit=1 uses polynomial fit else linear fit. """
    # X_trial : # neurons, # times 
    
    model = LinearRegression()
    fit_values_trial = []

    # indexes = range(0, X_trial.shape[1]) # neuron index 
    # values = np.mean(X_trial,axis=0) # mean fluo value 
    
    # indexes = np.reshape(indexes, (len(indexes), 1))
    
    # if poly_fit:
    #     poly = PolynomialFeatures(degree=degree) 
    #     indexes = poly.fit_transform(indexes) 
            
    # model.fit(indexes, values)
    # fit_values = model.predict(indexes) 
    # fit_values_trial = np.array(fit_values)
    
    for i in range(0, X_trial.shape[0]): # neurons 
        indexes = range(0, X_trial.shape[1]) # neuron index 
        values = X_trial[i] # fluo value 
                
        indexes = np.reshape(indexes, (len(indexes), 1))

        if poly_fit:
            poly = PolynomialFeatures(degree=degree) 
            indexes = poly.fit_transform(indexes) 

        model.fit(indexes, values)
        fit_values = model.predict(indexes) 
        
        fit_values_trial.append(fit_values) 
        
    fit_values_trial = np.array(fit_values_trial)
    return fit_values_trial

if __name__ == "__main__":
    
    IF_EPOCHS = 0
    
    if IF_EPOCHS==0 :
        gv.epochs = ['all']

    for gv.mouse in [gv.mice[0]] : 

        data.get_sessions_mouse() 
        data.get_stimuli_times() 
        data.get_delays_times() 

        for gv.session in [gv.sessions[-1]] : 
            X_data, y_labels = data.get_fluo_data() 
            print('mouse', gv.mouse, 'session', gv.session, 'data X', X_data.shape,'y', y_labels.shape) 
        
            data.get_delays_times() 
            data.get_frame_rate() 
            data.get_bins(t_start=0) 
        
            gv.duration = X_data.shape[2]/gv.frame_rate 
            time = np.linspace(0, gv.duration, X_data.shape[2]) ; 
        
            for gv.trial in [gv.trials[-1]] : 
                X_S1_trials, X_S2_trials = data.get_S1_S2_trials(X_data, y_labels) 
                X_trials, y_trials = data.get_X_y_epochs(X_S1_trials, X_S2_trials) 
            
                print('trial:', gv.trial, 'X', X_trials.shape,'y', y_trials.shape)
                
                X_avg = np.mean(X_trials, axis=1)

                fit_values = detrend_data(X_trials[0], poly_fit=1, degree=7)
                
                plt.figure()
                plt.plot(X_avg[0]) 
                plt.plot(np.mean(fit_values,axis=0),'--k') 
