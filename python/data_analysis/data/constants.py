import numpy as np 

global mouse, mice, session, sessions, trial, trials 
mouse = []
mice = ['C57_2_DualTask','ChRM04','JawsM15'] 
session = [] 
sessions = [] 
trial = 'ND' 
trials = ['ND', 'D1', 'D2']

global t_early_delay, t_DRT_delay, t_late_delay
t_early_delay = []
t_DRT_delay = []
t_late_delay = []

global frame_rate, n_bin, duration, time
frame_rate = []
n_bin = []
duration = []
time = []

global t_sample, t_test, t_distractor, t_cue, t_DRT_reward
t_sample = [2,3]
t_test = []
t_distractor = []
t_cue = []
t_DRT_reward = []

global epochs
# epochs = ['all']
# epochs = ['Baseline','Stim','ED','Dist','MD','Cue','LD','Test'] 
# epochs = ['ED','Dist','MD','Cue','LD','Test']
epochs = ['ED','MD','LD'] 

global bins, bins_baseline, bins_stim, bins_ED, bins_dist, bins_MD, bins_LD, bins_cue, bins_DRT_rwd, bins_test
bins = []
bins_baseline = []
bins_stim=[]
bins_ED=[]
bins_dist=[]
bins_MD=[]
bins_cue = []
bins_DRT_rwd = []
bins_LD=[]
bins_test = []

global dum
dum = -1

global IF_SAVE
IF_SAVE=0

global laser_on
laser_on = 0

global  n_neurons, n_trials, trial_type, trial_size
n_neurons = []
n_trials= [] 
trial_type = [] 
trial_size = [] 

global samples
samples=['S1', 'S2']

global data_type
data_type = 'fluo' # 'rates'

global n_boot
n_boot = 10

global correct_trial
correct_trial = 0

global n_components
n_components = 3

global eps
eps = np.finfo(float).eps
