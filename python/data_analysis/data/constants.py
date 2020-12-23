import numpy as np 
import multiprocessing 

global mouse, mice, session, sessions, days, trial, trials 
mouse = []
mice = ['C57_2_DualTask','ChRM04','JawsM15'] 
session = [] 
sessions = []
days = [1,2,3,4,5,6]
trial = 'ND' 
trials = ['ND', 'D1', 'D2']

global t_ED, t_MD, t_LD
t_ED = []
t_MD = []
t_LD = []

global frame_rate, n_bin, duration, time
frame_rate = []
n_bin = []
duration = []
time = []

global t_BL, t_STIM, t_test, t_DIST, t_cue, t_DRT_reward
t_BL = [0,2]
t_STIM = [2,3]
t_test = []
t_DIST = []
t_cue = []
t_DRT_reward = []

global epochs
# epochs = ['all']
# epochs = ['Baseline','Stim','ED','Dist','MD','Cue','LD','Test'] 
# epochs = ['ED','Dist','MD','Cue','LD','Test']
epochs = ['ED','MD','LD'] 

global bins, bins_BL, bins_STIM, bins_ED, bins_DIST, bins_MD, bins_LD, bins_cue, bins_DRT_rwd, bins_test
bins = []
bins_BL = []
bins_STIM=[]
bins_ED=[]
bins_DIST=[]
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

global n_boots, bootstrap_method 
n_boots = 10
bootstrap_method = 'bagging' 

global correct_trial
correct_trial = 0

global n_components
n_components = 3

global eps
eps = np.finfo(float).eps

global DELAY_ONLY, DELAY_AND_STIM, bins_delay, t_delay, AVG_EPOCHS, bins_ED_MD_LD, t_ED_MD_LD, ED_MD_LD, bins_stim_delay, t_stim_delay
DELAY_ONLY = 0
DELAY_AND_STIM = 0
bins_delay = []
t_delay = [] 
bin_start = np.array(0)
t_start = np.array(0)
AVG_EPOCHS=0
bins_ED_MD_LD = []
t_ED_MD_LD = []
ED_MD_LD = 0
bins_stim_delay = []
t_stim_delay = []

global scriptdir, figdir, filedir
scriptdir = ''
figdir = '/homecentral/alexandre.mahrach/IDIBAPS/python/data_analysis/figs' 
filedir = '/homecentral/alexandre.mahrach/IDIBAPS/python/data_analysis/data' 

global scaling
scaling = None

global IF_PCA, pca_method, detrend, bootstrap_trials
IF_PCA = 0 
pca_method = 'hybrid' 
detrend = 0
bootstrap_trials=0

global num_cores
num_cores = -int(multiprocessing.cpu_count()/2) 

global SELECTIVE
SELECTIVE=0

global explained_variance
explained_variance = 0.1

global clf_name
clf_name = 'LogisticRegression' 

global trialsXepochs
trialsXepochs=0

global EDvsLD
EDvsLD=0

global my_decoder
my_decoder=0

global pal
pal = ['r','b','y']

global bins_epochs
bins_epochs= []

global T_WINDOW
T_WINDOW = 0 

global SAVGOL
SAVGOL=0

global FEATURE_SELECTION
FEATURE_SELECTION=0

global TIBSHIRANI_TRICK
TIBSHIRANI_TRICK=0

global Z_SCORE
Z_SCORE=0

global pls_method, pls_max_comp, pls_cv
pls_method = 'PLSRegression'
pls_max_comp = 100
pls_cv = 5
