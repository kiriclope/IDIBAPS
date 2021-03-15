import numpy as np 
import multiprocessing 

global path, scriptdir, figdir, filedir 
path = '/homecentral/alexandre.mahrach/IDIBAPS/python/data_analysis' 
figdir = path + '/figs' 
filedir = path + '/data' 

global mouse, mice, session, sessions, days, trial, trials, n_days
mouse = []
mice = ['C57_2_DualTask','ChRM04','JawsM15','JawsM18','ACCM03','ACCM04'] 
n_days = 6
days = np.arange(1, n_days+1) 
day=-1 
tasks = ['DPA', 'dual GO', 'dual No-GO']
session =-1
sessions = [] 
trial = 'ND' 
trials = ['ND', 'D1', 'D2']

global SAME_DAYS, SAME_DECODER
SAME_DAYS = 1
SAME_DECODER = 0 

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
epoch = None

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
IF_SAVE = 1 

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
data_type = 'raw' # 'rates'

global n_boots, bootstrap_method 
n_boots = 10
bootstrap_method = 'bagging' 

global correct_trial
correct_trial = 0

global n_components
n_components = None 

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

global scaling
scaling = None

global pca_method, DETREND, bootstrap_trials, inflection
pca_method = 'hybrid' 
DETREND = 0
bootstrap_trials=0
inflection= False 

global num_cores
num_cores = int(0.9*multiprocessing.cpu_count()) 

global SELECTIVE
SELECTIVE=0

global explained_variance
explained_variance = 0.1

global clf_name
clf_name = 'LogisticRegressionCV' 
clf = None 

global trialsXepochs
trialsXepochs=0

global CONCAT_BINS
CONCAT_BINS=''

global EDvsLD
EDvsLD=1

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

global Z_SCORE, Z_SCORE_BL, DECONVOLVE, DCV_THRESHOLD, bins_z_score, NORMALIZE 
Z_SCORE_BL = 0
Z_SCORE=0
bins_z_score = 0
DECONVOLVE=0
DCV_THRESHOLD=.1
NORMALIZE=0

global pls_method, pls_max_comp, pls_cv
pls_method = None
pls_max_comp = 100
pls_cv = 5

global max_threshold, n_thresholds, spca_scoring, spca_cv
max_threshold = 10
n_thresholds = 10 
spca_scoring = 'mse'
spca_cv = 5

global LASSOCV, lassoCV 
LASSOCV = False 
lassoCV = None 

global scoring, fold_type, n_iter, inner_scoring
scoring='accuracy'
fold_type='stratified'
n_iter = 1
inner_scoring='accuracy'

global SYNTHETIC
SYNTHETIC=0

global standardize
standardize=True

global pair_trials
pair_trials = 0

global minka_mle
minka_mle=0

global cos_trials, bootstrap_cos, n_cos_boots
cos_trials=0 
bootstrap_cos=0
n_cos_boots= int(1e3)

global list_n_components
list_n_components = None

global pca_model, sparse_alpha, ridge_alpha
pca_model = None
sparse_alpha=1
ridge_alpha=.01 

global scores_trials
scores_trials=0

global F0_THRESHOLD, AVG_F0_TRIALS
F0_THRESHOLD=None
AVG_F0_TRIALS=0

global AVG_BEFORE_PCA
AVG_BEFORE_PCA=0

global fix_alpha_lbd
fix_alpha_lbd = 0

global inter_trials
inter_trials=1

global first_days, last_days, all_days
first_days=0
last_days=0 
all_days=0

