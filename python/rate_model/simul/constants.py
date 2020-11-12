import numpy as np

global n_pop, n_neurons, K, m0
n_pop = 1
n_neurons = 1
K = 500

global ext_inputs, J, Tsyn
ext_inputs = []
J = []
Tsyn = []

global J0, I0, folder
J0 = 0.1
I0 = 0.1
folder = 'I0_%.2f_J0_%.2f' % (I0,J0)

global filename
filename = 'inputs.dat'

global IF_TRIALS, TRIAL_ID
IF_TRIALS = 1 
TRIAL_ID = 1 

global IF_LOW_RANK, MEAN_XI, VAR_XI
IF_LOW_RANK = 1
MEAN_XI = -0.0
VAR_XI = 5.0 

global IF_LEFT_RIGHT, MEAN_XI_LEFT, MEAN_XI_RIGHT, VAR_XI_LEFT, VAR_XI_RIGHT, RHO
IF_LEFT_RIGHT = 0
MEAN_XI_LEFT = -0.0
VAR_XI_LEFT = 5.0 
MEAN_XI_RIGHT = -0.0
VAR_XI_RIGHT = 5.0 
RHO = 0.5

global IF_FF, MEAN_FF, VAR_FF, VAR_ORTHO, IF_RHO_FF, RH0_FF_XI
IF_FF = 0
MEAN_FF = 1.0
VAR_FF = 1.0
VAR_ORTHO = 0.0

IF_RHO_FF = 1
RHO_FF_XI = 1.0 

def init_param():
    global folder
    folder = 'I0_%.2f_J0_%.2f' % (I0,J0)

    global path
    path = '' 
    
    if(n_pop!=1):
        print("reading parameters from:")
        file_name = "/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/cpp/rate_model/parameters/%dpop/%s.txt" % (n_pop, folder)
        print(file_name)
        
        i=0
        with open(file_name, 'r') as file:  # Open file for read
            for line in file:           # Read line-by-line
                line = line.strip().split()  # Strip the leading/trailing whitespaces and newline
                line.pop(0)
                if i==0:
                    ext_inputs = np.asarray([float(j) for j in line])
                    # print(ext_inputs)
                if i==1:
                    J = np.asarray([float(j) for j in line])
                    J = J.reshape(n_pop, n_pop)
                    # print(J)
                if i==2:
                    Tsyn = np.asarray([float(j) for j in line])
                    Tsyn = Tsyn.reshape(n_pop, n_pop)
                    # print(Tsyn)
                i=i+1
    else:
        ext_inputs = I0
        J = J0
        Tsyn = 2 ;
        
    path = '/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/cpp/rate_model/simulations/%dpop/%s/N%d/K%d/' % (n_pop, folder, n_neurons, K)
    if(IF_LOW_RANK):
        if(IF_LEFT_RIGHT):
            path += 'low_rank/xi_left_mean_%.2f_var_%.2f' % (MEAN_XI_LEFT,VAR_XI_LEFT)
            path += '_xi_right_mean_%.2f_var_%.2f/' % (MEAN_XI_RIGHT,VAR_XI_RIGHT)
            path += 'rho_%.2f/' % RHO
        else:
            path += 'low_rank/xi_mean_%.2f_var_%.2f/' % (MEAN_XI,VAR_XI)
        if(IF_FF):
            if(IF_RHO_FF):
                path += 'ff_mean_%.2f_var_%.2f_rho_%.2f/' % (MEAN_FF,VAR_FF,RHO_FF_XI)
            else:
                path += 'ff_mean_%.2f_var_%.2f_ortho_%.2f/' % (MEAN_FF,VAR_FF,VAR_ORTHO)
        if(IF_TRIALS):
            path += 'trial_%d/' % TRIAL_ID ;

    print('reading simulations data from:')
    print(path)
