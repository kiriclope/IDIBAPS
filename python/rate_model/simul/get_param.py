import numpy as np
import constants as gv

def init_param():

    print("reading parameters from:")
    file_name = "/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/cpp/rate_model/parameters/%dpop/%s.txt" % (gv.n_pop, gv.folder)
    print(file_name)

    i=0
    with open(file_name, 'r') as file:  # Open file for read
        for line in file:           # Read line-by-line
            line = line.strip().split()  # Strip the leading/trailing whitespaces and newline
            line.pop(0)
            if i==0:
                gv.ext_inputs = np.asarray([float(j) for j in line])
                # print(ext_inputs)
            if i==1:
                gv.J = np.asarray([float(j) for j in line])
                J = J.reshape(n_pop,n_pop)
                # print(J)
            if i==2:
                gv.Tsyn = np.asarray([float(j) for j in line])
                Tsyn = Tsyn.reshape(n_pop,n_pop)
                # print(Tsyn)
            i=i+1

    path = '/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/cpp/rate_model/simulations/%dpop/%s/N%d/K%d/' %( gv.n_pop, gv.folder, gv.n_neurons, gv.K)
    print('reading simulations data from:')
    print(path)


