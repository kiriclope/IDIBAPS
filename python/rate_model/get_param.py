import numpy as np

def get_param(n_pop, dir):

    print("reading parameters from:")
    file_name = "../cpp/rate_model/parameters/%dpop/%s.txt" % (n_pop, dir)
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
                J = J.reshape(n_pop,n_pop)
                # print(J)
            if i==2:
                Tsyn = np.asarray([float(j) for j in line])
                Tsyn = Tsyn.reshape(n_pop,n_pop)
                # print(Tsyn)
            i=i+1

    return ext_inputs, J, Tsyn
