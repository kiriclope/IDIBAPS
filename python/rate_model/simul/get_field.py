import importlib
from importlib import reload

import numpy as np
import matplotlib.pyplot as plt

import constants as gv
gv.init_param()

def get_field():
    print(gv.VAR_XI)
    print(gv.path + gv.filename)
    time_field = np.loadtxt(gv.path + gv.filename) ; 
    time = time_field[:,0]
    print(time)
    
    field = np.delete(time_field,[0],axis=1)
    print(field.shape)

    return field
