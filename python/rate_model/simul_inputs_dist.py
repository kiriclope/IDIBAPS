import numpy as np

import globals
from globals import *

from importlib import reload
importlib.reload(sys.modules['globals'])

mean_rates = np.loadtxt(path + 'mean_rates.dat') ;
filter_rates = np.loadtxt(path + 'filter_rates.dat') ;
