import time
from libs import * 
sys.path.insert(1, '/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/python/data_analysis') 

from EDvsLD import *

def loop(X_proj, NO_PCA):
    for IF_CONCAT in [0,1]:
            EDvsLD(X_proj, IF_CONCAT, 1, 0, NO_PCA) 
            plt.clf() 
            for C in [1e-2, 1e-1,1e0]: 
                EDvsLD(X_proj, IF_CONCAT, 1, 1, NO_PCA, C=C) 
                plt.clf() 
