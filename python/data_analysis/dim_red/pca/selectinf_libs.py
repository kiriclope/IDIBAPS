import numpy as np
import nose.tools as nt
# import numpy.testing.decorators as dec
import numpy.testing._private.decorators as dec

from itertools import product
from selection.algorithms.lasso import (lasso, 
                                        data_carving, 
                                        data_splitting,
                                        instance, 
                                        split_model, 
                                        standard_lasso,
                                        instance, 
                                        nominal_intervals,
                                        gaussian_sandwich_estimator,
                                        gaussian_parametric_estimator)

from selection.algorithms.sqrt_lasso import (solve_sqrt_lasso, choose_lambda)

import regreg.api as rr

from selection.tests.decorators import set_sampling_params_iftrue
    
