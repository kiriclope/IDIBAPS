import numpy as np, pandas as pd
import nose.tools as nt

import numpy.testing._private.decorators as dec
from itertools import product

from ...tests.flags import SMALL_SAMPLES

from ...tests.instance import (gaussian_instance as instance,
                                      logistic_instance)

from ...tests.decorators import (set_sampling_params_iftrue, 
                                 wait_for_return_value,
                                 set_seed_iftrue)

from ..lasso import (lasso, 
                     ROSI,
                     data_carving, 
                     data_splitting,
                     split_model, 
                     standard_lasso,
                     nominal_intervals,
                     glm_sandwich_estimator,
                     glm_parametric_estimator)

import regreg.api as rr

try:
    import statsmodels.api
    statsmodels_available = True
except ImportError:
    statsmodels_available = False
