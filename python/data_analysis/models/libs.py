import sys

import numpy as np
import multiprocessing

from sklearn.svm import LinearSVC
from sklearn.cross_decomposition import PLSRegression, PLSSVD, PLSCanonical
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Lasso, LassoCV, LassoLars, LassoLarsCV 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from relaxed_lasso import RelaxedLassoLarsCV 
