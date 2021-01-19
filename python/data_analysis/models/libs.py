import sys

import numpy as np
import multiprocessing

from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.cross_decomposition import PLSRegression, PLSSVD, PLSCanonical 
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Lasso, LassoCV, LassoLars, LassoLarsCV, LassoLarsIC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from relaxed_lasso import RelaxedLassoLarsCV 

import pycasso

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline 
