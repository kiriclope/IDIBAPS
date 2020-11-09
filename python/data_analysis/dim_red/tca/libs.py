import os, sys, importlib
from importlib import reload

import numpy as np
import seaborn as sns

import matplotlib
matplotlib.use('GTK3cairo')

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D 
import random

from scipy.io import loadmat
from scipy.ndimage.filters import gaussian_filter1d
import scipy.stats as stats

from sklearn import svm
from sklearn.decomposition import PCA, SparsePCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.model_selection import cross_validate

import tensortools as tt

import warnings
warnings.filterwarnings("ignore")

