from sklearn import svm 
from sklearn.model_selection import cross_validate, GridSearchCV 
from sklearn.pipeline import make_pipeline, Pipeline 
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
