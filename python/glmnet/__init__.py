import pkg_resources

from .logistic import LogitNet
from .logistic_alphaCV import LogitNetAlphaCV
from .logistic_off_diag import LogitNetOffDiag 
from .linear import ElasticNet

__all__ = ['LogitNet', 'ElasticNet', 'LogitNetOffDiag', 'LogitNetAlphaCV']

__version__ = pkg_resources.get_distribution("glmnet").version
