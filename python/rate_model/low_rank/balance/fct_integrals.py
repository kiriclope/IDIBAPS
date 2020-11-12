import numpy as np
import scipy.special as special 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### Computing Gaussian integrals through Gauss-Hermite quadrature
# here Phi(x) = 1/2*(1+erf(x/sqrt(2))

# Global variables for Gaussian quadrature

gaussian_norm = (1/np.sqrt(np.pi))
gauss_points, gauss_weights = np.polynomial.hermite.hermgauss(200)
gauss_points = gauss_points*np.sqrt(2)

#### Input-output transfert function
def Phi(x): # CDF of the normal distribution
    return 0.5 *(1.0 + special.erf(x/np.sqrt(2)) )

def phi(x): # normal distribution
    return np.exp(-0.5*x**2) / np.sqrt(2*np.pi)

#### Single Gaussian intergrals

def intPhi (mu, delta0):
    return Phi( mu/np.sqrt(1.0 + delta0) ) 
    # integrand = Phi(mu+np.sqrt(delta0)*gauss_points)
    # return gaussian_norm * np.dot (integrand,gauss_weights)

def intPhiSq (mu, delta0):
    integrand = Phi(mu+np.sqrt(delta0)*gauss_points)**2
    return gaussian_norm * np.dot (integrand,gauss_weights)
