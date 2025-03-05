from copy import copy
import operator

import numpy as np
from numpy import random
from numbers import Number, Real
from scipy import stats
from warnings import warn

from holopy.core.metadata import get_extents, get_spacing
from holopy.core.utils import ensure_listlike
from holopy.core.process import center_find
from holopy.core.holopy_object import HoloPyObject
from holopy.scattering.errors import ParameterSpecificationError
from holopy.core.prior import Prior

EPS = 1e-6

"""
    Info about prior class

    Base class for Bayesian priors in holopy.

    Prior subclasses should define at least the following methods:
    - guess
    - sample
    - prob
    - lnprob
"""

class Angles(Prior):
    def __init__(self, concentration_parameter, mean_theta, mean_phi, guess=None, name=None):
        """
        Joint prior of phi and theta to implement the von Mises-Fisher distribution.
        Aim is to avoid angles getting stuck due to finite range limits.

        Parameters
        ----------
        concentration_parameter: float greater than or equal to zero 
        Influences rate distribution falls off as we move away from mean vector 
        position, 0 gives uniform distribution on a sphere.
        mean_theta, mean_phi: floats 
        These set the mean vector position, will take these modulo 
        2 pi or pi as appropriate.
        guess : float or None, optional
            The value to take as an initial guess from the prior. If
            guess is None, defaults to the midpoint of the prior for
            proper priors.
        name : string or None, optional
            The name of the parameter.   
        """
        if concentration_parameter < 0:
            raise ParameterSpecificationError(
                    "Concentration parameter {} is not greater than or equal to zero".format(
                    concentration_parameter))
        self.concentration_parameter = concentration_parameter
        self.mean_theta = mean_theta
        self.mean_phi = mean_phi
        self.name = name
        # maybe we could set a way for theta and phi distributions to be extracted, but this
        # feels like it defeats the purpose of the implementation
        # self.theta = 
        # self.phi = 
        # log of normalization of von Mises-Fisher in 3 dimensions
        self.log_normalization = np.log(concentration_parameter)-np.log(
            2*np.pi*(np.exp(concentration_parameter)-np.exp(-concentration_parameter)))
        # not sure about this bit
        if guess is None:
            self.guess = mean_theta, mean_phi
        else:
            self.guess = guess

    def lnprob(self, angles):
        # take dot product between past phi and theta and proposed phi and theta
        new_theta, new_phi = angles
        dot_product = (np.cos(new_phi-self.mean_phi)*np.sin(new_theta)*np.sin(self.mean_theta)
                        + np.cos(new_theta)*np.cos(self.mean_theta))
        # add log of normalization and dot product times k parameter
        return (self.log_normalization + self.concentration_parameter*dot_product)
        
    
    # don't know if I need a prob defined
    def prob(self, p):
        return "prob undefined for Angles prior"
    
    # I might also need to define a sample method
    def sample(self, size=None):
        return "sample undefined for Angles prior"

class Theta(Gaussian):
    def __init__(self, mu, k, name="theta"):
        """
        Prior for theta to implement the von Mises-Fisher distribution, while
        keeping phi and theta as seperate priors before implementing lnprior 
        manually in a class. Aim is to avoid angles getting stuck due to finite
        range limits.

        Parameters
        ----------
        k: float greater than or equal to zero 
        concentration_parameter, influences rate distribution falls off as we 
        move away from mean vector position, 0 gives uniform distribution on a sphere.
        mu: float
        This sets the theta value for the mean vector position could take these modulo 2 pi 
        name : string or None, optional
            The name of the parameter, assumed to be theta by default  
        """
        self.concentration_parameter = k
        # set sd so can use Gaussian but won't use
        # this for anything meaningful (likely bettte way to do this)
        sd = 1
        super().__init__(mu,sd,name)

class Phi(Gaussian):
    def __init__(self, mu, name="phi"):
        """
        Prior for theta to implement the von Mises-Fisher distribution, while
        keeping phi and theta as seperate priors before implementing lnprior 
        manually in a class. Aim is to avoid angles getting stuck due to finite
        range limits.

        Parameters
        ----------
        mu: float
        This sets the phi value for the mean vector position could take these modulo pi 
        name : string or None, optional
            The name of the parameter, assumed to be phi by default  
        """
        # set sd so can use Gaussian but won't use
        # this for anything meaningful (likely bettte way to do this)
        sd = 1
        super().__init__(mu,sd,name)