import warnings

import numpy as np

from holopy.core.metadata import make_subset_data
from holopy.core.utils import ensure_array, ensure_listlike, ensure_scalar
from holopy.core.holopy_object import HoloPyObject
from holopy.core.errors import raise_fitting_api_error
from holopy.scattering.errors import (MultisphereFailure, TmatrixFailure,
                                      InvalidScatterer, MissingParameter)
from holopy.scattering.interface import calc_holo, interpret_theory
from holopy.inference import prior, AlphaModel
from holopy.core.mapping import Mapper, read_map, edit_map_indices


OPTICS_KEYS = ['medium_index', 'illum_wavelen',
               'illum_polarization', 'noise_sd']

class KaiModel(AlphaModel):
    """
    Model of hologram image that enables joint priors for the angles
    """
    # initialize with same arguments as AlphaModel __init__ and just pass to that method
    def __init__(self, scatterer, alpha=1, noise_sd=None, medium_index=None,
                 illum_wavelen=None, illum_polarization=None, theory='auto',
                 constraints=[]):
        super().__init__(scatterer, alpha, noise_sd, medium_index, illum_wavelen,
                         illum_polarization, theory, constraints)
        
    # now for the tricky bit of overwriting how lnprior (lnposterior?) works so it can handle joint pdf
    def lnprior(self, pars):
        """
        Compute the log-prior probability of pars

        Parameters
        ----------
        pars: dict or list
            list - values for each parameter in the order of self._parameters
            dict - keys should match self.parameters
        Returns
        -------
        lnprior: float
        """
        pars = self.ensure_parameters_are_listlike(pars)
        return self._lnprior(pars)

    def _lnprior(self, pars):
        """
        Internal function taking pars as a list only
        """
        if 'scatterer' in self._maps:
            try:
                par_scat = self._scatterer_from_parameters(pars)
            except InvalidScatterer:
                return -np.inf

        for constraint in self.constraints:
            if not constraint.check(par_scat):
                return -np.inf
            
        sum_of_lnprob = 0
        theta = np.nan
        phi = np.nan
        # currently requires theta and phi to be input as gaussian or bounded gaussian priors
        # even though we then treat them as a joint prior von Mises–Fisher distribution

        # loop through parameter values and corresponding priors
        for p, val in zip(self._parameters, pars):
            # if the prior is not for phi or theta
            if p.name != "theta" and p.name != "phi":
                sum_of_lnprob = p.lnprob(val) + sum_of_lnprob
            # if the prior is theta
            elif p.name == "theta":
                previous_theta = p.mu
                theta = val
            # if the prior is phi
            elif p.name == "phi":
                previous_phi = p.mu
                phi = val
        # if phi and theta are both parameters then use von Mises-Fisher joint distribution
        if phi != np.nan and theta != np.nan:
            # take dot product between past phi and theta and proposed phi and theta
            dot_product = (np.cos(phi-previous_phi)*np.sin(theta)*np.sin(previous_theta)
                        + np.cos(theta)*np.cos(previous_theta))
            # add log of normalization and dot product times k parameter
            # k is concentration parameter k=0 is uniform distribution, k>0 is unimodal around average direction (set by p.mu)
            k_param = 1
            # log of normalization of von Mises-Fisher in 3 dimensions
            ln_normal = np.log(k_param)-np.log(2*np.pi*(np.exp(k_param)-np.exp(-k_param)))
            ln_von_mises_fisher = ln_normal + k_param*dot_product
            sum_of_lnprob = ln_von_mises_fisher + sum_of_lnprob
        return sum_of_lnprob
        # return sum([p.lnprob(val) for p, val in zip(self._parameters, pars)])
    # Want to sum over all p.lnprob(val) of all ._parameters except phi and theta
    # For phi and theta define a lnprob for both together -> is p a prior object? 
    # seems like might need to define prior object to allow for different parameter K to be used
    # could also try to implement new prior object (under prior.py in holopy core)
