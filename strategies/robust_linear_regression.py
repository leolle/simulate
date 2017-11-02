# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 19:06:31 2017

@author: gft
"""

sm.RLM(data.endog, data.exog, M=sm.robust.norms.HuberT())

statsmodels.robust.robust_linear_model.RLM??

class RLM(base.LikelihoodModel):
    __doc__ = """
    Robust Linear Models

    Estimate a robust linear model via iteratively reweighted least squares
    given a robust criterion estimator.

    %(params)s
    M : statsmodels.robust.norms.RobustNorm, optional
        The robust criterion function for downweighting outliers.
        The current options are LeastSquares, HuberT, RamsayE, AndrewWave,
        TrimmedMean, Hampel, and TukeyBiweight.  The default is HuberT().
        See statsmodels.robust.norms for more information.
    %(extra_params)s

    Notes
    -----

    **Attributes**

    df_model : float
        The degrees of freedom of the model.  The number of regressors p less
        one for the intercept.  Note that the reported model degrees
        of freedom does not count the intercept as a regressor, though
        the model is assumed to have an intercept.
    df_resid : float
        The residual degrees of freedom.  The number of observations n
        less the number of regressors p.  Note that here p does include
        the intercept as using a degree of freedom.
    endog : array
        See above.  Note that endog is a reference to the data so that if
        data is already an array and it is changed, then `endog` changes
        as well.
    exog : array
        See above.  Note that endog is a reference to the data so that if
        data is already an array and it is changed, then `endog` changes
        as well.
    M : statsmodels.robust.norms.RobustNorm
         See above.  Robust estimator instance instantiated.
    nobs : float
        The number of observations n
    pinv_wexog : array
        The pseudoinverse of the design / exogenous data array.  Note that
        RLM has no whiten method, so this is just the pseudo inverse of the
        design.
    normalized_cov_params : array
        The p x p normalized covariance of the design / exogenous data.
        This is approximately equal to (X.T X)^(-1)


    Examples
    ---------
import statsmodels.api as sm
data = sm.datasets.stackloss.load()
data.exog = sm.add_constant(data.exog)
rlm_model = sm.RLM(data.endog, data.exog,
                           M=sm.robust.norms.HuberT())

rlm_results = rlm_model.fit()
rlm_results.params
    array([  0.82938433,   0.92606597,  -0.12784672, -41.02649835])
rlm_results.bse
    array([ 0.11100521,  0.30293016,  0.12864961,  9.79189854])
rlm_results_HC2 = rlm_model.fit(cov="H2")
rlm_results_HC2.params
    array([  0.82938433,   0.92606597,  -0.12784672, -41.02649835])
rlm_results_HC2.bse
    array([ 0.11945975,  0.32235497,  0.11796313,  9.08950419])
    >>>
rlm_hamp_hub = sm.RLM(data.endog, data.exog,
                          M=sm.robust.norms.Hampel()).fit(
                          sm.robust.scale.HuberScale())

rlm_hamp_hub.params
    array([  0.73175452,   1.25082038,  -0.14794399, -40.27122257])
    """ % {'params' : base._model_params_doc,
            'extra_params' : base._missing_param_doc}

    def __init__(self, endog, exog, M=norms.HuberT(), missing='none',
                 **kwargs):
        self.M = M
        super(base.LikelihoodModel, self).__init__(endog, exog,
                missing=missing, **kwargs)
        self._initialize()
        #things to remove_data
        self._data_attr.extend(['weights', 'pinv_wexog'])

    def _initialize(self):
        """
        Initializes the model for the IRLS fit.

        Resets the history and number of iterations.
        """
        self.pinv_wexog = np.linalg.pinv(self.exog)
        self.normalized_cov_params = np.dot(self.pinv_wexog,
                                        np.transpose(self.pinv_wexog))
        self.df_resid = (np.float(self.exog.shape[0] -
                         np_matrix_rank(self.exog)))
        self.df_model = np.float(np_matrix_rank(self.exog)-1)
        self.nobs = float(self.endog.shape[0])

    def score(self, params):
        raise NotImplementedError

    def information(self, params):
        raise NotImplementedError

    def predict(self, params, exog=None):
        """
        Return linear predicted values from a design matrix.

        Parameters
        ----------
        params : array-like, optional after fit has been called
            Parameters of a linear model
        exog : array-like, optional.
            Design / exogenous data. Model exog is used if None.

        Returns
        -------
        An array of fitted values

        Notes
        -----
        If the model as not yet been fit, params is not optional.
        """
        #copied from linear_model
        if exog is None:
            exog = self.exog
        return np.dot(exog, params)

    def loglike(self, params):
        raise NotImplementedError

    def deviance(self, tmp_results):
        """
        Returns the (unnormalized) log-likelihood from the M estimator.
        """
        return self.M((self.endog - tmp_results.fittedvalues) /
                          tmp_results.scale).sum()

    def _update_history(self, tmp_results, history, conv):
        history['params'].append(tmp_results.params)
        history['scale'].append(tmp_results.scale)
        if conv == 'dev':
            history['deviance'].append(self.deviance(tmp_results))
        elif conv == 'sresid':
            history['sresid'].append(tmp_results.resid/tmp_results.scale)
        elif conv == 'weights':
            history['weights'].append(tmp_results.model.weights)
        return history

    def _estimate_scale(self, resid):
        """
        Estimates the scale based on the option provided to the fit method.
        """
        if isinstance(self.scale_est, str):
            if self.scale_est.lower() == 'mad':
                return scale.mad(resid, center=0)
            if self.scale_est.lower() == 'stand_mad':
                return scale.mad(resid)
            else:
                raise ValueError("Option %s for scale_est not understood" %
                                 self.scale_est)
        elif isinstance(self.scale_est, scale.HuberScale):
            return self.scale_est(self.df_resid, self.nobs, resid)
        else:
            return scale.scale_est(self, resid)**2

    def fit(self, maxiter=50, tol=1e-8, scale_est='mad', init=None, cov='H1',
            update_scale=True, conv='dev'):
        """
        Fits the model using iteratively reweighted least squares.

        The IRLS routine runs until the specified objective converges to `tol`
        or `maxiter` has been reached.

        Parameters
        ----------
        conv : string
            Indicates the convergence criteria.
            Available options are "coefs" (the coefficients), "weights" (the
            weights in the iteration), "sresid" (the standardized residuals),
            and "dev" (the un-normalized log-likelihood for the M
            estimator).  The default is "dev".
        cov : string, optional
            'H1', 'H2', or 'H3'
            Indicates how the covariance matrix is estimated.  Default is 'H1'.
            See rlm.RLMResults for more information.
        init : string
            Specifies method for the initial estimates of the parameters.
            Default is None, which means that the least squares estimate
            is used.  Currently it is the only available choice.
        maxiter : int
            The maximum number of iterations to try. Default is 50.
        scale_est : string or HuberScale()
            'mad' or HuberScale()
            Indicates the estimate to use for scaling the weights in the IRLS.
            The default is 'mad' (median absolute deviation.  Other options are
            'HuberScale' for Huber's proposal 2. Huber's proposal 2 has
            optional keyword arguments d, tol, and maxiter for specifying the
            tuning constant, the convergence tolerance, and the maximum number
            of iterations. See statsmodels.robust.scale for more information.
        tol : float
            The convergence tolerance of the estimate.  Default is 1e-8.
        update_scale : Bool
            If `update_scale` is False then the scale estimate for the
            weights is held constant over the iteration.  Otherwise, it
            is updated for each fit in the iteration.  Default is True.

        Returns
        -------
        results : object
            statsmodels.rlm.RLMresults
        """
        if not cov.upper() in ["H1","H2","H3"]:
            raise ValueError("Covariance matrix %s not understood" % cov)
        else:
            self.cov = cov.upper()
        conv = conv.lower()
        if not conv in ["weights","coefs","dev","sresid"]:
            raise ValueError("Convergence argument %s not understood" \
                % conv)
        self.scale_est = scale_est
        if (isinstance(scale_est,
                       string_types) and scale_est.lower() == "stand_mad"):
            from warnings import warn
            warn("stand_mad is deprecated and will be removed in 0.7.0",
                 FutureWarning)

        wls_results = lm.WLS(self.endog, self.exog).fit()
        if not init:
            self.scale = self._estimate_scale(wls_results.resid)

        history = dict(params = [np.inf], scale = [])
        if conv == 'coefs':
            criterion = history['params']
        elif conv == 'dev':
            history.update(dict(deviance = [np.inf]))
            criterion = history['deviance']
        elif conv == 'sresid':
            history.update(dict(sresid = [np.inf]))
            criterion = history['sresid']
        elif conv == 'weights':
            history.update(dict(weights = [np.inf]))
            criterion = history['weights']

        # done one iteration so update
        history = self._update_history(wls_results, history, conv)
        iteration = 1
        converged = 0
        while not converged:
            self.weights = self.M.weights(wls_results.resid/self.scale)
            wls_results = lm.WLS(self.endog, self.exog,
                                 weights=self.weights).fit()
            if update_scale is True:
                self.scale = self._estimate_scale(wls_results.resid)
            history = self._update_history(wls_results, history, conv)
            iteration += 1
            converged = _check_convergence(criterion, iteration, tol, maxiter)
        results = RLMResults(self, wls_results.params,
                            self.normalized_cov_params, self.scale)

        history['iteration'] = iteration
        results.fit_history = history
        results.fit_options = dict(cov=cov.upper(), scale_est=scale_est,
                                   norm=self.M.__class__.__name__, conv=conv)
        #norm is not changed in fit, no old state

        #doing the next causes exception
        #self.cov = self.scale_est = None #reset for additional fits
        #iteration and history could contain wrong state with repeated fit
        return RLMResultsWrapper(results)
