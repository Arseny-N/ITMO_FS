from sklearn.feature_selection.univariate_selection import _BaseFilter, f_classif, check_is_fitted

import numpy as np

class SelectFunc(_BaseFilter):
    """Select features according to a costum rule
    Parameters
    ----------
    mask_func : callable
        Function taking an array of scores (n_features, ) and returining a mask of 
        selected elements 

    score_func : callable
        Function taking two arrays X and y, and returning a pair of arrays
        (scores, pvalues) or a single array with scores.
        Default is f_classif (see below "See also"). The default function only
        works with classification tasks.
    
    Attributes
    ----------
    scores_ : array-like, shape=(n_features,)
        Scores of features.
    pvalues_ : array-like, shape=(n_features,)
        p-values of feature scores, None if `score_func` returned only scores.
    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.feature_selection import SelectKBest, chi2
    >>> X, y = load_digits(return_X_y=True)
    >>> X.shape
    (1797, 64)
    >>> # Drop best 10 and worst 10 features
    >>> def select_range(begin, end):
    >>>     def fn(scores):
    >>>         mask = np.zeros_like(scores, dtype=bool)
    >>>         mask[begin:end] = 1
    >>>         return mask
    >>>     return fn
    >>> 
    >>> X_new = SelectFunc(chi2, select_range(10, -10), sorted=True).fit_transform(X, y)
    >>> X_new.shape
    (1797, 44)
    >>> # Select features with socres highter 500
    >>> X_new = SelectFunc(chi2, lambda scores: scores > 500).fit_transform(X, y)
    >>> X_new.shape
    (1797, 49)
    >>> # Drop baes 10 and wors 10 features and take every second feature
    >>> def select_ixs(ixs):
    >>>     def fn(scores):
    >>>         mask = np.zeros_like(scores, dtype=bool)
    >>>         mask[ixs] = 1
    >>>         return mask
    >>>     return fn
    >>> 
    >>> X_new = SelectFunc(chi2, select_ixs(np.r_[10:X.shape[-1]-10:2]), sorted=True).fit_transform(X, y)
    >>> X_new.shape
    (1797, 22)
    See also
    --------
    f_classif: ANOVA F-value between label/feature for classification tasks.
    mutual_info_classif: Mutual information for a discrete target.
    chi2: Chi-squared stats of non-negative features for classification tasks.
    f_regression: F-value between label/feature for regression tasks.
    mutual_info_regression: Mutual information for a continuous target.
    SelectPercentile: Select features based on percentile of the highest scores.
    SelectFpr: Select features based on a false positive rate test.
    SelectFdr: Select features based on an estimated false discovery rate.
    SelectFwe: Select features based on family-wise error rate.
    GenericUnivariateSelect: Univariate feature selector with configurable mode.
    """

    def __init__(self, score_func=f_classif, mask_func=None, sorted=False):
        super().__init__(score_func)
        self.mask_func = mask_func
        self.sorted = sorted
        # assert mask_func is not None 


    def _get_support_mask(self):
        check_is_fitted(self, 'scores_')

        if self.sorted:
            ixs = np.argsort(self.scores_)
            inv_perm = np.zeros_like(ixs)
            inv_perm[ixs] = np.arange(len(ixs))

            return self.mask_func(self.scores_[ixs])[inv_perm]
                
        return self.mask_func(self.scores_)



from sklearn.feature_selection.univariate_selection import _BaseFilter, f_classif, check_is_fitted, SelectPercentile, SelectKBest, SelectFpr, SelectFdr, SelectFwe, f_classif


def _is_select_obj(x):
    return all( hasattr(x, attr) 
            for attr in (
                '_get_param_names', 'set_params', 
                '_check_params', '_get_support_mask'))

class GenericUnivariateSelect(_BaseFilter):
    """Univariate feature selector with configurable strategy.
    Read more in the :ref:`User Guide <univariate_feature_selection>`.
    Parameters
    ----------
    score_func : callable
        Function taking two arrays X and y, and returning a pair of arrays
        (scores, pvalues). For modes 'percentile' or 'kbest' it can return
        a single array scores.
    mode : {'percentile', 'k_best', 'fpr', 'fdr', 'fwe'}
        Feature selection mode.
    param : float or int depending on the feature selection mode
        Parameter of the corresponding mode.
    Attributes
    ----------
    scores_ : array-like, shape=(n_features,)
        Scores of features.
    pvalues_ : array-like, shape=(n_features,)
        p-values of feature scores, None if `score_func` returned scores only.
    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.feature_selection import GenericUnivariateSelect, chi2
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> X.shape
    (569, 30)
    >>> transformer = GenericUnivariateSelect(chi2, lambda scores: scores > 500)
    >>> X_new = transformer.fit_transform(X, y)
    >>> X_new.shape
    (569, 5)
    >>> # Drop best 10 and worst 10 features
    >>> def select_range(begin, end):
    >>>     def fn(scores):
    >>>         mask = np.zeros_like(scores, dtype=bool)
    >>>         mask[begin:end] = 1
    >>>         return mask
    >>>     return fn
    >>> 
    >>> transformer = GenericUnivariateSelect(chi2, select_range(1, 10), param=True)
    >>> X_new = transformer.fit_transform(X, y)
    >>> X_new.shape
    (1797, 9)
    See also
    --------
    f_classif: ANOVA F-value between label/feature for classification tasks.
    mutual_info_classif: Mutual information for a discrete target.
    chi2: Chi-squared stats of non-negative features for classification tasks.
    f_regression: F-value between label/feature for regression tasks.
    mutual_info_regression: Mutual information for a continuous target.
    SelectPercentile: Select features based on percentile of the highest scores.
    SelectKBest: Select features based on the k highest scores.
    SelectFpr: Select features based on a false positive rate test.
    SelectFdr: Select features based on an estimated false discovery rate.
    SelectFwe: Select features based on family-wise error rate.
    """
    #
    # TODO: write docs
    #

    _selection_modes = {'percentile': SelectPercentile,
                        'k_best': SelectKBest,
                        'fpr': SelectFpr,
                        'fdr': SelectFdr,
                        'fwe': SelectFwe}

    def __init__(self, score_func=f_classif, mode='percentile', param=1e-5):
        super().__init__(score_func)
        self.mode = mode
        self.param = param

    def _make_selector(self):
        if _is_select_obj(self.mode):
            selector = self.mode
        elif callable(self.mode):
            selector = SelectFunc(score_func=f_classif, mask_func=self.mode)
        else:    
            selector = self._selection_modes[self.mode](score_func=self.score_func)

        # Now perform some acrobatics to set the right named parameter in
        # the selector

        if _is_select_obj(self.mode):
            return selector

        possible_params = selector._get_param_names()
        possible_params.remove('score_func')

        if callable(self.mode):
            if not isinstance(self.param, bool):
                return selector

            possible_params.remove('mask_func')
                

        selector.set_params(**{possible_params[0]: self.param})

        return selector

    def _check_params(self, X, y):

        if not (self.mode in self._selection_modes or _is_select_obj(self.mode) or callable(self.mode)):
            raise ValueError("The mode passed should be one of %s, %r,"
                             " (type %s) was passed."
                             % (self._selection_modes.keys(), self.mode,
                                type(self.mode)))

        self._make_selector()._check_params(X, y)

    def _get_support_mask(self):
        check_is_fitted(self, 'scores_')

        selector = self._make_selector()
        selector.pvalues_ = self.pvalues_
        selector.scores_ = self.scores_

        return selector._get_support_mask()