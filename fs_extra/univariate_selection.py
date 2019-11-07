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
