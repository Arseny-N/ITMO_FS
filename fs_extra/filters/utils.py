from functools import partial
from math import log

import numpy as np
from scipy import sparse as sp
from sklearn.feature_selection import mutual_info_classif as MI

import filters
from utils import generate_features


class DefaultMeasures:
    @staticmethod
    def fit_criterion_measure(X, y):
        x = np.asarray(X)  # Converting input data to numpy array
        y = np.asarray(y.reshape((-1,)))

        fc = np.zeros(x.shape[1])  # Array with amounts of correct predictions for each feature

        tokens_n = np.max(y) + 1  # Number of different class tokens

        centers = np.empty(tokens_n)  # Array with centers of sets of feature values for each class token
        variances = np.empty(tokens_n)  # Array with variances of sets of feature values for each class token
        # Each of arrays above will be separately calculated for each feature

        distances = np.empty(tokens_n)  # Array with distances between sample's value and each class's center
        # This array will be separately calculated for each feature and each sample

        for feature_index, feature in enumerate(x.T):  # For each feature
            # Initializing utility structures
            class_values = [[] for _ in range(tokens_n)]  # Array with lists of feature values for each class token
            for index, value in enumerate(y):  # Filling array
                class_values[value].append(feature[index])
            for token, values in enumerate(class_values):  # For each class token's list of feature values
                tmp_arr = np.array(values)
                centers[token] = np.mean(tmp_arr)
                variances[token] = np.var(tmp_arr)

            # Main calculations
            for sample_index, value in enumerate(feature):  # For each sample value
                for i in range(tokens_n):  # For each class token
                    # Here can be raise warnings by 0/0 division. In this case, default results
                    # are interpreted correctly
                    distances[i] = np.abs(value - centers[i]) / variances[i]
                fc[feature_index] += np.argmin(distances) == y[sample_index]

        fc /= y.shape[0]
        return dict(zip(generate_features(x), fc))

    @staticmethod
    def __calculate_F_ratio(row, y_data):
        """
        Calculates the Fisher ratio of the row passed to the data
        :param row: ndarray, feature
        :param y_data: ndarray, labels
        :return: int, fisher_ratio
        """
        inter_class = 0.0
        intra_class = 0.0
        for value in np.unique(y_data):
            index_for_this_value = np.where(y_data == value)[0]
            n = np.sum(row[index_for_this_value])
            mu = np.mean(row[index_for_this_value])
            var = np.var(row[index_for_this_value])
            inter_class += n * np.power((mu - mu), 2)
            intra_class += (n - 1) * var

        f_ratio = inter_class / intra_class
        return f_ratio

    @classmethod
    def __f_ratio_measure(cls, X, y, n):
        assert not 1 < X.shape[1] < n, 'incorrect number of features'
        f_ratios = []
        for feature in X.T:
            f_ratio = DefaultMeasures.__calculate_F_ratio(feature, y.T)
            f_ratios.append(f_ratio)
        f_ratios = np.array(f_ratios)
        return np.argpartition(f_ratios, -n)[-n:]

    @staticmethod
    def f_ratio_measure(n):
        return partial(DefaultMeasures.__f_ratio_measure, n=n)

    @staticmethod
    def gini_index(X, y):
        cum_x = np.cumsum(X / np.linalg.norm(X, 1, axis=0), axis=0)
        cum_y = np.cumsum(y / np.linalg.norm(y, 1))
        diff_x = (cum_x[1:] - cum_x[:-1])
        diff_y = (cum_y[1:] + cum_y[:-1])
        return np.abs(1 - np.sum(np.multiply(diff_x.T, diff_y).T, axis=0))

    # Calculate the entropy of y.
    @staticmethod
    def __cal_entropy(y):
        dict_label = dict()
        for label in y:
            if label not in dict_label:
                dict_label.update({label: 1})
            else:
                dict_label[label] += 1
        entropy = 0.0
        for i in dict_label.values():
            entropy += -i / len(y) * log(i / len(y), 2)
        return entropy

    @staticmethod
    def ig_measure(X, y):
        y = y.reshape((-1,))
        entropy = DefaultMeasures.__cal_entropy(y)
        list_f = np.empty(X.shape[1])
        for index in range(len(X.T)):
            dict_i = dict()
            for i in range(len(X.T[index])):
                if X.T[index][i] not in dict_i:
                    dict_i.update({X.T[index][i]: [i]})
                else:
                    dict_i[X.T[index][i]].append(i)
            # print(dict_i)

            # Conditional entropy of a feature.
            con_entropy = 0.0
            # get corresponding values in y.
            for f in dict_i.values():
                # Probability of each class in a feature.
                p = len(f) / len(X.T[0])
                # Dictionary of corresponding probability in labels.
                dict_y = dict()
                for i in f:
                    if y[i] not in dict_y:
                        dict_y.update({y[i]: 1})
                    else:
                        dict_y[y[i]] += 1

                # calculate the probability of corresponding label.
                sub_entropy = 0.0
                for l in dict_y.values():
                    sub_entropy += -l / sum(dict_y.values()) * log(l / sum(dict_y.values()), 2)

                con_entropy += sub_entropy * p
            list_f[index] = entropy - con_entropy
        return list_f

    @staticmethod
    def __contingency_matrix(labels_true, labels_pred):
        """Build a contingency matrix describing the relationship between labels.

            Parameters
            ----------
            labels_true : int array, shape = [n_samples]
                Ground truth class labels to be used as a reference

            labels_pred : array, shape = [n_samples]
                Cluster labels to evaluate

            Returns
            -------
            contingency : {array-like, sparse}, shape=[n_classes_true, n_classes_pred]
                Matrix :math:`C` such that :math:`C_{i, j}` is the number of samples in
                true class :math:`i` and in predicted class :math:`j`. If
                ``eps is None``, the dtype of this array will be integer. If ``eps`` is
                given, the dtype will be float.
            """
        classes, class_idx = np.unique(labels_true, return_inverse=True)
        clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
        n_classes = classes.shape[0]
        n_clusters = clusters.shape[0]
        # Using coo_matrix to accelerate simple histogram calculation,
        # i.e. bins are consecutive integers
        # Currently, coo_matrix is faster than histogram2d for simple cases
        # TODO redo it with numpy
        contingency = sp.coo_matrix((np.ones(class_idx.shape[0]),
                                     (class_idx, cluster_idx)),
                                    shape=(n_classes, n_clusters),
                                    dtype=np.int)
        contingency = contingency.tocsr()
        contingency.sum_duplicates()
        return contingency

    @staticmethod
    def __mi(U, V):
        contingency = DefaultMeasures.__contingency_matrix(U, V)
        nzx, nzy, nz_val = sp.find(contingency)
        contingency_sum = contingency.sum()
        pi = np.ravel(contingency.sum(axis=1))
        pj = np.ravel(contingency.sum(axis=0))
        log_contingency_nm = np.log(nz_val)
        contingency_nm = nz_val / contingency_sum
        # Don't need to calculate the full outer product, just for non-zeroes
        outer = (pi.take(nzx).astype(np.int64, copy=False)
                 * pj.take(nzy).astype(np.int64, copy=False))
        log_outer = -np.log(outer) + log(pi.sum()) + log(pj.sum())
        mi = (contingency_nm * (log_contingency_nm - log(contingency_sum)) +
              contingency_nm * log_outer)
        return mi.sum()

    # return list(index)
    # i'm used MID as default. Or we need add ability to choose info_gain?
    @classmethod
    def __mrmr_measure(cls, X, y, n):
        assert not 1 < X.shape[1] < n, 'incorrect number of features'

        x = np.array(X)
        y = np.array(y).ravel()

        # print([DefaultMeasures.__mi(X[:, j].reshape(-1, 1), y) for j in range(X.shape[1])])
        return [MI(x[:, j].reshape(-1, 1), y) for j in range(x.shape[1])]

    @staticmethod
    def mrmr_measure(n):
        return partial(DefaultMeasures.__mrmr_measure, n=n)

    # RandomFilter = filters.RandomFilter() # TODO: bad .run() interface; .run() feature_names; no default constructor

    # SymmetricUncertainty = filters.SymmetricUncertainty()  # TODO

    @staticmethod
    def spearman_corr(X, y):
        n = X.shape[0]
        c = 6 / (n * (n - 1) * (n + 1))
        dif = X - np.hstack(tuple([y] * X.shape[1]))
        return 1 - c * np.sum(dif * dif, axis=0)

    @staticmethod
    def pearson_corr(X, y):
        x_dev = X - np.mean(X, axis=0)
        y_dev = y - np.mean(y)
        sum_dev = y_dev.T.dot(x_dev)
        sq_dev_x = x_dev * x_dev
        sq_dev_y = y_dev * y_dev
        return (sum_dev / np.sqrt(np.sum(sq_dev_y) * np.sum(sq_dev_x))).reshape((-1,))

    # TODO Fehner correlation,concordation coef
    VDM = filters.VDM()  # TODO: probably not a filter


# print(DefaultMeasures.SpearmanCorrelation)

GLOB_MEASURE = {"FitCriterion": DefaultMeasures.fit_criterion_measure,
                "FRatio": DefaultMeasures.f_ratio_measure,
                "GiniIndex": DefaultMeasures.gini_index,
                "InformationGain": DefaultMeasures.ig_measure,
                "MrmrDiscrete": DefaultMeasures.mrmr_measure,
                "SpearmanCorr": DefaultMeasures.spearman_corr,
                "PearsonCorr": DefaultMeasures.pearson_corr}


class _DefaultCuttingRules:
    @staticmethod
    def select_best_by_value(value):
        return partial(_DefaultCuttingRules.__select_by_value, value=value, more=True)

    @staticmethod
    def select_worst_by_value(value):
        return partial(_DefaultCuttingRules.__select_by_value, value=value, more=False)

    @staticmethod
    def __select_by_value(scores, value, more=True):
        features = []
        for key, sc_value in scores.items():
            if more:
                if sc_value >= value:
                    features.append(key)
            else:
                if sc_value <= value:
                    features.append(key)
        return features

    @staticmethod
    def select_k_best(k):
        return partial(_DefaultCuttingRules.__select_k, k=k, reverse=True)

    @staticmethod
    def select_k_worst(k):
        return partial(_DefaultCuttingRules.__select_k, k=k)

    @classmethod
    def __select_k(cls, scores, k, reverse=False):
        if type(k) != int:
            raise TypeError("Number of features should be integer")
        return [keys[0] for keys in sorted(scores.items(), key=lambda kv: kv[1], reverse=reverse)[:k]]


GLOB_CR = {"Best by value": _DefaultCuttingRules.select_best_by_value,
           "Worst by value": _DefaultCuttingRules.select_worst_by_value,
           "K best": _DefaultCuttingRules.select_k_best,
           "K worst": _DefaultCuttingRules.select_k_worst}
