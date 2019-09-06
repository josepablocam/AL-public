# Construct meta-features for learning task
from collections import defaultdict
import random
random.seed(0)

import numpy as np
from scipy import stats, sparse
import pandas as pd

# Features drawn from:
# A meta-learning approach to automatic kernel selection for support vector machines (Ali, Smith-Miles)


# max size of a nominal column. We can end up with really expensive operations otherwise
class Memoizer(object):
    def __init__(self):
        self.memo = {}

    def get(self, arg, fun=None):
        id_arg = id(arg)
        val = self.memo.get(id_arg, None)
        if val is None and fun is not None:
            val = fun(arg)
            self.memo[id_arg] = val
        return val


class DirectSummarizer(object):
    def __init__(self):
        self.memoizer = Memoizer()

    def summarize(self, arg):
        return self.memoizer.get(arg, get_features)


class SampleSummarizer(object):
    def __init__(self, n, fraction):
        self.n = n
        self.fraction = fraction
        self.memoizer = Memoizer()

    def summarize(self, arg):
        return self.memoizer.get(arg, self._summarize)

    def _combine_sample_features(self, sample_features):
        combined = defaultdict(lambda: [])
        for feats in sample_features:
            for k, v in feats.items():
                combined[k].append(v)
        # average numeric features, take first of other types
        return {
            k:
            (np.nanmean(v) if is_numeric_type(np.dtype(type(v[0]))) else v[0])
            for k, v in combined.items()
        }

    def _summarize(self, arg):
        # get underlying data
        data = get_data(arg)
        if data is not None:
            # resample a n times
            sample_size = int(data.shape[0] * self.fraction)
            samples, _ = sample_multiple(
                data, n_obs=sample_size, n_iters=self.n
            )
            print(len(samples))
            acc = []
            for s in samples:
                acc.append(get_features(s))
            # combine feature summaries
            feats = self._combine_sample_features(acc)
            feats['sample_params'] = (self.n, self.fraction)
            # override the sampled value number
            feats['num_rows'] = data.shape[0]
            return feats
        else:
            return get_features(arg)


MAX_NOMINAL_LEN = 100


# type helpers
def canonical_type_name(t):
    return np.dtype(t).kind


def is_numeric_type(dtype):
    # based on https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.dtypes.html
    numeric_kinds = set(['i', 'u', 'f', 'c'])
    return dtype.kind in numeric_kinds


def is_nominal_type(dtype):
    nominal_kinds = set(['O', 'S', 'a', 'U'])
    return dtype.kind in nominal_kinds


def is_boolean_type(dtype):
    bool_types = set(['?', 'b'])
    return dtype.kind in bool_types


# start of meta feature functions
def zscore_stats(x, axis):
    results = {}
    zscore = stats.zscore(x, axis=axis)
    results['min_zscore'] = np.nanmean(np.min(zscore, axis=axis))
    results['max_zscore'] = np.nanmean(np.max(zscore, axis=axis))
    results['zscore'] = np.nanmean(zscore)
    return results


def count_nonzero(m, axis):
    return np.unique(np.nonzero(m)[1], return_counts=True)[1]


def dense_numeric_ops():
    ops = {
        'mean': np.mean,
        'geometric_mean': stats.mstats.gmean,
        'median': np.median,
        'zscore_stats': zscore_stats,
        'iqr': stats.iqr,
        'std': np.std,
        'max': np.max,
        'min': np.min,
        'count_nonzero': count_nonzero,
        'skew': stats.skew,
        'kurtosis': stats.kurtosis,
        'missing': lambda x, axis: np.mean(pd.isnull(x), axis=axis),
    }
    return ops


def sparse_spread_stats(m, axis):
    results = {}
    m_squared = m.copy()
    m_squared.data **= 2.0

    col_mean = m.mean(axis=axis)
    results['mean'] = np.mean(col_mean)

    col_var = m_squared.mean(axis=axis) - np.square(col_mean)
    col_sigma = np.sqrt(col_var)
    results['std'] = np.mean(np.sqrt(col_var))

    sigma_3 = np.multiply(col_var, col_sigma)
    m_cubed = m.copy()
    m_cubed.data **= 3.0
    col_skew = (
        m_cubed.mean(axis=axis) - 3 * np.multiply(col_mean, col_var) -
        np.power(col_mean, 3)
    ) / sigma_3
    results['skew'] = np.mean(col_skew)
    return results


def sparse_numeric_ops():
    ops = {
        'sparse_spread_stats':
        sparse_spread_stats,
        'count_nonzero':
        lambda m, axis: np.unique(m.nonzero()[1], return_counts=True)[1],
        'missing':
        lambda m, axis: np.isnan(m).mean(axis=axis).mean(),
    }
    return ops


def columnwise_numeric_descriptive_stats(mat):
    results = {}

    # what we can calculate here depends on whether
    # matrix is sparse or not
    if sparse.issparse(mat):
        ops = sparse_numeric_ops()
    else:
        ops = dense_numeric_ops()

    for nm, fun in ops.items():
        try:
            # apply them column-wise and then average across columns
            val = fun(mat, axis=0)
            if not isinstance(val, dict):
                val = {nm: np.mean(val)}
            # prefix keys indicating how we are aggregating this over matrix
            val = {("colwise_avg_%s" % nm): v for k, v in val.items()}
            results.update(val)
        except:
            print("Failed computing %s" % nm)
    return results


def columnwise_boolean_stats(mat):
    results = {}
    ops = {
        'colwise_max_fraction_true': np.max,
        'colwise_min_fraction_true': np.min,
        'colwise_avg_fraction_true': np.mean,
    }
    frac_true = np.mean(mat, axis=0)
    try:
        for nm, fun in ops.items():
            results[nm] = fun(frac_true)
    except:
        print("Failed computing %s" % nm)
        pass
    return results


def columnwise_nominal_stats(mat):
    nominal_stats = {}
    if sparse.issparse(mat):
        return nominal_stats
    if mat.shape[0] > MAX_NOMINAL_LEN:
        mat, info = sample(mat, MAX_NOMINAL_LEN)
        nominal_stats.update(info)
    # must apply reduction op before to ensure size can be reshaped appropriately by numpy
    freq_along_domain = lambda op, c: op(np.unique(c, return_counts=True)[1])
    ops = {
        'size_domain_nominal': lambda c: len(np.unique(c)),
        'mean_freq_nominal': lambda c: freq_along_domain(np.mean, c),
        'min_freq_nominal': lambda c: freq_along_domain(np.min, c),
        'max_freq_nominal': lambda c: freq_along_domain(np.max, c),
        'colwise_missing_nominal': lambda c: np.mean(pd.isnull(c)),
    }
    for nm, fun in ops.items():
        try:
            full_nm = "colwise_avg_%s" % nm
            nominal_stats[full_nm] = np.mean(np.apply_along_axis(fun, 0, mat))
        except:
            print("Failed computing colwise_avg_%s" % nm)
    return nominal_stats


def matrixwise_numeric_distribution_stats(mat):
    results = {}
    if sparse.issparse(mat):
        return results
    flat_mat = mat.flatten()
    if len(flat_mat) > 2e6:
        flat_mat, info = sample(flat_mat, int(100e3))
        assert (len(flat_mat) <= int(100e3))
        info = {("matrixwise_numeric_distribution_%s" % k): v
                for k, v in info.items()}
        results.update(info)
    ops = {
        'dist_norm': stats.norm(0, 1),
        'dist_chi2': stats.chi2(1),
        'dist_expon': stats.expon(0, 1),
        'dist_gamma': stats.gamma(1.0),
    }
    for nm, fun in ops.items():
        try:
            results['matrixwise_avg_%s_pdf' % nm] = np.mean(fun.pdf(flat_mat))
            results['matrixwise_avg_%s_cdf' % nm] = np.mean(fun.cdf(flat_mat))
        except:
            print("Failed computing %s" % nm)
            pass
    return results


def is_large_matrix(mat):
    return not (mat.shape[0] <= 1000 and mat.shape[1] < 100)


def matrixwise_numeric_descriptive_stats(mat):
    # numeric columns only
    results = {}
    # we don't want to compute this for large collections, too expensive
    try:
        if not is_large_matrix(mat):
            if sparse.issparse(mat):
                mat = mat.todense()
            if mat.shape[1] > 1:
                cor = np.corrcoef(mat)
                cor_vec = cor[np.triu_indices_from(cor, 1)]
                results['matrixwise_max_corr'] = np.max(cor_vec)
                results['matrixwise_min_corr'] = np.min(cor_vec)
                results['matrixwise_avg_corr'] = np.mean(cor_vec)
    except:
        print("Failed computing correlation")
        pass
    return results


# features on a collection of vectors (e.g. a matrix/table)


def features_numeric_matrix(mat):
    features = {}
    features.update(columnwise_numeric_descriptive_stats(mat))
    features.update(matrixwise_numeric_descriptive_stats(mat))
    features.update(matrixwise_numeric_distribution_stats(mat))
    return features


def features_nominal_matrix(mat):
    features = {}
    features.update(columnwise_nominal_stats(mat))
    return features


def features_boolean_matrix(mat):
    features = {}
    features.update(columnwise_boolean_stats(mat))
    return features


def get_features_pandas_df(df):
    features = {}
    column_type_predicates = {
        is_numeric_type, is_nominal_type, is_boolean_type
    }
    for type_pred in column_type_predicates:
        correct_type_cols = [
            c for c, t in zip(df.columns, df.dtypes) if type_pred(t)
        ]
        mat = df[correct_type_cols].values
        features.update(get_features_numpy_matrix(mat))

    return features


def get_features_numpy_matrix(mat):
    if mat.size == 0:
        return {}

    features = {}
    features['count_column_type_%s' % canonical_type_name(mat.dtype)
             ] = mat.shape[1]
    features['num_rows'] = mat.shape[0]
    # dispatch python
    type_based_features = {
        is_numeric_type: features_numeric_matrix,
        is_nominal_type: features_nominal_matrix,
        is_boolean_type: features_boolean_matrix
    }

    for type_pred, meta_feature_extractor in type_based_features.items():
        if type_pred(mat.dtype):
            features.update(meta_feature_extractor(mat))
            break
    return features


def sample(x, n):
    try:
        if isinstance(x, np.ndarray):
            idx = random.sample(range(x.shape[0]), n)
            return x[idx], {'num_rows_sampled': n}
        elif isinstance(x, pd.DataFrame):
            # sample data frames here
            return x.sample(n=n)
        else:
            idx = random.sample(range(len(x)), n)
            return [x[i] for i in idx], {'num_rows_sampled': n}
    except:
        # if couldn't sample, then just return original and we pay price with slow run
        return x, {}


def sample_multiple(x, n_obs, n_iters):
    assert (n_iters > 0)
    xs = []
    while n_iters > 0:
        s, info = sample(x, n_obs)
        xs.append(s)
        n_iters -= 1
    return xs, info


def get_data(arg):
    # return underlying array/matrix or dataframe
    if isinstance(arg, list) or isinstance(arg, set) or isinstance(arg,
                                                                   frozenset):
        return np.array(list(arg)).reshape(-1, 1)
    elif isinstance(arg, pd.DataFrame):
        # split into numeric and nonnumeric here
        return arg
    elif isinstance(arg, pd.core.groupby.DataFrameGroupBy):
        return arg.obj
    elif isinstance(arg, pd.Series):
        return arg.to_frame()
    elif isinstance(arg, pd.core.groupby.SeriesGroupBy):
        return arg.obj.to_frame()
    elif sparse.issparse(arg):
        # a lot of the stats will fail on sparse matrices, but get what we can
        # its too expensive to make this dense everywhere then compute, not worth it...
        # this is really expensive....
        return arg
    elif isinstance(arg, np.ndarray):
        # take all numeric columsn etc
        shape = arg.shape
        if len(shape) == 1:
            return arg.reshape(-1, 1)
        elif len(shape) == 2:
            return arg
        else:
            return None
    else:
        return None


def get_features(arg):
    feats = {}
    feats['arg_type'] = str(type(arg))
    data = get_data(arg)
    if isinstance(data, pd.DataFrame):
        feats.update(get_features_pandas_df(data))
    elif isinstance(data, np.ndarray) or sparse.issparse(data):
        feats.update(get_features_numpy_matrix(data))
    else:
        feats['unhandled'] = True
    return feats
