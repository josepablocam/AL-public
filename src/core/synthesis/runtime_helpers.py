import copy

import numpy as np
import pandas as pd
from scipy import sparse


def put_column(base, ix, v):
    if isinstance(v, pd.Series):
        v = v.values
    if isinstance(base, np.ndarray) or sparse.issparse(base):
        shape = base[:, ix].shape
        base[:, ix] = v.reshape(shape)
    elif isinstance(base, pd.DataFrame):
        base[ix] = v
    else:
        raise Exception("Unhandled base type")


def get_columns(base, ix):
    if isinstance(base, np.ndarray) or sparse.issparse(base):
        selected = base[:, ix]
        if len(selected.shape) == 1:
            selected = selected.reshape(-1, 1)
        return selected
    elif isinstance(base, pd.DataFrame):
        selected = base[ix]
        if len(selected.shape) == 1:
            selected = selected.values.reshape(-1, 1)
        return selected
    else:
        raise Exception("Unhandled base type")


def get_column_ixs(base):
    if isinstance(base, np.ndarray):
        return range(0, base.shape[1])
    elif sparse.issparse(base):
        return np.unique(base.nonzero()[1])
    elif isinstance(base, pd.DataFrame):
        return base.columns
    else:
        raise Exception("Unhandled base type")


def remove_column_ixs(base, remove_ixs):
    new_column_ixs = [i for i in get_column_ixs(base) if not i in remove_ixs]
    return get_columns(base, new_column_ixs)


def combine_values(base, results):
    init = results[0]
    # if new results are sparse
    # we are forced to make everything sparse
    if sparse.issparse(init):
        if not sparse.issparse(base):
            if isinstance(base, pd.DataFrame):
                base = base.values
            # needs to be numeric type to convert to sparse matrix
            if base.dtype not in [np.dtype(float), np.dtype(int)]:
                base = base.astype(np.dtype(float))
                base = sparse.csr_matrix(base)
        # return csr since it is column indexable
        return sparse.hstack([base] + results, format='csr')
    # for other cases, favor type of base
    elif isinstance(init, np.ndarray):
        if isinstance(base, np.ndarray):
            return np.hstack([base] + results)
        elif isinstance(base, pd.DataFrame):
            new_cols = np.hstack(results)
            new_df = pd.DataFrame(new_cols)
            return pd.concat([base, new_df], axis=1)
        else:
            raise Exception("Unhandled base type")
    elif isinstance(init, pd.DataFrame):
        if isinstance(base, np.ndarray):
            results = [v.values for v in results]
            return np.hstack([base] + results)
        elif isinstance(base, pd.DataFrame):
            return pd.concat([base] + results, axis=1)
        else:
            raise Exception("Unhandled base type")
    else:
        raise Exception("Unhandled result type")


def smudge_column(col):
    return np.apply_along_axis(lambda x: '__%s' % str(x[0]), 1, col)


# an upper limit on the size of the table in num of columns
# that we're willing to iterate over
COLUMN_LOOP_LIMIT = 1000


class ColumnLoop(object):
    def __init__(self, sklearn_op):
        self.base_op = sklearn_op
        self.ops = []
        self.ixs = []
        self.flatten = []
        self.smudge = []
        self._fit = False
        self._import = 'runtime_helpers'

    @staticmethod
    def code():
        return 'runtime_helpers.ColumnLoop'

    @staticmethod
    def is_allowed(arg, state):
        ncols = len(get_column_ixs(arg.value(state)))
        return ncols <= COLUMN_LOOP_LIMIT

    def _fit_column(self, X, ix):
        o = self.base_op()
        col = get_columns(X, ix)
        # we may need to massage the column
        for smudge in [False, True]:
            for flatten in [False, True]:
                try:
                    mod_col = col
                    is_str_col = col.dtype == np.dtype('object')
                    if smudge and is_str_col:
                        mod_col = smudge_column(mod_col)
                    if flatten:
                        mod_col = mod_col.flatten()
                    o.fit(mod_col)
                    self.ixs.append(ix)
                    self.ops.append(o)
                    self.smudge.append(smudge and is_str_col)
                    self.flatten.append(flatten)
                    # at least one transform was succesfully fit
                    self._fit = True
                    return
                except:
                    pass
        return

    def fit(self, X):
        # don't re-fit if already done
        # if self._fit:
        #   return self
        ixs = []
        ops = []
        for i in get_column_ixs(X):
            self._fit_column(X, i)
        return self

    def transform(self, X):
        if not self._fit:
            raise Exception("Must fit first")
        results = []
        for o, must_smudge, must_flatten, i in zip(self.ops, self.smudge,
                                                   self.flatten, self.ixs):
            col = get_columns(X, i)
            if must_smudge:
                col = smudge_column(col)
            if must_flatten:
                col = col.flatten()
            transformed_col = o.transform(col)
            # make it a column vector explicitly, if necessary
            if len(transformed_col.shape) == 1:
                transformed_col = transformed_col.reshape(-1, 1)
            results.append(transformed_col)
        return self.update_values(X, results)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def update_values(self, X, results):
        X = copy.deepcopy(X)
        max_num_cols = max([r.shape[1] for r in results])
        # can just override directly
        if max_num_cols == 1:
            for i, v in zip(self.ixs, results):
                put_column(X, i, v)
        else:
            # remove columns that may have been modified
            X_removed = remove_column_ixs(X, self.ixs)
            # concatenate column-wise at the end, the results
            X = combine_values(X_removed, results)
        return X
