from collections import Counter
import sys
import argparse
import io
import os
import pickle
import logging

import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import tqdm

from synthesis.generate import ProgramGenerator
from synthesis.learn import NGramCallScorer
from synthesis.program import run_with_timeout_in_thread

# starts sampling after this many rows
MAX_NUM_ROWS = 10000
# won't try to fit final version of model
# for more than this amount of minutes
MAX_FINAL_FIT_MINS = 2

# w.r.t to this location
PREDICTOR_PATH = os.path.join(
    os.path.dirname(__file__),
    "resources",
    "predictor.pkl",
)


def wrapped_f1_score(y_true, y_pred, average='macro'):
    # f1 can fail if y_pred is not the same type etc
    try:
        return f1_score(y_true, y_pred, average=average)
    except:
        return 0


def subsample_data(X, y, seed=42):
    ixs = np.arange(0, X.shape[0])
    # note that this has repeats as well
    chosen_ixs = np.random.choice(ixs, size=MAX_NUM_ROWS)
    if isinstance(X, np.ndarray):
        X = X[chosen_ixs, :]
    elif isinstance(X, pd.DataFrame):
        X = X.iloc[chosen_ixs].copy()
    elif scipy.issparse(X):
        X = X.todense()[chosen_ixs, :]
    else:
        raise Exception("Unhandled X")
    if isinstance(y, np.ndarray):
        y = y[chosen_ixs]
    elif isinstance(y, pd.Series):
        y = y.iloc[chosen_ixs].copy()
    else:
        raise Exception("Unhandled y")
    return X, y


def prune_ill_pipelines(progs, X, y):
    fitted_progs = []
    for p in tqdm.tqdm(progs):
        # use variant of timeout that spins up separate process for this
        # seems to work, version with stopit.ThreadingTimeout doesn't
        # interrupt atomic ops and hangs here often
        try:
            fitted_p = run_with_timeout_in_thread(
                MAX_FINAL_FIT_MINS * 60, p.fit_final, X, y)
        except TimeoutError:
            print("Timedout fitting final dataset")
            fitted_p = None
        if fitted_p is not None:
            # only include pipelines where can call predict at the end
            try:
                fitted_p.predict(X)
                # toss out bad programss that change size of y_test
                # by calling score and seeing if it works
                fitted_p.score(X, y)
                if fitted_p.score_ > 0:
                    fitted_progs.append(fitted_p)
            except:
                pass
    return fitted_progs


def predict_ensemble(programs, X):
    predictions = []
    ixs = np.arange(0, X.shape[0])
    for p in programs:
        p_preds = list(zip(ixs, p.predict(X)))
        predictions.extend(p_preds)
    df = pd.DataFrame(predictions, columns=["ix", "y"])
    most_common = df.groupby("ix").agg(lambda x: x.value_counts().index[0])
    most_common = most_common.reset_index()
    return most_common.y.values


def predict_proba_ensemble(programs, X):
    predictions = []
    ixs = np.arange(0, X.shape[0]).reshape(-1, 1)
    for p in programs:
        try:
            p_preds = np.hstack((ixs, p.predict_proba(X)))
            predictions.append(p_preds)
        except AttributeError:
            pass

    count_classes = Counter()
    for entry in predictions:
        count_classes[entry.shape[1] - 1] += 1

    # most common determines
    num_classes = count_classes.most_common()[0][0]
    clean_predictions = []
    for entry in predictions:
        if (entry.shape[1] - 1) == num_classes:
            clean_predictions.extend(entry)

    cols = ["ix"] + ["p_{}".format(i) for i in range(num_classes)]
    df = pd.DataFrame(clean_predictions, columns=cols)
    probs = df.groupby("ix").mean().values
    # normalize these to one
    return probs / probs.sum(axis=1).reshape(-1, 1)


class AL(object):
    def __init__(
            self,
            predictor_path=PREDICTOR_PATH,
            max_num_rows=MAX_NUM_ROWS,
            task_type="classification",
    ):
        self.max_num_rows = max_num_rows
        with open(predictor_path, "rb") as fin:
            self.predictor = pickle.load(fin)
        self.task_type = task_type

    def fit(self, X, y, test_size=0.2, depth=3, beam=10, timeout=60):
        orig_X, orig_y = X, y
        if X.shape[0] > self.max_num_rows:
            X, y = subsample_data(X, y)

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=test_size,
        )
        if self.task_type == "classification":
            metric = wrapped_f1_score
        else:
            metric = None

        generator = ProgramGenerator(
            self.predictor,
            X_train,
            y_train,
            X_val,
            y_val,
            depth=depth,
            beam=beam,
            timeout=timeout,
            metric=metric,
        )
        progs = generator.generate()
        self.programs = prune_ill_pipelines(progs, orig_X, orig_y)
        return self

    def get_programs(self):
        return self.programs

    def predict(self, X, index=None):
        if index is None:
            return predict_ensemble(self.programs, X)
        else:
            return self.programs[index].predict(X)

    def predict_proba(self, X, index=None):
        if index is None:
            return predict_proba_ensemble(self.programs, X)
        else:
            return self.programs[index].predict_proba(X)
