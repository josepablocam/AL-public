import sys
import os
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../traces/')
)

import copy
from collections import defaultdict
import glob
import itertools
import pickle

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn.model_selection import KFold

import synthesis.meta_features as mf
from synthesis import tag
from traces import *

START_LABEL = '<START>'


def load_data(trace_dir):
    pattern = '%s/project_*/scripts/*.pkl' % trace_dir
    file_names = glob.glob(pattern)
    acc = []
    for fn in file_names:
        try:
            with open(fn, 'rb') as f:
                ts = pickle.load(f)
                for t in ts:
                    t['script'] = fn
                    t['project'] = fn.split("/")[1]
                acc.append(ts)
        except Exception as err:
            print(err)
    # tag the entries
    acc = [tag.tag_entries(ts) for ts in acc]
    # extract canonical programs from each trace
    canonical_progs = [t for ts in acc for t in tag.linearize(ts)]
    # add in extra info
    canonical_progs = [add_extra_info(prog) for prog in canonical_progs]
    return canonical_progs


def remove_bad_features(d):
    clean_d = {}
    for k, v in d.items():
        bad = False
        # bad features, remove
        bad = k in set([
            'num_rows_sampled',
        ])

        # bad values
        try:
            bad |= np.isnan(v)
        except:
            pass
        try:
            bad |= np.isinf(v)
        except:
            pass
        if not bad:
            clean_d[k] = v
    return clean_d


def add_extra_info(l):
    acc = []
    for c in l:
        # complete name:
        c = copy.deepcopy(c)
        c['component_name'] = tag.get_qualified_name_component(c)
        c['nargs'] = len(c['args'])
        acc.append(c)
    return acc


def prepend_keys(d, prefix, sep='_'):
    return {(prefix + sep + k): v for k, v in d.items()}


def get_component(call):
    return call['component_name']


def construct_ngrams(xs, n):
    # extend with start label if necessary
    assert (n >= 1)
    ext = [START_LABEL] * (n - 1) + xs
    return list(zip(*[ext[i:] for i in range(n)]))


def construct_seq_unique_ngrams(xs, n):
    """
  This removes sequentially repeated elements in xs before
  constructing xs, and align these with the original sequence.
  for example
  bigrams of 1 1 1 2 => (START, 1) (START, 1) (START, 1) (1, 2)
  """
    seq_unique = list(map(lambda x: x[0], itertools.groupby(xs)))
    unique_ngrams = construct_ngrams(seq_unique, n)
    ngrams = []
    prev = None
    i = 0
    for x in xs:
        if prev and x != prev:
            # if not a sequentially repeated element
            # fetch a new ngram
            i += 1
        ngrams.append(unique_ngrams[i])
        prev = x
    return ngrams


def create_obs_set(canonical_progs, n):
    acc = []
    for prog in canonical_progs:
        components = [get_component(call) for call in prog]
        ngrams = construct_seq_unique_ngrams(components, n)
        for i, call in enumerate(prog):
            role = call['tag']
            # ngrams include self
            prev_components = ngrams[i][:-1]
            component = components[i]
            obs = (role, prev_components, call['args'], component)
            acc.append(obs)
    return [o for o in acc if o[0] != 'O']


def feature_matrix(obs, with_data=True, dict_vectorizer=None):
    X = []
    for role, prev_components, meta_feat_args in obs:
        # features for this observation
        features = {}
        # add in previous components as observation info
        for i, component in enumerate(prev_components):
            features['prev_component_%d' % i] = component
        # add in the role (tag) we're looking for
        features['role'] = role
        # arguments represented as meta features
        if with_data and meta_feat_args:
            for i, meta_arg in enumerate(meta_feat_args):
                clean_meta_arg = remove_bad_features(meta_arg)
                # add information on position of arg
                clean_meta_arg = prepend_keys(clean_meta_arg, "arg_%d" % i)
                features.update(clean_meta_arg)
        # add in to the matrix of observations
        X.append(features)
    if dict_vectorizer is None:
        dict_vectorizer = DictVectorizer(sparse=False)
        dict_vectorizer.fit(X)
    return dict_vectorizer.transform(X), dict_vectorizer


def target_matrix(components, label_encoder=None):
    if label_encoder is None:
        label_encoder = LabelEncoder()
        label_encoder.fit(components)
    return label_encoder.transform(components), label_encoder


def featurize(obs, with_data=True, dict_vectorizer=None, label_encoder=None):
    abstract_calls = [o[:-1] for o in obs]
    component_names = [o[-1] for o in obs]
    X, dv = feature_matrix(
        abstract_calls, with_data=with_data, dict_vectorizer=dict_vectorizer
    )
    y, le = target_matrix(component_names, label_encoder=label_encoder)
    return X, y, dv, le


class NGramCallScorer(object):
    def __init__(
            self,
            n,
            transform_predictor,
            model_predictor,
            transform_vectorizer,
            model_vectorizer,
            transform_encoder,
            model_encoder,
            with_data=True
    ):
        self.n = n
        # components to predict transformations
        self.transform_predictor = transform_predictor
        self.transform_vectorizer = transform_vectorizer
        self.transform_encoder = transform_encoder
        # components to predict models
        self.model_predictor = model_predictor
        self.model_vectorizer = model_vectorizer
        self.model_encoder = model_encoder
        self.with_data = with_data

    def sort_and_label(self, probs, labels, get_name):
        return [(get_name(labels[i]), p)
                for i, p in sorted(enumerate(probs), key=lambda x: -x[1])]

    def get_previous_components(self, prog):
        if self.n == 1:
            return []
        else:
            offset = self.n - 1
            ext = [START_LABEL] * offset + [c.f for c in prog.trace]
            return ext[-offset:]

    def predict_component(
            self, prog, args, role, component_model,
            component_feature_vectorizer, component_label_encoder
    ):
        # collect n previous calls from program
        prev_components = self.get_previous_components(prog)
        data = None
        if self.with_data:
            # extract current argument values
            arg_vals = [arg.value(prog.state) for arg in args]
            # compute argument meta features
            data = [mf.get_features(arg_val) for arg_val in arg_vals]
        obs = (role, prev_components, data)
        X, _ = feature_matrix([obs],
                              dict_vectorizer=component_feature_vectorizer,
                              with_data=self.with_data)
        probs = component_model.predict_log_proba(X)[0]
        return self.sort_and_label(
            probs, component_model.classes_,
            component_label_encoder.inverse_transform
        )

    def predict_transforms(self, prog, arg):
        return self.predict_component(
            prog, [arg], tag.TRANSFORM, self.transform_predictor,
            self.transform_vectorizer, self.transform_encoder
        )

    def predict_models(self, prog, arg1, arg2):
        return self.predict_component(
            prog, [arg1, arg2], tag.MODEL, self.model_predictor,
            self.model_vectorizer, self.model_encoder
        )

    def compute_log_prob(self, prog):
        return prog.log_prob


# use this to evaluate null hypothesis
class RandomScorer(object):
    """
  Returns components (transforms/models) in random order and with random (uniform) probabilities.
  Scores programs randomly using uniform distribution.
  """

    def __init__(self, transform_encoder, model_encoder, seed=None):
        self.transform_encoder = transform_encoder
        self.model_encoder = model_encoder
        self.random_state = np.random.RandomState(seed=seed)

    def _get_random_prob(self):
        return self.random_state.uniform(low=0., high=1.)

    def _get_random_logprob(self):
        prob = self._get_random_prob()
        return self._get_random_logprob() if prob == 0. else np.log(prob)

    def predict_component(self, labels):
        shuffled = self.random_state.permutation(labels)
        return [(l, self._get_random_logprob()) for l in shuffled]

    def predict_transforms(self, prog, arg):
        return self.predict_component(self.transform_encoder.classes_)

    def predict_models(self, prog, arg1, arg2):
        return self.predict_component(self.model_encoder.classes_)

    def compute_log_prob(self, prog):
        return self._get_random_logprob()


def get_all_transforms(predictor):
    return predictor.transform_encoder.classes_


def get_all_models(predictor):
    return predictor.model_encoder.classes_


def fit_model(model, obs, role, with_data=True):
    relevant_obs = [o for o in obs if o[0] == role]
    X, y, dv, le = featurize(relevant_obs, with_data=with_data)
    trained = model().fit(X, y) if model else None
    return (X, y, dv, le, trained)


def get_predictor(
        model,
        n,
        with_data=True,
        traces_path='../../../data/executed/large_sample/'
):
    progs = load_data(traces_path)
    obs = create_obs_set(progs, n)
    X_t, y_t, dv_t, le_t, transforms_predictor = fit_model(
        model, obs, tag.TRANSFORM, with_data=with_data
    )
    X_m, y_m, dv_m, le_m, models_predictor = fit_model(
        model, obs, tag.MODEL, with_data=with_data
    )
    return NGramCallScorer(
        n,
        transforms_predictor,
        models_predictor,
        dv_t,
        dv_m,
        le_t,
        le_m,
        with_data=with_data
    )


def get_random_predictor(traces_path='../../../data/executed/large_sample/'):
    progs = load_data(traces_path)
    obs = create_obs_set(progs, 1)
    transforms = fit_model(None, obs, tag.TRANSFORM, with_data=False)
    models = fit_model(None, obs, tag.MODEL, with_data=False)
    return RandomScorer(transforms[3], models[3])


def get_all_sklearn_in_module(module):
    return [
        '%s.%s' % (module.__name__, m) for m in dir(module) if m[0].isupper()
    ]


def get_random_all_sklearn():
    """
  Same as random scorer, but returns all transforms/models from sklearn, not just
  those that we saw in Kaggle examples. Based on
  http://scikit-learn.org/stable/supervised_learning.html#supervised-learning
  """
    models = []
    import sklearn.linear_model
    import sklearn.svm
    import sklearn.neighbors
    import sklearn.gaussian_process
    import sklearn.naive_bayes
    import sklearn.tree
    import sklearn.ensemble
    import sklearn.neural_network
    import xgboost

    models.extend(get_all_sklearn_in_module(sklearn.linear_model))
    # http://scikit-learn.org/stable/modules/lda_qda.html
    models.extend([
        'sklearn.discriminant_analysis.LineadDiscriminantAnalysis',
        'sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis'
    ])
    # http://scikit-learn.org/stable/modules/kernel_ridge.html
    models.extend(['sklearn.kernel_ridge.KernelRidge'])
    models.extend(get_all_sklearn_in_module(sklearn.svm))
    models.extend(get_all_sklearn_in_module(sklearn.neighbors))
    models.extend(get_all_sklearn_in_module(sklearn.gaussian_process))
    models.extend(get_all_sklearn_in_module(sklearn.naive_bayes))
    models.extend(get_all_sklearn_in_module(sklearn.tree))
    models.extend(get_all_sklearn_in_module(sklearn.ensemble))
    models.extend(get_all_sklearn_in_module(sklearn.neural_network))
    # add in xgboost
    models.extend([
        'xgboost.sklearn.XGBClassifier', 'xgboost.sklearn.XGBRegressor'
    ])

    transforms = []
    import sklearn.preprocessing
    import sklearn.feature_selection
    import sklearn.feature_extraction
    import sklearn.feature_extraction.text
    import sklearn.decomposition
    transforms.extend(get_all_sklearn_in_module(sklearn.preprocessing))
    transforms.extend(get_all_sklearn_in_module(sklearn.feature_selection))
    transforms.extend(get_all_sklearn_in_module(sklearn.feature_extraction))
    transforms.extend(
        get_all_sklearn_in_module(sklearn.feature_extraction.text)
    )
    transforms.extend(get_all_sklearn_in_module(sklearn.decomposition))

    transform_label_encoder = LabelEncoder().fit(transforms)
    model_label_encoder = LabelEncoder().fit(models)
    return RandomScorer(transform_label_encoder, model_label_encoder)
