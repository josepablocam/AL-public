import copy
import dill
import importlib
import inspect
import logging
import multiprocessing as mp
import stopit
import traceback
import time

import numpy as np
import pandas as pd

# change start method to avoid issues with crashes/freezes
# discussed in
# http://scikit-learn.org/stable/faq.html#why-do-i-sometime-get-a-crash-freeze-with-n-jobs-1-under-osx-or-linux
try:
    mp.set_start_method('spawn')
except RuntimeError:
    # already set
    pass


# traverse imports necessary to get a name
# TODO there must be a better way to do this
def _import_necessary(nm, context):
    """ import necessary packages """
    module_name = None
    success = False
    while not success and len(nm) > 0:
        try:
            # if it was already in the state passed into the program, may not be a module
            if nm in context:
                success = True
                if inspect.ismodule(eval(nm, context)):
                    module_name = nm
                else:
                    module_name = None
            else:
                context[nm] = importlib.import_module(nm)
                success = True
                module_name = nm
        except ModuleNotFoundError:
            # chop off last bit and try again
            nm = '.'.join(nm.split('.')[:-1])
    return success, module_name


# we use multiprocessing to enforce timeouts
# since signals won't necessarily work
# as calls to signal handlers can be delayed arbitrary amounts of time
# if there is long running computation in C/C++ etc
def run_with_timeout_in_pool(seconds, fun, *args, **kwargs):
    if seconds > 0:
        pool = mp.Pool(processes=1)
        try:
            proc = pool.apply_async(fun, args, kwargs)
            result = proc.get(seconds)
            return result
        finally:
            pool.terminate()
            pool.close()
    else:
        # if no timeout, then no point
        # in incurring cost of running as separate process
        # so call locally
        return fun(*args, **kwargs)


# doesn't use processes instead uses threads
def run_with_timeout_in_thread(seconds, fun, *args, **kwargs):
    if seconds > 0:
        try:
            with stopit.ThreadingTimeout(seconds, swallow_exc=False):
                result = fun(*args, **kwargs)
                return result
        except stopit.TimeoutException:
            # to match behavior of old run with timeout
            raise TimeoutError
    else:
        return fun(*args, **kwargs)


def store_result_in_queue(queue, func, args, kwargs):
    try:
        result = func(*args, **kwargs)
        # we put in dill str if Program, so that we can pickle modules
        # which are used in Program
        if isinstance(result, Program):
            queue.put(dill.dumps(result))
        else:
            queue.put(result)
    except:
        # any exception here should just result in None as result
        result = None
        queue.put(result)


# uses processes but instead of mp.Pool uses mp.Process and mp.Queue
def run_with_timeout_in_separate_process(seconds, fun, *args, **kwargs):
    if seconds > 0:
        q = mp.Queue()
        proc = mp.Process(
            target=store_result_in_queue,
            args=(q, fun, args, kwargs),
        )
        proc.start()
        try:
            result = q.get(timeout=seconds)
        except mp.queues.Empty:
            result = None
        if proc.is_alive():
            proc.terminate()
        if result is not None:
            if isinstance(result, bytes):
                return dill.loads(result)
            else:
                return result
        else:
            raise TimeoutError
    else:
        return fun(*args, **kwargs)


# set which version of running with timeout we'll use
run_with_timeout = run_with_timeout_in_separate_process


# AST for our programs
# atoms
class Name(object):
    """ Variable in our AST """

    def __init__(self, name):
        self.name = name

    def value(self, state):
        return state[self.name]

    def code(self):
        return self.name

    def __str__(self):
        return self.code()

    def __eq__(self, other):
        return isinstance(other, self.__class__) and other.name == self.name

    def __hash__(self):
        return hash(self.name)


# code fragments
class Transform(object):
    """ Transform a data object and store transformer to local variable """

    def __init__(self, instr_id, f, arg, wrap=None):
        self.name = '_t%s' % str(instr_id)
        self.f = f
        self.arg = arg
        self._import = None
        self.wrap = wrap

    def execute(self, state, timeout=-1):
        success, _import = _import_necessary(self.f, state)
        if not success:
            logger = logging.getLogger(__name__)
            logger.warning("Failed to import transform:%s" % self.f)
            return False, (None, None)
        self._import = _import
        # build the object
        try:
            # import pdb
            # pdb.set_trace()
            transform_builder = eval(self.f, state)
            if self.wrap:
                if not self.wrap.is_allowed(self.arg, state):
                    return False, {}
                transformer = self.wrap(transform_builder)
            else:
                transformer = transform_builder()
            arg_val = self.arg.value(state)
            fitted_transformer = run_with_timeout(
                timeout, transformer.fit, arg_val
            )
            next_arg_val = run_with_timeout(
                timeout, fitted_transformer.transform, arg_val
            )
            return True, {
                self.name: fitted_transformer,
                self.arg.name: next_arg_val
            }
        except:
            return False, {}

    def code(self):
        if self.wrap:
            _code_builder = f'{self.wrap.code()}({self.f})'
        else:
            _code_builder = f'{self.f}()'
        _code = f'{self.name} = {_code_builder}.fit({self.arg.code()})\n'
        _code += f'{self.arg.code()} = {self.name}.transform({self.arg.code()})\n'
        return _code

    def pipeline_code(self):
        if self.wrap:
            return f'{self.wrap.code()}({self.f})'
        else:
            return f'{self.f}()'

    def construct_apply_transform(self, new_arg, state):
        return ApplyTransform(self.name, new_arg)

    def __str__(self):
        return self.code()

    def __eq__(self, other):
        return isinstance(
            other, self.__class__
        ) and self.f == other.f and self.arg == other.arg

    def __hash__(self):
        return hash((self.f, hash(self.arg)))


class ApplyTransform(object):
    def __init__(self, t, arg):
        self.t = t
        self.arg = arg

    def execute(self, state, timeout=-1):
        # there is nothing to import for this
        try:
            transformer = eval(self.t, state)
            arg_val = self.arg.value(state)
            next_arg_val = run_with_timeout(
                timeout, transformer.transform, arg_val
            )
            return True, {self.arg.name: next_arg_val}
        except:
            return False, {}

    def code(self):
        return f'{self.arg.code()} = {self.t}.transform({self.arg.code()})\n'

    def __str__(self):
        return self.code()

    def __eq__(self, other):
        return isinstance(
            other, self.__class__
        ) and self.t == other.t and self.arg == other.arg

    def __hash__(self):
        return hash((self.t, hash(self.arg)))


class Model(object):
    """ Fit a model to a data matrix and a target vector """

    def __init__(self, instr_id, f, arg1, arg2):
        self.name = '_m%s' % str(instr_id)
        self.f = f
        self.arg1 = arg1
        self.arg2 = arg2
        self._import = None

    def execute(self, state, timeout=-1):
        success, _import = _import_necessary(self.f, state)
        if not success:
            logger = logging.getLogger(__name__)
            logger.warn("Failed to import model:%s" % self.f)
            return False, None
        self._import = _import
        try:
            model_builder = eval(self.f, state)
            model = model_builder()
            X_arg_val = self.arg1.value(state)
            y_arg_val = self.arg2.value(state)
            fitted_model = run_with_timeout(
                timeout, model.fit, X_arg_val, y_arg_val
            )
            return True, {self.name: fitted_model}
        except:
            return False, {}

    def code(self):
        return f'{self.name} = {self.f}().fit({self.arg1.code()}, {self.arg2.code()})\n'

    def pipeline_code(self):
        return f'{self.f}()'

    def construct_score_model(self, new_arg_1, new_arg_2, metric):
        return ScoreModel(self.name, new_arg_1, new_arg_2, metric)

    def __str__(self):
        return self.code()

    def __eq__(self, other):
        return isinstance(
            other, self.__class__
        ) and self.f == other.f and self.arg1 == other.arg1 and self.arg2 == other.arg2

    def __hash__(self):
        return hash((self.f, hash(self.arg1), hash(self.arg2)))


def model_metric_score(model, metric, X, y):
    y_pred = model.predict(X)
    return metric(y, y_pred)


class ScoreModel(object):
    def __init__(self, m, arg1, arg2, metric):
        self.name = '%s_score' % m
        self.m = m
        self.arg1 = arg1
        self.arg2 = arg2
        self.metric = metric

    def execute(self, state, timeout=-1):
        try:
            model = eval(self.m, state)
            X_arg_val = self.arg1.value(state)
            y_arg_val = self.arg2.value(state)
            if self.metric is None:
                score = run_with_timeout(
                    timeout, model.score, X_arg_val, y_arg_val
                )
            else:
                score = run_with_timeout(
                    timeout,
                    model_metric_score,
                    model,
                    self.metric,
                    X_arg_val,
                    y_arg_val,
                )
            return True, {self.name: score}
        except:
            return False, {}

    def code(self):
        return f'{self.m}.score({self.arg1.name}, {self.arg2.name})'


class Program(object):
    def __init__(self, state=None, _imports=None, _trace=None):
        if state is None:
            state = {}
        self.state = state
        if _imports is None:
            default_imports = ['sklearn', 'xgboost', 'runtime_helpers']
            _imports = [
                _import_necessary(p, self.state)[1] for p in default_imports
            ]
        self.imports = set(_imports)
        # depth of a program defined in number of transformations before model
        self.depth = 0
        if _trace is None:
            _trace = []
        self.trace = _trace
        # keep track of operations performed on a given argument
        # we do not accept repeated operations on the same argument
        self.ops = set([])
        self.instr_id = 0
        self.log_prob = 0.0
        self.score_ = None
        self.model_ = None

    def __len__(self):
        return len(self.trace)

    def __deepcopy__(self, memo):
        return self.copy_with_sharing(must_copy=None)

    def copy_with_sharing(self, must_copy):
        p = Program()
        # share as much of the state as we can
        p.state = {}
        for name, val in self.state.items():
            if must_copy and name in must_copy:
                try:
                    p.state[name] = copy.deepcopy(val)
                except TypeError:
                    # some things (e.g. modules) cannot be copied
                    p.state[name] = val
            else:
                p.state[name] = val
        # create new containers with the same elements
        p.imports = set(self.imports)
        p.trace = [e for e in self.trace]
        p.ops = set(self.ops)
        p.log_prob = self.log_prob
        p.instr_id = self.instr_id
        p.depth = self.depth
        p.score_ = self.score_
        p.model_ = copy.deepcopy(self.model_)
        return p

    def add_transform(
            self, component_name, arg, log_prob=0.0, wrap=None, timeout=-1
    ):
        logger = logging.getLogger(__name__)
        logger.info(
            "Transforming %s(%s) [wrap=%s]" %
            (component_name, str(arg), str(wrap))
        )
        transform = Transform(self.instr_id, component_name, arg, wrap=wrap)
        # assume the transform fails
        success = False
        if not transform in self.ops:
            # no repeated transformations in program
            # if t_nm == 'sklearn.feature_extraction.text.TfidfVectorizer' and wrap is not None:
            #   import pdb
            #   pdb.set_trace()
            success, state_update = transform.execute(
                self.state, timeout=timeout
            )
            # import pdb
            # pdb.set_trace()
        if success:
            # TODO: only need to deepcopy things in state_update
            # all else can remain the same, to share, reduce memory load
            modified = self.copy_with_sharing(must_copy=None)
            modified.imports.add(transform._import)
            modified.trace.append(transform)
            modified.ops.add(transform)
            modified.state.update(state_update)
            modified.log_prob += log_prob
            modified.instr_id += 1
            modified.depth += 1
            return True, modified
        else:
            logger.warn(
                "Failed to apply transform(%s): %s" % (component_name, wrap)
            )
            return False, None

    def add_model(
            self, component_name, X_arg, y_arg, log_prob=0.0, timeout=-1
    ):
        logger = logging.getLogger(__name__)
        logger.info(
            "Modeling %s(%s, %s)" % (component_name, str(X_arg), str(y_arg))
        )
        model = Model(self.instr_id, component_name, X_arg, y_arg)
        success = False
        if not model in self.ops:
            # no repeated models in program
            success, state_update = model.execute(self.state, timeout=timeout)
        if success:
            modified = self.copy_with_sharing(must_copy=None)
            modified.imports.add(model._import)
            modified.trace.append(model)
            modified.ops.add(model)
            modified.state.update(state_update)
            modified.instr_id += 1
            modified.log_prob += log_prob
            # save down pointer to trained model
            modified.model_ = state_update[model.name]
            return True, modified
        else:
            logger.warn("Failed to apply model: %s" % component_name)
            return False, None

    def add_test_data(self, X_test_arg, y_test_arg, timeout=-1, metric=None):
        # having a program that doesn't work on test data
        # and that provides a score for the program
        # we should use this sort them at the end
        # only copy once, because if any step fails, entire pipeline fails
        modified = copy.deepcopy(self)
        for node_ix in range(0, len(modified.trace)):
            node = modified.trace[node_ix]
            if isinstance(node, Transform):
                # pick arg to use analogous variable in test setting
                # TODO: we currently only modify X...
                applied = node.construct_apply_transform(
                    X_test_arg, modified.state
                )
                success, state_update = applied.execute(
                    modified.state, timeout=timeout
                )
                if success:
                    modified.state.update(state_update)
                    modified.trace.append(applied)
                else:
                    raise Exception('Ill-formed program, fails on test data')
            elif isinstance(node, Model):
                # we may want to add data with out testing
                if y_test_arg is not None:
                    applied = node.construct_score_model(
                        X_test_arg,
                        y_test_arg,
                        metric=metric,
                    )
                    success, state_update = applied.execute(
                        modified.state, timeout=timeout
                    )
                    if success:
                        modified.trace.append(applied)
                        assert (len(state_update.values()) == 1)
                        [score] = state_update.values()
                        modified.score_ = score
                    else:
                        raise Exception(
                            'Ill-formed program, fails on test data'
                        )
            else:
                pass
        return modified

    def fit_final(self, X_train, y_train, timeout=-1):
        modified = self.copy_with_sharing(must_copy=None)
        modified.state['X_train'] = X_train
        modified.state['y_train'] = y_train
        for op in modified.trace:
            success, state_update = op.execute(modified.state, timeout=timeout)
            if not success:
                raise Exception("Failed during fit_final")
                # raise Exception("Failed during fit_final")
            modified.state.update(state_update)
        # overwrite the model_ with the new model
        for op in modified.trace:
            if isinstance(op, Model):
                modified.model_ = modified.state[op.name]
        return modified

    def score(self, X_test, y_test, timeout=-1, metric=None):
        # add this data to state...
        # TODO: fix this, horrible hack
        modified = self.copy_with_sharing(must_copy=None)
        modified.state['new_X'] = X_test
        modified.state['new_y'] = y_test
        X_test_arg = Name('new_X')
        y_test_arg = Name('new_y')
        with_new_test_data = modified.add_test_data(
            X_test_arg,
            y_test_arg,
            timeout=timeout,
            metric=metric,
        )
        return with_new_test_data.score_

    def predict(self, X_test, timeout=-1):
        modified = self.copy_with_sharing(must_copy=None)
        modified.state['new_X'] = X_test
        X_test_arg = Name('new_X')
        with_new_test_data = modified.add_test_data(
            X_test_arg, None, timeout=timeout
        )
        model = with_new_test_data.model_
        X_test_transformed = X_test_arg.value(with_new_test_data.state)
        return model.predict(X_test_transformed)

    def predict_proba(self, X_test, timeout=-1):
        modified = self.copy_with_sharing(must_copy=None)
        modified.state['new_X'] = X_test
        X_test_arg = Name('new_X')
        with_new_test_data = modified.add_test_data(
            X_test_arg, None, timeout=timeout
        )
        model = with_new_test_data.model_
        X_test_transformed = X_test_arg.value(with_new_test_data.state)
        return model.predict_proba(X_test_transformed)

    def imports_code(self):
        acc = []
        for i in self.imports:
            if i is not None:
                acc.append('import %s\n' % i)
        return ''.join(acc) + '\n'

    def code(self):
        acc = self.imports_code()
        for e in self.trace:
            acc += e.code()
        return acc

    def pipeline_code(self):
        acc = self.imports_code()
        acc += "from sklearn.pipeline import Pipeline"
        # build up the pipeline
        steps = []
        step_ct = 0
        X_name, y_name = None, None
        for e in self.trace:
            if isinstance(e, Transform):
                if X_name is None:
                    X_name = e.arg,
                step_name = "t{}".format(step_ct)
                step_code = e.pipeline_code()
                steps.append("('{}', {})".format(step_name, step_code))
                step_ct += 1
            elif isinstance(e, Model):
                step_code = e.pipeline_code()
                steps.append("('model', {})".format(step_code))
                step_ct += 1
            else:
                continue
        steps_code = ",".join(steps)
        pipeline_code = "\np = Pipeline([{}])".format(steps_code)
        acc += pipeline_code
        # fit pipeline on X_train, y_train
        acc += "\np.fit(X_train, y_train)"
        # validate on val data
        acc += "\nprint(p.score(X_val, y_val))"
        # train on entire X
        acc += "\np.fit(X, y)"
        # add function to predict for new X
        acc += "\ndef predict(X_test): return p.predict(X_test)"
        return acc
