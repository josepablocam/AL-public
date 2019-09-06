from collections import defaultdict
from copy import deepcopy
import itertools
import logging
import numpy
import sklearn
import time
from tqdm import tqdm
tqdm.monitor_interval = 0
import warnings

from synthesis import program
from synthesis.learn import get_predictor, get_all_transforms, get_all_models
from sklearn.linear_model import LogisticRegression
from synthesis.runtime_helpers import ColumnLoop
from synthesis import tag

MAX_COLS_LOOP = 3000


def silence_warnings():
    # sklearn has a lot of annoying deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings(
        "ignore", category=sklearn.exceptions.DataConversionWarning
    )
    warnings.filterwarnings("ignore", category=RuntimeWarning)


silence_warnings()


def get_timestamp():
    return time.time()


class ProgramGenerator(object):
    def __init__(
            self,
            predictor,
            X,
            y,
            X_test=None,
            y_test=None,
            X_name='X_train',
            y_name='y_train',
            X_test_name='X_test',
            y_test_name='y_test',
            depth=4,
            beam=None,
            timeout=5,
            progress=True,
            metric=None,
    ):
        # factors in generation
        self.predictor = predictor
        self.depth = depth
        self.beam = beam
        # timeout in seconds for add any operation to program
        self.timeout = timeout
        # training data
        self.X = X
        self.y = y
        self.X_arg = program.Name(X_name)
        self.y_arg = program.Name(y_name)
        # test data if provided
        self.X_test = X_test
        self.y_test = y_test
        self.X_test_arg = program.Name(X_test_name)
        self.y_test_arg = program.Name(y_test_name)
        self.metric = metric
        # save down state for initializing empty programs
        self.init_state = {
            X_name: self.X,
            y_name: self.y,
            X_test_name: self.X_test,
            y_test_name: self.y_test
        }
        # tracking stats
        self.tried = 0
        self.components = []
        self.component_outcomes = []
        self.component_timestamps = []
        self.component_execution_times = []
        self.start_timestamp = None
        # display tqdm progress bar
        self.progress = progress
        self.logger = logging.getLogger(__name__)

    def generate(self, progress=None):
        if progress is not None:
            self.progress = progress
        self.start_timestamp = get_timestamp()
        return self.generate_model(self.generate_transforms())

    def init_programs(self):
        return [program.Program(state=deepcopy(self.init_state))]

    def from_configuration(self, config):
        """
    Build program from a configuration, which is a list of components and
    appropriate arguments
    """
        curr_prog = self.init_programs()[0]
        for component in config:
            role = component['role']
            args = {k: v for k, v in component.items() if k != 'role'}
            if role == tag.TRANSFORM:
                success, new_prog = curr_prog.add_transform(**args)
            elif role == tag.MODEL:
                success, new_prog = curr_prog.add_model(**args)
            else:
                raise Exception(
                    "Cannot generate from configuration with unknown role: %s"
                    % role
                )
            if not success:
                return False, None
            curr_prog = new_prog
        return True, curr_prog

    def generate_exhaustive_configurations(self):
        transforms = sorted(get_all_transforms(self.predictor))
        models = sorted(get_all_models(self.predictor))
        # start with an empty configuration
        current_configs = [[]]
        configs = []
        ct = 0
        # transformations
        for depth in tqdm(list(range(self.depth)), disable=not self.progress):
            # add in all current configurations
            next_configs = []
            components = itertools.product(transforms, [None, ColumnLoop])
            for trans, wrap in components:
                component = dict(
                    role=tag.TRANSFORM,
                    component_name=trans,
                    arg=self.X_arg,
                    wrap=wrap,
                    timeout=self.timeout
                )
                for config in current_configs:
                    new_config = deepcopy(config)
                    new_config.append(component)
                    # add them to final set, and
                    # also keep track to extend at next depth
                    configs.append(new_config)
                    next_configs.append(new_config)
                    ct += 1
            current_configs = next_configs
        # models
        current_configs = configs
        configs = []
        print("Adding models to %d configurations" % ct)
        for model in tqdm(models, disable=not self.progress):
            component = dict(
                role=tag.MODEL,
                component_name=model,
                X_arg=self.X_arg,
                y_arg=self.y_arg,
                timeout=self.timeout
            )
            for config in tqdm(current_configs, disable=not self.progress):
                new_config = deepcopy(config)
                new_config.append(component)
                configs.append(new_config)
                ct += 1
        print("Final configuration count: %d" % ct)
        return configs

    def generate_transforms(self):
        print("Generating transforms")
        # start with an empty program
        working_set = self.init_programs()
        # program with zero transformations
        programs = deepcopy(working_set)
        for i in range(self.depth):
            print("Programs of depth=%d" % i)
            new_programs = []
            for prog in tqdm(working_set, disable=not self.progress):
                # TODO: can only modify X, not y for now
                for var in [self.X_arg]:
                    # we multiply by two because we search whole table transforms and columnwise
                    need = -1 if self.beam is None else self.beam * 2
                    # predict transformations based on the variable to transform
                    predicted_transforms = self.predictor.predict_transforms(
                        prog, var
                    )
                    for t, log_prob in tqdm(predicted_transforms,
                                            disable=not self.progress):
                        # whole X or loop over columns
                        # note that transform itself is predicted based on X as a whole
                        if self.X_arg.value(prog.state
                                            ).shape[1] < MAX_COLS_LOOP:
                            possible_wraps = [None, ColumnLoop]
                        else:
                            possible_wraps = [None]
                        for wrap in possible_wraps:
                            start = get_timestamp()
                            success, next_prog = prog.add_transform(
                                t,
                                var,
                                wrap=wrap,
                                log_prob=log_prob,
                                timeout=self.timeout
                            )
                            end = get_timestamp()
                            self.tried += 1
                            # don't keep track of timestamp if exhaustive search, blows up memory
                            if self.beam > 0:
                                self.components.append(
                                    (tag.TRANSFORM, t, wrap is None)
                                )
                                self.component_timestamps.append(start)
                                self.component_execution_times.append(
                                    end - start
                                )
                                self.component_outcomes.append(success)
                            if success:
                                new_programs.append(next_prog)
                                need -= 1
                            if need == 0:
                                break
            print("[Transforms] Pruning programs of depth %i" % i)
            pruned = self.prune(new_programs)
            # only continue with pruned programs for next depth iteration
            working_set = pruned
            programs.extend(pruned)
        return programs

    def generate_model(self, working_set):
        print("Generating modeling")
        programs = []
        for prog in tqdm(working_set, disable=not self.progress):
            need = -1 if self.beam is None else self.beam
            predicted_models = self.predictor.predict_models(
                prog, self.X_arg, self.y_arg
            )
            for m, log_prob in tqdm(predicted_models,
                                    disable=not self.progress):
                start = get_timestamp()
                success, next_prog = prog.add_model(
                    m,
                    self.X_arg,
                    self.y_arg,
                    log_prob=log_prob,
                    timeout=self.timeout
                )
                end = get_timestamp()
                self.tried += 1
                if self.beam > 0:
                    self.components.append((tag.MODEL, m))
                    self.component_timestamps.append(start)
                    self.component_execution_times.append(end - start)
                    self.component_outcomes.append(success)
                if success:
                    programs.append(next_prog)
                    need -= 1
                if need == 0:
                    break
        # group them by length before pruning each group
        progs_by_len = defaultdict(lambda: [])
        for prog in programs:
            progs_by_len[len(prog)].append(prog)

        for k, progs in progs_by_len.items():
            print("[Model] Pruning programs of depth %d" % (k - 1))
            pruned = self.prune(progs)
            progs_by_len[k] = pruned

        progs = [p for ps in progs_by_len.values() for p in ps]
        # add in test data for each if provided
        if self.X_test is not None and self.y_test is not None:
            print("Adding test data")
            for prog in tqdm(progs, disable=not self.progress):
                try:
                    scored_prog = prog.add_test_data(
                        self.X_test_arg,
                        self.y_test_arg,
                        timeout=self.timeout,
                        metric=self.metric,
                    )
                    # keep track of the score
                    prog.score_ = scored_prog.score_
                except:
                    prog.score_ = None
                    self.logger.warning(
                        "Failed to add test data, program depth: %d" %
                        prog.depth
                    )
            # sort original programs based on the score given by the test data given
            return sorted([p for p in progs if p.score_ is not None],
                          key=lambda x: x.score_,
                          reverse=True)
        return progs

    def prune(self, programs):
        # want most likely programs first
        sorted_programs = sorted(
            programs, key=self.predictor.compute_log_prob, reverse=True
        )
        if self.beam is not None and self.beam > 0:
            return sorted_programs[:self.beam]
        else:
            return sorted_programs


def example(X, y, X_test=None, y_test=None, ngram_size=1, depth=2, beam=10):
    predictor3 = get_predictor(
        lambda: LogisticRegression(penalty='l1', random_state=0), ngram_size
    )
    generator3 = ProgramGenerator(
        predictor3, X, y, X_test, y_test, depth=depth, beam=beam
    )
    return generator3.generate(), generator3.tried
