# Tag entries in a dynamically collected trace as
# transformations, model building/fitting, or evaluation
from collections import defaultdict

UNKNOWN = {-1}

# Possible tags
TRANSFORM = 'T'
MODEL = 'M'
# model combines both, no real need to distinguish components anymore
# MODEL_BUILD = 'MB'
# MODEL_FIT = 'MF'
EVAL = 'E'
OTHER = 'O'

# predefined assignment for some sklearn modules
# based on http://scikit-learn.org/stable/modules/classes.html
# accessed 09/13/2017
predefined_mapping = {
    'sklearn.calibration': EVAL,
    'sklearn.decomposition': TRANSFORM,
    'sklearn.feature_extraction': TRANSFORM,
    'sklearn.feature_selection': TRANSFORM,
    'sklearn.metrics': EVAL,
    'sklearn.model_selection': EVAL,
    'sklearn.pipeline': TRANSFORM,
    'sklearn.preprocessing': TRANSFORM,
    'sklearn.random_projection': TRANSFORM
}


def to_dict(calls):
    return {c['call_id']: c for c in calls}


def find_calls_to_methods(calls, methods):
    return [
        c for c in calls.values()
        if c['func_name'] and c['func_name'].split('.')[-1] in methods
    ]


def get_qualified_name_component(call):
    if call['caller_type_name'] == type(None).__name__:
        name = '%s.%s' % (call['func_module'], call['func_name'])
    else:
        name = '%s.%s' % (call['caller_type_module'], call['caller_type_name'])
    return name


def backward_deps(calls, seed_call_id):
    """ Find all dependencies for seed_call_id"""
    acc = set()
    working_set = {seed_call_id}
    avail = set(calls.keys())
    while len(working_set) > 0:
        working_set = {
            dep
            for call_id in working_set
            for dep in calls[call_id]['dependencies'].difference(UNKNOWN)
        }
        working_set = working_set.difference(acc).intersection(avail)
        acc = acc.union(working_set)
    return acc


def forward_deps(calls, seed_call_id):
    acc = set()
    working_set = {seed_call_id}
    avail = set(calls.keys())
    while len(working_set) > 0:
        # get entries that use this
        working_set = {
            c['call_id']
            for c in calls.values()
            if c['dependencies'].intersection(working_set)
        }
        # avoid picking up stuff we've already checked
        working_set = working_set.difference(acc).intersection(avail)
        acc = acc.union(working_set)
    return acc


def set_tag(entry, tag):
    if not 'tag' in entry:
        func_module = entry.get('func_module', '')
        caller_module = entry.get('caller_type_module', '')
        for k in predefined_mapping.keys():
            if func_module[:len(k)] == k or caller_module[:len(k)] == k:
                entry['source_tag'] = 'predefined'
                entry['tag'] = predefined_mapping[k]
                return
        # if no predefined match, then assign desired tag
        entry['tag'] = tag
        entry['source_tag'] = 'set'


def add_model_id(entry, call_id):
    model_ids = entry.get('model_ids', set([]))
    model_ids.add(call_id)
    entry['model_ids'] = model_ids


def eval_label_pass(calls):
    # 1- any entry that calls .predict/.score is eval
    predict_or_score = find_calls_to_methods(calls, ['predict', 'score'])
    for c in predict_or_score:
        set_tag(c, EVAL)


def label_model_based_on_component_names(calls, model_names):
    # label any component with one of the names as a model
    ids = []
    for _id, call in calls.items():
        if get_qualified_name_component(call) in model_names:
            set_tag(call, MODEL)
            # make sure setting call succeeded before we add to model ids
            if call['tag'] == MODEL:
                call['source_tag'] = 'name_based'
                add_model_id(call, _id)
                ids.append(_id)
    return ids


def model_label_pass_one(calls):
    # 1 - Label as a model any component that is every seen with a .predict method call
    # or a .score method call
    predict_or_score = find_calls_to_methods(calls, ['predict', 'score'])
    model_names = [get_qualified_name_component(c) for c in predict_or_score]
    return label_model_based_on_component_names(calls, model_names)


def model_label_pass_two(calls):
    # 2 - Label as a model any component that is seen to make a .fit/.train method call
    # but never a .fit_transform, or .transform call
    fits = find_calls_to_methods(calls, ['fit', 'train'])
    transforms = find_calls_to_methods(calls, ['fit_transform', 'transform'])
    fits = set([get_qualified_name_component(c) for c in fits])
    transforms = [get_qualified_name_component(c) for c in transforms]
    fits.difference_update(transforms)
    return label_model_based_on_component_names(calls, fits)


def tag_entries(calls):
    if not isinstance(calls, dict):
        calls = to_dict(calls)
    # label evaluation
    eval_label_pass(calls)
    # label models
    model_ids = set([])
    model_ids.update(model_label_pass_one(calls))
    model_ids.update(model_label_pass_two(calls))
    # traverse dependencies for remaining labeling
    for m_id in model_ids:
        # this can be different if one of the predefined libs
        # need to check the tag to make sure its actually a model
        model = calls[m_id]
        if model['tag'] == MODEL:
            for bd_id in backward_deps(calls, m_id):
                e = calls[bd_id]
                add_model_id(e, m_id)
                set_tag(e, TRANSFORM)
                # # TODO: this assumes constructor called directly....
                # # if the function of e matches the caller type, of the current
                # # fit in accumulation, we call this the model building stage
                # if fit['caller_type_name'] == e['func_name']:
                #   set_tag(e, MODEL)
                # else:
                #   set_tag(e, TRANSFORM)
            for fd_id in forward_deps(calls, m_id):
                e = calls[fd_id]
                add_model_id(e, m_id)
                set_tag(e, EVAL)
                # components labeled with EVAL can also propagate model id backwards
                # but they do not propagate any other labeling
                if e['tag'] == EVAL:
                    for bd_id in backward_deps(calls, fd_id):
                        e = calls[bd_id]
                        add_model_id(e, m_id)
    # default to OTHER if no tag set so far
    # and add negative model id (i.e. couldn't associate it with anything)
    for e in calls.values():
        set_tag(e, OTHER)
        if not 'model_ids' in e:
            e['model_ids'] = set(UNKNOWN)
    # TODO: we need to avoid producing duplicate pipelines
    # basically when we linearize, we need to make sure
    # that a model_id is not in the backward/fwd dependencies of another
    # model_id that we have already done
    # a model_id cannot be in another model_ids linearized pipeline
    return calls


def is_unknown(v):
    return v == list(UNKNOWN)[0]


def linearize(calls, fun=None):
    if isinstance(calls, dict):
        calls = sorted(calls.values(), key=lambda x: x['call_id'])
    if fun is None:
        fun = lambda x: x
    # for each call, we can append it to the pipeline associated with
    # a particular model_id component (these have been accumulate for each)
    # call in the tag_entries function
    acc = defaultdict(lambda: [])
    for c in calls:
        c['qualified_name'] = get_qualified_name_component(c)
        for m in c['model_ids']:
            acc[m].append(c)
    # we only want unique progs, sometimes we can end up
    # with duplicates based on how we slice
    progs = {}
    for m_id, p in acc.items():
        # if unknown, we don't want different components to be
        # considered part of a program, they are independent
        # they just end up in the same 'linearized' program because
        # of the way we accumulate above
        if not is_unknown(m_id):
            prog_key = sorted([e['call_id'] for e in p])
            prog_key = tuple(prog_key)
            progs[prog_key] = p
        else:
            for e in p:
                progs[e['call_id']] = [e]
    return list(progs.values())


def just_keys(prog, keys):
    return [
        tuple([
            call[k] if isinstance(k, str) else k[1](call[k[0]]) for k in keys
        ]) for call in prog
    ]
