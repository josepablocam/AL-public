# Library with static instrumenting functions (which modify source code to add calls to runtime instrumenting functions)
# and dynamic instrumenting functions

import ast
import astunparse
import copy
import inspect
import pickle
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class NameCollector(ast.NodeVisitor):
    def __init__(self):
        self.acc = []

    def visit_Name(self, node):
        self.acc.append(node)

    def collect(self, node):
        self.acc = []
        self.visit(node)
        return self.acc


class StaticInstrumentation(object):
    """ Creates AST nodes that add in instrumentation calls """
    # methods names matching definitions in instrument.DynamicInstrumentation
    REGISTER_CALL = "_register_call"
    UPDATE_DATA_DEP = "_update_data_dep"
    ENTER_LOOP = '_enter_loop'
    EXIT_LOOP = '_exit_loop'

    def __init__(self, dynamic_instance_nm="_instr"):
        self.dynamic_instance_nm = dynamic_instance_nm
        self.name_collector = NameCollector()
        self.static_instr_id = 0
        self.static_loop_id = 0

    def create_instr_fun(self, instr_fun_nm, args, **kwargs):
        # this is a call on the dynamic instrumentation instance
        instr_full_nm = "%s.%s" % (self.dynamic_instance_nm, instr_fun_nm)
        instr_fun = ast.Name(id=instr_full_nm, ctx=ast.Load())
        named_args = [ast.keyword(k, v) for k, v in kwargs.items()]
        # built it piecewise to be backwards compatible
        call = ast.Call()
        call.func = instr_fun
        call.args = args

        # add static instrumentation id to arguments, if we're registering a call
        if instr_fun_nm == StaticInstrumentation.REGISTER_CALL:
            call.args = [ast.Num(self.static_instr_id)] + call.args
            self.static_instr_id += 1

        call.keywords = named_args
        call.kwargs = []
        call.starargs = []
        return ast.Expr(call)

    def create_list_node(self, elems):
        return ast.List(elts=copy.deepcopy(elems), ctx=ast.Load())

    def create_register_call(self, node):
        assert (type(node) == ast.Call)
        func = node.func
        if type(func) == ast.Name:
            caller = ast.parse('None', mode='eval').body
        elif type(func) == ast.Attribute:
            caller = ast.Name(
                id=astunparse.unparse(func.value).strip(), ctx=ast.Load()
            )
        elif type(func) == ast.Subscript:
            caller = ast.parse('None', mode='eval').body
        else:
            raise Exception(
                "Trying to instrument an invalid call node: %s" %
                ast.dump(node)
            )
        # create tuples of (name, val)
        pos_args = self.create_list_node(node.args)
        named_args = self.create_list_node([
            ast.Tuple(elts=[ast.Str(e.arg), e.value], ctx=ast.Load())
            for e in node.keywords
        ])
        # for position arguments that are subscripts, or attributes, we look at the source
        # to add additional potential dependencies
        # e.g. a[*], a.* => {a} added
        extra = []
        for arg in node.args:
            if type(arg) in [ast.Subscript, ast.Attribute]:
                extra.append(arg.value)
        extra = self.create_list_node(extra)
        # keep a copy of the actual source code used in the call
        src_code = ast.Str(astunparse.unparse(node).strip())
        instr_args = [caller, func, pos_args, named_args, extra, src_code]
        instr_node = self.create_instr_fun(
            StaticInstrumentation.REGISTER_CALL, instr_args
        )
        return instr_node

    def collect_names(self, nodes):
        names = []
        for node in nodes:
            names.extend(self.name_collector.collect(node))
        return names

    def create_update_data_dependence(self, node):
        # we don't care about these instructions, so we don't increase he instruction id
        targets = node.targets
        lhs_names = self.collect_names(targets)
        lhs_names_arg = self.create_list_node(lhs_names)
        if type(node.value) == ast.Call:
            # parse none to avoid issues with python3/2 versions of ast
            none_node = ast.parse('None').body[0].value
            # use the instrumentation id
            return self.create_instr_fun(
                StaticInstrumentation.UPDATE_DATA_DEP,
                [lhs_names_arg, none_node]
            )
        else:
            # collect names in RHS
            values = [node.value]
            rhs_names = self.collect_names(values)
            rhs_names_arg = self.create_list_node(rhs_names)
            return self.create_instr_fun(
                StaticInstrumentation.UPDATE_DATA_DEP,
                [lhs_names_arg, rhs_names_arg]
            )
        #
        # elif type(node.value) == ast.Name:
        #   # look up the RHS in existing dependencies
        #   return self.create_instr_fun(StaticInstrumentation.UPDATE_DATA_DEP, [names, node.value])
        # else:
        #   raise Exception("Trying to update data dependencies with unhandled RHS`")

    def create_enter_loop(self):
        loop_id_arg = ast.Num(self.static_loop_id)
        self.static_loop_id += 1
        return self.create_instr_fun(
            StaticInstrumentation.ENTER_LOOP, [loop_id_arg]
        )

    def create_exit_loop(self):
        return self.create_instr_fun(StaticInstrumentation.EXIT_LOOP, [])


class DynamicInstrumentation(object):
    """" Actually executes the instrumentation calls at runtime """

    def __init__(self, relevant_modules=None, arg_summarizer=None):
        if relevant_modules is None:
            relevant_modules = ['sklearn', 'xgboost']
        self.relevant_modules = relevant_modules
        self.arg_summarizer = arg_summarizer
        self.call_id = -1
        self.data_deps = {}
        self.acc = []
        self.counter = 0
        self.loop_stack = []
        self.loop_instrs = {}

    def write(self, msg):
        self.acc.append(msg)

    def flush(self, file_name):
        print("Saving trace file to %s" % file_name)
        with open(file_name, 'wb') as f:
            pickle.dump(self.acc, f)

    def reset(self):
        self.call_id = 0
        self.data_deps = {}
        self.acc = []

    def _update_data_dep(self, lhs_vals, existing_deps):
        if existing_deps is None:
            # we recorded a call, and if relevant, we'll collect other argument dependencies etc
            # during instrumentation, this just allows the chain of dependencies to be resolved
            dep_info = set([self.call_id])
        else:
            dep_info = self._resolve_data_deps(existing_deps)

        for val in lhs_vals:
            self.data_deps[id(val)] = dep_info

    def _resolve_data_deps(self, vals):
        # look up data dependencies and flatten set
        return set([d for v in vals for d in self.data_deps.get(id(v), {-1})])

    def _relevant_module(self, func_module):
        # in this case we consider all functions not just those from a particular module
        if self.relevant_modules is False:
            return True
        else:
            return str(func_module).split('.')[0] in self.relevant_modules

    def _qualify_name(self, e):
        try:
            module = e.__module__
        except:
            print("Failed to get module: %s" % e)
            module = None

        try:
            name = e.__qualname__
        except:
            try:
                name = e.__name__
            except:
                print("Failed to get name: %s" % e)
                name = None
        return module, name

    def _register_call(
            self, static_instr_id, caller, func, pos_args, named_args,
            extra_deps, src_code
    ):
        # always increase the call id, regardless of whether we plan to record
        self.call_id += 1
        module, name = self._qualify_name(func)
        try:
            if self._relevant_module(
                    module) and not self._repeating_loop(static_instr_id):
                result = {}
                result['static_instr_id'] = static_instr_id
                result['static_loop_id'
                       ] = None if not self.loop_stack else self.loop_stack[-1]
                result['call_id'] = self.call_id

                caller_module, caller_name = self._qualify_name(type(caller))
                result['caller_type_module'] = caller_module
                result['caller_type_name'] = caller_name
                result['func_module'] = module
                result['func_name'] = name
                result['dependencies'] = self._resolve_data_deps(
                    list(pos_args) + [v for _, v in named_args] + [caller] +
                    extra_deps
                )
                if self.arg_summarizer is not None:
                    result['args'] = [
                        self.arg_summarizer.summarize(a) for a in pos_args
                    ]
                else:
                    result['args'] = pos_args
                result['src_code'] = src_code
                self.write(result)
        except:
            traceback.print_exc()
            return None

    def _repeating_loop(self, static_instr_id):
        # can't be repeating if we're not in a loop
        if len(self.loop_stack) == 0:
            return False
        loop_id = self.loop_stack[-1]
        instrs_seen = self.loop_instrs.get(loop_id, set([]))
        # we already have this instrumentation call site associated with this loop level
        if static_instr_id in instrs_seen:
            return True
        # in a loop and hadn't seen this instrumentation call site before, so record it
        instrs_seen.add(static_instr_id)
        self.loop_instrs[loop_id] = instrs_seen

    def _enter_loop(self, loop_id):
        self.loop_stack.append(loop_id)

    def _exit_loop(self):
        # remove loop id from stack
        self.loop_stack.pop()
        # if we are out of all loops, remove the instruction tracking. If we hit outer loop again, I want to record
        # e.g. loop is inside a function which is called multiple times, each time we call function, treat as if
        # new loop in source code
        if len(self.loop_stack) == 0:
            self.loop_instrs = {}
