# Library to perform relevant transformations on Kaggle scripts
# This includes lifting nested calls and adding instrumentation to code

import ast
import astunparse
import copy
import inspect

from instrumentation import instrument


class NodeExtender(ast.NodeTransformer):
    def __init__(self, fresh_var_ct=0, fresh_var='_x'):
        self.acc = []
        self.fresh_var_ct = fresh_var_ct
        self.fresh_var = fresh_var
        self.extending = False

    # ignore these as they bind local variables which
    # we can't hoist easily
    IGNORE = set([
        ast.Lambda, ast.GeneratorExp, ast.ListComp, ast.SetComp, ast.DictComp
    ])

    def visit(self, node):
        if type(node) in NodeExtender.IGNORE:
            return node
        else:
            return super(NodeExtender, self).visit(node)

    def transform(self, tree):
        # TODO: this is super expensive...
        self.extending = False
        t = copy.deepcopy(tree)
        self.visit(t)
        return ast.parse(astunparse.unparse(t))

    def _extend_elem(self, node):
        raise Exception("Implement in subclass")

    def visit_Expr(self, node):
        self.extending = True
        self.generic_visit(node)
        new_nodes = self.acc
        self.acc = []
        self.extending = False
        return new_nodes + [node]

    def visit_Assign(self, node):
        self.extending = True
        # only extend the RHS of the assignment
        node.value = self.visit(node.value)
        new_nodes = self.acc
        self.acc = []
        self.extending = False
        return new_nodes + [node]

    def visit_Call(self, node):
        if self.extending:
            return self._extend_elem(node)
        else:
            self.generic_visit(node)
            return node


class HoistCalls(NodeExtender):
    """ Hoist calls and nested data accesses """

    def _extend_elem(self, node):
        self.generic_visit(node)
        # add definition for local variable
        orig_node = astunparse.unparse(node)
        # execute and assign to new variable
        new_def = ast.parse(
            "%s%d = %s" % (self.fresh_var, self.fresh_var_ct, orig_node)
        ).body[0]
        self.acc.append(new_def)
        # use new assignment in place of original node
        new_node = ast.parse("%s%d" % (self.fresh_var, self.fresh_var_ct)
                             ).body[0].value
        self.fresh_var_ct += 1
        # return the new node using the definition
        return new_node


class Instrument(NodeExtender):
    """ Instrument calls and data accesses """

    def __init__(self, dynamic_instance_nm="_instr"):
        super(Instrument, self).__init__()
        self.instrumenter = instrument.StaticInstrumentation(
            dynamic_instance_nm
        )

    def transform(self, tree):
        t = copy.deepcopy(tree)
        self.visit(t)
        return self._repair(t)

    def _repair(self, t):
        # avoid blocking on plot display
        fix_matplotlib = ast.parse(
            "import matplotlib; matplotlib.pyplot.ion()"
        ).body
        # append it after the first set of imports
        # otherwise python may complain if there is a `from __future__` import
        for i, stmt in enumerate(t.body):
            if type(stmt) not in set([ast.Import, ast.ImportFrom]):
                imp_stmts = fix_matplotlib
                for j, stmt in enumerate(imp_stmts):
                    t.body.insert(i + j, stmt)
                break
        return ast.parse(astunparse.unparse(t))

    def visit(self, node):
        # all extendable calls should be transformed, regardless of context
        self.context = True
        return super(Instrument, self).visit(node)

    def visit_Loops(self, node):
        record_loop_entry = self.instrumenter.create_enter_loop()
        # visit body of original node
        node = super(Instrument, self).generic_visit(node)
        record_loop_exit = self.instrumenter.create_exit_loop()
        return [record_loop_entry, node, record_loop_exit]

    def visit_For(self, node):
        return self.visit_Loops(node)

    def visit_While(self, node):
        return self.visit_Loops(node)

    def visit_Assign(self, node):
        # only instrument assignments of the form:
        # _ = x
        # we've hoisted up relevant function calls to capture _x = f(*)
        # add standard instrumentation
        nodes = super(Instrument, self).visit_Assign(node)
        # add in data dependency tracking
        data_dep_node = self.instrumenter.create_update_data_dependence(node)
        return nodes + [data_dep_node]

    def _extend_elem(self, node):
        # instrument the current element
        if type(node) == ast.Call:
            new_instr = self.instrumenter.create_register_call(node)
            self.acc.append(new_instr)
        else:
            raise Exception(
                'trying to instrument wrong node type: %s' % ast.dump(node)
            )
        return node


def compile_instrumented(dynamic_instance_nm, src):
    t = ast.parse(src)
    ht = HoistCalls().transform(t)
    hti = Instrument(dynamic_instance_nm).transform(ht)
    return compile(hti, filename="<ast>", mode="exec"), astunparse.unparse(hti)
