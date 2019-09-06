#!/usr/bin/env python3

import sqlite3
import ast
import builtins
from tqdm import tqdm
import collections
from astunparse import unparse
import copy
import abc
from pprint import pprint
import os
import pickle
import inspect
import nbformat
import dill
from argparse import ArgumentParser

def fn(f):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), f)

class Call(object):
    def __init__(self, ast, func=None, err=None):
        self.ast = ast
        if func is None:
            self.func = None
        else:
            try:
                sig = inspect.signature(func)
            except (ValueError, TypeError):
                sig = None
            try:
                mod = func.__module__
            except AttributeError:
                mod = None
            try:
                name = func.__name__
            except:
                name = None
            try:
                qualname = func.__qualname__
            except:
                qualname = None
            self.func = (name, mod, sig, qualname)
        self.err = err

    def __str__(self):
        return unparse(ast)

    def __repr__(self):
        if self.func is None and self.err is None:
            return f'Call({self.ast})'
        if self.func is None:
            return f'Call({self.ast}, err={self.err})'
        if self.err is None:
            return f'Call({self.ast}, {self.func})'
        return f'Call({self.ast}, {self.func}, {self.err})'



class Trace(object):
    def __init__(self, imports, calls):
        self.imports = imports
        self.calls = calls

    def __str__(self):
        return '\n'.join(str(c) for c in self.calls)


class LiftParallel(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if type(node.func) == ast.Call:
            if (type(node.func.func) == ast.Name and node.func.func.id == 'delayed') or \
                (type(node.func.func) == ast.Attribute and node.func.func.attr == 'delayed') or \
                (type(node.func.func) == ast.Attribute and node.func.func.attr == 'vectorize'):
                # Ex: delayed(convolve)(feat[i], boxcar(nwin), 'full')
                assert len(node.func.args) == 1
                return ast.Call(node.func.args[0], node.args, node.keywords)
            if (type(node.func.func) == ast.Name and node.func.func.id == 'Parallel') or \
                (type(node.func.func) == ast.Attribute and node.func.func.attr == 'Parallel'):
                # Ex: Parallel(n_jobs=(- 1))((delayed(lfilter)(b, a, raw._data[i]) for i in picks))
                assert len(node.args) == 1 and len(node.keywords) == 0
                return node.args[0]
        return node

class ExtractCallTrace(ast.NodeVisitor):
    def __init__(self):
        self.calls = []
        self.imports = []
        self._names = copy.copy(builtins.__dict__)

    def _exec_node(self, node):
        try:
            return exec(unparse(node), self._names)
        except Exception as e:
            print(e)
            return None

    def _eval_node(self, node):
        try:
            return eval(unparse(node), self._names)
        except Exception as e:
            print(e)
            return None

    def _lookup(self, node):
        return self._eval_node(node)

    def visit_Import(self, node):
        self.generic_visit(node)
        for alias in node.names:
            self.imports.append(alias.name)

        self._exec_node(node)

    def visit_ImportFrom(self, node):
        self.generic_visit(node)
        self.imports.append(node.module)

        self._exec_node(node)

    def visit_With(self, node):
        self.generic_visit(node)
        print('Warn: ignored with statement')

    def visit_Assign(self, node):
        self.generic_visit(node)

        if len(node.targets) > 1:
            print(f'Warn: multi-assignment {unparse(node).strip()}')
            return
        target = node.targets[0]
        if type(target) != ast.Name:
            print(f'Warn: non-name assignment {unparse(node).strip()}')
            return

        if type(node.value) == ast.Name:
            val = self._lookup(node.value.id)
            self._names[target.id] = val
        elif type(node.value) == ast.Dict or type(node.value) == ast.DictComp:
            self._names[target.id] = dict
        elif type(node.value) == ast.Set or type(node.value) == ast.SetComp:
            self._names[target.id] = set
        elif type(node.value) == ast.List or type(node.value) == ast.ListComp:
            self._names[target.id] = list
        elif type(node.value) == ast.Tuple:
            self._names[target.id] = tuple
        elif type(node.value) == ast.Num:
            self._names[target.id] = type(node.value.n)
        elif type(node.value) in [ast.Str, ast.FormattedValue, ast.JoinedStr]:
            self._names[target.id] = str
        elif type(node.value) == ast.Call:
            func = self._lookup(node.value.func)
            if func is None:
                print(f'Warn: looking up func failed {unparse(node).strip()}')
                return
            if type(func) == type or type(func) == abc.ABCMeta:
                self._names[target.id] = func
            else:
                print(f'Warn: not type {func}')
        else:
            print(f'Warn: unknown assignment {unparse(node).strip()}')
            return

    def visit_Call(self, node):
        self.generic_visit(node)
        if type(node.func) == ast.Name or type(node.func) == ast.Attribute:
            func = self._lookup(node.func)
            if func is None:
                print(f'Warn: looking up func failed {unparse(node).strip()}')
                call = Call(node, func, 'func lookup failed')
            else:
                call = Call(node, func)
        elif type(node.func) == ast.Call:
            call = Call(node, err='calling returned func')
        elif type(node.func) == ast.Subscript:
            call = Call(node, err='calling subscripted func')
        elif type(node.func) == ast.Lambda: # Fine to ignore these
            call = Call(node, err='calling lambda')
        else:
            print(f'Warn: unexpected func type {unparse(node.func).strip()}.')
            call = Call(node, err='unexpected type')

        self.calls.append(call)

class NotebookTracer(object):
  """ Extract trace per-source code cell """
  def __init__(self, nb_raw):
    self.nb_raw = nb_raw
    self.parsed_cells = []
    self.calls = []
    self.imports = []

  def _parse_nb(self, raw):
    return nbformat.reads(raw, nbformat.NO_CONVERT)

  def _cells(self, nb):
      """Yield all cells in an nbformat-insensitive manner"""
      if nb.nbformat < 4:
          for ws in nb.worksheets:
              for cell in ws.cells:
                  yield cell
      else:
          for cell in nb.cells:
            yield cell

  def _parse(self):
    # convert raw textbook to notebook
    nb = self._parse_nb(self.nb_raw)
    is_code_cell = lambda x: 'cell_type' in x and x['cell_type'] == 'code'
    for cell in filter(is_code_cell, self._cells(nb)):
      src = cell['source']
      if len(src) > 0:
        parsed_cell = ast.parse(src)
        lifted_cell = LiftParallel().visit(parsed_cell)
        self.parsed_cells.append(lifted_cell)

  def _collect_trace(self):
    # single call tracer, to keep state (since resolves library functions etc)
    ct = ExtractCallTrace()
    cell_calls = []
    for parsed_cell in self.parsed_cells:
      ct.visit(parsed_cell)
      # append calls in cell as a list
      if len(ct.calls) > 0:
        cell_calls.append(ct.calls)
        # clear calls
        ct.calls = []
    self.calls = cell_calls
    self.imports = ct.imports

  def trace(self):
    if len(self.nb_raw.strip()) > 0:
      self._parse()
      self._collect_trace()

def query_db(dbfile, language="Python"):
  """
  Query kaggle database for scripts associated with a particular language
  :param dbfile: location of kaggle db
  :param language: language to query, default is Python. IPython Notebook retrieves notebooks instead.
  :returns: database cursor
  """
  conn = sqlite3.connect(dbfile)
  c = conn.cursor()
  py_scripts_query = f"""
       select 
         Scripts.AuthorUserId as user_id,
         Scripts.ScriptProjectId AS project_id,
         ScriptVersions.id as script_id,
         Scripts.ForkParentScriptVersionId as parent_id,
         ScriptVersions.ScriptContent as script
      from ScriptVersions, Scripts
        where Scripts.CurrentScriptVersionId = ScriptVersions.Id
        and Scripts.Id IS NOT NULL
        and ScriptVersions.ScriptLanguageId = (select Id from ScriptLanguages where Name = "{language}")
        group by ScriptContent"""
  print('Querying db...')
  c.execute(py_scripts_query)
  return c

def trace_from_script(code):
  tree = ast.parse(code)
  tree = LiftParallel().visit(tree)
  ct = ExtractCallTrace()
  ct.visit(tree)
  return ct

def trace_from_notebook(code):
  ct = NotebookTracer(code)
  ct.trace()
  return ct

class ScriptInfo(object):
  # contains info that we use about script to remove duplicates later on
  def __init__(self, user_id, project_id, script_id, parent_id=None):
    self.user_id = user_id
    self.project_id = project_id
    self.script_id = script_id
    self.parent_id = int(parent_id) if parent_id is not None else parent_id


def main(dbfile, tracefile, language):
    print("Collecting %s traces and saving to %s" % (language, tracefile))
    dbfile = fn(dbfile)
    tracefile = fn(tracefile)
    c = query_db(dbfile, language=language)
    traces = []
    # use appropriate trace extraction
    get_trace = trace_from_script if language == "Python" else trace_from_notebook
    for (user_id, project_id, script_id, parent_id, script) in tqdm(c.fetchall()):
        script_info = ScriptInfo(user_id, project_id, script_id, parent_id)
        try:
            print(50 * "*")
            ct = get_trace(script)
            try:
                pprint(ct.calls)
            except TypeError:
                print('BUG: inspect __repr__ failed')
            traces.append((script_info, script, ct.calls, ct.imports))
        except SyntaxError:
            pass

    print('Writing traces...')
    try:
      with open(tracefile, 'wb') as f:
          pickle.dump(traces, f, protocol=pickle.HIGHEST_PROTOCOL)
    except pickle.PicklingError as err:
      # pickle can fail if there are things like lambdas in the trace
      tracefile = os.path.splitext(tracefile)[0] + '.dill'
      print("Pickle failed, trying dill. Saving to %s" % tracefile)
      with open(tracefile, 'wb') as f:
          dill.dump(traces, f, protocol=dill.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    try:
        argparser = ArgumentParser(description="Collect traces from kaggle kernels")
        argparser.add_argument('dbfile', nargs='?', type=str, help='Path to database', default='../../data/meta-kaggle/database.sqlite')
        argparser.add_argument('--output', type=str, help='Path to save pickled traces', default='../../data/meta-kaggle/traces.pkl')
        argparser.add_argument('--language', type=str, help='Language for scripts [Python, IPython Notebook]', default='Python')
        args = argparser.parse_args()
        main(args.dbfile, args.output, args.language)
    except SystemExit as ex:
      if ex.code != 0:
        import pdb
        pdb.post_mortem()
    except:
        import pdb
        pdb.post_mortem()
