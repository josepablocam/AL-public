#!/usr/local/bin/python3

import sys 
import os
script_path = os.path.abspath(os.path.dirname(sys.argv[0]))
sys.path.insert(0, os.path.join(script_path, '../../src/'))

from argparse import ArgumentParser
import astunparse
from chunker import utils
from chunker.utils import mapl, flatten, map_nested, get_fun_lib, get_fun_name
from difflib import SequenceMatcher
import pandas as pd
import pickle
import random
from tqdm import tqdm
from traces import *

# SequenceMatcher.quick_ratio threshold
# values above this are considered to be too similar for our
# purposes
SIM_THRESHOLD = 0.75

def standardize(src):
  """ Simple heuristic to standardize python scripts for similarity comparison """
  # parse them
  s = astunparse.unparse(ast.parse(src))
  # janky standardization by removing leading white space etc
  return '\n'.join([e.strip() for e in s.split('\n') if len(e.strip()) > 0])  

def show_issue():
  """
  Kaggle scripts have "pseudo-duplicates". These are scripts that have only marginally
  changed from their parent fork, or from prior versions written by the same user, despite
  saving down with different title. A lot of these are produced due to the ad-hoc
  version control to reflect parameter changes, new features etc.
  
  The query below shows an example of this in the Kaggle database
  
  create temp table t as
  select Scripts.Id as id, Scripts.AuthorUserId as user_id, ScriptContent, IsChange, title
          from ScriptVersions, Scripts
          where Scripts.CurrentScriptVersionId = ScriptVersions.Id
          and Scripts.Id IS NOT NULL
          and ScriptVersions.ScriptLanguageId = (select Id from ScriptLanguages where Name = "Python")
          group by ScriptContent;
  select title from t where user_id=81892;

  ```
  xgboost_v0
  python_xgboost
  Beating the Benchmark v1.0
  Beating the Benchmark v1.1
  ExtraTreesClassifier002
  ExtraTreesClassifier (score 0.45911)
  RFR Features (0.47203)
  ```
  """
  
  with open('../../data/meta-kaggle/traces.pkl', 'rb') as f:
    traces = pickle.load(f)
    src = [t[1] for t in traces if len(t[2]) > 0]
    src = [s.strip() for s in src]
    src = [s for s in src if len(s) > 0]
    sorted_src = sorted(src)
    dists = []
    random.seed(1)
    indices = set(range(0, len(standard)))
    batch_size = 50
    for batches in [1, 2, 3, 4]:
      sample = list(random.sample(indices, batch_size))
    for s_i, i in tqdm(enumerate(sample)):
      for s_j in tqdm(range(s_i + 1, batch_size)):
        j = sample[s_j]
        s1 = standard[i]
        s2 = standard[j]
        d = editdistance.eval(s1, s2)
        dists.append((i, j, d))
    indices.difference_update(sample)
  dists_df = pd.DataFrame(dists, columns=['i', 'j', 'd'])
  dists_df = dists_df.sort_values('d')
  return dists_df
  
def transitive_closure(pairs):
  """ 
  Returns transitive closure of sets based on inclusion 
  e.g. {1, 2}, {3, 5}, {2, 4} , {6, 5}, {7, 8} => {1, 2, 4}, {3, 5, 6}, {7, 8}
  """
  changed = True
  pairs = mapl(set, pairs)
  while changed:
    changed = False
    acc = []
    for p in pairs:
      added = False
      for e in acc:
        # if already considered by a current set, just ignore
        # but make sure we don't add later
        if e.issuperset(p):
          added = True
          break
        # if it has a link to an existing set, merge it in
        # make sure we note there was a change and not to add later
        if e.intersection(p):
          e |= p
          changed = True
          added = True
          break
      # if the entry wasnt already there, nor merged, then keep
      if not added:
        acc.append(p) 
    pairs = acc
  return acc

def remove_similar_user_project(df):
  """
  Given a df for a single user and project id with all script ids,
  return the list of script ids to remove. We keep the single largest
  script within groups of "pseudo-duplicates".
  """
  ratios = []
  nrows = df.shape[0]
  for i in range(nrows):
    for j in range(i + 1, nrows):
      row1 = df.iloc[i]
      row2 = df.iloc[j]
      id1 = row1.script_id
      id2 = row2.script_id
      r = SequenceMatcher(lambda x: x == ' ', row1.script, row2.script).quick_ratio()
      ratios.append((id1, id2, r))
  ratios = pd.DataFrame(ratios, columns=['id1', 'id2', 'r'])
  bad_df = ratios[ratios.r > SIM_THRESHOLD]
  bad_pairs = list(zip(bad_df.id1, bad_df.id2))
  bad_ids = transitive_closure(bad_pairs)
  remove = []
  # line of code use to pick single within each pseudo-duplicate class
  loc = dict(zip(df.script_id, df.len))
  for pseudo_class in bad_ids:
    # keep the largest
    sorted_ids = sorted([i for i in pseudo_class], key=lambda x: loc[x])
    # remove all bad largest
    remove.append(sorted_ids[:-1])
  return flatten(remove)

  
def get_clean_ids(traces):
  """ Return script ids for all scripts that are *not* pseudo-duplicates """
  # "standardize" before comparing
  print("Standardizing scripts")
  traces = [(e[0], standardize(e[1])) + e[2:] for e in traces]
  traces = [e for e in traces if len(e[1].strip()) > 0]
  data = [(info.user_id, info.project_id, info.parent_id, info.script_id, script) for (info, script, _, _) in traces]
  df = pd.DataFrame(data, columns=['user_id', 'project_id', 'parent_id', 'script_id', 'script'])
  df['len'] = df['script'].map(len)
  
  #### Remove duplicates based on their parent fork ####
  print("Removing fork-based pseudo-duplicates")
  df_w_parent = df.merge(df, how='inner', left_on='parent_id', right_on='script_id', suffixes=('_child', '_parent'))
  for i, row in tqdm(list(df_w_parent.iterrows())):
    # only check if the number length is significantly more
    compare = SequenceMatcher(lambda x: x == ' ', row.script_child, row.script_parent)
    r = compare.quick_ratio()
    df_w_parent.loc[i, 'r'] = r
  # remove child ids with high similarity ratio
  dupes_of_fork = set(df_w_parent[df_w_parent.r > SIM_THRESHOLD].script_id_child.values)
  df = df[~df.script_id.isin(dupes_of_fork)]
  
  #### Remove duplicates based on user and project ####
  print("Removing user-project-based pseudo-duplicates")
  grouped_scripts = df.groupby(['user_id', 'project_id'])['script_id'].apply(frozenset).to_frame('ids')
  # only consider scripts that have multiple entries for same user and project
  multiple = grouped_scripts[grouped_scripts['ids'].map(len) > 1]
  user_dupes = []
  for _, row in tqdm(list(multiple.iterrows())):
    grp_df = df[df.script_id.isin(row.ids)]
    user_dupes.append(remove_similar_user_project(grp_df))
  
  user_dupes = flatten(user_dupes)
  df = df[~df.script_id.isin(user_dupes)]
  return set(df.script_id.values)

def clean(traces):
  """ Remove pseudo-duplicates from kaggle scripts """
  clean_ids = get_clean_ids(traces)
  print("%d/%d scripts conserved" % (len(clean_ids), len(traces)))
  return [e for e in traces if e[0].script_id in clean_ids]
  
def main(traces_file, output):
    with open(traces_file, 'rb') as f:
      traces = pickle.load(f)
    clean_traces = clean(traces)
    
    with open(output, 'wb') as f:
      pickle.dump(clean_traces, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
  argparser = ArgumentParser(description="Remove pseudo-duplicates from Kaggle scripts")
  argparser.add_argument('input', type=str, help="Pickled file of traces")
  argparser.add_argument('output', type=str, help="File name for output of cleaned traces")
  args = argparser.parse_args()
  main(args.input, args.output)

    
  