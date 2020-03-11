import sys
import os
import json
import pickle
from Tree import *
import copy
from multiprocessing import Pool
import gc
import logging


# Special vocabulary symbols
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_NT = b"_NT"
_LEFT_BRACKET = b"("
_RIGHT_BRACKET = b")"
_START_VOCAB = [_UNK, _GO, _EOS, _PAD, _NT, _LEFT_BRACKET, _RIGHT_BRACKET]


# DO NOT change UNK_ID, since the oov id code depends on it..
UNK_ID = 0
GO_ID = 1
EOS_ID = 2
PAD_ID = 3
NT_ID = 4
LEFT_BRACKET_ID = 5
RIGHT_BRACKET_ID = 6

def add_tokens_from_code(code, vocab, format):
  if format == 'tree':
    tok = str(code["root"])
    if tok not in vocab:
      vocab.append(tok)
    for sub_tree in code["children"]:
      vocab = add_tokens_from_code(sub_tree, vocab, format)
  else:
    for tok in code:
      if not (str(tok) in vocab):
        vocab.append(str(tok))
  return vocab

def add_tokens_from_code_pointergen(code, vocab, format, parent=""):
  if format == 'tree':
    tok = str(code["root"])
    if (tok not in vocab) and ('Identifier' not in parent) and ('Literal' not in parent):
      vocab.append(tok)
    for sub_tree in code["children"]:
      vocab = add_tokens_from_code_pointergen(sub_tree, vocab, format, parent=tok)
  else:
    for tok in code:
      if not (str(tok) in vocab):
        vocab.append(str(tok))
  return vocab

def get_max_num_children(tree):
  max_num_children = len(tree['children'])
  for child in tree['children']:
    t = get_max_num_children(child)
    max_num_children = max(t, max_num_children)
  return max_num_children

def calStat(data, serialize):
  if serialize:
    min_len = 10000
    max_len = 0
    avg_len = 0
    for seq in data:
      l = len(seq)
      min_len = min(min_len, l)
      max_len = max(max_len, l)
      avg_len += l
    return min_len, max_len, avg_len * 1.0 / len(data)
  min_len = 0
  max_len = 0
  avg_len = 0
  for tree in data:
    current_len = get_max_num_children(tree)
    avg_len = avg_len + current_len
    max_len = max(max_len, current_len)
  return min_len, max_len, avg_len * 1.0 / len(data)

def build_vocab(train_data, vocab_filename, input_format, output_format, pointergen):
  if not vocab_filename:
    source_vocab_list = []
    target_vocab_list = []
    for prog in train_data:
      if input_format == 'seq':
        source_prog = prog['source_prog']
      else:
        source_prog = prog['source_ast']
      if output_format == 'seq':
        target_prog = prog['target_prog']
      else:
        target_prog = prog['target_ast']
      source_vocab_list = add_tokens_from_code_pointergen(
          source_prog, source_vocab_list, input_format) if pointergen else add_tokens_from_code(source_prog, source_vocab_list, input_format)
      target_vocab_list = add_tokens_from_code_pointergen(
          target_prog, target_vocab_list, output_format) if pointergen else add_tokens_from_code(target_prog, target_vocab_list, output_format)
  else:
    vocab = pickle.load(open(vocab_filename))
    source_vocab_list = vocab["source"]
    target_vocab_list = vocab["target"]

  source_vocab_list = _START_VOCAB[:] + source_vocab_list
  target_vocab_list = _START_VOCAB[:] + target_vocab_list
  source_vocab_dict = {}
  target_vocab_dict = {}
  for idx, token in enumerate(source_vocab_list):
    source_vocab_dict[token] = idx
  for idx, token in enumerate(target_vocab_list):
    target_vocab_dict[token] = idx
  return source_vocab_dict, target_vocab_dict

def ast_to_token_ids(code, vocab, serialize):
  if serialize:
    current = []
    current.append(vocab.get(str(code['root']), UNK_ID))
    if len(code['children']) > 0:
      current.append(LEFT_BRACKET_ID)
      for sub_tree in code['children']:
        child, child_extended = ast_to_token_ids(sub_tree, vocab, serialize)
        current = current + child
      current.append(RIGHT_BRACKET_ID)
    return current
  else:
    current = {}
    current['root'] = vocab.get(str(code['root']), UNK_ID)
    current['children'] = []
    for sub_tree in code['children']:
      current['children'].append(ast_to_token_ids(sub_tree, vocab, serialize))
    return current

def serialize_tree(tree):
  current = []
  current.append(LEFT_BRACKET_ID)
  current.append(tree['root'])
  if len(tree['children']) > 0:
    for sub_tree in tree['children']:
      child = serialize_tree(sub_tree)
      current = current + child
  current.append(RIGHT_BRACKET_ID)
  return current

def raw_program_to_token_ids(prog, vocab):
  return [vocab[str(t)] for t in prog] + [EOS_ID]

def build_vocab_oovs(code, source_vocab, current_target_vocab, format, vocab_oovs = None):
  if vocab_oovs == None:
    vocab_oovs = {}

  current_vocab_oovs_id = len(vocab_oovs) + 1 # starts from 1(0 means not OOV)
  current_target_vocab_id = len(current_target_vocab)
  if format == 'tree':
    tok = str(code["root"])
    if (tok not in source_vocab) and (tok not in vocab_oovs):
      vocab_oovs[tok] = current_vocab_oovs_id
      current_target_vocab[tok] = current_target_vocab_id
    for sub_tree in code["children"]:
      vocab_oovs, current_target_vocab = build_vocab_oovs(sub_tree, source_vocab, current_target_vocab, format, vocab_oovs)
  else:
    for tok in code:
      if (str(tok) not in source_vocab) and (str(tok) not in vocab_oovs):
        vocab_oovs[str(tok)] = current_vocab_oovs_id
        current_vocab_oovs_id += 1
        current_target_vocab[str(tok)] = current_target_vocab_id
        current_target_vocab_id += 1
  return vocab_oovs, current_target_vocab

def get_ext_id2target_ext_id(vocab_oovs, target_vocab):
  # Converts id starting from 1 to extended target_vocab's id
  diff = len(target_vocab) - (len(vocab_oovs) + 1) # because it starts from 1
  def _ext_id2target_ext_id(id):
    return id+diff
  return _ext_id2target_ext_id

def _prepare_data(arguments):
  prog, source_vocab, target_vocab, input_format, output_format, source_serialize, target_serialize, pointer_gen = arguments
  if input_format == 'seq':
    source_prog = prog['source_prog']
  else:
    source_prog = prog['source_ast']
  if output_format == 'seq':
    target_prog = prog['target_prog']
  else:
    target_prog = prog['target_ast']

  vocab_oovs, target_vocab_extended = build_vocab_oovs(source_prog, copy.deepcopy(
      source_vocab), copy.deepcopy(target_vocab), input_format) if pointer_gen else (None, None)

  # UNK token is 0, so the result of *_to_token_ids(source_prog, vocab_oovs) 
  # includes 0s where the token is not included in the vocab_oovs(i.e. is included in the original vocab).

  if input_format == 'seq':
    source_prog = raw_program_to_token_ids(source_prog, source_vocab)
    target_prog_oov_ids = raw_program_to_token_ids(source_prog, vocab_oovs) if pointer_gen else None
  else:
    source_prog = ast_to_token_ids(source_prog, source_vocab, source_serialize)
    source_prog_oov_ids = ast_to_token_ids(source_prog, vocab_oovs, source_serialize) if pointer_gen else None
  if output_format == 'seq':
    target_prog = raw_program_to_token_ids(target_prog, target_vocab)
    target_prog_extended = raw_program_to_token_ids(target_prog, target_vocab_extended) if pointer_gen else None
  else:
    target_prog = ast_to_token_ids(target_prog, target_vocab, target_serialize)
    target_prog_extended = ast_to_token_ids(target_prog, target_vocab_extended, target_serialize) if pointer_gen else None
  return (source_prog, target_prog, source_prog_oov_ids, target_prog_extended, vocab_oovs)

def prepare_data(init_data, source_vocab, target_vocab, input_format, output_format, source_serialize, target_serialize, pointer_gen, n_cpus):
  data = []

  global _prepare_data


  pool = Pool(n_cpus)
  logging.info("Generating input values for _prepare_data function")
  init_data_len = len(init_data)
  source_vocab_dup = [source_vocab] * init_data_len
  target_vocab_dup = [target_vocab] * init_data_len
  input_format_dup = [input_format] * init_data_len
  output_format_dup = [output_format] * init_data_len
  source_serialize_dup = [source_serialize] * init_data_len
  target_serialize_dup = [target_serialize] * init_data_len
  pointer_gen_dup = [pointer_gen] * init_data_len
  logging.info("_prepare_data start")
  for res in pool.map(_prepare_data, zip(init_data, source_vocab_dup, target_vocab_dup, input_format_dup, output_format_dup, source_serialize_dup, target_serialize_dup, pointer_gen_dup)):
    data.append(res)
  logging.info("_prepare_data finished")
  
  gc.collect()
    
  if input_format == 'tree' and (not source_serialize):
    logging.info("build_trees start")
    data = build_trees(data, n_cpus, target_serialize)
    logging.info("build_trees finished")
  return data

def _build_trees(arguments):
  (source, target, source_oov_ids, target_extended, vocab_oovs), target_serialize = arguments
  source_trees = TreeManager()
  source_trees.build_binary_tree_from_dict(source)
  if source_oov_ids is not None:
    source_trees_oov_ids = TreeManager()
    source_trees_oov_ids.build_binary_tree_from_dict(source_oov_ids)
  else:
    source_trees_oov_ids = None

  if target_serialize:
    target_seq = target[:]
    if target_extended is not None:
      target_seq_extended = target_extended[:]
    else:
      target_trees_extended = None

    return (source_trees, target_seq, source_trees_oov_ids, target_seq_extended, vocab_oovs)

  else:
    target_trees = TreeManager()
    target_trees.build_binary_tree_from_dict(target)
    if target_extended is not None:
      target_trees_extended = TreeManager()
      target_trees_extended.build_binary_tree_from_dict(target_extended)
    else:
      target_trees_extended = None
    logging.ino("DEBUGGING")
    exit(0)
    return (source_trees, target_trees, source_trees_oov_ids, target_trees_extended, vocab_oovs)



def build_trees(init_dataset, n_cpus, target_serialize=False):
  data_set = []
  pool = Pool(n_cpus)
  target_serialize_dup = [target_serialize] * len(init_dataset)
  for res in pool.map(_build_trees, zip(init_dataset, target_serialize_dup)):
    data_set.append(res)
  return data_set
