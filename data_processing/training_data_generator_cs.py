"""
Copyright 2017 Rahul Gupta, Soham Pal, Aditya Kanade, Shirish Shevade.
Indian Institute of Science.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from util.cs_tokenizer import CS_Tokenizer
from util.helpers import get_rev_dict, make_dir_if_not_exists
import os
import argparse
import sqlite3
import numpy as np
from functools import partial


class FixIDNotFoundInSource(Exception):
    pass


tokenize = CS_Tokenizer().tokenize


def get_cs_tokenized():
    with open('data/cs-data-dict') as f:
        result = eval(f.read())
    for problem_id, code in result.items():
        result[problem_id], _, _ = tokenize(code)
    return result


def rename_ids_(rng, corrupt_program, fix):
    corrupt_program_new = ''
    fix_new = ''

    names = []
    for token in corrupt_program.split():
        if '_<id>_' in token:
            if token not in names:
                names.append(token)

    rng.shuffle(names)
    name_dictionary = {}

    for token in corrupt_program.split():
        if '_<id>_' in token:
            if token not in name_dictionary:
                name_dictionary[token] = '_<id>_' + \
                    str(names.index(token) + 1) + '@'

    for token in fix.split():
        if '_<id>_' in token:
            if token not in name_dictionary:
                raise FixIDNotFoundInSource

    # Rename
    for token in corrupt_program.split():
        if '_<id>_' in token:
            corrupt_program_new += name_dictionary[token] + " "
        else:
            corrupt_program_new += token + " "

    for token in fix.split():
        if '_<id>_' in token:
            fix_new += name_dictionary[token] + " "
        else:
            fix_new += token + " "

    return corrupt_program_new, fix_new


def generate_training_data(bins, min_program_length, max_program_length,
                           max_fix_length, kind_mutations, max_mutations, max_variants, seed):
    rng = np.random.RandomState(seed)

    if kind_mutations == 'typo':
        from data_processing.typo_mutator import LoopCountThresholdExceededException, FailedToMutateException, Typo_Mutate, typo_mutate
        mutator_obj = Typo_Mutate(rng)
        mutate = partial(typo_mutate, mutator_obj)
        def rename_ids(x, y): return x, y
    else:
        from data_processing.undeclared_mutator_cs import LoopCountThresholdExceededException, FailedToMutateException, id_mutate
        mutate = partial(id_mutate, rng)
        rename_ids = partial(rename_ids_, rng)

    token_strings = {'train': {}, 'validation': {}}

    exceptions_in_mutate_call = 0
    total_mutate_calls = 0
    program_lengths, fix_lengths = [], []

    for problem_id, tokenized_code in get_cs_tokenized().items():
        program_length = len(tokenized_code.split())
        key = 'validation' if rng.rand() < 0.2 else 'train'

        program_lengths.append(program_length)

        id_renamed_correct_program, _ = rename_ids(tokenized_code, '')

        # Correct pairs
        token_strings[key].setdefault(problem_id, []).append((id_renamed_correct_program, '-1'))

        # Mutate
        total_mutate_calls += 1
        try:
            iterator = mutate(tokenized_code, max_mutations, max_variants)
        except (FailedToMutateException, LoopCountThresholdExceededException):
            exceptions_in_mutate_call += 1
        except Exception:
            exceptions_in_mutate_call += 1
            if kind_mutations == 'typo':
                raise
        else:
            for corrupt_program, fix in iterator:
                corrupt_program_length = len(corrupt_program.split())
                fix_length = len(fix.split())
                fix_lengths.append(fix_length)
                if (min_program_length <= corrupt_program_length <= max_program_length
                        and fix_length <= max_fix_length):
                    try:
                        corrupt_program, fix = rename_ids(corrupt_program, fix)
                    except FixIDNotFoundInSource:
                        exceptions_in_mutate_call += 1
                    token_strings[key].setdefault(problem_id, []).append((corrupt_program, fix))

    program_lengths = np.sort(program_lengths)
    fix_lengths = np.sort(fix_lengths)

    print 'Statistics'
    print '----------'
    print 'Program length:  Mean =', np.mean(program_lengths), '\t95th %ile =', program_lengths[int(0.95 * len(program_lengths))]
    try:
        print 'Mean fix length: Mean =', np.mean(fix_lengths), '\t95th %ile = ', fix_lengths[int(0.95 * len(fix_lengths))]
    except Exception as e:
        print e
        print 'fix_lengths'
        print fix_lengths
    print 'Total mutate calls:', total_mutate_calls
    print 'Exceptions in mutate() call:', exceptions_in_mutate_call, '\n'

    return token_strings, mutator_obj.get_mutation_distribution() if kind_mutations == 'typo' else {}, rng


def build_dictionary(token_strings, drop_ids, tl_dict={}):

    def build_dict(list_generator, dict_ref):
        for tokenized_code in list_generator:
            for token in tokenized_code.split():
                if drop_ids and '_<id>_' in token:
                    continue
                token = token.strip()
                if token not in dict_ref:
                    dict_ref[token] = len(dict_ref)

    tl_dict['_pad_'] = 0
    tl_dict['_eos_'] = 1
    tl_dict['~'] = 2

    if drop_ids:
        tl_dict['_<id>_@'] = 3

    for key in token_strings:
        for pfs in token_strings[key].values():
            build_dict((prog + ' ' + fix for prog, fix in pfs), tl_dict)

    print 'dictionary size:', len(tl_dict)
    assert len(tl_dict) > 4
    return tl_dict


def vectorize(tokens, tl_dict, max_vector_length, drop_ids, reverse, vecFor='encoder'):
    assert vecFor == 'encoder' or not reverse, 'reverse passed as True for decoder sequence'

    vec_tokens = []
    for token in tokens.split():
        if drop_ids and '_<id>_' in token:
            token = '_<id>_@'

        try:
            vec_tokens.append(tl_dict[token])
        except Exception:
            raise

    if len(vec_tokens) > max_vector_length:
        return None

    if reverse:
        vec_tokens = vec_tokens[::-1]

    return vec_tokens


def vectorize_data(token_strings, tl_dict, max_program_length, max_fix_length, drop_ids):
    token_vectors = {}
    skipped = 0

    for key in token_strings:
        token_vectors[key] = {}
        for problem_id in token_strings[key]:
            token_vectors[key][problem_id] = []

    for key in token_strings:
        for problem_id in token_strings[key]:
            for prog_tokens, fix_tokens in token_strings[key][problem_id]:
                prog_vector = vectorize(
                    prog_tokens, tl_dict, max_program_length, drop_ids, reverse=True, vecFor='encoder')
                fix_vector = vectorize(
                    fix_tokens, tl_dict,  max_fix_length, drop_ids, reverse=False, vecFor='decoder')

                if (prog_vector is not None) and (fix_vector is not None):
                    token_vectors[key][problem_id].append((prog_vector, fix_vector))
                else:
                    skipped += 1

    print 'skipped during vectorization:', skipped
    return token_vectors


def save_dictionaries(destination, tl_dict):
    all_dicts = (tl_dict, get_rev_dict(tl_dict))
    np.save(os.path.join(destination, 'all_dicts.npy'), all_dicts)


def load_dictionaries(destination):
    tl_dict, rev_tl_dict = np.load(os.path.join(destination, 'all_dicts.npy'), allow_pickle=True)
    return tl_dict, rev_tl_dict


def save_pairs(destination, token_vectors, tl_dict):
    for key in token_vectors.keys():
        np.save(os.path.join(destination, ('examples-%s.npy' % key)),
                token_vectors[key])
        save_dictionaries(destination, tl_dict)


def save_bins(destination, tl_dict, token_vectors, rng):
    fold_n = 5
    bins = []
    full_list = get_cs_tokenized().keys()
    rng.shuffle(full_list)
    for i in range(fold_n):
        bins.append(full_list[len(full_list)*i//fold_n:len(full_list)*(i+1)//fold_n])
    for i, bin_ in enumerate(bins):
        test_problems = bin_
        training_problems = list(set(full_list) - set(bin_))

        token_vectors_this_fold = {'train': [], 'validation': [], 'test': []}

        for problem_id in training_problems:
            if problem_id in token_vectors['train']:
                token_vectors_this_fold['train'] += token_vectors['train'][problem_id]
            if problem_id in token_vectors['validation']:
                token_vectors_this_fold['validation'] += token_vectors['validation'][problem_id]

        for problem_id in test_problems:
            if problem_id in token_vectors['validation']:
                token_vectors_this_fold['test'] += token_vectors['validation'][problem_id]

        make_dir_if_not_exists(os.path.join(destination, 'bin_%d' % i))

        print "Fold %d: Train:%d Validation:%d Test:%d" % (i, len(token_vectors_this_fold['train']),
                                                           len(token_vectors_this_fold['validation']), len(token_vectors_this_fold['test']))

        save_pairs(os.path.join(destination, 'bin_%d' %
                                i), token_vectors_this_fold, tl_dict)

######## data generation #########

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Process 'C' dataset to be used in repair tasks")
    parser.add_argument(
        "-i", "--ids", help="Generate inputs for undeclared-ids-neural-network", action="store_true")
    args = parser.parse_args()

    kind_mutations = 'ids' if args.ids else 'typo'

    drop_ids = kind_mutations == 'typo'
    max_program_length = 450
    min_program_length = 75
    max_fix_length = 25

    max_mutations = 5
    max_variants = 4000 if kind_mutations == 'ids' else 2000

    db_path = os.path.join('data', 'iitk-dataset', 'dataset.db')
    validation_users = np.load(os.path.join('data', 'iitk-dataset', 'validation_users.npy'), allow_pickle=True).item()
    bins = np.load(os.path.join('data', 'iitk-dataset', 'bins.npy'), allow_pickle=True)

    seed = 1189

    output_directory = os.path.join('data/network_inputs', 'iitk-%s-%d' % (kind_mutations, seed))

    print 'output_directory:', output_directory
    make_dir_if_not_exists(os.path.join(output_directory))

    token_strings, mutations_distribution, rng = generate_training_data(bins, min_program_length, max_program_length, max_fix_length,
                                                                   kind_mutations, max_mutations, max_variants, seed)

    np.save(os.path.join(output_directory, 'tokenized-examples.npy'), token_strings)
    np.save(os.path.join(output_directory, 'error-seeding-distribution.npy'), mutations_distribution)

    # token_strings = np.load(os.path.join(output_directory, 'tokenized-examples.npy'), allow_pickle=True).item()

    tl_dict = build_dictionary(token_strings, drop_ids)

    # Tokenize
    token_vectors = vectorize_data(token_strings, tl_dict, max_program_length, max_fix_length, drop_ids)

    # Save
    save_bins(output_directory, tl_dict, token_vectors, rng)

    print '\n\n--------------- all outputs written to {} ---------------\n\n'.format(output_directory)
