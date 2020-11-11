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

import regex as re

from util.helpers import fix_ids_are_in_program, extract_line_number, get_rev_dict, get_lines, recompose_program


class FailedToMutateException(Exception):
    pass


class CouldNotFindUsesForEitherException(Exception):
    pass


class NothingToMutateException(FailedToMutateException):
    pass


class LoopCountThresholdExceededException(Exception):
    pass


def which_fix_goes_first(program, fix1, fix2):
    try:
        fix1_location = extract_line_number(' '.join(fix1.split()[1:]))
        fix2_location = extract_line_number(' '.join(fix2.split()[1:]))
    except Exception:
        raise

    if not fix_ids_are_in_program(recompose_program(get_lines(program)[fix2_location:]), fix2) and fix_ids_are_in_program(recompose_program(get_lines(program)[fix1_location:]), fix1):
        return fix1

    if not fix_ids_are_in_program(recompose_program(get_lines(program)[fix1_location:]), fix1) and fix_ids_are_in_program(recompose_program(get_lines(program)[fix2_location:]), fix2):
        return fix2

    if not fix_ids_are_in_program(recompose_program(get_lines(program)[fix1_location:]), fix1) and not fix_ids_are_in_program(recompose_program(get_lines(program)[fix2_location:]), fix2):
        raise CouldNotFindUsesForEitherException

    if fix1_location < fix2_location:
        return fix1
    elif fix2_location < fix1_location:
        return fix2

    prog_lines = get_lines(program)
    id_in_fix1 = None
    id_in_fix2 = None

    for token in fix1.split():
        if '_<id>_' in token:
            assert id_in_fix1 is None, fix1
            id_in_fix1 = token
        elif token == '_<op>_[':
            break

    for token in fix2.split():
        if '_<id>_' in token:
            assert id_in_fix2 is None, fix2
            id_in_fix2 = token
        elif token == '_<op>_[':
            break

    assert id_in_fix1 != id_in_fix2, fix1 + ' & ' + fix2
    assert fix1_location == fix2_location

    for i in range(fix1_location, len(prog_lines)):
        for token in prog_lines[i].split():
            if token == id_in_fix1:
                return fix1
            elif token == id_in_fix2:
                return fix2

    assert False, 'unreachable code'


def find_declaration(rng, variables, line_indexes, orig_lines):
    rng.shuffle(variables)
    for variable in variables:
        rng.shuffle(line_indexes)
        declaration_regex = re.compile(r'''
            (?P<decl>
                (?P<type>
                    _<type>_\w+
                    (
                        \ _<op>_\[
                        \ _<op>_\]
                    )*
                )
              \ {}
            )
          \ _<op>_=
          \ [^,;]+
          \ _<op>_;
        '''.format(variable), re.X)
        for i in line_indexes:
            regex_matches = list(declaration_regex.finditer(orig_lines[i]))
            if len(regex_matches) != 1:
                continue
            declaration = regex_matches[0].group('decl') + ' _<op>_;'
            l, r = regex_matches[0].span('type')
            orig_lines[i] = orig_lines[i][:l] + orig_lines[i][r+1:]
            return declaration, i
    raise NothingToMutateException


def insert_fix(declaration_pos, orig_lines):
    function_regex = re.compile(r'''
            (
                _<type>_\w+
              | _<keyword>_void
              | _<id>_\d+@
            )
            (
                \ _<op>_\[
                \ _<op>_\]
            )*
          \ _<id>_\d+@
          \ _<op>_\(
        ''', re.X)
    function_start_regex = re.compile(r'_<op>_\{')
    for i in range(declaration_pos-1, -1, -1):
        if len(function_regex.findall(orig_lines[i])) == 1:
            for j in range(i, declaration_pos+1):
                if next(function_start_regex.search(orig_lines[j])) is not None:
                    return j
    raise FailedToMutateException


def undeclare_variable(rng, program_string):
    # Lines
    orig_lines = get_lines(program_string)
    # Variables
    variables = []
    for token in program_string.split():
        if '_<id>_' in token and token not in variables:
            variables.append(token)
    # Look for a declaration
    declaration, declaration_pos = find_declaration(
        rng, variables, list(range(len(orig_lines))), orig_lines)
    # Find the function signature
    fix_line = insert_fix(declaration_pos, orig_lines)
    fix = '_<insertion>_ {} ~ {}'.format(' '.join(str(fix_line)), declaration)
    # ...
    if orig_lines[declaration_pos].strip() == '':
        del orig_lines[declaration_pos]
    return recompose_program(orig_lines), fix, fix_line


def id_mutate(rng, prog, max_num_mutations, num_mutated_progs, exact=False, name_dict=None):
    assert max_num_mutations > 0 and num_mutated_progs > 0, "Invalid argument(s) supplied to the function token_mutate"
    corrupted = []
    fixes = []

    for _ in range(num_mutated_progs):
        tokens = prog

        if exact:
            num_mutations = max_num_mutations
        else:
            num_mutations = rng.choice(range(max_num_mutations)) + 1
        mutation_count = 0

        fix_line = None

        for _ in range(num_mutations):
            # Step 2: Induce _[ONE]_ mutation, removing empty lines and shifting program if required
            try:
                mutated, this_fix, _ = undeclare_variable(rng, tokens)
                mutation_count += 1
            # Couldn't delete anything
            except NothingToMutateException:
                break
            # Insertion or something failed
            except FailedToMutateException:
                raise

            # Deleted something that can't be fixed (all uses gone from the program)
            else:
                # REPLACE! program with mutated version
                tokens = mutated

                if not fix_ids_are_in_program(mutated, this_fix):
                    # Discarding previous fix: all uses are gone
                    continue

                # Update fix line
                if fix_line is not None:
                    fix_line = which_fix_goes_first(
                        mutated, fix_line, this_fix)
                else:
                    fix_line = this_fix

        if fix_line is not None:
            corrupted.append(tokens)
            fixes.append(fix_line)

    for fix in fixes:
        assert fix is not None

    return zip(corrupted, fixes)
