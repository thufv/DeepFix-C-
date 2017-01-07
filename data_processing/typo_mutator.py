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

import re, random, numpy as np
from util.tokenizer import C_Tokenizer
from util.helpers import isolate_line, fetch_line, extract_line_number, get_lines, recompose_program

class FailedToMutateException(Exception):
    pass

class LoopCountThresholdExceededException(Exception):
    pass

class TypeInferenceFailedException(Exception):
    pass

def get_last_id(prog):
    prog = prog.split()
    last_id = None
    
    for word in prog:
        if '_<id>_' in word:
            the_id = int(word.lstrip('_<id>_').rstrip('@'))
            
            if last_id == None or the_id > last_id:
                last_id = the_id
                
    if last_id is not None:
        return last_id
    else:
        return 1
pass


def do_fix_at_line(corrupted_prog, line, fix):
    try:
        lines = get_lines(corrupted_prog)
    except Exception:
        print corrupted_prog
        raise
    
    try:
        lines[line] = fix
    except IndexError:
        raise
        
    return recompose_program(lines)
pass


def add_fix_number(corrupted_prog, fix_number):
    try:
        lines = get_lines(corrupted_prog)
    except Exception:
        print corrupted_prog
        raise
    
    last_line = '_<directive>_#include _<include>_<FixNumber_%d>' % fix_number
    lines.append(last_line)
        
    return recompose_program(lines)
pass

def get_min(alist):
    if len(alist) == 0:
        return None, None
    m, mi = alist[0], 0
    for idx in range(len(alist)):
        if alist[idx] < m:
            m = alist[idx]
            mi = idx
        pass
    return m, mi
pass  

def token_mutate(prog, max_num_mutations, num_mutated_progs, include_kind=False):
    raise NotImplementedError

class Typo_Mutate:    
    __actions = {
           'delete (' : ("_<op>_\(", "", 1),
           'delete )' : ("_<op>_\)", "", 1),
           'delete ,' : ("_<op>_,", "", 1),
           'delete ;' : ("_<op>_;", "", 1),
           'duplicate (' : ("_<op>_\(", "_<op>_( _<op>_(", 1),
           'duplicate )' : ("_<op>_\)", "_<op>_) _<op>_)", 1),
           'duplicate ,' : ("_<op>_,", "_<op>_, _<op>_,", 1),
           'replace ; with ,' : ("_<op>_;", "_<op>_,", 1),
           'replace , with ;' : ("_<op>_,", "_<op>_;", 1),
           'replace ; with .' : ("_<op>_;", "_<op>_.", 1),
           'replace ); with ;)' : ("_<op>_\) _<op>_;", "_<op>_; _<op>_)", 1),
           'delete {' : ("_<op>_\{", "", 1),
           'delete }' : ("_<op>_\}", "", 1),
           'duplicate {' : ("_<op>_\{", "_<op>_{ _<op>_{", 1),
           'duplicate }' : ("_<op>_\}", "_<op>_} _<op>_}", 1),
          }

    __mutation_distribution = None
    __pmf = None
    

    def find_and_replace(self, org_prog, corrupted_prog, regex, replacement='', name='', include_kind=False):
        # for pointer mutate
        if regex == '[^)@,#\]] (_<op>_\*)(?! _<number>_)':
            positions = [m.span(1) for m in re.finditer(regex, corrupted_prog)]
        else:
            positions = [m.span() for m in re.finditer(regex, corrupted_prog)]
                
        if len(positions) > 1:
            to_corrupt = np.random.randint(len(positions))
        elif len(positions) == 1:
            to_corrupt = 0
        elif include_kind:
            return corrupted_prog, None, None, None
        else:
            return corrupted_prog, None, None
                
        corrupted_prog = corrupted_prog[:positions[to_corrupt][0]] + replacement + corrupted_prog[positions[to_corrupt][1]:]
        
        fix = isolate_line(org_prog, positions[to_corrupt][0])
        line = extract_line_number(fix)
        
        if include_kind:
            return corrupted_prog, fix, line, name
        else:
            return corrupted_prog, fix, line
    pass    

    def __get_pmf(self):                

        _actions = Typo_Mutate.__actions
        
        pmf = []
        denominator = 0
    
        for _, value in _actions.iteritems():
            pmf.append(value[2])
            denominator += value[2]
    
        for i in range(len(pmf)):
            pmf[i] = float(pmf[i])/float(denominator)
            
        return pmf
    
    def __init__(self):
        if Typo_Mutate.__pmf == None:
            Typo_Mutate.__pmf = self.__get_pmf()
        assert Typo_Mutate.__pmf != None, 'pmf is None'
        
        if Typo_Mutate.__mutation_distribution == None:
            Typo_Mutate.__mutation_distribution = {action:0 for action in Typo_Mutate.__actions.keys()}
        assert Typo_Mutate.__mutation_distribution != None, '_mutation_distribution is None'
        

    def easy_mutate2(self, org_prog, corrupted_prog, include_kind=False):
        actions = Typo_Mutate.__actions
        action = np.random.choice(actions.keys(), 1, p=Typo_Mutate.__pmf)[0]
        Typo_Mutate.__mutation_distribution[action] += 1
        
        return self.find_and_replace(org_prog, corrupted_prog, actions[action][0], actions[action][1], name=action, include_kind=include_kind)
    
    def get_mutation_distribution(self):
        return Typo_Mutate.__mutation_distribution
    
    def specific_mutate(self, org_prog, corrupted_prog, action, include_kind=False):
        actions = Typo_Mutate.__actions
        assert action in actions, "action passed as arg: %s, was not found in the allowed list of actions" % action
        return self.find_and_replace(org_prog, corrupted_prog, actions[action][0], actions[action][1], name=action, include_kind=include_kind)
pass


mutator_obj = Typo_Mutate()


def get_mutation_distribution():
    global mutator_obj
    return mutator_obj.get_mutation_distribution()
    
# Includes incremental incorrect-fix pairs for a program in a decreasing order of number of errors
#  until correct-noFix pair is generated!! (not generating correct-nofix pair for compatibility with other functions)
def token_mutate_series(prog, max_num_mutations, num_mutated_progs, include_kind=False, flag_last_fix=False):
    assert max_num_mutations > 0 and num_mutated_progs > 0, "Invalid argument(s) supplied to the function token_mutate"
    
    global mutator_obj    
    
    corrupt_fix_pair = set()
    
    for _ in range(num_mutated_progs):
        num_mutations = random.choice(range(max_num_mutations)) + 1
        this_corrupted = prog
        lines = set()
        mutation_count = 0
        loop_counter = 0
        loop_count_threshold = 50

        if include_kind:
            fix_kinds = {}        
        
        while(mutation_count < num_mutations):
            loop_counter += 1
            if loop_counter == loop_count_threshold:
                print "mutation_count", mutation_count                
                raise LoopCountThresholdExceededException
            line = None
            
            if include_kind:
                this_corrupted, fix, line, kind = mutator_obj.easy_mutate2(prog, this_corrupted, include_kind=True)
            else:
                this_corrupted, fix, line = mutator_obj.easy_mutate2(prog, this_corrupted)

            if line is not None:
                fix = fetch_line(prog, line)
                corrupt_line = fetch_line(this_corrupted, line)

                if fix != corrupt_line:
                    lines.add(line)
                    mutation_count += 1

                    if include_kind:
                        if str(line) not in fix_kinds:
                            fix_kinds[str(line)] = [kind]
                        else:
                            fix_kinds[str(line)].append(kind)
    
        assert len(lines) > 0, "Could not mutate!"
        
        flag_empty_line_in_corrupted = False
        for _line_ in get_lines(this_corrupted):
            if _line_.strip() == '':
                flag_empty_line_in_corrupted = True
                break
                
        if flag_empty_line_in_corrupted:
            continue
        
        sorted_lines = reversed(sorted(lines)) if flag_last_fix else sorted(lines)

        for line in sorted_lines:
            fix = fetch_line(prog, line)
            corrupt_line = fetch_line(this_corrupted, line)
            assert len(fetch_line(prog, line, include_line_number=False).strip()) != 0, "empty fix" 
            assert len(fetch_line(this_corrupted, line, include_line_number=False).strip()) != 0, "empty corrupted line"
            if fix != corrupt_line:
                if include_kind:
                    if len(fix_kinds[str(line)]) == 1: # remove later
                        for kind in fix_kinds[str(line)]:
                            corrupt_fix_pair.add((this_corrupted, fix, kind))
                else:
                    corrupt_fix_pair.add((this_corrupted, fix))
            
                try:
                    this_corrupted = do_fix_at_line(this_corrupted, line, fetch_line(prog, line, include_line_number=False))
                except IndexError:
                    raise
        
    return list(corrupt_fix_pair)


def token_mutate_series_any_fix(prog, max_num_mutations, num_mutated_progs, include_kind=False):
    assert max_num_mutations > 0 and num_mutated_progs > 0, "Invalid argument(s) supplied to the function token_mutate"
    
    global mutator_obj    
    
    corrupt_fix_pair = set()
    
    for _ in range(num_mutated_progs):
        num_mutations = random.choice(range(max_num_mutations)) + 1
        this_corrupted = prog
        lines = set()
        mutation_count = 0
        loop_counter = 0
        loop_count_threshold = 50

        if include_kind:
            fix_kinds = {}        
        
        while(mutation_count < num_mutations):
            loop_counter += 1
            if loop_counter == loop_count_threshold:
                print "mutation_count", mutation_count                
                raise LoopCountThresholdExceededException
            line = None
            
            if include_kind:
                this_corrupted, fix, line, kind = mutator_obj.easy_mutate2(prog, this_corrupted, include_kind=True)
            else:
                this_corrupted, fix, line = mutator_obj.easy_mutate2(prog, this_corrupted)

            if line is not None:
                fix = fetch_line(prog, line)
                corrupt_line = fetch_line(this_corrupted, line)

                if fix != corrupt_line:
                    lines.add(line)
                    mutation_count += 1

                    if include_kind:
                        if str(line) not in fix_kinds:
                            fix_kinds[str(line)] = [kind]
                        else:
                            fix_kinds[str(line)].append(kind)
    
        assert len(lines) > 0, "Could not mutate!"
        
        flag_empty_line_in_corrupted = False
        for _line_ in get_lines(this_corrupted):
            if _line_.strip() == '':
                flag_empty_line_in_corrupted = True
                break
                
        if flag_empty_line_in_corrupted:
            continue
        
        lines = sorted(lines)
        ranked_lines = map(lambda x:(x,lines.index(x)+1), lines)
        random.shuffle(ranked_lines)
        random.shuffle(lines)

        for line, fix_number in ranked_lines:
            fix = fetch_line(prog, line)
            corrupt_line = fetch_line(this_corrupted, line)
            assert len(fetch_line(prog, line, include_line_number=False).strip()) != 0, "empty fix" 
            assert len(fetch_line(this_corrupted, line, include_line_number=False).strip()) != 0, "empty corrupted line"
            if fix != corrupt_line:
                if include_kind:
                    if len(fix_kinds[str(line)]) == 1: # remove later
                        for kind in fix_kinds[str(line)]:
                            corrupt_fix_pair.add((this_corrupted, fix, fix_number, kind))
                else:
                    corrupt_fix_pair.add((this_corrupted, fix, fix_number))
            
                try:
                    this_corrupted = do_fix_at_line(this_corrupted, line, fetch_line(prog, line, include_line_number=False))
                except IndexError:
                    raise
                
    if include_kind:
        return map( lambda (w,x,y,z):(add_fix_number(w, y), x, z), list(corrupt_fix_pair))
    else:
        return map( lambda (w,x,y):(add_fix_number(w, y), x), list(corrupt_fix_pair))

def token_mutate_for_tsne(prog, num_mutations, include_kind=False):
    assert num_mutations > 0, "Invalid argument(s) supplied to the function token_mutate"
    global mutator_obj
    
    corrupt_fix_pair = set()
    
    for _ in range(1):
        this_corrupted = prog
        lines = set()
        mutation_count = 0
        loop_counter = 0
        loop_count_threshold = 50

        if include_kind:
            fix_kinds = {}        
        
        while(mutation_count < num_mutations):
            loop_counter += 1
            if loop_counter == loop_count_threshold:
                print "mutation_count", mutation_count                
                raise LoopCountThresholdExceededException
            line = None
            
            if include_kind:
                this_corrupted, fix, line, kind = mutator_obj.easy_mutate2(prog, this_corrupted, include_kind=True)
            else:
                this_corrupted, fix, line = mutator_obj.easy_mutate2(prog, this_corrupted)

            if line is not None:
                fix = fetch_line(prog, line)
                corrupt_line = fetch_line(this_corrupted, line)

                if fix != corrupt_line:
                    lines.add(line)
                    mutation_count += 1

                    if include_kind:
                        if str(line) not in fix_kinds:
                            fix_kinds[str(line)] = [kind]
                        else:
                            fix_kinds[str(line)].append(kind)
    
        assert len(lines) > 0, "Could not mutate!"
        
        empty_line_in_corrupted = False
        for _line_ in get_lines(this_corrupted):
            if _line_.strip() == '':
                empty_line_in_corrupted = True
                break
                
        if empty_line_in_corrupted:
            continue
        
        sorted_lines = sorted(lines)

        for line in sorted_lines:
            fix = fetch_line(prog, line)
            corrupt_line = fetch_line(this_corrupted, line)
            assert len(fetch_line(prog, line, include_line_number=False).strip()) != 0, "empty fix" 
            assert len(fetch_line(this_corrupted, line, include_line_number=False).strip()) != 0, "empty corrupted line"
            if fix != corrupt_line:
                corrupt_fix_pair.add((this_corrupted, fix))
                break
        
    return list(corrupt_fix_pair)


def token_mutate_for_tsne_with_specific_errors(prog, num_mutations, action, include_kind=False):
    assert num_mutations > 0, "Invalid argument(s) supplied to the function token_mutate"        
    global mutator_obj
    specific_mutate = mutator_obj.specific_mutate
    corrupt_fix_pair = set()
    
    for _ in range(1):
        this_corrupted = prog
        lines = set()
        mutation_count = 0
        loop_counter = 0
        loop_count_threshold = 50

        if include_kind:
            fix_kinds = {}        
        
        while(mutation_count < num_mutations):
            loop_counter += 1
            if loop_counter == loop_count_threshold:
                print "mutation_count", mutation_count                
                raise LoopCountThresholdExceededException
            line = None
            
            if include_kind:
                this_corrupted, fix, line, kind = specific_mutate(prog, this_corrupted, action, include_kind=True)
            else:
                this_corrupted, fix, line = specific_mutate(prog, this_corrupted, action)

            if line is not None:
                fix = fetch_line(prog, line)
                corrupt_line = fetch_line(this_corrupted, line)

                if fix != corrupt_line:
                    lines.add(line)
                    mutation_count += 1

                    if include_kind:
                        if str(line) not in fix_kinds:
                            fix_kinds[str(line)] = [kind]
                        else:
                            fix_kinds[str(line)].append(kind)
    
        assert len(lines) > 0, "Could not mutate!"
        
        empty_line_in_corrupted = False
        for _line_ in get_lines(this_corrupted):
            if _line_.strip() == '':
                empty_line_in_corrupted = True
                break
                
        if empty_line_in_corrupted:
            continue
        
        sorted_lines = sorted(lines)

        for line in sorted_lines:
            fix = fetch_line(prog, line)
            corrupt_line = fetch_line(this_corrupted, line)
            assert len(fetch_line(prog, line, include_line_number=False).strip()) != 0, "empty fix" 
            assert len(fetch_line(this_corrupted, line, include_line_number=False).strip()) != 0, "empty corrupted line"
            if fix != corrupt_line:
                corrupt_fix_pair.add((this_corrupted, fix))
                break
        
    return list(corrupt_fix_pair)
