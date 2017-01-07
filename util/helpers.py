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

import subprocess32 as subprocess
import os, tempfile, time
import numpy as np

# Exceptions used by helpers
class FailedToGetLineNumberException(Exception):
    pass

class InvalidFixLocationException(Exception):
    pass

class SubstitutionFailedException(Exception):
    pass

def compilation_errors(string):
    name1 = int(time.time() * 10**6)
    name2 = np.random.randint(0, 1000 + 1)
    filename = 'tempfile_%d_%d.c' % (name1, name2)
    
    with open(filename, 'w+') as f:
        f.write(string)
    
    shell_string = "gcc -w -std=c99 -pedantic %s -lm" % (filename,)
    
    try:
        result = subprocess.check_output(shell_string, timeout=30, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        result = e.output
    
    os.unlink('%s' % (filename,))
    
    error_set = []
    
    for line in result.splitlines():
        if 'error:' in line:
            error_set.append(line)
    
    return error_set, result

# Input: tokenized program
# Returns: array of lines, each line is tokenized
def get_lines(program_string):
    tokens = program_string.split()
    ignore_tokens = ['~'] + [chr(n + ord('0')) for n in range(10)]
    
    # Split token string into lines
    lines = []
    
    for token in tokens:
        if token in ignore_tokens and token == '~':
            if len(lines) > 0:
                lines[-1] = lines[-1].rstrip(' ')
            lines.append('')
        elif token not in ignore_tokens:
            lines[-1] += token + ' '

    #for line in lines:
    #    assert line.strip() != '', "program_string: \n" + program_string             
    return lines

# Input: output of get_lines() (tokenized lines)
# Result: Tokenized program
def recompose_program(lines):
    recomposed_program = ''

    for i, line in enumerate(lines):
        for digit in str(i):
            recomposed_program += digit + ' '

        recomposed_program += '~ '
        recomposed_program += line + ' '
        
    return recomposed_program

# Fetches one specific line from the program
def fetch_line(program_string, line_number, include_line_number=True):
    result = ''

    if include_line_number:
        for digit in str(line_number):
            result += digit + ' '

        result += '~ '

    result += get_lines(program_string)[line_number]
    #assert result.strip() != ''
    return result

def fix_to_source(fix, tokens, name_dict, name_seq=None, literal_seq=None, clang_format=False):
    result = ''
    type_ = None

    reverse_name_dict = {}
    name_count = 0

    for k,v in name_dict.iteritems():
        reverse_name_dict[v] = k

    line_number = extract_line_number(fix)

    tokens = recompose_program(get_lines(tokens)[:line_number])

    for token in tokens.split():
        try:
            type_, content = token.split('>_')
            type_ = type_.lstrip('_<')

            if type_ == 'id':
                if name_seq is not None:
                    name_count += 1

            if type_ == 'number':
                if literal_seq is not None:
                    literal_seq = literal_seq[1:]

            elif type_ == 'string':
                if literal_seq is not None:
                    literal_seq = literal_seq[1:]

            elif type_ == 'char':
                if literal_seq is not None:
                    literal_seq = literal_seq[1:]

        except ValueError:
            if token == '~':
                pass

    for token in fix.split():
        try:
            prev_type_was_op = (type_ == 'op')

            type_, content = token.split('>_')
            type_ = type_.lstrip('_<')

            if type_ == 'id':
                if name_seq is not None:
                    content = name_seq[name_count]
                    name_count += 1
                else:
                    try:
                        content = reverse_name_dict[content.rstrip('@')]
                    except KeyError:
                        content = 'new_id_' + content.rstrip('@')

            elif type_ == 'number':
                content = content.rstrip('#')

            if type_ == 'directive' or type_ == 'include' or type_ == 'op' or type_ == 'type' or type_ == 'keyword' or type_ == 'APIcall':
                if type_ == 'op' and prev_type_was_op:
                    result = result[:-1] + content + ' '
                else:
                    result += content + ' '

            elif type_ == 'id':
                result += content + ' '

            elif type_ == 'number':
                if literal_seq is None:
                    result += '0 '
                else:
                    result += '%s ' % literal_seq[0]
                    literal_seq = literal_seq[1:]

            elif type_ == 'string':
                if literal_seq is None:
                    result += '"String" '
                else:
                    result += '%s ' % literal_seq[0]
                    literal_seq = literal_seq[1:]

            elif type_ == 'char':
                if literal_seq is None:
                    result += "'c' "
                else:
                    result += '%s ' % literal_seq[0]
                    literal_seq = literal_seq[1:]

        except ValueError:
            if token == '~':
                pass

    if not clang_format:
        return result

    source_file = tempfile.NamedTemporaryFile(suffix=".c", delete=False)
    source_file.write(result)
    source_file.close()

    shell_string = 'clang-format %s' % source_file.name
    clang_output = subprocess.check_output(shell_string, timeout=30, shell=True)
    os.unlink(source_file.name)

    return clang_output

# Input: tokenized program
# Returns: source code, optionally clang-formatted
def tokens_to_source(tokens, name_dict, clang_format=False, name_seq=None, literal_seq=None):
    result = ''
    type_ = None
    
    reverse_name_dict = {}
    name_count = 0
    
    for k,v in name_dict.iteritems():
        reverse_name_dict[v] = k
    
    for token in tokens.split():
        try:
            prev_type_was_op = (type_ == 'op')
            
            type_, content = token.split('>_')
            type_ = type_.lstrip('_<')
            
            if type_ == 'id':
                if name_seq is not None:
                    content = name_seq[name_count]
                    name_count += 1
                else:
                    try:
                        content = reverse_name_dict[content.rstrip('@')]
                    except KeyError:
                        content = 'new_id_' + content.rstrip('@')
            elif type_ == 'number':
                content = content.rstrip('#')
                
            if type_ == 'directive' or type_ == 'include' or type_ == 'op' or type_ == 'type' or type_ == 'keyword' or type_ == 'APIcall':
                if type_ == 'op' and prev_type_was_op:
                    result = result[:-1] + content + ' '
                else:
                    result += content + ' '
            elif type_ == 'id':
                result += content + ' '
            elif type_ == 'number':
                if literal_seq is None:
                    result += '0 '
                else:
                    result += '%s ' % literal_seq[0]
                    literal_seq = literal_seq[1:]
            elif type_ == 'string':
                if literal_seq is None:
                    result += '"String" '
                else:
                    result += '%s ' % literal_seq[0]
                    literal_seq = literal_seq[1:]
            elif type_ == 'char':
                if literal_seq is None:
                    result += "'c' "
                else:
                    result += '%s ' % literal_seq[0]
                    literal_seq = literal_seq[1:]
        except ValueError:
            if token == '~':
                result += '\n'
    
    if not clang_format:
        return result
    
    source_file = tempfile.NamedTemporaryFile(suffix=".c", delete=False)
    source_file.write(result)
    source_file.close()
    
    shell_string = 'clang-format %s' % source_file.name
    clang_output = subprocess.check_output(shell_string, timeout=30, shell=True)
    os.unlink(source_file.name)
    
    return clang_output

# This function returns the line where we indexed into the string
# [program_string] is a tokenized program
# [char_index] is an index into program_string
def isolate_line(program_string, char_index):
    begin = program_string[:char_index].rfind('~') - 2
    
    while begin - 2 > 0 and program_string[begin - 2] in [str(i) for i in range(10)]:
        begin -= 2
        
    if program_string[char_index:].find('~') == -1:
        end = len(program_string)
    else:
        end = char_index + program_string[char_index:].find('~') - 2

        while end - 2 > 0 and program_string[end - 2] in [str(i) for i in range(10)]:
            end -= 2

        end -= 1
    
    return program_string[begin:end]

# Extracts the line number for a tokenized line, e.g. '1 2 ~ _<token>_ ...' returns 12
def extract_line_number(line):
    number = 0
    never_entered = True
    
    for token in line.split('~')[0].split():
        never_entered = False
        number *= 10
        try:
            number += int(token) - int('0')
        except ValueError:
            raise
        
    if never_entered:
        raise FailedToGetLineNumberException(line)
        
    return number


def done(msg=''):
    import time
    if msg == '':
        print 'done at', time.strftime("%d/%m/%Y"), time.strftime("%H:%M:%S")
    else:
        print msg, ',done at', time.strftime("%d/%m/%Y"), time.strftime("%H:%M:%S")

# Checks if all of the ids in the fix are present in the original program
# fix_string is the token string of the fix
# program_string is the token string of the program
def fix_ids_are_in_program(program_string, fix_string):
    prog_ids = []
    fix_ids  = []
    
    for token in program_string.split():
        if '_<id>_' in token and token not in prog_ids:
            prog_ids.append(token)
            
    for token in fix_string.split():
        if '_<id>_' in token and token not in fix_ids:
            fix_ids.append(token)
    
    for fix_id in fix_ids:
        if fix_id not in prog_ids:
            return False
        
    return True

# Does what it says
def reverse_name_dictionary(dictionary):
    rev = {}
    
    for x, y in dictionary.iteritems():
        rev['_<id>_' + y + '@'] = x
        
    return rev

# Prints a pretty version of a fix
def pretty_fix(fix, name_dict):
    fix_tokens = fix.split()
    rev_name_dict = reverse_name_dictionary(name_dict) 
    
    try:
        for i in range(len(fix_tokens)):
            if '_<id>_' in fix_tokens[i]:
                fix_tokens[i] = rev_name_dict[fix_tokens[i]]
            elif '_<insertion>_' in fix_tokens[i]:
                fix_tokens[i] = 'Insert at'
            elif fix_tokens[i] == '~':
                fix_tokens[i] = ':'
            elif '_<type>_' in fix_tokens[i]:
                fix_tokens[i] = fix_tokens[i][8:]
            elif '_<op>_' in fix_tokens[i]:
                fix_tokens[i] = fix_tokens[i][6:]
    except KeyError:
        print name_dict
            
    return ' '.join(fix_tokens)

def create_dir(dir_path):
    try:
        import os
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    except:
        pass

def replace_ids(new_line, old_line):
    ids = []
    
    for token in old_line.split():
        if '_<id>_' in token:
            ids.append(token)
    
    result = ''
    counter = 0
    
    for token in new_line.split():
        if '_<id>_' not in token:
            result += token + ' '
        else:
            result += ids[counter] + ' '
            counter += 1
    
    return result.strip()

def apply_fix(program_string, fix_string, type_):
    # Break up program string into lines
    orig_lines = get_lines(program_string)
    
    # Retrieve insertion location
    fix_location = 0
    fix = ''
    hit_tilde = False
    
    if type_ == 'insert':
        first_token = True
    
    for token in fix_string.split():
        if type_ == 'insert':
            if first_token:
                first_token = False
                
                if token == '_<insertion>_':
                    continue
                else:
                    print "Warning: First token did not suggest insertion (should not happen)"
        else:
            assert type_ == 'replace'
        
        if not hit_tilde:
            if token in [chr(n + ord('0')) for n in range(10)]:
                fix_location *= 10
                fix_location += ord(token) - ord('0')
            elif token == '~':
                hit_tilde = True
            else:
                raise InvalidFixLocationException
        else:
            # We are done!
            if token == '_eos_':
                fix = fix[:-1]
                break
            else:
                fix += token + ' '
                
    # Insert the fix
    if type_ == 'insert':        
        orig_lines.insert(fix_location+1, fix)
    else:
        if orig_lines[fix_location].count('_<id>_') != fix.count('_<id>_'):
            raise SubstitutionFailedException
        
        assert type_ == 'replace'
        orig_lines[fix_location] = replace_ids(fix, orig_lines[fix_location]) #fix
        
    return recompose_program(orig_lines)

def make_dir_if_not_exists(path):
    try:
        os.makedirs(path)
    except:
        pass

def calculate_token_level_accuracy(Y, Y_hat):
    token_accuracy = float(np.count_nonzero(np.equal(Y, Y_hat)))/np.prod(np.shape(Y))
    return token_accuracy

def line_equal(y, y_hat, tl_dict):
    tilde_token = tl_dict['~']
    
    y_line = []
    y_hat_line = []
    
    for token in y:
        if token != tilde_token:
            y_line.append(token)
        else:
            break
    
    for token in y_hat:
        if token != tilde_token:
            y_hat_line.append(token)
        else:
            break

    return np.array_equal(y_line, y_hat_line)

def calculate_localization_accuracy(Y, Y_hat, dictionary):
    localization_accuracy = 0

    for y, y_hat in zip(Y, Y_hat):
        if np.array_equal(y, y_hat):
            localization_accuracy += 1
        elif line_equal(y, y_hat, dictionary):
            localization_accuracy += 1

    localization_accuracy = float(localization_accuracy)/float(len(Y))
    return localization_accuracy

def calculate_repair_accuracy(Y, Y_hat):
    repair_accuracy = 0

    for y, y_hat in zip(Y, Y_hat):
        if np.array_equal(y, y_hat):
            repair_accuracy += 1

    repair_accuracy = float(repair_accuracy)/float(len(Y))
    return repair_accuracy
