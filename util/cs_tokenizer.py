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

import collections
import regex as re
from util.helpers import get_lines, recompose_program
from util.tokenizer import Tokenizer, UnexpectedTokenException, EmptyProgramException

Token = collections.namedtuple('Token', ['typ', 'value', 'line', 'column'])


class CS_Tokenizer(Tokenizer):
    _keywords = ['abstract', 'add', 'alias', 'as', 'ascending', 'async',
                 'await', 'base', 'break', 'by', 'case', 'catch', 'checked',
                 'class', 'const', 'continue', 'default', 'delegate',
                 'descending', 'do', 'else', 'enum', 'equals', 'event',
                 'explicit', 'extern', 'false', 'finally', 'fixed', 'for',
                 'foreach', 'from', 'get', 'global', 'goto', 'group', 'if',
                 'implicit', 'in', 'interface', 'internal', 'into', 'is',
                 'join', 'let', 'lock', 'nameof', 'namespace', 'new',
                 'notnull', 'null', 'on', 'operator', 'orderby', 'out',
                 'override', 'params', 'partial', 'private', 'protected',
                 'public', 'readonly', 'ref', 'remove', 'return', 'sealed',
                 'select', 'set', 'sizeof', 'stackalloc', 'static', 'struct',
                 'switch', 'this', 'throw', 'true', 'try', 'typeof',
                 'unchecked', 'unmanaged', 'unsafe', 'using', 'value', 'var',
                 'virtual', 'void', 'volatile', 'when', 'where', 'while',
                 'yield']
    _types = ['bool', 'byte', 'char', 'decimal', 'double', 'dynamic', 'float',
              'int', 'long', 'object', 'sbyte', 'short', 'string', 'uint',
              'ulong', 'ushort']

    def _escape(self, string):
        return repr(string)[1:-1]

    def _tokenize_code(self, code):
        token_specification = [
            ('comment',
             r'\/\*(?:[^*]|\*(?!\/))*\*\/|\/\*([^*]|\*(?!\/))*\*?|\/\/[^\n]*'),
            # ('directive', r'#\w+'),
            ('string', r'[$@]?"(?:[^"\n]|\\")*"?'),
            ('char', r"'(?:\\?[^'\n]|\\')'"),
            ('char_continue', r"'[^']*"),
            ('number',  r'[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'),
            # ('include',  r'(?<=\#include) *<([_A-Za-z]\w*(?:\.h))?>'),
            ('op',
             r'\(|\)|\[|\]|{|}|->|=>|<<|>>|\*\*|\|\||&&|--|\+\+|[-+*|&%\/=]=|[-<>~!%^&*\/+=?|.,:;#]'),
            ('name',  r'[_A-Za-z]\w*'),
            ('whitespace',  r'\s+'),
            ('nl', r'\\\n?'),
            ('MISMATCH', r'.'),            # Any other character
        ]
        tok_regex = '|'.join('(?P<%s>%s)' %
                             pair for pair in token_specification)
        for mo in re.finditer(tok_regex, code):
            kind = mo.lastgroup
            value = mo.group(kind)
            if kind == 'MISMATCH':
                yield UnexpectedTokenException('%r unexpected on line %d' % (value, 1))
            else:
                yield Token(kind, value, 1, mo.start())

    def _sanitize_brackets(self, tokens_string):
        lines = get_lines(tokens_string)

        if lines == ['']:
            raise EmptyProgramException(tokens_string)

        for i in range(len(lines) - 1, -1, -1):
            line = lines[i]

            if line.strip() == '_<op>_}' or line.strip() == '_<op>_} _<op>_}' \
               or line.strip() == '_<op>_} _<op>_} _<op>_}' or line.strip() == '_<op>_} _<op>_;' \
               or line.strip() == '_<op>_} _<op>_} _<op>_} _<op>_}' \
               or line.strip() == '_<op>_{' \
               or line.strip() == '_<op>_{ _<op>_{':
                if i > 0:
                    lines[i - 1] += ' ' + line.strip()
                    lines[i] = ''
                else:
                    # can't handle this case!
                    return ''

        # Remove empty lines
        for i in range(len(lines) - 1, -1, -1):
            if lines[i] == '':
                del lines[i]

        for line in lines:
            assert(lines[i].strip() != '')
            # Should be line instead of lines[i]???

        return recompose_program(lines)

    def tokenize(self, code, keep_format_specifiers=False, keep_names=True,
                 keep_literals=False):
        result = '0 ~ '

        line_count = 1
        name_dict = {}
        name_sequence = []

        isNewLine = True

        # Get the iterable
        my_gen = self._tokenize_code(code)

        while True:
            try:
                token = next(my_gen)
            except StopIteration:
                break

            if isinstance(token, Exception):
                return '', '', ''

            type_ = token[0]
            value = token[1]

            if value in self._keywords:
                result += '_<keyword>_' + self._escape(value) + ' '
                isNewLine = False

            elif value in self._types:
                result += '_<type>_' + self._escape(value) + ' '
                isNewLine = False

            elif type_ == 'whitespace' and (('\n' in value) or ('\r' in value)):
                if isNewLine:
                    continue

                result += ' '.join(list(str(line_count))) + ' ~ '
                line_count += 1
                isNewLine = True

            elif type_ == 'whitespace' or type_ == 'comment' or type_ == 'nl':
                pass

            elif 'string' in type_:
                matchObj = [m.group().strip() for m in re.finditer(
                            '%(d|i|f|c|s|u|g|G|e|p|llu|ll|ld|l|o|x|X)', value)]
                if matchObj and keep_format_specifiers:
                    for each in matchObj:
                        result += each + ' '
                else:
                    result += '_<string>_' + ' '
                isNewLine = False

            elif type_ == 'name':
                if keep_names:
                    if self._escape(value) not in name_dict:
                        name_dict[self._escape(value)] = str(
                            len(name_dict) + 1)

                    name_sequence.append(self._escape(value))
                    result += '_<id>_' + name_dict[self._escape(value)] + '@ '
                else:
                    result += '_<id>_' + '@ '
                isNewLine = False

            elif type_ == 'number':
                if keep_literals:
                    result += '_<number>_' + self._escape(value) + '# '
                else:
                    result += '_<number>_' + '# '
                isNewLine = False

            elif 'char' in type_ or value == '':
                result += '_<' + type_.lower() + '>_' + ' '
                isNewLine = False

            else:
                converted_value = self._escape(value).replace('~', 'TiLddE')
                result += '_<' + type_ + '>_' + converted_value + ' '

                isNewLine = False

        result = result[:-1]

        if result.endswith('~'):
            idx = result.rfind('}')
            result = result[:idx + 1]

        return self._sanitize_brackets(result), name_dict, name_sequence
