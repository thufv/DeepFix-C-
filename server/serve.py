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

from backend import DeepFixServer as DeepFix
from bottle import run, post, request, response, get, route, static_file
import json, argparse

parser = argparse.ArgumentParser(description="Host a server that suggests fixes")
parser.add_argument('fold', help='Model to serve', type=int)
parser.add_argument('embedding_dim', type=int)
parser.add_argument('memory_dim', type=int)
parser.add_argument('num_layers', type=int)
parser.add_argument('rnn_cell', help='One of RNN, LSTM or GRU.')
parser.add_argument('dropout', help='Probability to use for dropout', type=float)
parser.add_argument('task', help="One of 'typo' or 'ids' (undeclared)", choices=['typo', 'ids'])
parser.add_argument('-v', '--vram', help='Fraction of GPU memory to use', type=float, default=0.2)

args    = parser.parse_args()
deepfix = DeepFix(args.fold, args.embedding_dim, args.memory_dim, \
                  args.num_layers, args.rnn_cell, args.dropout, args.task, args.vram)

@route('/')
def index():
    return static_file('server/demo.html', root='.')

@route('/run', method = 'POST')
def process():
    global deepfix
    fixes, repaired, status_codes = deepfix.process([request.forms.get('source_code')])

    result = {}
    result['fixes'] = fixes[0]
    result['repaired'] = repaired[0]

    return json.dumps(result)

if __name__ == "__main__":
    print "Ready!"
    run(host='0.0.0.0', port=8081, server='paste', debug=True)
