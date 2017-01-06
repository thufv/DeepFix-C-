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
