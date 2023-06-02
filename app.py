from flask import Flask
from flask import request
import json
import time
from solver import solve_sudoku
import os
from flask_cors import CORS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
CORS(app)

@app.route('/', methods =('GET', 'POST'))
def handle_request():
    grid = solve_sudoku()
    print(grid)
    data_set = {'timestamp': time.time(), 'solution':grid}
    json_dump = json.dumps(data_set)
    return json_dump

if __name__ == '__main__':
    app.run(host='0.0.0.0')