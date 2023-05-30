from flask import Flask
from flask import request
import json
import time
import cv2
import numpy as np
# from solver import solve_sudoku
from solver import solve_sudoku


app = Flask(__name__)

@app.route('/', methods =('GET', 'POST'))
def handle_request():
    grid = solve_sudoku()
    data_set = {'timestamp': time.time(), 'just_check':grid}
    json_dump = json.dumps(data_set)
    return json_dump


if __name__ == '__main__':
    app.run()