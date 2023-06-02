import cv2
import numpy as np
from tensorflow.keras.models import load_model
import imutils
import os
import urllib.request as urllib

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model-OCR.h5')


classes = np.arange(0, 10)

model = load_model(model_path)
# print(model.summary())
input_size = 48


# DataFlair Sudoku solver
def find_empty(board):
    """checkes where is an empty or unsolved block"""
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 0:
                return (i, j)  # row, col

    return None

def valid(board, num, pos):
    # Check row
    for i in range(len(board[0])):
        if board[pos[0]][i] == num and pos[1] != i:
            return False

    # Check column
    for i in range(len(board)):
        if board[i][pos[1]] == num and pos[0] != i:
            return False

    # Check box
    box_x = pos[1] // 3
    box_y = pos[0] // 3

    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x * 3, box_x*3 + 3):
            if board[i][j] == num and (i,j) != pos:
                return False

    return True


def solve(board):
    find = find_empty(board)
    if not find:
        return True
    else:
        row, col = find

    for i in range(1,10):
        if valid(board, i, (row, col)):
            board[row][col] = i

            if solve(board):
                return True
            board[row][col] = 0
    return False


def get_board(bo):
    """Takes a 9x9 matrix unsolved sudoku board and returns a fully solved board."""
    if solve(bo):
        return bo
    else:
        raise ValueError


def get_perspective(img, location, height = 900, width = 900):
    """Takes an image and location os interested region.
        And return the only the selected region with a perspective transformation"""
    pts1 = np.float32([location[0], location[3], location[1], location[2]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (width, height))
    return result

def get_InvPerspective(img, masked_num, location, height = 900, width = 900):
    """Takes original image as input"""
    pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    pts2 = np.float32([location[0], location[3], location[1], location[2]])

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(masked_num, matrix, (img.shape[1], img.shape[0]))
    return result

def find_board(img):
    """Takes an image as input and finds a sudoku board inside of the image"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 13, 20, 20)
    edged = cv2.Canny(bfilter, 30, 180)
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours  = imutils.grab_contours(keypoints)

    # newimg = cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 3)
    # cv2.imshow("Contour", newimg)


    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
    location = None

    # Finds rectangular contour
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 15, True)
        if len(approx) == 4:
            location = approx
            break
    result = get_perspective(img, location)
    return result, location


def split_boxes(board):
    """Takes a sudoku board and split it into 81 cells.
        each cell contains an element of that board either given or an empty cell."""
    rows = np.vsplit(board,9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r,9)
        for box in cols:
            box = cv2.resize(box, (input_size, input_size))/255.0
            # cv2.imshow("Splitted block", box)
            # cv2.waitKey(50)
            boxes.append(box)
    # cv2.destroyAllWindows()
    return boxes



def displayNumbers(img, numbers, color=(0, 255, 0)):
    """Displays 81 numbers in an image or mask at the same position of each cell of the board"""
    W = int(img.shape[1]/9)
    H = int(img.shape[0]/9)
    for i in range (9):
        for j in range (9):
            if numbers[(j*9)+i] !=0:
                cv2.putText(img, str(numbers[(j*9)+i]), (i*W+int(W/2)-int((W/4)), int((j+0.7)*H)), cv2.FONT_HERSHEY_COMPLEX, 2, color, 2, cv2.LINE_AA)
    return img

def solve_sudoku():

    firebaseURL = "https://firebasestorage.googleapis.com/v0/b/sudokuforever-b9936.appspot.com/o/"

    image_url = firebaseURL+ "sudoku.jpg?alt=media&token=https://firebasestorage.googleapis.com/v0/b/sudokuforever-b9936.appspot.com/o/sudoku.jpg?alt=media&token=1e41c789-536d-4c81-b172-37d3d2650a4d"
    
    resp = urllib.urlopen(image_url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)

    
    # image_path = os.path.join(current_dir, 'sudoku1.jpg')
    # Read image

    # img = cv2.imread(image_path)

    # ... the rest of your code ...
    board, location = find_board(img)
    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    # print(gray.shape)
    rois = split_boxes(gray)
    rois = np.array(rois).reshape(-1, input_size, input_size, 1)

    # get prediction
    prediction = model.predict(rois)
    # print(prediction)

    predicted_numbers = []
    # get classes from prediction
    for i in prediction:
        index = (np.argmax(i)) # returns the index of the maximum number of the array
        predicted_number = classes[index]
        predicted_numbers.append(predicted_number)

    # print(predicted_numbers)

    # reshape the list
    board_num = np.array(predicted_numbers).astype('uint8').reshape(9, 9)
    # Solve the board and return the solved numbers
    try:
        solved_board_nums = get_board(board_num)
        solved_board_nums = solved_board_nums.tolist()
        # print(solved_board_nums)
        return solved_board_nums
    except:
        return "Solution doesn't exist. Model misread digits."