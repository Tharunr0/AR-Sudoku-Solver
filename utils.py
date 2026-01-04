import cv2
import numpy as np

#### 1. IMAGE PROCESSING ####

def preProcess(img):
    """
    Converts image to grayscale, blurs it, and applies adaptive thresholding
    to find the grid lines.
    """
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)
    return imgThreshold

def biggestContour(contours):
    """
    Finds the largest 4-sided contour in the image (assumed to be the Sudoku grid).
    """
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area

def reorder(myPoints):
    """
    Reorders the 4 corner points of the grid to:
    [Top-Left, Top-Right, Bottom-Left, Bottom-Right]
    """
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

#### 2. AI RECOGNITION ####

def get_board_from_image(imgWarpGray, model):
    """
    Splits the grid into 81 cells and asks the model to predict the digit in each.
    """
    # Split into 81 boxes
    rows = np.vsplit(imgWarpGray, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            # Crop center to remove grid lines
            box = box[4:24, 4:24] 
            box = cv2.resize(box, (28, 28))
            # Normalize pixel values (0 to 1)
            box = box / 255.0
            box = box.reshape(28, 28, 1)
            boxes.append(box)
    
    # Predict all 81 boxes at once (batch processing)
    boxes = np.array(boxes)
    preds = model.predict(boxes)
    
    numbers = []
    for i, pred in enumerate(preds):
        prob = np.amax(pred)
        # Confidence threshold: 0.7 (70%)
        if prob > 0.7:
            numbers.append(np.argmax(pred))
        else:
            numbers.append(0)
    return numbers

#### 3. SUDOKU SOLVER LOGIC ####

def solve_logic(bo):
    """
    Solves the sudoku board using backtracking.
    Returns True if solved, False if impossible.
    """
    find = find_empty(bo)
    if not find:
        return True
    else:
        row, col = find

    for i in range(1, 10):
        if valid(bo, i, (row, col)):
            bo[row][col] = i

            if solve_logic(bo):
                return True

            bo[row][col] = 0

    return False

def valid(bo, num, pos):
    # Check row
    for i in range(len(bo[0])):
        if bo[pos[0]][i] == num and pos[1] != i:
            return False
    # Check column
    for i in range(len(bo)):
        if bo[i][pos[1]] == num and pos[0] != i:
            return False
    # Check box
    box_x = pos[1] // 3
    box_y = pos[0] // 3
    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x*3, box_x*3 + 3):
            if bo[i][j] == num and (i,j) != pos:
                return False
    return True

def find_empty(bo):
    for i in range(len(bo)):
        for j in range(len(bo[0])):
            if bo[i][j] == 0:
                return (i, j)  # row, col
    return None

#### 4. OVERLAY / DRAWING ####

def displayNumbers(img, numbers, color=(0, 255, 0)):
    """
    Draws the solved numbers onto the blank image.
    """
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for x in range (0,9):
        for y in range (0,9):
            if numbers[(y*9)+x] != 0:
                 text = str(numbers[(y*9)+x])
                 
                 # Calculate text size to center it perfectly
                 (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)
                 
                 text_x = int(x * secW + (secW - text_w) / 2)
                 text_y = int(y * secH + (secH + text_h) / 2)
                 
                 cv2.putText(img, text, (text_x, text_y), 
                             cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
    return img