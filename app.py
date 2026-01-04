import base64
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model

# Import custom helper functions from utils.py
from utils import preProcess, biggestContour, reorder, solve_logic, get_board_from_image, displayNumbers

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_PATH = 'models/digit_model.h5'  # Make sure this matches your folder structure
# If your model is in the root folder, just use 'digit_model.h5'

print("Loading AI Model...")
try:
    model = load_model(MODEL_PATH)
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Make sure 'digit_model.h5' exists in the correct folder.")
    model = None # Handle this gracefully in the route

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    if model is None:
        return jsonify({'status': 'Error', 'error': 'Model not loaded on server.'})

    try:
        # 1. RECEIVE IMAGE FROM FRONTEND
        data = request.json['image']
        header, encoded = data.split(",", 1)
        data = base64.b64decode(encoded)
        np_arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # 2. IMAGE PROCESSING
        imgContour = img.copy()
        imgThreshold = preProcess(img)
        
        # Find the Grid
        contours, _ = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        biggest, maxArea = biggestContour(contours)

        status = "No Grid Found"
        
        if biggest.size != 0:
            status = "Solving..."
            biggest = reorder(biggest)
            
            # Warp Perspective (Flatten the grid)
            h, w = 450, 450
            pts1 = np.float32(biggest)
            pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgWarp = cv2.warpPerspective(img, matrix, (w, h))
            imgWarpGray = cv2.cvtColor(imgWarp, cv2.COLOR_BGR2GRAY)
            
            # 3. AI DIGIT RECOGNITION
            # We pass the 'model' here so the utils function can use it
            numbers = get_board_from_image(imgWarpGray, model)
            board = np.array(numbers).reshape(9,9).tolist()
            
            # Create a mask of where the empty slots (0s) were
            # We only want to draw the solution numbers, not overwrite the printed ones
            zeros_mask = [1 if n == 0 else 0 for n in numbers]
            
            # 4. SOLVE LOGIC
            if solve_logic(board):
                status = "Solved!"
                flat_solution = [item for sublist in board for item in sublist]
                
                # Prepare Overlay Image
                solution_overlay = np.zeros((w, h, 3), np.uint8)
                
                # Filter: Only show numbers that were originally 0
                final_display = [n if z == 1 else 0 for n, z in zip(flat_solution, zeros_mask)]
                
                solution_overlay = displayNumbers(solution_overlay, final_display)
                
                # Warp Back to Camera Perspective
                inv_matrix = cv2.getPerspectiveTransform(pts2, pts1)
                imgOverlay = cv2.warpPerspective(solution_overlay, inv_matrix, (img.shape[1], img.shape[0]))
                
                # Combine Overlay with Original Image
                img = cv2.addWeighted(imgOverlay, 1, img, 0.5, 1)
            else:
                status = "Unsolvable Grid"

        # 5. SEND RESPONSE BACK
        _, buffer = cv2.imencode('.jpg', img)
        response_img = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({'status': status, 'image': 'data:image/jpeg;base64,' + response_img})

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({'status': 'Error', 'error': str(e)})

if __name__ == '__main__':
    # Debug=True allows auto-restart when you change code
    app.run(debug=True, port=5000)