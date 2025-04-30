import io
import cv2
import numpy as np
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from rembg import remove
import os
app = Flask(__name__)
CORS(app)
application = app
def enhance_image(img):
    """Enhance image quality by reducing noise and sharpening."""
    img = cv2.GaussianBlur(img, (5, 5), 0)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)
    return img

def is_cartoon(img):
    """Determine if the image is a cartoon based on edge variance and color clusters."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Laplacian(gray, cv2.CV_64F).var()
    reduced = cv2.resize(img, (100, 100))
    pixels = reduced.reshape(-1, 3)
    _, _, centers = cv2.kmeans(pixels.astype(np.float32), 8, None,
                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                               10, cv2.KMEANS_RANDOM_CENTERS)
    unique_colors = len(np.unique(centers, axis=0))
    return edges > 1000 and unique_colors < 6

def is_logo(img):
    """Detect if the image is a logo based on white background ratio."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 60, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    ratio = cv2.countNonZero(mask) / (img.shape[0] * img.shape[1])
    return ratio > 0.7

def remove_background_cartoon(img):
    """Remove background from cartoon images using GrabCut."""
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    mask_initial = np.zeros(img.shape[:2], np.uint8)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.drawContours(mask_initial, [contour], -1, 255, -1)
    mask = np.where(mask_initial == 255, cv2.GC_FGD, cv2.GC_BGD).astype('uint8')
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error as e:
        print("GrabCut error:", e)
        return img
    final_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
    result = cv2.cvtColor(original, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = final_mask
    return result

def remove_background_real_world(img):
    """Use rembg to remove background from real-world images."""
    success, encoded_image = cv2.imencode('.png', img)
    if not success:
        raise ValueError("Image encoding failed")
    input_bytes = encoded_image.tobytes()
    output_bytes = remove(input_bytes)
    decoded_img = cv2.imdecode(np.frombuffer(output_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    return decoded_img

def remove_logo_background(img):
    """Remove white background from logo images."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 60, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    mask = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(img, img, mask=mask)
    result_bgra = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result_bgra[:, :, 3] = mask
    return enhance_image(result_bgra)

@app.route('/', methods=['POST'])
def process_image():
    force_cartoon = request.args.get('mode') == 'cartoon'
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        file_bytes = file.read()
        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Invalid image file'}), 400
        print("[INFO] Processing image...")
        if force_cartoon:
            print("[INFO] Forced cartoon mode.")
            result_img = remove_background_cartoon(img)
        elif is_logo(img):
            print("[INFO] Detected logo image.")
            result_img = remove_logo_background(img)
        elif is_cartoon(img):
            print("[INFO] Detected cartoon image.")
            result_img = remove_background_cartoon(img)
        else:
            print("[INFO] Detected real-world image.")
            result_img = remove_background_real_world(img)
        _, buffer = cv2.imencode('.png', result_img)
        return send_file(
            io.BytesIO(buffer.tobytes()),
            mimetype='image/png',
            as_attachment=False,
            download_name='output.png'
        )
    except Exception as e:
        print("[ERROR] Processing failed:", str(e))
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Render uses $PORT
    app.run(host='0.0.0.0', port=port)
