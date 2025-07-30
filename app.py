from flask import Flask, request, send_file
import keras_ocr
import numpy as np
import cv2
import math
import io
from PIL import Image

app = Flask(__name__)
pipeline = keras_ocr.pipeline.Pipeline()

def midpoint(x1, y1, x2, y2):
    return (int((x1 + x2)/2), int((y1 + y2)/2))

def inpaint_text(img):
    prediction_groups = pipeline.recognize([img])
    mask = np.zeros(img.shape[:2], dtype="uint8")

    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1] 
        x2, y2 = box[1][2]
        x3, y3 = box[1][3]

        mid1 = midpoint(x1, y1, x2, y2)
        mid2 = midpoint(x0, y0, x3, y3)
        thickness = int(math.sqrt((x2 - x1)**2 + (y2 - y1)**2))
        cv2.line(mask, mid1, mid2, 255, thickness)

    return cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)

@app.route('/remove-text', methods=['POST'])
def remove_text():
     try:
        if 'image' not in request.files:
            return 'No image uploaded', 400

        file = request.files['image']
        image = keras_ocr.tools.read(file.stream)
        result = inpaint_text(image)

        _, buffer = cv2.imencode('.png', result)
        return send_file(io.BytesIO(buffer), mimetype='image/png')

    except Exception as e:
        return str(e), 500

@app.route('/')
def index():
    return 'Text Removal API is running!'

if __name__ == '__main__':
    app.run()
