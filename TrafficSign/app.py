import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
from tensorflow import keras

app = Flask(__name__)

# Configurer les chemins locaux
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Utiliser le modèle généré s'il existe, sinon le modèle par défaut
MODEL_PATH = os.path.join(BASE_DIR, 'traffic_sign_model.h5')
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, 'traffif_sign_model.h5')

print(f"[*] Chargement du modèle depuis : {MODEL_PATH}")
model = keras.models.load_model(MODEL_PATH)
threshold = 0.75

# Fonctions de prétraitement identiques à celles du projet original
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255.0
    return img

def getClassName(classNo):
    class_names = {
        0: 'Speed Limit 20 km/h', 1: 'Speed Limit 30 km/h', 2: 'Speed Limit 50 km/h',
        3: 'Speed Limit 60 km/h', 4: 'Speed Limit 70 km/h', 5: 'Speed Limit 80 km/h',
        6: 'End of Speed Limit 80 km/h', 7: 'Speed Limit 100 km/h', 8: 'Speed Limit 120 km/h',
        9: 'No passing', 10: 'No passing for vechiles over 3.5 metric tons',
        11: 'Right-of-way at the next intersection', 12: 'Priority road', 13: 'Yield',
        14: 'Stop', 15: 'No vechiles', 16: 'Vechiles over 3.5 metric tons prohibited',
        17: 'No entry', 18: 'General caution', 19: 'Dangerous curve to the left',
        20: 'Dangerous curve to the right', 21: 'Double curve', 22: 'Bumpy road',
        23: 'Slippery road', 24: 'Road narrows on the right', 25: 'Road work',
        26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
        29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing',
        32: 'End of all speed and passing limits', 33: 'Turn right ahead',
        34: 'Turn left ahead', 35: 'Ahead only', 36: 'Go straight or right',
        37: 'Go straight or left', 38: 'Keep right', 39: 'Keep left',
        40: 'Roundabout mandatory', 41: 'End of no passing',
        42: 'End of no passing by vechiles over 3.5 metric tons'
    }
    return class_names.get(classNo, "Unknown")

# Algorithme original de détection de contours (rouge/bleu)
def preprocess_detection_img(imgBGR):
    imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)
    Bmin = np.array([100, 43, 46])
    Bmax = np.array([124, 255, 255])
    img_Bbin = cv2.inRange(imgHSV, Bmin, Bmax)

    Rmin1 = np.array([0, 43, 46])
    Rmax1 = np.array([10, 255, 255])
    img_Rbin1 = cv2.inRange(imgHSV, Rmin1, Rmax1)

    Rmin2 = np.array([156, 43, 46])
    Rmax2 = np.array([180, 255, 255])
    img_Rbin2 = cv2.inRange(imgHSV, Rmin2, Rmax2)
    img_Rbin = np.maximum(img_Rbin1, img_Rbin2)
    img_bin = np.maximum(img_Bbin, img_Rbin)

    kernelErosion = np.ones((3, 3), np.uint8)
    kernelDilation = np.ones((3, 3), np.uint8)
    img_bin = cv2.erode(img_bin, kernelErosion, iterations=2)
    img_bin = cv2.dilate(img_bin, kernelDilation, iterations=2)
    return img_bin

def contour_detect(img_bin, min_area, wh_ratio=2.0):
    rects = []
    contours, _ = cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(contour)
            if 1.0 * w / h < wh_ratio and 1.0 * h / w < wh_ratio:
                rects.append([x, y, w, h])
    return rects

# Générateur vidéo Flask
def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error] Impossible d'accéder à la webcam.")
        return

    cols = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rows = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        success, img = cap.read()
        if not success:
            break
        else:
            img_bin = preprocess_detection_img(img)
            min_area = img_bin.shape[0] * img.shape[1] / (25 * 25)
            rects = contour_detect(img_bin, min_area=min_area)
            img_bbx = img.copy()

            for rect in rects:
                xc = int(rect[0] + rect[2] / 2)
                yc = int(rect[1] + rect[3] / 2)
                size = max(rect[2], rect[3])
                x1 = max(0, int(xc - size / 2))
                y1 = max(0, int(yc - size / 2))
                x2 = min(cols, int(xc + size / 2))
                y2 = min(rows, int(yc + size / 2))

                if rect[2] > 60 and rect[3] > 60:  # Sensibilité améliorée à 60px
                    cv2.rectangle(img_bbx, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 2)
                    crop_img = np.asarray(img[y1:y2, x1:x2])
                    if crop_img.size > 0:
                        crop_img = cv2.resize(crop_img, (32, 32))
                        crop_img = preprocessing(crop_img)
                        crop_img = crop_img.reshape(1, 32, 32, 1)
                        predictions = model.predict(crop_img, verbose=0)
                        classIndex = int(np.argmax(predictions, axis=-1)[0])
                        probabilityValue = np.amax(predictions)

                        if probabilityValue > threshold:
                            label_str = f"{getClassName(classIndex)} ({round(probabilityValue*100, 1)}%)"
                            cv2.putText(img_bbx, label_str, (rect[0], rect[1] - 10),
                                        font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', img_bbx)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Lire l'image
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        # Prétraitement
        resized = cv2.resize(img, (32, 32))
        processed = preprocessing(resized)
        processed = processed.reshape(1, 32, 32, 1)
        
        # Prédiction
        predictions = model.predict(processed, verbose=0)
        classIndex = int(np.argmax(predictions, axis=-1)[0])
        probabilityValue = float(np.amax(predictions))
        
        return jsonify({
            'success': True,
            'class_index': classIndex,
            'class_name': getClassName(classIndex),
            'probability': probabilityValue
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("[*] Serveur de test Web lancé sur http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
